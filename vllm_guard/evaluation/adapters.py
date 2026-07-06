from pathlib import Path
from typing import Any

from vllm_guard.evaluation.benchmarks import get_benchmark_spec
from vllm_guard.evaluation.model_registry import get_model_info, resolve_model_path


class VLLMAdapter:
    def __init__(self, config):
        from vllm import LLM, SamplingParams

        model_path = config.model_path
        if not model_path:
            try:
                resolved_path, _, _ = resolve_model_path(config.model_name)
                model_path = resolved_path
            except ValueError:
                model_path = config.model_name
        tokenizer_path = config.tokenizer_path or model_path
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=config.vllm_tensor_parallel,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=config.vllm_max_model_len,
            max_num_seqs=max(1, config.batch_size),
            limit_mm_per_prompt={"image": 1},
            load_format="auto",
        )
        self.sampling_params = SamplingParams(temperature=config.temperature, max_tokens=config.max_tokens)
        self.tokenizer = self.llm.get_tokenizer()
        self.enable_thinking = config.enable_thinking

    def _build_prompt(self, text: str) -> str:
        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        try:
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except Exception:
            messages_plain = [{"role": "user", "content": text}]
            prompt_text = self.tokenizer.apply_chat_template(messages_plain, **template_kwargs)
            img_token = "<|vision_start|><|image_pad|><|vision_end|>"
            if img_token not in prompt_text:
                prompt_text = prompt_text.replace(text, img_token + "\n" + text, 1)
            return prompt_text

    def generate(self, prompts_and_images):
        inputs = [{"prompt": self._build_prompt(item["text"]), "multi_modal_data": {"image": item["image"]}} for item in prompts_and_images]
        outputs = self.llm.generate(inputs, self.sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]


class _HFImageTextAdapter:
    model_cls_name = "AutoModelForImageTextToText"
    min_new_tokens = 64
    use_cache = True
    cache_implementation = "__unchanged__"
    attention_chunk_size = None

    def __init__(self, config):
        import torch
        import transformers
        from transformers import AutoProcessor

        model_path = config.model_path or config.model_name
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_cls = getattr(transformers, self.model_cls_name)
        self.model = model_cls.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={"": 0},
            trust_remote_code=True,
        ).eval()
        if self.cache_implementation != "__unchanged__":
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.cache_implementation = self.cache_implementation
            text_config = getattr(getattr(self.model, "config", None), "text_config", None)
            if text_config is not None and hasattr(text_config, "cache_implementation"):
                text_config.cache_implementation = self.cache_implementation
        if self.attention_chunk_size is not None:
            text_config = getattr(getattr(self.model, "config", None), "text_config", None)
            if text_config is not None and getattr(text_config, "attention_chunk_size", None) is None:
                text_config.attention_chunk_size = self.attention_chunk_size
            if hasattr(self.model.config, "attention_chunk_size") and getattr(self.model.config, "attention_chunk_size", None) is None:
                self.model.config.attention_chunk_size = self.attention_chunk_size
        self.max_new_tokens = max(int(config.max_tokens), self.min_new_tokens)
        self.do_sample = float(config.temperature) > 0.0
        self.temperature = max(float(config.temperature), 1e-5)

    def _device(self):
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.model.parameters()).device

    def _batch_generate(self, items: list[dict[str, Any]]) -> list[str]:
        import torch

        prompts, images = [], []
        for item in items:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": item["text"]},
                    ],
                }
            ]
            prompts.append(self.processor.apply_chat_template(conversation, add_generation_prompt=True))
            images.append(item["image"])

        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        device = self._device()
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "use_cache": self.use_cache,
        }
        if self.do_sample:
            gen_kwargs.update({"temperature": self.temperature, "top_p": 0.95, "top_k": 50})
        pad_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
        eos_id = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None)
        if pad_id is not None or eos_id is not None:
            gen_kwargs["pad_token_id"] = pad_id if pad_id is not None else eos_id

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        return [x.strip() for x in self.processor.batch_decode(generated, skip_special_tokens=True)]

    def generate(self, prompts_and_images):
        if not prompts_and_images:
            return []
        try:
            return self._batch_generate(prompts_and_images)
        except Exception as exc:
            print(f"  WARN: HF image-text batch inference failed ({exc}); retry one-by-one.")
            outputs = []
            for item in prompts_and_images:
                try:
                    outputs.extend(self._batch_generate([item]))
                except Exception as item_exc:
                    print(f"  WARN: HF image-text single-sample inference failed: {item_exc}")
                    outputs.append("")
            return outputs


class LLaVAGuardAdapter(_HFImageTextAdapter):
    model_cls_name = "LlavaOnevisionForConditionalGeneration"


class LlamaGuard4Adapter(_HFImageTextAdapter):
    model_cls_name = "Llama4ForConditionalGeneration"
    use_cache = False
    cache_implementation = None
    attention_chunk_size = 8192


class ClsAdapter:
    def __init__(self, config):
        import torch
        from transformers import AutoProcessor

        from qwen_vl_utils import process_vision_info
        from vllm_guard.training.cls.pipeline import Qwen2_5_VLBinaryClassifier

        model_path = config.model_path or config.model_name
        self.model = Qwen2_5_VLBinaryClassifier.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.process_vision_info = process_vision_info

    def generate(self, prompts_and_images):
        import torch

        responses = []
        output_instructions = get_benchmark_spec().output_instructions
        for item in prompts_and_images:
            text = item["text"].replace(output_instructions, "").strip()
            image = item["image"]
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
            prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = self.process_vision_info(messages)
            inputs = self.processor(text=[prompt_text], images=image_inputs, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                pred_label = torch.argmax(self.model(**inputs)["logits"], dim=-1).item()
            responses.append("true" if pred_label == 1 else "false")
        return responses


def create_adapter(config):
    if not config.model_type:
        config.model_type = get_model_info(config.model_name)["type"]
    if config.model_type == "vllm":
        return VLLMAdapter(config)
    if config.model_type == "llavaguard":
        return LLaVAGuardAdapter(config)
    if config.model_type == "llamaguard4":
        return LlamaGuard4Adapter(config)
    if config.model_type == "cls":
        return ClsAdapter(config)
    raise ValueError(f"Unknown model_type: {config.model_type}")
