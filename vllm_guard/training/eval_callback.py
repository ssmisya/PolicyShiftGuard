import json
import time
import io
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from vllm_guard.common.constants import CANONICAL_DATASET_DIR
from vllm_guard.evaluation.metrics import compute_all_metrics
from vllm_guard.evaluation.miniset import create_minisets
from vllm_guard.evaluation.parsing import parse_predicted_categories, parse_response
from vllm_guard.evaluation.prompts import resolve_output_instructions


class EvaluationCallback(TrainerCallback):
    """Run lightweight miniset eval during training and log results to wandb."""

    def __init__(
        self,
        eval_steps: int = 50,
        miniset_dir: str | None = None,
        processor=None,
        model_type: str = "sft",
        max_new_tokens: int | None = None,
        response_format: str = "nothink",
        wandb_run=None,
    ):
        self.eval_steps = eval_steps
        self.miniset_dir = str(miniset_dir or CANONICAL_DATASET_DIR)
        self.processor = processor
        self.model_type = model_type
        self.response_format = response_format
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else (5 if model_type == "sft" else 2)
        self.output_instructions = resolve_output_instructions(response_format)
        self._wandb_run = wandb_run
        self.id_test = None
        self.ood_test = None
        self._load_minisets()

    @property
    def wandb_run(self):
        if self._wandb_run is not None:
            return self._wandb_run
        try:
            import wandb

            return wandb.run
        except Exception:
            return None

    def _load_minisets(self) -> None:
        create_minisets(self.miniset_dir, sample_size=100, seed=42)
        id_path = Path(self.miniset_dir) / "id_test_mini.parquet"
        ood_path = Path(self.miniset_dir) / "ood_test_mini.parquet"
        self.id_test = load_dataset("parquet", data_files=str(id_path), split="train")
        self.ood_test = load_dataset("parquet", data_files=str(ood_path), split="train")

    def _prepare_input(self, example, model):
        from qwen_vl_utils import process_vision_info

        image = example["image"]
        if not isinstance(image, Image.Image):
            if isinstance(image, dict):
                if image.get("bytes") is not None:
                    image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
                elif image.get("path"):
                    image = Image.open(image["path"]).convert("RGB")
                else:
                    raise ValueError("Unsupported image dict without bytes/path")
            else:
                image = Image.open(image).convert("RGB")

        prompt_text = f"{example['question']}{self.output_instructions}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt")
        device = next(model.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}

    def _generate_response(self, model, inputs):
        if self.model_type == "cls":
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs["logits"]
                pred_label = torch.argmax(logits, dim=-1).item()
                return "true" if pred_label == 1 else "false"

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0][input_len:]
            return self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _parse_response(self, response: str) -> tuple[str, list[int]]:
        prediction = parse_response(response, response_format=self.response_format)
        categories = parse_predicted_categories(response, response_format=self.response_format)
        return prediction, categories

    def _evaluate_split(self, model, dataset, split_name: str) -> dict[str, Any]:
        model.eval()
        results = []
        started = time.time()
        for example in dataset:
            try:
                inputs = self._prepare_input(example, model)
                response = self._generate_response(model, inputs)
                prediction, pred_cats = self._parse_response(response)
                results.append(
                    {
                        "image_idx": int(example["image_idx"]),
                        "section_id": int(example["section_id"]),
                        "section_title": example["section_title"],
                        "policy_name": example["policy_name"],
                        "tier": example.get("tier", ""),
                        "label": "unsafe" if example["label"] == "block" else "safe",
                        "prediction": prediction,
                        "raw_response": response,
                        "violated_categories": example.get("violated_categories", []),
                        "predicted_categories": pred_cats,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "image_idx": int(example["image_idx"]),
                        "section_id": int(example["section_id"]),
                        "section_title": example["section_title"],
                        "policy_name": example["policy_name"],
                        "tier": example.get("tier", ""),
                        "label": "unsafe" if example["label"] == "block" else "safe",
                        "prediction": "invalid",
                        "raw_response": f"ERROR: {exc}",
                        "violated_categories": example.get("violated_categories", []),
                        "predicted_categories": [],
                    }
                )

        metrics = compute_all_metrics(results)
        metrics["overall"]["eval_time"] = time.time() - started
        metrics["overall"]["split"] = split_name
        model.train()
        return {"metrics": metrics, "results": results}

    def _write_report(self, args: TrainingArguments, state: TrainerState, payload: dict[str, Any]) -> None:
        output_dir = Path(args.output_dir) / "miniset_eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"step_{state.global_step:06d}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero:
            return control
        if self.eval_steps <= 0 or state.global_step <= 0 or state.global_step % self.eval_steps != 0:
            return control

        model = kwargs.get("model")
        if model is None or self.id_test is None or self.ood_test is None:
            return control
        if hasattr(model, "module"):
            model = model.module

        id_payload = self._evaluate_split(model, self.id_test, "id_test_mini")
        ood_payload = self._evaluate_split(model, self.ood_test, "ood_test_mini")

        summary = {
            "step": state.global_step,
            "epoch": state.epoch,
            "id_test_mini": id_payload["metrics"]["overall"],
            "ood_test_mini": ood_payload["metrics"]["overall"],
        }
        self._write_report(args, state, summary)

        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "train/step": state.global_step,
                    "train/epoch": state.epoch,
                    "miniset/id_accuracy": summary["id_test_mini"]["accuracy"],
                    "miniset/id_pca_disagree": summary["id_test_mini"].get("pca_disagree", 0.0),
                    "miniset/id_pss": summary["id_test_mini"].get("pss", 0.0),
                    "miniset/ood_accuracy": summary["ood_test_mini"]["accuracy"],
                    "miniset/ood_pca_disagree": summary["ood_test_mini"].get("pca_disagree", 0.0),
                    "miniset/ood_pss": summary["ood_test_mini"].get("pss", 0.0),
                    "miniset/id_invalid_rate": summary["id_test_mini"].get("invalid_rate", 0.0),
                    "miniset/ood_invalid_rate": summary["ood_test_mini"].get("invalid_rate", 0.0),
                }
            )
        return control
