import logging
import os
import io
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoConfig,
    AutoModelForVision2Seq,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_image_for_qwen(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
        raise ValueError("Unsupported image dict without bytes/path")
    if isinstance(image, (str, bytes, os.PathLike)):
        return Image.open(image).convert("RGB")
    return image

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to model"})
    trust_remote_code: bool = field(default=True)


@dataclass
class DataArguments:
    dataset_path: str = field(metadata={"help": "Path to dataset"})
    max_pixels: int = field(default=1003520)
    eval_steps_callback: int = field(default=20)
    train_split: str = field(default="train")


class Qwen2_5_VLBinaryClassifier(nn.Module):
    def __init__(self, model_name_or_path, torch_dtype):
        super().__init__()
        self.base_model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        hidden_size = self.base_model.config.hidden_size
        self.base_model.lm_head = nn.Linear(hidden_size, 2, bias=False).to(torch_dtype)
        self.original_vocab_size = self.base_model.config.vocab_size
        self.base_model.config.num_labels = 2
        self.base_model.config.problem_type = "single_label_classification"
        self.config = self.base_model.config
        for name, param in self.base_model.named_parameters():
            param.requires_grad = "lm_head" in name
        self._processor = None

    def forward(self, input_ids, pixel_values, image_grid_thw, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = outputs.hidden_states[-1]
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
        pooled_logits = self.base_model.lm_head(last_hidden_state[torch.arange(last_hidden_state.shape[0]), sequence_lengths])
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(pooled_logits, labels)
        return {"loss": loss, "logits": pooled_logits}

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def save_pretrained(self, output_dir, processor=None):
        import json

        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "original_vocab_size": self.original_vocab_size,
                "num_labels": 2,
                "model_type": "binary_classifier",
            },
            os.path.join(output_dir, "pytorch_model.bin"),
        )
        self.config.save_pretrained(output_dir)
        proc = processor or self._processor
        if proc is not None:
            proc.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "model_info.json"), "w") as f:
            json.dump(
                {
                    "architecture": "Qwen2_5_VLBinaryClassifier",
                    "base_model": self.config._name_or_path,
                    "num_labels": 2,
                    "classifier_type": "sequence_classification",
                    "trainable_params": "~4K (lm_head only)",
                },
                f,
                indent=2,
            )

    @classmethod
    def from_pretrained(cls, model_path, torch_dtype=torch.bfloat16):
        ckpt = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
        config = AutoConfig.from_pretrained(model_path)
        model = cls(config._name_or_path, torch_dtype)
        model.load_state_dict(ckpt["model_state_dict"])
        return model


def preprocess_function(examples, processor):
    batch_input_ids, batch_labels, batch_pixel_values, batch_image_grid_thw = [], [], [], []
    for i in range(len(examples["question"])):
        image = _normalize_image_for_qwen(examples["image"][i])
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": examples["question"][i]}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt")
        batch_input_ids.append(inputs["input_ids"][0])
        batch_labels.append(1 if examples["label"][i] == "block" else 0)
        batch_pixel_values.append(inputs["pixel_values"])
        batch_image_grid_thw.append(inputs["image_grid_thw"])
    return {
        "input_ids": batch_input_ids,
        "labels": batch_labels,
        "pixel_values": batch_pixel_values,
        "image_grid_thw": batch_image_grid_thw,
    }


class ClsDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_ids = [torch.as_tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
            "labels": labels,
            "pixel_values": torch.cat([torch.as_tensor(f["pixel_values"]) for f in features], dim=0),
            "image_grid_thw": torch.cat([torch.as_tensor(f["image_grid_thw"]) for f in features], dim=0),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [torch.ones(ids.size(0), dtype=torch.long) for ids in input_ids],
                batch_first=True,
                padding_value=0,
            ).bool(),
        }


def load_training_split(dataset_path: str, split_name: str):
    split_dir = os.path.join(dataset_path, split_name)
    split_parquet = os.path.join(dataset_path, f"{split_name}.parquet")
    if os.path.isdir(split_dir) and os.path.exists(os.path.join(split_dir, "dataset_info.json")):
        return {"train": load_from_disk(split_dir)}
    if os.path.exists(split_parquet):
        return load_dataset("parquet", data_files={"train": split_parquet})
    train_dir = os.path.join(dataset_path, "train")
    train_parquet = os.path.join(dataset_path, "train.parquet")
    if os.path.isdir(train_dir) and os.path.exists(os.path.join(train_dir, "dataset_info.json")):
        return {"train": load_from_disk(train_dir)}
    if os.path.exists(train_parquet):
        return load_dataset("parquet", data_files={"train": train_parquet})
    return load_dataset("parquet", data_files={"train": os.path.join(dataset_path, "data/train-*.parquet")})


def run_training(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments) -> str:
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, max_pixels=data_args.max_pixels)
    model = Qwen2_5_VLBinaryClassifier(model_args.model_name_or_path, torch_dtype=torch.bfloat16)
    model._processor = processor
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable parameter: {name}")

    ds = load_training_split(data_args.dataset_path, data_args.train_split)
    train_dataset = ds["train"].map(
        lambda x: preprocess_function(x, processor),
        batched=True,
        batch_size=4,
        remove_columns=ds["train"].column_names,
    )

    callbacks = []
    try:
        from vllm_guard.training.eval_callback import EvaluationCallback

        callbacks.append(EvaluationCallback(eval_steps=data_args.eval_steps_callback, processor=processor, model_type="cls"))
    except Exception as exc:
        logger.warning(f"Could not load EvaluationCallback: {exc}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=ClsDataCollator(processor),
        callbacks=callbacks if callbacks else None,
    )

    checkpoint = get_last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=checkpoint)
    output_dir = os.path.join(training_args.output_dir, "final_model")
    model.save_pretrained(output_dir, processor=processor)
    logger.info(f"Model saved to {output_dir}")
    return output_dir


def parse_hf_train_args(argv: Optional[Sequence[str]] = None):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    return parser.parse_args_into_dataclasses(args=list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    model_args, data_args, training_args = parse_hf_train_args(argv)
    run_training(model_args, data_args, training_args)


if __name__ == "__main__":
    main()
