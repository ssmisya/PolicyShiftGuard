import logging
import os
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from vllm_guard.training.formatting import (
    build_output_instructions,
    build_supervised_answer,
    randomize_policy_descriptions,
)


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
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return image


class ProcessorCheckpointCallback(TrainerCallback):
    def __init__(self, processor, output_dir: str):
        self.processor = processor
        self.output_dir = output_dir

    def _save_processor_assets(self, checkpoint_dir: str) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.processor.save_pretrained(checkpoint_dir)
        logger.info(f"Processor assets saved to {checkpoint_dir}")

    def on_save(self, args, state, control, **kwargs):
        if state.global_step is None:
            return control
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        self._save_processor_assets(checkpoint_dir)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._save_processor_assets(self.output_dir)
        return control


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to model"})
    trust_remote_code: bool = field(default=True)


@dataclass
class DataArguments:
    dataset_path: str = field(metadata={"help": "Path to dataset"})
    max_pixels: int = field(default=1003520)
    no_reason: bool = field(default=False, metadata={"help": "If True, omit reason from answer"})
    use_think_tags: bool = field(default=False, metadata={"help": "If True, train on '<think>reason</think> answer' format"})
    eval_steps_callback: int = field(default=50, metadata={"help": "Run miniset evaluation every N training steps"})
    train_split: str = field(default="train", metadata={"help": "Split name to use for training (e.g. 'sft', 'sft_think')"})
    policy_rephrase_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional policy rephrase JSON. If set, policy descriptions in prompts are randomized."},
    )
    policy_rephrase_seed: int = field(default=0, metadata={"help": "Seed for deterministic policy rephrase selection"})
    sft_loss_mode: str = field(default="ce", metadata={"help": "SFT loss mode: 'ce' or 'pair_contrast'."})
    pair_contrast_label_weight: float = field(default=0.1, metadata={"help": "Weight for per-sample true/false margin loss."})
    pair_contrast_pair_weight: float = field(default=0.2, metadata={"help": "Weight for block/pass pair margin loss."})
    pair_contrast_category_weight: float = field(default=0.0, metadata={"help": "Weight for block-side category margin loss."})
    pair_contrast_label_margin: float = field(default=0.5, metadata={"help": "Margin for per-sample label contrast."})
    pair_contrast_pair_margin: float = field(default=0.5, metadata={"help": "Margin for block/pass pair contrast."})
    pair_contrast_category_margin: float = field(default=0.5, metadata={"help": "Margin for category contrast."})
    pair_contrast_category_ids: str = field(
        default="01,02,03,04,05,06,07",
        metadata={"help": "Comma-separated category ids used as negative candidates for category contrast."},
    )


def _find_subsequence(sequence: Sequence[int], pattern: Sequence[int], start: int = 0) -> int:
    if not pattern:
        return -1
    max_start = len(sequence) - len(pattern)
    for pos in range(max(0, start), max_start + 1):
        if list(sequence[pos : pos + len(pattern)]) == list(pattern):
            return pos
    return -1


def _first_supervised_position(labels: torch.Tensor) -> int:
    positions = torch.nonzero(labels != -100, as_tuple=False)
    if positions.numel() == 0:
        return -1
    return int(positions[0].item())


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _first_category_id(*, violated_categories, section_id) -> str:
    if violated_categories:
        try:
            return f"{int(violated_categories[0]):02d}"
        except Exception:
            pass
    return f"{_safe_int(section_id):02d}"


def preprocess_function(
    examples,
    processor,
    *,
    no_reason: bool = False,
    use_think_tags: bool = False,
    policy_rephrase_path: Optional[str] = None,
    policy_rephrase_seed: int = 0,
    include_pair_metadata: bool = False,
):
    batch_input_ids = []
    batch_labels = []
    batch_pixel_values = []
    batch_image_grid_thw = []
    batch_boundary_group_ids = []
    batch_boundary_pair_roles = []
    batch_decision_token_positions = []
    batch_category_token_positions = []
    batch_target_category_token_ids = []

    def get_optional_value(name, index, default=""):
        values = examples.get(name)
        if values is None or index >= len(values):
            return default
        value = values[index]
        return default if value is None else value

    has_target_text = "target_text" in examples and any(str(x).strip() for x in examples.get("target_text", []))
    output_instructions = build_output_instructions(no_reason=no_reason, use_think_tags=use_think_tags)

    for i in range(len(examples["question"])):
        image = _normalize_image_for_qwen(examples["image"][i])
        example_key = "|".join(
            str(get_optional_value(name, i, ""))
            for name in ("image_idx", "section_id", "policy_name")
        )
        question = randomize_policy_descriptions(
            examples["question"][i],
            rephrase_path=policy_rephrase_path,
            seed=policy_rephrase_seed,
            example_key=example_key,
        )
        instruction = question + output_instructions
        ans_str = build_supervised_answer(
            label=examples["label"][i],
            violated_categories=get_optional_value("violated_categories", i, []),
            section_id=examples["section_id"][i],
            reason=get_optional_value("reason", i, ""),
            target_text=get_optional_value("target_text", i, "") if has_target_text else None,
            no_reason=no_reason,
            use_think_tags=use_think_tags,
        )

        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
            {"role": "assistant", "content": [{"type": "text", "text": ans_str}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        header_ids = processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        header_tensor = torch.tensor(header_ids)
        found = False
        for j in range(len(input_ids) - len(header_ids)):
            if torch.equal(input_ids[j:j + len(header_ids)], header_tensor):
                labels[: j + len(header_ids) + 1] = -100
                found = True
                break
        if not found:
            labels[:-5] = -100

        if include_pair_metadata:
            input_id_list = input_ids.tolist()
            response_start = _first_supervised_position(labels)
            label_value = str(examples["label"][i])
            decision_text = "true" if label_value == "block" else "false"
            decision_ids = processor.tokenizer.encode(decision_text, add_special_tokens=False)
            decision_pos = _find_subsequence(input_id_list, decision_ids, start=max(response_start, 0))
            if decision_pos < 0:
                decision_pos = response_start

            category_pos = -1
            target_category_token_id = -1
            if label_value == "block":
                category_id = _first_category_id(
                    violated_categories=get_optional_value("violated_categories", i, []),
                    section_id=get_optional_value("section_id", i, 0),
                )
                category_ids = processor.tokenizer.encode(category_id, add_special_tokens=False)
                category_start = _find_subsequence(input_id_list, category_ids, start=max(decision_pos, 0))
                if category_start >= 0:
                    category_pos = category_start + len(category_ids) - 1
                    target_category_token_id = int(category_ids[-1])

            batch_boundary_group_ids.append(str(get_optional_value("boundary_group_id", i, "")))
            batch_boundary_pair_roles.append(str(get_optional_value("boundary_pair_role", i, "")))
            batch_decision_token_positions.append(int(decision_pos))
            batch_category_token_positions.append(int(category_pos))
            batch_target_category_token_ids.append(int(target_category_token_id))

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_pixel_values.append(inputs["pixel_values"])
        batch_image_grid_thw.append(inputs["image_grid_thw"])

    output = {
        "input_ids": batch_input_ids,
        "labels": batch_labels,
        "pixel_values": batch_pixel_values,
        "image_grid_thw": batch_image_grid_thw,
    }
    if include_pair_metadata:
        output.update(
            {
                "boundary_group_id": batch_boundary_group_ids,
                "boundary_pair_role": batch_boundary_pair_roles,
                "decision_token_position": batch_decision_token_positions,
                "category_token_position": batch_category_token_positions,
                "target_category_token_id": batch_target_category_token_ids,
            }
        )
    return output


class VLMDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_ids = [torch.as_tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.as_tensor(f["labels"], dtype=torch.long) for f in features]
        pixel_values = [torch.as_tensor(f["pixel_values"]) for f in features]
        image_grid_thw = [torch.as_tensor(f["image_grid_thw"]) for f in features]
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_thw": torch.cat(image_grid_thw, dim=0),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [torch.ones(ids.size(0), dtype=torch.long) for ids in input_ids],
                batch_first=True,
                padding_value=0,
            ).bool(),
        }


class PairContrastiveDataset(Dataset):
    """Expose one boundary group per item while reusing preprocessed row features."""

    def __init__(self, dataset):
        self.dataset = dataset
        by_group: dict[str, dict[str, int]] = {}
        try:
            group_ids = dataset["boundary_group_id"]
            roles = dataset["boundary_pair_role"]
        except Exception:
            # Fallback for Dataset-like objects without efficient column access.
            group_ids = []
            roles = []
            for idx in range(len(dataset)):
                row = dataset[idx]
                group_ids.append(row.get("boundary_group_id", ""))
                roles.append(row.get("boundary_pair_role", ""))

        for idx, (group_id_raw, role_raw) in enumerate(zip(group_ids, roles)):
            group_id = str(group_id_raw).strip()
            role = str(role_raw).strip().lower()
            if not group_id or role not in {"block", "pass"}:
                continue
            by_group.setdefault(group_id, {})[role] = idx
        self.pair_indices = [
            (roles["block"], roles["pass"])
            for _, roles in sorted(by_group.items())
            if "block" in roles and "pass" in roles
        ]
        if not self.pair_indices:
            raise ValueError("pair_contrast mode requires boundary_group_id with both block/pass rows")
        logger.info("PairContrastiveDataset built %d block/pass pairs from %d rows", len(self.pair_indices), len(dataset))

    def __len__(self) -> int:
        return len(self.pair_indices)

    def __getitem__(self, index: int):
        block_idx, pass_idx = self.pair_indices[index]
        return {"block": self.dataset[block_idx], "pass": self.dataset[pass_idx]}


class PairContrastiveDataCollator(VLMDataCollator):
    def __call__(self, features):
        flat_features = []
        pair_indices = []
        pair_roles = []
        decision_positions = []
        category_positions = []
        target_category_token_ids = []

        for feature in features:
            start = len(flat_features)
            block = feature["block"]
            passed = feature["pass"]
            flat_features.extend([block, passed])
            pair_indices.append([start, start + 1])
            pair_roles.extend([1, 0])
            for item in (block, passed):
                decision_positions.append(int(item.get("decision_token_position", -1)))
                category_positions.append(int(item.get("category_token_position", -1)))
                target_category_token_ids.append(int(item.get("target_category_token_id", -1)))

        batch = super().__call__(flat_features)
        batch.update(
            {
                "pair_indices": torch.tensor(pair_indices, dtype=torch.long),
                "pair_roles": torch.tensor(pair_roles, dtype=torch.long),
                "decision_token_positions": torch.tensor(decision_positions, dtype=torch.long),
                "category_token_positions": torch.tensor(category_positions, dtype=torch.long),
                "target_category_token_ids": torch.tensor(target_category_token_ids, dtype=torch.long),
            }
        )
        return batch


class PairContrastiveTrainer(Trainer):
    def __init__(
        self,
        *args,
        true_token_id: int,
        false_token_id: int,
        category_token_ids: Sequence[int],
        label_weight: float,
        pair_weight: float,
        category_weight: float,
        label_margin: float,
        pair_margin: float,
        category_margin: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.true_token_id = int(true_token_id)
        self.false_token_id = int(false_token_id)
        self.category_token_ids = tuple(int(x) for x in category_token_ids)
        self.label_weight = float(label_weight)
        self.pair_weight = float(pair_weight)
        self.category_weight = float(category_weight)
        self.label_margin = float(label_margin)
        self.pair_margin = float(pair_margin)
        self.category_margin = float(category_margin)

    @staticmethod
    def _zero_like(loss: torch.Tensor) -> torch.Tensor:
        return loss.new_zeros(())

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pair_indices = inputs.pop("pair_indices")
        pair_roles = inputs.pop("pair_roles")
        decision_positions = inputs.pop("decision_token_positions")
        category_positions = inputs.pop("category_token_positions")
        target_category_token_ids = inputs.pop("target_category_token_ids")

        outputs = model(**inputs)
        ce_loss = outputs.loss
        logits = outputs.logits.float()
        device = logits.device
        batch_size = logits.size(0)
        batch_indices = torch.arange(batch_size, device=device)

        decision_positions = decision_positions.to(device)
        valid_decision = decision_positions > 0
        safe_decision_positions = decision_positions.clamp(min=1)
        decision_logits = logits[batch_indices, safe_decision_positions - 1]
        true_scores = decision_logits[:, self.true_token_id]
        false_scores = decision_logits[:, self.false_token_id]

        pair_roles = pair_roles.to(device)
        block_mask = (pair_roles == 1) & valid_decision
        pass_mask = (pair_roles == 0) & valid_decision
        label_losses = []
        if block_mask.any():
            label_losses.append(F.relu(self.label_margin - (true_scores[block_mask] - false_scores[block_mask])))
        if pass_mask.any():
            label_losses.append(F.relu(self.label_margin - (false_scores[pass_mask] - true_scores[pass_mask])))
        label_loss = torch.cat(label_losses).mean() if label_losses else self._zero_like(ce_loss)

        pair_indices = pair_indices.to(device)
        block_indices = pair_indices[:, 0]
        pass_indices = pair_indices[:, 1]
        valid_pairs = valid_decision[block_indices] & valid_decision[pass_indices]
        if valid_pairs.any():
            block_indices = block_indices[valid_pairs]
            pass_indices = pass_indices[valid_pairs]
            true_pair_loss = F.relu(self.pair_margin - (true_scores[block_indices] - true_scores[pass_indices]))
            false_pair_loss = F.relu(self.pair_margin - (false_scores[pass_indices] - false_scores[block_indices]))
            pair_loss = 0.5 * (true_pair_loss + false_pair_loss).mean()
        else:
            pair_loss = self._zero_like(ce_loss)

        category_loss = self._zero_like(ce_loss)
        if self.category_weight > 0 and self.category_token_ids:
            category_positions = category_positions.to(device)
            target_category_token_ids = target_category_token_ids.to(device)
            valid_category = (pair_roles == 1) & (category_positions > 0) & (target_category_token_ids >= 0)
            if valid_category.any():
                category_batch = batch_indices[valid_category]
                category_logits = logits[category_batch, category_positions[valid_category] - 1]
                candidate_ids = torch.tensor(self.category_token_ids, dtype=torch.long, device=device)
                candidate_scores = category_logits.index_select(dim=-1, index=candidate_ids)
                target_scores = category_logits.gather(dim=-1, index=target_category_token_ids[valid_category].unsqueeze(-1)).squeeze(-1)
                target_token_matches = candidate_ids.unsqueeze(0) == target_category_token_ids[valid_category].unsqueeze(-1)
                other_scores = candidate_scores.masked_fill(target_token_matches, torch.finfo(candidate_scores.dtype).min)
                hardest_other = other_scores.max(dim=-1).values
                category_loss = F.relu(self.category_margin - (target_scores - hardest_other)).mean()

        loss = ce_loss
        if self.label_weight:
            loss = loss + self.label_weight * label_loss
        if self.pair_weight:
            loss = loss + self.pair_weight * pair_loss
        if self.category_weight:
            loss = loss + self.category_weight * category_loss

        if return_outputs:
            return loss, outputs
        return loss


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


def create_trainer(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments) -> Trainer:
    loss_mode = data_args.sft_loss_mode.strip().lower()
    if loss_mode not in {"ce", "pair_contrast"}:
        raise ValueError(f"Unsupported sft_loss_mode={data_args.sft_loss_mode!r}; expected 'ce' or 'pair_contrast'")
    if loss_mode == "pair_contrast" and (not data_args.no_reason or data_args.use_think_tags):
        raise ValueError("pair_contrast SFT is currently only supported for no-think training")

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, max_pixels=data_args.max_pixels)
    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ds = load_training_split(data_args.dataset_path, data_args.train_split)
    train_dataset = ds["train"].map(
        lambda x: preprocess_function(
            x,
            processor,
            no_reason=data_args.no_reason,
            use_think_tags=data_args.use_think_tags,
            policy_rephrase_path=data_args.policy_rephrase_path,
            policy_rephrase_seed=data_args.policy_rephrase_seed,
            include_pair_metadata=loss_mode == "pair_contrast",
        ),
        batched=True,
        batch_size=4,
        remove_columns=ds["train"].column_names,
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    data_collator = VLMDataCollator(processor)
    trainer_cls = Trainer
    trainer_kwargs = {}
    if loss_mode == "pair_contrast":
        train_dataset = PairContrastiveDataset(train_dataset)
        data_collator = PairContrastiveDataCollator(processor)
        training_args.remove_unused_columns = False
        category_ids = [x.strip() for x in data_args.pair_contrast_category_ids.split(",") if x.strip()]
        category_token_ids = [
            processor.tokenizer.encode(category_id, add_special_tokens=False)[-1]
            for category_id in category_ids
        ]
        trainer_cls = PairContrastiveTrainer
        trainer_kwargs.update(
            {
                "true_token_id": processor.tokenizer.encode("true", add_special_tokens=False)[0],
                "false_token_id": processor.tokenizer.encode("false", add_special_tokens=False)[0],
                "category_token_ids": category_token_ids,
                "label_weight": data_args.pair_contrast_label_weight,
                "pair_weight": data_args.pair_contrast_pair_weight,
                "category_weight": data_args.pair_contrast_category_weight,
                "label_margin": data_args.pair_contrast_label_margin,
                "pair_margin": data_args.pair_contrast_pair_margin,
                "category_margin": data_args.pair_contrast_category_margin,
            }
        )
        logger.info(
            "pair_contrast SFT enabled: pairs=%d label_weight=%.4f pair_weight=%.4f category_weight=%.4f",
            len(train_dataset),
            data_args.pair_contrast_label_weight,
            data_args.pair_contrast_pair_weight,
            data_args.pair_contrast_category_weight,
        )

    callbacks = []
    is_main_process = getattr(training_args, "local_rank", -1) in (-1, 0)
    if data_args.eval_steps_callback > 0 and is_main_process:
        try:
            from vllm_guard.training.eval_callback import EvaluationCallback

            callbacks.append(
                EvaluationCallback(
                    eval_steps=data_args.eval_steps_callback,
                    processor=processor,
                    model_type="sft",
                    max_new_tokens=5 if data_args.no_reason else 128,
                    response_format="think" if data_args.use_think_tags else ("nothink" if data_args.no_reason else "reasoned"),
                )
            )
        except Exception as exc:
            logger.warning(f"Could not load EvaluationCallback: {exc}")

    if is_main_process:
        callbacks.append(ProcessorCheckpointCallback(processor, training_args.output_dir))

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
        **trainer_kwargs,
    )
    trainer._vllm_guard_processor = processor
    return trainer


def run_training(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments) -> str:
    trainer = create_trainer(model_args, data_args, training_args)
    resume_mode = os.environ.get("SFT_RESUME_FROM_CHECKPOINT", "").strip().lower()
    checkpoint = None
    if resume_mode in {"1", "true", "yes", "auto"}:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        logger.info(f"Resuming from checkpoint: {checkpoint}")
    else:
        logger.info("Starting a fresh training run without auto-resume.")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    processor = getattr(trainer, "_vllm_guard_processor", None)
    if processor is not None:
        processor.save_pretrained(training_args.output_dir)
        logger.info(f"Processor saved to {training_args.output_dir}")
    return training_args.output_dir


def parse_hf_train_args(argv: Optional[Sequence[str]] = None):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    return parser.parse_args_into_dataclasses(args=list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    model_args, data_args, training_args = parse_hf_train_args(argv)
    run_training(model_args, data_args, training_args)


if __name__ == "__main__":
    main()
