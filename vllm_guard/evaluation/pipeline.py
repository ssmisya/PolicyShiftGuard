import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vllm_guard.evaluation.adapters import create_adapter
from vllm_guard.evaluation.loaders import EvalDataConfig, ImageLoader, get_instance_image, load_instances
from vllm_guard.evaluation.parsing import parse_predicted_categories, parse_response
from vllm_guard.evaluation.prompts import build_llavaguard_prompt, build_prompt, resolve_output_instructions
from vllm_guard.evaluation.reporting import save_results_bundle


@dataclass
class CheckpointManager:
    output_dir: str

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.output_dir / "checkpoint.jsonl"

    def load(self):
        completed = {}
        if self.ckpt_path.exists():
            with open(self.ckpt_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    completed[(r["image_idx"], r["section_id"], r["policy_name"])] = r
        return completed

    def append(self, result):
        with open(self.ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def run_evaluation(config) -> tuple[list[dict], float]:
    instances = load_instances(EvalDataConfig(config.dataset_dir, config.dataset_repo, config.dataset_parquet, config.split))
    ckpt = CheckpointManager(config.output_dir)
    completed = ckpt.load() if config.resume else {}
    done_results, pending = [], []
    for inst in instances:
        key = (inst["image_idx"], inst["section_id"], inst["policy_name"])
        (done_results if key in completed else pending).append(completed[key] if key in completed else inst)

    if pending and "image_idx" not in pending[0]:
        pending = [x for x in pending if isinstance(x, dict) and "image_idx" in x]

    model = create_adapter(config)
    image_loader = ImageLoader(config.image_source, config.image_source_type, config.split) if config.image_source else None
    output_instructions = resolve_output_instructions(config.response_format)
    prompt_fn = build_llavaguard_prompt if config.model_type == "llavaguard" else build_prompt

    total_inference_time = 0.0
    n_inference_calls = 0
    for i in range(0, len(pending), config.batch_size):
        batch = pending[i:i + config.batch_size]
        batch_inputs, batch_valid = [], []
        for inst in batch:
            try:
                batch_inputs.append({"text": prompt_fn(inst, output_instructions), "image": get_instance_image(inst, image_loader)})
                batch_valid.append(inst)
            except Exception:
                result = {
                    "image_idx": inst["image_idx"],
                    "section_id": inst["section_id"],
                    "section_title": inst["section_title"],
                    "policy_name": inst["policy_name"],
                    "tier": inst["tier"],
                    "label": inst["label"],
                    "prediction": "invalid",
                    "raw_response": "IMAGE_LOAD_ERROR",
                    "violated_categories": inst.get("violated_categories", []),
                    "predicted_categories": [],
                }
                done_results.append(result)
                ckpt.append(result)
        if not batch_inputs:
            continue
        start_time = time.time()
        responses = model.generate(batch_inputs)
        total_inference_time += time.time() - start_time
        n_inference_calls += len(batch_inputs)
        for inst, resp in zip(batch_valid, responses):
            result = {
                "image_idx": inst["image_idx"],
                "section_id": inst["section_id"],
                "section_title": inst["section_title"],
                "policy_name": inst["policy_name"],
                "tier": inst["tier"],
                "label": inst["label"],
                "discrimination_score": inst.get("discrimination_score", 0.0),
                "prediction": parse_response(resp, model_type=config.model_type, response_format=config.response_format),
                "raw_response": resp,
                "violated_categories": inst.get("violated_categories", []),
                "predicted_categories": parse_predicted_categories(resp, response_format=config.response_format),
            }
            done_results.append(result)
            ckpt.append(result)
    avg = total_inference_time / n_inference_calls if n_inference_calls else 0.0
    return done_results, avg


def run_and_save(config):
    results, avg = run_evaluation(config)
    metrics = save_results_bundle(config.output_dir, results, avg)
    return results, metrics
