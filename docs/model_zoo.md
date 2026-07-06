# Model Zoo

This page records public model/checkpoint entry points for PolicyShiftGuard-style models.

## Released Models

| Model | Backbone | Status | Link |
|---|---:|---|---|
| PolicyShiftGuard-3B | Qwen2.5-VL-3B-Instruct | Pending public release | TBD |
| PolicyShiftGuard-7B | Qwen2.5-VL-7B-Instruct | Pending public release | TBD |

Update this table when checkpoints are published.

## Evaluating a Released Checkpoint

```bash
python -m vllm_guard.evaluation.runner \
  --model-name policyshiftguard-3b \
  --model-path /path/or/hf/repo \
  --dataset-repo PolicyShiftBench/PolicyShiftBench \
  --split id_test \
  --output-dir outputs/eval/policyshiftguard3b/id_test \
  --model-type vllm \
  --max-tokens 128
```

For local checkpoints, use `--model-path /path/to/checkpoint`. For Hugging Face repositories, use the repo ID as `--model-path` or register it in `vllm_guard/evaluation/model_registry.py`.

## Release Checklist

Before publishing a checkpoint:

- include tokenizer and processor files;
- include model config and generation config;
- document base model and training data version;
- document response format and parser assumptions;
- run `id_test` and `ood_test` evaluation;
- run `python -m vllm_guard.validator.sft_validator --model-dir <checkpoint>`.
