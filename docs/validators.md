# Validators

Validators are lightweight sanity checks for public PolicyShiftGuard and PolicyShiftBench artifacts.

## Code Validator

```bash
python -m vllm_guard.validator.code_validator
```

Checks that the package still follows the canonical benchmark, dataset, and documentation assumptions.

## Dataset Validator

```bash
python -m vllm_guard.validator.dataset_validator \
  --dataset-path /path/to/policyshiftbench \
  --output-dir outputs/validation/dataset
```

Checks:

- required fields for each split;
- `sft` versus `sft_think` field semantics;
- image-level and instance-level split hygiene;
- label balance;
- generated miniset examples and HTML spot-checks.

## Evaluation Validator

```bash
python -m vllm_guard.validator.eval_validator \
  --eval-root outputs/eval
```

Checks:

- per-split prediction files;
- `metrics.json` structure;
- summary tables;
- qualitative visualization artifacts.

## SFT Validator

```bash
python -m vllm_guard.validator.sft_validator \
  --model-dir outputs/sft/qwen25vl3b
```

Checks:

- model directory exists;
- required config, tokenizer, and processor files exist;
- checkpoint directories are discoverable.

## RL Validator

```bash
python -m vllm_guard.validator.rl_validator \
  --rl-dir outputs/rl_data
```

Checks:

- `train.parquet` and validation parquet files;
- required `verl` columns;
- split-specific `data_source` values;
- optional adjacent pass/block pair packing.
