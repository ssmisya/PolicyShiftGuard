# Validators

Validators catch common mistakes before publishing datasets, model outputs, or tables.

## Code

```bash
python -m vllm_guard.validator.code_validator
```

Checks that the public package still follows the canonical dataset and benchmark contracts.

## Dataset

```bash
python -m vllm_guard.validator.dataset_validator \
  --dataset-path /path/to/policyshiftbench \
  --output-dir outputs/validation/dataset
```

Checks required fields, split hygiene, label balance, and generates HTML examples.

## Evaluation

```bash
python -m vllm_guard.validator.eval_validator \
  --eval-root outputs/eval
```

Checks metrics, per-split result files, summary tables, and visualizations.

## SFT

```bash
python -m vllm_guard.validator.sft_validator \
  --model-dir outputs/sft/qwen25vl3b
```

Checks model/tokenizer artifacts and checkpoint layout.

## RL

```bash
python -m vllm_guard.validator.rl_validator \
  --rl-dir outputs/rl_data
```

Checks verl parquet files, required columns, validation split names, and optional pair-packing integrity.
