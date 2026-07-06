# Datasets

This module defines the dataset schema and registry used by PolicyShiftBench.

## Key Files

- `schema.py`: required fields for base samples and reasoning samples.
- `registry.py`: canonical split names and dataset locations.

## Expected Splits

- `id_test`: adaptive-branch evaluation split.
- `ood_test`: shift-branch evaluation split.
- `sft`: supervised fine-tuning split without reasoning-only fields.
- `sft_think`: supervised fine-tuning split with `reason` and `target_text`.
- `rl`: policy adaptation / RL preparation split.

The validator enforces split hygiene and the `sft` vs `sft_think` field contract.
