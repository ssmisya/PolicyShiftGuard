# Training

This module contains the public training utilities for PolicyShiftGuard-style guardrail models.

## Submodules

- `sft/`: generative supervised fine-tuning for no-think and think targets.
- `cls/`: binary classifier baseline on top of a frozen VLM backbone.
- `rl/`: GRPO/verl data preparation, launch glue, and PRM-compatible reward functions.
- `formatting.py`: shared answer and prompt formatting logic.
- `eval_callback.py`: miniset callback evaluation helpers.

## Common Dataset Assumption

Training expects a dataset root containing the canonical splits:

```text
sft/
sft_think/
rl/
id_test/
ood_test/
```

You can also use parquet-backed splits when the runner supports them.

## Output Hygiene

Training outputs, checkpoints, logs, and wandb runs are ignored by Git. Keep them under `outputs/`, `checkpoints/`, or external object storage.
