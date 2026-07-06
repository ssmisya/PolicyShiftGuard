# RL / GRPO Preparation

This module prepares PolicyShiftBench data for GRPO-style training and provides reward utilities for policy-conditioned RL.

## Prepare verl Parquet Files

```bash
python -m vllm_guard.training.rl.runner \
  --dataset-path /path/to/policyshiftbench \
  --train-split rl \
  --val-splits id_test,ood_test \
  --output-dir outputs/rl_data \
  --response-format think
```

The output directory contains:

- `train.parquet`
- `val_id_test.parquet`
- `val_ood_test.parquet`
- `stats.json`

## Pair-Packed Training

For boundary-pair adaptation, keep adjacent pass/block pairs:

```bash
--pair-pack-train
```

## verl Backend

The public repository does not vendor `verl`. Install it in your environment or place it under `third_party/verl` yourself. The launcher checks both options.

## PRM Rewards

PRM reward judging requires runtime configuration for image roots, API endpoints, API keys, and model names. Pass these as arguments or environment variables; do not hard-code them into repository files.
