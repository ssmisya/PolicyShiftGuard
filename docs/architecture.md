# Architecture

PolicyShiftGuard is organized around policy-conditioned multimodal guardrail evaluation and training. The public codebase keeps only stable package code, minimal configs, policy rules, and validators. Large datasets, checkpoints, logs, and cluster-specific launch scripts are intentionally excluded.

## Core Contracts

PolicyShiftGuard relies on four stable contracts:

- **Policy contract**: each risk category defines field-level metadata, policy variants, and deterministic label logic.
- **Dataset contract**: each sample contains image, question, active policy, label, category metadata, and optional reasoning target fields.
- **Model contract**: model adapters expose a common `generate(prompts_and_images)` interface.
- **Evaluation contract**: raw model outputs are parsed into `pass/block`, optional category IDs, invalid-rate flags, and aggregate metrics.

## Package Layout

```text
vllm_guard/
  common/          Shared constants.
  datasets/        Dataset schema and registry.
  data_curation/   Metadata, policy application, sampling, export, and upload.
  evaluation/      Benchmark loading, adapters, parsers, metrics, reporting.
  training/        SFT, classifier, RL preparation, reward functions.
  validator/       Code, dataset, eval, SFT, and RL validators.
```

## Data Curation Flow

1. Annotate images with field-level metadata using multiple VLM annotators.
2. Vote over field metadata.
3. Apply category-specific policy rules to compute policy-conditioned labels.
4. Sample `id_test`, `ood_test`, `sft`, `sft_think`, and `rl` splits.
5. Export Hugging Face `datasets` splits and validation artifacts.

## Evaluation Flow

1. Load benchmark split.
2. Build prompt with the active policy bundle.
3. Run a model adapter.
4. Parse structured output.
5. Compute metrics and save tables.

## Training Flow

- **Generative SFT**: trains structured `pass/block + category + reason` outputs.
- **Classifier baseline**: freezes the VLM backbone and trains a binary head.
- **RL preparation**: converts PolicyShiftBench splits into `verl`-compatible parquet files.
- **PRM reward**: scores policy-following behavior through exact format/category checks and optional judge models.

## Open-Source Boundary

The repository does not include:

- raw images or generated metadata;
- benchmark split artifacts;
- model checkpoints;
- wandb runs and cluster logs;
- vendored RL frameworks;
- private agent notes or machine-specific paths.

Users should install optional backends such as `vllm` or `verl` in their own environment.
