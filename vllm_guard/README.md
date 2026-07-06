# `vllm_guard`

`vllm_guard` is the canonical Python package for PolicyShiftGuard and PolicyShiftBench.

## Modules

- `common/`: repository constants and shared paths.
- `datasets/`: dataset schema and registry.
- `data_curation/`: metadata ingestion, rule application, sampling, export, and HF upload utilities.
- `evaluation/`: model adapters, benchmark loading, parsing, metrics, reporting, and tables.
- `training/`: SFT, classifier baseline, RL data preparation, and PRM reward utilities.
- `validator/`: validators for code, datasets, evaluation results, SFT outputs, and RL artifacts.

## Design Contract

The package is organized around stable contracts:

- dataset rows carry image, policy, label, category, and prompt fields;
- model outputs are parsed into `block/pass` plus optional violated category IDs;
- training formatters are shared across SFT, CLS, and RL pipelines;
- validators should be run before publishing datasets, checkpoints, or tables.

## Import Style

Prefer module entrypoints over legacy scripts:

```bash
python -m vllm_guard.data_curation.pipeline --help
python -m vllm_guard.evaluation.runner --help
python -m vllm_guard.training.sft.runner --help
python -m vllm_guard.validator.code_validator --help
```
