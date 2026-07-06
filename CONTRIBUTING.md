# Contributing

This repository is the public implementation for PolicyShiftGuard and PolicyShiftBench. Contributions are welcome when they preserve the benchmark contracts and keep the repository reproducible.

## Good Contributions

- Bug fixes for parsing, metrics, validators, or data loading.
- New model adapters that follow the existing evaluation contract.
- Documentation improvements and reproduction notes.
- Additional validators or visualization tools for public artifacts.

## Before Opening a Pull Request

Run the lightweight checks:

```bash
python -m compileall -q vllm_guard
python -m vllm_guard.validator.code_validator
```

If your change touches data or evaluation outputs, also run the relevant validator:

```bash
python -m vllm_guard.validator.dataset_validator --dataset-path /path/to/policyshiftbench
python -m vllm_guard.validator.eval_validator --eval-root outputs/eval
```

## Repository Hygiene

Do not commit:

- raw images, generated datasets, model checkpoints, or evaluation dumps;
- API keys, proxy settings, private paths, or cluster-specific scripts;
- large third-party training frameworks;
- notebook scratch work or temporary logs.

Use Hugging Face Hub, object storage, or release assets for large artifacts.

## Code Style

- Keep public entrypoints under `vllm_guard/`.
- Prefer small, typed functions over one-off scripts.
- Preserve the structured output contract:

```text
true | <two-digit risk category id> | <short reason>
false | <short reason>
```

- Add or update validators when changing dataset, evaluation, training, or reporting logic.

## Reporting Issues

When reporting a bug, include:

- command used;
- model name or adapter type;
- dataset split;
- relevant config;
- minimal error log;
- whether the issue affects parsing, metrics, model inference, or data loading.
