<div align="center">

# PolicyShiftGuard

**Benchmarking and Improving Policy-Adaptive Image Guardrails**

Official repository for the PolicyShiftGuard paper and the PolicyShiftBench benchmark.

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2607.05910-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2607.05910)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-PolicyShiftGuard-ffcc00?style=for-the-badge)](https://huggingface.co/PolicyShiftGuard)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-PolicyShiftBench-ffcc00?style=for-the-badge)](https://huggingface.co/datasets/PolicyShiftGuard/PolicyShiftBench)
[![Code](https://img.shields.io/badge/Code-GitHub-24292f?style=for-the-badge)](https://github.com/ssmisya/PolicyShiftGuard)
[![License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)](LICENSE)

</div>

---

PolicyShiftGuard studies a practical failure mode of image guardrails: the correct decision for the same image can change when the active moderation policy changes. This repository contains the public implementation for:

- **PolicyShiftBench**, a policy-adaptive image guardrail benchmark with category-specific policies and same-image policy flips.
- **PolicyShiftGuard**, a compact policy-conditioned VLM guardrail trained with randomized policy SFT and boundary-pair policy adaptation.
- Evaluation, training, data-curation, and validation utilities for reproducing the paper-style experiments.

## Links

| Resource | Status |
|---|---|
| Paper | [`arXiv:2607.05910`](https://arxiv.org/abs/2607.05910) |
| Models | [`PolicyShiftGuard`](https://huggingface.co/PolicyShiftGuard) (7B, 3B, 7B-RP-SFT, 3B-RP-SFT) |
| Dataset | [`PolicyShiftGuard/PolicyShiftBench`](https://huggingface.co/datasets/PolicyShiftGuard/PolicyShiftBench) |
| Training data | [`PolicyShiftGuard/adaptive-policy-v2.8-withreason`](https://huggingface.co/datasets/PolicyShiftGuard/adaptive-policy-v2.8-withreason) |
| Code | [`ssmisya/PolicyShiftGuard`](https://github.com/ssmisya/PolicyShiftGuard) |
| Policy rules | [`data_curation/rules/`](data_curation/rules/) |
| Reproducibility guide | [`docs/reproducibility.md`](docs/reproducibility.md) |
| Data format | [`docs/data.md`](docs/data.md) |
| Model releases | [`docs/model_zoo.md`](docs/model_zoo.md) |

## News

- **2026-07-07**: Paper released on [arXiv:2607.05910](https://arxiv.org/abs/2607.05910); models and datasets released on the [PolicyShiftGuard](https://huggingface.co/PolicyShiftGuard) Hugging Face organization.
- **2026-07-05**: Initial public repository cleanup for PolicyShiftGuard and PolicyShiftBench.

## Why This Benchmark

Most visual safety benchmarks evaluate whether an image is unsafe under a fixed taxonomy. Real deployments are different: moderation policies vary across social platforms, child-safe products, commercial applications, regional norms, legal baselines, and specialized domains.

PolicyShiftBench evaluates whether a model can bind image evidence to the currently active policy. A model must answer in a structured format:

```text
true | <two-digit risk category id> | <short reason>
false | <short reason>
```

This allows the benchmark to score both binary pass/block decisions and category attribution for unsafe outputs.

## Repository Layout

```text
vllm_guard/
  datasets/        Dataset schemas and registries
  data_curation/   Metadata processing, policy execution, sampling, export
  evaluation/      Model adapters, prompting, parsing, metrics, reporting
  training/        SFT, classifier baseline, RL data prep, reward utilities
  validator/       Dataset/eval/SFT/RL/code validators

configs/
  datasets/        Dataset config examples
  eval/            Evaluation defaults
  train/           Training defaults

data_curation/rules/
  basic_rules_v2.* Canonical policy rules and policy rephrases

docs/
  architecture.md       Package design and contracts
  data.md               Dataset format and expected fields
  model_zoo.md          Model/checkpoint release placeholders
  reproducibility.md    End-to-end reproduction guide
  validators.md         Validation commands
```

Large datasets, checkpoints, logs, raw images, and cluster-specific scripts are intentionally not tracked in Git.

## Installation

```bash
git clone https://github.com/ssmisya/PolicyShiftGuard.git
cd PolicyShiftGuard
python -m pip install -e .
```

For the minimal utilities:

```bash
python -m pip install -r requirements.txt
```

Install runtime extras only when needed:

```bash
python -m pip install "vllm>=0.8.0"   # fast VLM evaluation
python -m pip install verl            # GRPO/RL experiments
python -m pip install wandb           # experiment tracking
```

`torch` is not pinned because the correct wheel depends on your CUDA/runtime stack.

## Quick Start

Run a single-split evaluation with a Hugging Face or local model:

```bash
python -m vllm_guard.evaluation.runner \
  --model-name qwen2.5-vl-3b \
  --dataset-repo PolicyShiftGuard/PolicyShiftBench \
  --split id_test \
  --output-dir outputs/eval/qwen25vl3b/id_test \
  --model-type vllm \
  --batch-size 1 \
  --max-tokens 128
```

Evaluate a local checkpoint:

```bash
python -m vllm_guard.evaluation.runner \
  --model-name policyshiftguard-local \
  --model-path /path/to/checkpoint \
  --dataset-repo PolicyShiftGuard/PolicyShiftBench \
  --split ood_test \
  --output-dir outputs/eval/local/ood_test \
  --model-type vllm
```

Evaluation runs write parsed predictions and metrics into `--output-dir`. Use the validator before reporting numbers:

```bash
python -m vllm_guard.validator.eval_validator --eval-root outputs/eval
```

## Dataset

PolicyShiftBench rows contain an image reference, an active policy bundle, a gold pass/block label, optional violated category IDs, and prompt/target fields for training. The canonical policies are executable JSON rules under [`data_curation/rules/`](data_curation/rules/).

See [`docs/data.md`](docs/data.md) for the expected split names, fields, and policy-rule contract.

## Training

Run randomized policy SFT:

```bash
python -m vllm_guard.training.sft.runner \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset_path /path/to/policyshiftbench \
  --train_split sft \
  --output_dir outputs/sft/qwen25vl3b
```

Run the classifier baseline:

```bash
python -m vllm_guard.training.cls.runner \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset_path /path/to/policyshiftbench \
  --train_split sft \
  --output_dir outputs/cls/qwen25vl3b
```

Prepare RL/GRPO parquet artifacts:

```bash
python -m vllm_guard.training.rl.runner \
  --dataset-path /path/to/policyshiftbench \
  --output-dir outputs/rl_data
```

See [`vllm_guard/training/README.md`](vllm_guard/training/README.md) and [`docs/reproducibility.md`](docs/reproducibility.md) for the paper-style workflow.

## Validation

Run these checks before publishing data, checkpoints, or tables:

```bash
python -m vllm_guard.validator.code_validator
python -m vllm_guard.validator.dataset_validator --dataset-path /path/to/policyshiftbench
python -m vllm_guard.validator.eval_validator --eval-root outputs/eval
python -m vllm_guard.validator.sft_validator --model-dir outputs/sft/qwen25vl3b
python -m vllm_guard.validator.rl_validator --rl-dir outputs/rl_data
```

Visualization helpers can generate HTML spot checks with images, prompts, gold labels, and model outputs.

## Official Release Boundary

This official code repository intentionally excludes:

- raw images and generated metadata dumps;
- model checkpoints and merged weights;
- experiment logs, wandb runs, and temporary outputs;
- cluster-specific launch scripts;
- credentials, proxy settings, and machine-local paths;
- vendored copies of large training frameworks.

Use Hugging Face Hub or object storage for large artifacts.

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{song2026policyshiftguard,
  title   = {PolicyShiftGuard: Benchmarking and Improving Policy-Adaptive Image Guardrails},
  author  = {Song, Mingyang and Xu, Luxin and Sun, Haoyu and Pan, Minzhou and Cheng, Yu and Li, Bo},
  journal = {arXiv preprint arXiv:2607.05910},
  year    = {2026}
}
```

## License

Code is released under the [MIT License](LICENSE). Dataset and model artifacts may be released under separate terms on their respective hosting platforms.
