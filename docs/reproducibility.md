# Reproducibility

This guide describes the public reproduction path for the PolicyShiftGuard repository. It is intentionally backend-agnostic: use the CUDA, `torch`, and `vllm` versions that match your machine.

## 1. Install

```bash
git clone https://github.com/ssmisya/PolicyShiftGuard.git
cd PolicyShiftGuard
python -m pip install -e .
python -m pip install -r requirements.txt
```

Optional backends:

```bash
python -m pip install "vllm>=0.8.0"
python -m pip install verl
python -m pip install wandb
```

## 2. Prepare Data

Use the released PolicyShiftBench dataset from Hugging Face, or export your local copy into the split layout described in [`docs/data.md`](data.md).

For a local dataset directory:

```text
/path/to/policyshiftbench/
  id_test/
  ood_test/
  sft/
  sft_think/
  rl/
```

Validate it:

```bash
python -m vllm_guard.validator.dataset_validator \
  --dataset-path /path/to/policyshiftbench \
  --output-dir outputs/validation/dataset
```

## 3. Evaluate a Model

Adaptive split:

```bash
python -m vllm_guard.evaluation.runner \
  --model-name qwen2.5-vl-3b \
  --dataset-repo PolicyShiftBench/PolicyShiftBench \
  --split id_test \
  --output-dir outputs/eval/qwen25vl3b/id_test \
  --model-type vllm \
  --batch-size 1 \
  --max-tokens 128
```

Shift split:

```bash
python -m vllm_guard.evaluation.runner \
  --model-name qwen2.5-vl-3b \
  --dataset-repo PolicyShiftBench/PolicyShiftBench \
  --split ood_test \
  --output-dir outputs/eval/qwen25vl3b/ood_test \
  --model-type vllm \
  --batch-size 1 \
  --max-tokens 128
```

Validate outputs:

```bash
python -m vllm_guard.validator.eval_validator --eval-root outputs/eval
```

## 4. Train PolicyShiftGuard-Style Models

Randomized policy SFT:

```bash
python -m vllm_guard.training.sft.runner \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset_path /path/to/policyshiftbench \
  --train_split sft \
  --output_dir outputs/sft/qwen25vl3b
```

Reasoning-style SFT:

```bash
python -m vllm_guard.training.sft.runner \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset_path /path/to/policyshiftbench \
  --train_split sft_think \
  --no_reason False \
  --use_think_tags True \
  --output_dir outputs/sft/qwen25vl3b_think
```

Boundary-pair / RL data preparation:

```bash
python -m vllm_guard.training.rl.runner \
  --dataset-path /path/to/policyshiftbench \
  --train-split rl \
  --val-splits id_test,ood_test \
  --output-dir outputs/rl_data \
  --pair-pack-train
```

## 5. Report Results

Evaluation outputs include parsed predictions and metrics. For a paper-style release, keep:

- command line and config;
- model identifier or checkpoint hash;
- split name and dataset revision;
- parser/response format;
- metrics JSON;
- representative HTML spot-checks.

## Notes

- Closed-source model results require provider-specific adapters and credentials and are not part of the default public workflow.
- Large checkpoints and raw dataset artifacts should be hosted outside Git.
- If your run uses a custom parser or prompt, report it explicitly because policy-adaptive metrics are sensitive to output parsing.
