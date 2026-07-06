# Configs

This directory contains lightweight example configs for the public PolicyShiftGuard code path.

## Files

- `datasets/adaptive_policy_v2.7_withreason.yaml`: canonical dataset registry example.
- `eval/adaptive_policy_default.yaml`: default evaluation settings.
- `train/sft_qwen25vl_3b.yaml`: minimal SFT config using `Qwen/Qwen2.5-VL-3B-Instruct`.

## Usage

The config files are intentionally simple YAML references. They are meant to document defaults and can be loaded by downstream launchers or copied into your own experiment manager.

Most canonical entrypoints also expose equivalent CLI arguments:

```bash
python -m vllm_guard.evaluation.runner --help
python -m vllm_guard.training.sft.runner --help
python -m vllm_guard.training.rl.runner --help
```

## Do Not Commit

Do not commit machine-specific paths, private endpoints, API keys, cluster job IDs, or checkpoint paths here. Use environment variables or external launch scripts for local infrastructure.
