# SFT Training

Generative SFT teaches a VLM to read the active policy bundle and output a structured guardrail decision.

## No-Think SFT

```bash
python -m vllm_guard.training.sft.runner \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset_path /path/to/policyshiftbench \
  --train_split sft \
  --output_dir outputs/sft/qwen25vl3b_nothink \
  --no_reason True \
  --use_think_tags False
```

## Think SFT

```bash
python -m vllm_guard.training.sft.runner \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset_path /path/to/policyshiftbench \
  --train_split sft_think \
  --output_dir outputs/sft/qwen25vl3b_think \
  --no_reason False \
  --use_think_tags True
```

Additional Hugging Face `TrainingArguments` can be passed through the runner.

## Policy Randomization

Use policy rephrases to reduce policy-ID memorization:

```bash
--policy_rephrase_path data_curation/rules/basic_rules_v2_policy_rephrases.json
--policy_rephrase_seed 42
```
