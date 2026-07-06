# Classifier Baseline

The classifier baseline freezes the VLM backbone and replaces the language-model head with a binary `pass/block` classifier.

## Train

```bash
python -m vllm_guard.training.cls.runner \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset_path /path/to/policyshiftbench \
  --train_split sft \
  --output_dir outputs/cls/qwen25vl3b
```

## Evaluate

```bash
python -m vllm_guard.evaluation.cls_runner \
  --model-path outputs/cls/qwen25vl3b/final_model \
  --dataset-path /path/to/policyshiftbench \
  --split id_test \
  --output-dir outputs/eval_cls/qwen25vl3b/id_test
```

Use this baseline to measure how much performance comes from binary separation alone versus policy-conditioned structured generation.
