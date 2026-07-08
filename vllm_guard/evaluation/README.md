# Evaluation

This module evaluates multimodal guardrail models on PolicyShiftBench splits.

## Flow

1. `benchmarks.py` resolves benchmark defaults.
2. `loaders.py` loads dataset rows and images.
3. `prompts.py` builds policy-conditioned prompts.
4. `adapters.py` calls vLLM, Hugging Face, LLaVA-Guard, Llama-Guard, or CLS models.
5. `parsing.py` parses structured model outputs.
6. `metrics.py` computes accuracy, F1, category accuracy, PSS/PCA, invalid rate, and latency.
7. `reporting.py` and `tables.py` write JSON/Markdown/LaTeX reports.

## Run Evaluation

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

Use `--model-path /path/to/checkpoint` for local checkpoints.

## Output Contract

Model responses should follow:

```text
true | <two-digit category id> | <short reason>
false | <short reason>
```

The parser is tolerant to common formatting variants, but benchmark tables should be generated from parsed structured predictions rather than raw text.
