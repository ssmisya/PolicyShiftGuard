#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-qwen2.5-vl-3b}"
DATASET_REPO="${DATASET_REPO:-PolicyShiftBench/PolicyShiftBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
RUN_NAME="${RUN_NAME:-qwen25vl3b}"

for SPLIT in id_test ood_test; do
  python -m vllm_guard.evaluation.runner \
    --model-name "${MODEL_NAME}" \
    --dataset-repo "${DATASET_REPO}" \
    --split "${SPLIT}" \
    --output-dir "${OUTPUT_ROOT}/${RUN_NAME}/${SPLIT}" \
    --model-type vllm \
    --batch-size 1 \
    --max-tokens 128 \
    --temperature 0.0
done

python -m vllm_guard.validator.eval_validator --eval-root "${OUTPUT_ROOT}"
