# Data Curation

This module contains the canonical PolicyShiftBench dataset construction logic.

## Flow

1. `metadata.py` queries VLM annotators for field-level metadata.
2. `policy.py` votes metadata and applies deterministic policy rules.
3. `sampling.py` chooses ID/OOD/SFT/RL instances.
4. `export.py` builds model prompts and target answers.
5. `builder.py` assembles Hugging Face `datasets` splits.
6. `upload.py` pushes validated datasets to Hugging Face Hub.

## Build Dataset

```bash
python -m vllm_guard.data_curation.pipeline \
  --metadata-gpt /path/to/metadata_gpt.jsonl \
  --metadata-gemini /path/to/metadata_gemini.jsonl \
  --metadata-qwen /path/to/metadata_qwen.jsonl \
  --mapping /path/to/dataset_mapping.json \
  --rules data_curation/rules/basic_rules_v2.json \
  --image-source /path/or/hf_dataset \
  --image-source-type hf_dataset \
  --output-dir outputs/policyshiftbench
```

## Upload Dataset

```bash
python -m vllm_guard.data_curation.upload \
  --dataset-dir outputs/policyshiftbench \
  --repo-id your-org/your-dataset \
  --token "$HF_TOKEN" \
  --private
```

## Notes

Raw images, generated metadata, split outputs, and logs are intentionally ignored by Git. Store those artifacts in object storage or Hugging Face Hub.
