# Dataset

PolicyShiftBench is a policy-conditioned image guardrail benchmark. Each instance pairs an image with a runtime policy bundle and asks a model to decide whether the image violates any active policy.

## Splits

The public code expects these split names:

| Split | Purpose |
|---|---|
| `id_test` | Adaptive-policy evaluation split. Policies are in the supervised policy families. |
| `ood_test` | Held-out policy-shift evaluation split. Policies are not used as evaluation targets during supervised training. |
| `sft` | Concise randomized-policy SFT data. |
| `sft_think` | Reasoning-target SFT data with optional `<think>...</think>` text. |
| `rl` | Policy-conditioned data for RL/GRPO-style experiments and diagnostic rewards. |
| `id_test_mini`, `ood_test_mini` | Small callback-evaluation splits used during training. |

## Row Contract

Dataset rows should provide enough information to reconstruct the prompt, image, and gold answer. The exact schema can evolve, but public validators expect fields equivalent to:

- image reference or image payload;
- unique image/index identifiers;
- active policy text or policy bundle;
- gold label: pass/block;
- accepted violated category IDs for unsafe examples;
- prompt/question text;
- target answer text for SFT-style training;
- optional reason text for think-mode training.

The structured target format is:

```text
true | <two-digit risk category id> | <short reason>
false | <short reason>
```

## Policy Rules

The canonical policy catalog is stored under:

```text
data_curation/rules/
```

The rules separate image attributes from policy decisions. Field-level metadata is first inferred and voted, then deterministic category-specific rules compute policy-conditioned labels.

## Validation

Run:

```bash
python -m vllm_guard.validator.dataset_validator \
  --dataset-path /path/to/policyshiftbench \
  --output-dir outputs/validation/dataset
```

The validator checks required fields, split hygiene, label balance, miniset presence, and representative HTML spot-checks.
