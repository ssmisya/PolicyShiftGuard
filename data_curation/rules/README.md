# Policy Rules

This directory stores the public policy definitions used by PolicyShiftBench.

## Files

- `basic_rules_v2.json`: canonical risk categories, fields, policy variants, and deterministic label logic.
- `basic_rules_v2.md`: human-readable policy specification.
- `basic_rules_v2_policy_rephrases.json`: semantic-preserving policy rephrases for policy randomization.
- `basic_rules_v2_rl_ood_policies.json`: additional policy variants used for RL-side augmentation and policy adaptation experiments.

## Semantics

PolicyShiftBench separates image metadata from policy decisions:

1. VLM annotators produce field-level metadata for each image.
2. The rule engine maps metadata plus an active policy variant to `block` or `pass`.
3. The same image can therefore receive different labels under different policies.

This is the core mechanism behind policy-conditioned evaluation and training.

## Editing Rules

When changing policy rules:

- keep `section_id` stable unless you intentionally define a new benchmark version;
- keep category IDs two-digit compatible with evaluation output, e.g. `01`, `02`;
- update deterministic logic and human-readable text together;
- run the dataset validator after rebuilding splits.

Do not store generated metadata, split outputs, or private prompt logs in this directory.
