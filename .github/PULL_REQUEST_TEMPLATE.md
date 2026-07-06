## Summary

Describe the change and why it is needed.

## Area

- [ ] Data / policy rules
- [ ] Evaluation / metrics / parsing
- [ ] Training
- [ ] Validators
- [ ] Documentation

## Checks

- [ ] `python -m compileall -q vllm_guard`
- [ ] `python -m vllm_guard.validator.code_validator`
- [ ] Relevant dataset/eval/training validator, if applicable

## Artifact Hygiene

- [ ] No raw images, checkpoints, logs, or generated datasets are committed.
- [ ] No credentials, tokens, private paths, or proxy settings are committed.
- [ ] Public documentation links are correct.
