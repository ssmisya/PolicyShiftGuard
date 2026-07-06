# Security Policy

PolicyShiftGuard is a research codebase for evaluating and training image guardrails. Please do not use this repository as a sole production safety system without additional review, monitoring, and deployment-specific validation.

## Reporting Security Issues

If you find a vulnerability, credential leak, or reproducible misuse pathway introduced by this codebase, please open a private report through GitHub security advisories when available. If private reporting is not available, open an issue with minimal public details and request a maintainer contact.

## Sensitive Artifacts

Do not submit pull requests containing:

- API keys or service tokens;
- private proxy settings;
- raw sensitive images;
- non-public datasets or checkpoints;
- cluster-specific paths or logs.

## Intended Use

This repository is intended for research on policy-adaptive image guardrail evaluation and training. Released benchmarks and models should be used under the responsible-use terms provided with the corresponding dataset/model artifacts.
