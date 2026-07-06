# Examples

Minimal runnable examples for the public PolicyShiftGuard code path.

## Evaluate Qwen2.5-VL-3B on PolicyShiftBench

```bash
bash examples/evaluate_qwen25vl3b.sh
```

The script evaluates `id_test` and `ood_test` into:

```text
outputs/eval/qwen25vl3b/
```

Install `vllm` and a compatible CUDA/PyTorch stack before running the example.
