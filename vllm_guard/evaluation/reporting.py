import json
from pathlib import Path

from vllm_guard.evaluation.metrics import compute_all_metrics
from vllm_guard.evaluation.tables import save_tables


def save_results_bundle(output_dir: str, results: list[dict], avg_inference_time: float):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_all_metrics(results)
    metrics["overall"]["avg_inference_time"] = avg_inference_time
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    save_tables(metrics, str(out_dir))
    return metrics
