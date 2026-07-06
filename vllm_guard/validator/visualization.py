import base64
import glob
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from PIL import Image

from vllm_guard.training.formatting import randomize_policy_descriptions


def _to_data_uri(image, max_size: int = 320, quality: int = 85) -> str:
    if not isinstance(image, Image.Image):
        if isinstance(image, dict) and image.get("bytes"):
            image = Image.open(io.BytesIO(image["bytes"]))
        else:
            image = Image.open(image)
    image = image.convert("RGB")
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _html_escape(text: Any) -> str:
    text = "" if text is None else str(text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _truncate(text: Any, limit: int = 1200) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + " ..."


def save_dataset_examples_visualization(dataset_path: str | Path, output_html: str | Path) -> dict[str, Any]:
    root = Path(dataset_path)
    output_html = Path(output_html)
    selected: dict[str, list[dict[str, Any]]] = {}

    for split in ("id_test", "ood_test", "sft", "sft_think", "rl"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        ds = load_from_disk(str(split_dir))
        seen_labels = set()
        rows = []
        for row in ds:
            label = str(row.get("label", ""))
            if label in seen_labels and len(rows) >= 4:
                continue
            payload = {
                "split": split,
                "image_idx": row.get("image_idx"),
                "section_id": row.get("section_id"),
                "policy_name": row.get("policy_name"),
                "label": label,
                "answer": row.get("answer"),
                "violated_categories": row.get("violated_categories", []),
                "question": row.get("question", ""),
                "image_data_uri": _to_data_uri(row["image"]),
            }
            if "reason" in row:
                payload["reason"] = row.get("reason")
            if "target_text" in row:
                payload["target_text"] = row.get("target_text")
            rows.append(payload)
            seen_labels.add(label)
            if len(rows) >= 4:
                break
        selected[split] = rows

    lines = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:Arial,sans-serif;margin:24px;} .card{border:1px solid #ddd;padding:16px;margin:16px 0;border-radius:8px;}",
        "img{max-width:320px;max-height:320px;display:block;margin-bottom:12px;} pre{white-space:pre-wrap;background:#f7f7f7;padding:12px;border-radius:6px;}",
        ".meta{font-size:14px;color:#333;margin-bottom:8px;} .split{font-size:22px;margin-top:24px;}",
        "</style></head><body>",
        "<h1>Dataset Visualization</h1>",
        "<p>Representative examples showing image, question, and gold answer.</p>",
    ]
    for split, rows in selected.items():
        lines.append(f"<div class='split'>{_html_escape(split)}</div>")
        for row in rows:
            lines.extend(
                [
                    "<div class='card'>",
                    f"<div class='meta'><b>image_idx</b>: {_html_escape(row['image_idx'])} | <b>section</b>: {_html_escape(row['section_id'])} | <b>policy</b>: {_html_escape(row['policy_name'])}</div>",
                    f"<div class='meta'><b>label</b>: {_html_escape(row['label'])} | <b>answer</b>: {_html_escape(row['answer'])} | <b>violated_categories</b>: {_html_escape(row['violated_categories'])}</div>",
                    f"<img src='{row['image_data_uri']}' />",
                    "<b>Question</b>",
                    f"<pre>{_html_escape(_truncate(row['question']))}</pre>",
                ]
            )
            if row.get("reason"):
                lines.extend(["<b>Reason</b>", f"<pre>{_html_escape(_truncate(row['reason'], 400))}</pre>"])
            if row.get("target_text"):
                lines.extend(["<b>Target Text</b>", f"<pre>{_html_escape(_truncate(row['target_text'], 400))}</pre>"])
            lines.append("</div>")
    lines.append("</body></html>")
    output_html.write_text("\n".join(lines), encoding="utf-8")
    manifest = {"splits": {k: len(v) for k, v in selected.items()}, "output_html": str(output_html)}
    output_html.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def save_eval_examples_visualization(eval_root: str | Path, dataset_path: str | Path, output_html: str | Path) -> dict[str, Any]:
    eval_root = Path(eval_root)
    dataset_path = Path(dataset_path)
    output_html = Path(output_html)
    dataset_cache = {}
    model_results: dict[str, dict[str, dict[tuple[int, int, str], dict[str, Any]]]] = defaultdict(dict)

    for model_dir in sorted(p for p in eval_root.iterdir() if p.is_dir() and p.name not in {"tables"}):
        for split in ("id_test", "ood_test"):
            result_file = model_dir / split / "results.jsonl"
            if not result_file.exists():
                continue
            rows = {}
            with open(result_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    key = (int(row["image_idx"]), int(row["section_id"]), str(row["policy_name"]))
                    rows[key] = row
            model_results[split][model_dir.name] = rows

    selected_cases = {}
    for split, by_model in model_results.items():
        if not by_model:
            continue
        if split not in dataset_cache:
            dataset_cache[split] = load_from_disk(str(dataset_path / split))
        ds = dataset_cache[split]
        ds_lookup = {
            (int(row["image_idx"]), int(row["section_id"]), str(row["policy_name"])): row
            for row in ds
        }
        cases = []
        candidate_keys = set()
        for rows in by_model.values():
            candidate_keys.update(rows.keys())

        def case_rank(key):
            model_rows = [rows.get(key) for rows in by_model.values() if rows.get(key)]
            preds = {row["prediction"] for row in model_rows}
            golds = {row["label"] for row in model_rows}
            wrong = sum(1 for row in model_rows if row["prediction"] != row["label"])
            return (len(preds) > 1, wrong, len(golds), key[0])

        for key in sorted(candidate_keys, key=case_rank, reverse=True):
            base = ds_lookup.get(key)
            if base is None:
                continue
            model_answers = {}
            wrong_count = 0
            for model_name, rows in by_model.items():
                row = rows.get(key)
                if not row:
                    continue
                model_answers[model_name] = {
                    "prediction": row.get("prediction"),
                    "predicted_categories": row.get("predicted_categories", []),
                    "raw_response": row.get("raw_response", ""),
                }
                if row.get("prediction") != row.get("label"):
                    wrong_count += 1
            if wrong_count == 0 and len({v["prediction"] for v in model_answers.values()}) == 1 and len(cases) >= 4:
                continue
            cases.append(
                {
                    "split": split,
                    "image_idx": base.get("image_idx"),
                    "section_id": base.get("section_id"),
                    "policy_name": base.get("policy_name"),
                    "label": base.get("label"),
                    "answer": base.get("answer"),
                    "violated_categories": base.get("violated_categories", []),
                    "question": base.get("question", ""),
                    "image_data_uri": _to_data_uri(base["image"]),
                    "model_answers": model_answers,
                }
            )
            if len(cases) >= 8:
                break
        selected_cases[split] = cases

    lines = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:Arial,sans-serif;margin:24px;} .card{border:1px solid #ddd;padding:16px;margin:16px 0;border-radius:8px;}",
        "img{max-width:320px;max-height:320px;display:block;margin-bottom:12px;} pre{white-space:pre-wrap;background:#f7f7f7;padding:12px;border-radius:6px;}",
        "table{border-collapse:collapse;width:100%;margin-top:12px;} td,th{border:1px solid #ccc;padding:8px;vertical-align:top;} .split{font-size:22px;margin-top:24px;}",
        "</style></head><body>",
        "<h1>Eval Visualization</h1>",
        "<p>Representative cases showing image, question, gold answer, and model predictions.</p>",
    ]
    for split, cases in selected_cases.items():
        lines.append(f"<div class='split'>{_html_escape(split)}</div>")
        for case in cases:
            lines.extend(
                [
                    "<div class='card'>",
                    f"<div><b>image_idx</b>: {_html_escape(case['image_idx'])} | <b>section</b>: {_html_escape(case['section_id'])} | <b>policy</b>: {_html_escape(case['policy_name'])}</div>",
                    f"<div><b>gold label</b>: {_html_escape(case['label'])} | <b>gold answer</b>: {_html_escape(case['answer'])} | <b>violated_categories</b>: {_html_escape(case['violated_categories'])}</div>",
                    f"<img src='{case['image_data_uri']}' />",
                    "<b>Question</b>",
                    f"<pre>{_html_escape(_truncate(case['question']))}</pre>",
                    "<b>Model Answers</b>",
                    "<table><tr><th>Model</th><th>Prediction</th><th>Predicted Categories</th><th>Raw Response</th></tr>",
                ]
            )
            for model_name, answer in case["model_answers"].items():
                lines.append(
                    "<tr>"
                    f"<td>{_html_escape(model_name)}</td>"
                    f"<td>{_html_escape(answer.get('prediction'))}</td>"
                    f"<td>{_html_escape(answer.get('predicted_categories'))}</td>"
                    f"<td><pre>{_html_escape(_truncate(answer.get('raw_response'), 400))}</pre></td>"
                    "</tr>"
                )
            lines.append("</table></div>")
    lines.append("</body></html>")
    output_html.write_text("\n".join(lines), encoding="utf-8")
    manifest = {"splits": {k: len(v) for k, v in selected_cases.items()}, "output_html": str(output_html)}
    output_html.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _evenly_sample(items: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    indexes = sorted({round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)})
    return [items[i] for i in indexes]


def _normalize_json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _read_jsonl_samples(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"parse_error": str(exc), "raw_line": line[:1200]})
    return rows


def save_training_artifacts_visualization(
    dataset_path: str | Path,
    output_html: str | Path,
    *,
    rephrase_path: str | Path | None = None,
    rephrase_seed: int = 0,
    rl_root: str | Path = "outputs/rl",
    max_dataset_rows_per_split: int = 8,
    max_rollout_files_per_experiment: int = 10,
    max_rollout_rows_per_file: int = 2,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_html = Path(output_html)
    rl_root = Path(rl_root)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    dataset_examples: dict[str, list[dict[str, Any]]] = {}
    for split in ("sft", "sft_think", "rl", "id_test", "ood_test"):
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
        ds = load_from_disk(str(split_dir))
        rows = []
        for idx, row in enumerate(ds):
            if len(rows) >= max_dataset_rows_per_split:
                break
            example_key = f"{split}:{row.get('image_idx')}:{row.get('section_id')}:{row.get('policy_name')}:{idx}"
            original_question = row.get("question", "")
            randomized_question = randomize_policy_descriptions(
                original_question,
                rephrase_path=str(rephrase_path) if rephrase_path else None,
                seed=rephrase_seed,
                example_key=example_key,
            )
            payload = {
                "split": split,
                "image_idx": row.get("image_idx"),
                "section_id": row.get("section_id"),
                "section_title": row.get("section_title"),
                "policy_name": row.get("policy_name"),
                "label": row.get("label"),
                "answer": row.get("answer"),
                "target_text": row.get("target_text", ""),
                "reason": row.get("reason", ""),
                "violated_categories": _normalize_json_value(row.get("violated_categories", [])),
                "policy_description": row.get("policy_description", ""),
                "question": original_question,
                "randomized_question": randomized_question,
                "randomized_changed": original_question != randomized_question,
                "image_data_uri": _to_data_uri(row["image"]),
            }
            rows.append(payload)
        dataset_examples[split] = rows

    rollout_examples: dict[str, dict[str, Any]] = {}
    if rl_root.exists():
        for exp_dir in sorted(p for p in rl_root.iterdir() if p.is_dir()):
            exp_payload: dict[str, Any] = {"kinds": {}}
            for kind in ("rollout_generations", "validation_generations"):
                files = [Path(p) for p in sorted(glob.glob(str(exp_dir / kind / "*.jsonl")), key=lambda x: (len(Path(x).stem), Path(x).stem))]
                if not files:
                    continue
                selected_files = _evenly_sample(files, max_rollout_files_per_experiment)
                file_payloads = []
                for file_path in selected_files:
                    samples = _read_jsonl_samples(file_path, max_rollout_rows_per_file)
                    step = file_path.stem
                    file_payloads.append(
                        {
                            "file": str(file_path),
                            "step": step,
                            "samples": samples,
                        }
                    )
                exp_payload["kinds"][kind] = {
                    "total_files": len(files),
                    "selected_files": len(selected_files),
                    "files": file_payloads,
                }
            if exp_payload["kinds"]:
                rollout_examples[exp_dir.name] = exp_payload

    style = """
    body{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f4f0e8;color:#201b16;}
    header{padding:32px 40px;background:linear-gradient(135deg,#1f3d35,#7c5a2e);color:#fff;}
    main{padding:24px 40px 56px;}
    h1{margin:0 0 8px;font-size:34px;} h2{margin-top:36px;border-bottom:2px solid #cbb895;padding-bottom:8px;} h3{margin-top:24px;}
    .summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:18px;}
    .metric{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.28);border-radius:12px;padding:14px;}
    .card{background:#fffaf0;border:1px solid #dccaa7;border-radius:14px;padding:16px;margin:16px 0;box-shadow:0 8px 24px rgba(58,43,23,.08);}
    .grid{display:grid;grid-template-columns:280px minmax(0,1fr);gap:16px;align-items:start;}
    img{max-width:260px;max-height:260px;border-radius:10px;border:1px solid #d7c5a5;background:#eee;}
    pre{white-space:pre-wrap;word-break:break-word;background:#f0e7d8;border:1px solid #ddceb5;border-radius:10px;padding:12px;max-height:460px;overflow:auto;}
    table{border-collapse:collapse;width:100%;background:#fffaf0;margin:12px 0;} td,th{border:1px solid #d6c4a8;padding:8px;vertical-align:top;}
    .pill{display:inline-block;padding:3px 8px;border-radius:999px;background:#dfc88f;color:#30220c;font-size:12px;margin:2px;}
    .changed{background:#1f6b4f;color:#fff;} .unchanged{background:#8b7d6b;color:#fff;}
    .muted{color:#6f6255;font-size:13px;} .two{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
    @media(max-width:900px){main,header{padding:20px}.grid,.two{grid-template-columns:1fr}img{max-width:100%;}}
    """

    lines = [
        "<html><head><meta charset='utf-8'><title>V2.8 Dataset and RL Rollout Visualization</title>",
        f"<style>{style}</style></head><body>",
        "<header>",
        "<h1>V2.8 Dataset, Randomized Policy Prompts, and RL Rollouts</h1>",
        f"<div>Dataset: {_html_escape(dataset_path)}</div>",
        f"<div>Policy rephrase file: {_html_escape(rephrase_path or 'disabled')}</div>",
        f"<div>Rephrase seed: {_html_escape(rephrase_seed)}</div>",
        "<div class='summary'>",
        f"<div class='metric'><b>Dataset splits shown</b><br>{_html_escape(len(dataset_examples))}</div>",
        f"<div class='metric'><b>RL experiments shown</b><br>{_html_escape(len(rollout_examples))}</div>",
        f"<div class='metric'><b>RL root</b><br>{_html_escape(rl_root)}</div>",
        "</div></header><main>",
    ]

    lines.append("<h2>Dataset Samples</h2>")
    lines.append("<p class='muted'>Each card shows the original training/eval prompt and the deterministic randomized policy prompt used when policy rephrase is enabled. Output format and gold answer are unchanged.</p>")
    for split, rows in dataset_examples.items():
        changed_count = sum(1 for row in rows if row["randomized_changed"])
        lines.append(f"<h3>{_html_escape(split)} <span class='pill'>{len(rows)} samples</span> <span class='pill changed'>{changed_count} randomized</span></h3>")
        for row in rows:
            changed_cls = "changed" if row["randomized_changed"] else "unchanged"
            changed_text = "changed" if row["randomized_changed"] else "unchanged"
            lines.extend(
                [
                    "<div class='card'>",
                    "<div class='grid'>",
                    f"<div><img src='{row['image_data_uri']}'><div class='muted'>image_idx={_html_escape(row['image_idx'])}</div></div>",
                    "<div>",
                    f"<div><span class='pill {_html_escape(changed_cls)}'>{_html_escape(changed_text)}</span> <span class='pill'>section {_html_escape(row['section_id'])}</span> <span class='pill'>{_html_escape(row['label'])}</span></div>",
                    f"<div><b>Policy</b>: {_html_escape(row['policy_name'])}</div>",
                    f"<div><b>Section</b>: {_html_escape(row['section_title'])}</div>",
                    f"<div><b>Gold answer</b>: {_html_escape(row['answer'])}</div>",
                    f"<div><b>Violated categories</b>: {_html_escape(row['violated_categories'])}</div>",
                    "<b>Policy Description</b>",
                    f"<pre>{_html_escape(_truncate(row['policy_description'], 900))}</pre>",
                    "</div></div>",
                    "<div class='two'>",
                    f"<div><b>Original Question</b><pre>{_html_escape(_truncate(row['question'], 2600))}</pre></div>",
                    f"<div><b>Randomized Question</b><pre>{_html_escape(_truncate(row['randomized_question'], 2600))}</pre></div>",
                    "</div>",
                ]
            )
            if row.get("target_text"):
                lines.append(f"<b>Target Text</b><pre>{_html_escape(_truncate(row['target_text'], 700))}</pre>")
            elif row.get("reason"):
                lines.append(f"<b>Reason</b><pre>{_html_escape(_truncate(row['reason'], 700))}</pre>")
            lines.append("</div>")

    lines.append("<h2>RL Rollout and Validation Samples</h2>")
    lines.append("<p class='muted'>This section samples saved JSONL rollout artifacts from each experiment directory. The manifest lists total files so missing or sparse experiments are visible.</p>")
    for exp_name, exp_payload in rollout_examples.items():
        lines.append(f"<h3>{_html_escape(exp_name)}</h3>")
        lines.append("<table><tr><th>Kind</th><th>Total files</th><th>Selected files</th></tr>")
        for kind, kind_payload in exp_payload["kinds"].items():
            lines.append(
                "<tr>"
                f"<td>{_html_escape(kind)}</td>"
                f"<td>{_html_escape(kind_payload['total_files'])}</td>"
                f"<td>{_html_escape(kind_payload['selected_files'])}</td>"
                "</tr>"
            )
        lines.append("</table>")
        for kind, kind_payload in exp_payload["kinds"].items():
            lines.append(f"<h4>{_html_escape(kind)}</h4>")
            for file_payload in kind_payload["files"]:
                lines.append(
                    f"<div class='card'><div><span class='pill'>step {_html_escape(file_payload['step'])}</span> "
                    f"<span class='muted'>{_html_escape(file_payload['file'])}</span></div>"
                )
                for sample in file_payload["samples"]:
                    metrics = {
                        k: sample.get(k)
                        for k in (
                            "score",
                            "reward",
                            "format_gate",
                            "accuracy",
                            "think_length_score",
                            "think_tokens",
                            "excess_tokens",
                            "pred_label",
                            "pred_category",
                            "gt_label",
                            "accepted_category_ids",
                        )
                        if k in sample
                    }
                    lines.append("<div class='card'>")
                    lines.append(f"<div><b>Metrics</b>: {_html_escape(json.dumps(metrics, ensure_ascii=False))}</div>")
                    if "gts" in sample:
                        lines.append(f"<div><b>GT</b>: {_html_escape(sample.get('gts'))}</div>")
                    if "input" in sample:
                        lines.append(f"<b>Input</b><pre>{_html_escape(_truncate(sample.get('input'), 2200))}</pre>")
                    if "output" in sample:
                        lines.append(f"<b>Output</b><pre>{_html_escape(_truncate(sample.get('output'), 1200))}</pre>")
                    if "raw_line" in sample:
                        lines.append(f"<b>Raw Line</b><pre>{_html_escape(_truncate(sample.get('raw_line'), 1200))}</pre>")
                    lines.append("</div>")
                lines.append("</div>")

    lines.append("</main></body></html>")
    output_html.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "dataset_path": str(dataset_path),
        "output_html": str(output_html),
        "rephrase_path": str(rephrase_path) if rephrase_path else None,
        "rephrase_seed": rephrase_seed,
        "dataset_splits": {k: len(v) for k, v in dataset_examples.items()},
        "rl_root": str(rl_root),
        "rl_experiments": {
            exp_name: {
                kind: {
                    "total_files": payload["total_files"],
                    "selected_files": payload["selected_files"],
                }
                for kind, payload in exp_payload["kinds"].items()
            }
            for exp_name, exp_payload in rollout_examples.items()
        },
    }
    output_html.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
