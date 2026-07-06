import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path
from typing import Sequence


def _shorten(text: str, limit: int = 1200) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + " ..."


def _image_to_data_uri(path: str | Path) -> str:
    img_path = Path(path)
    mime = mimetypes.guess_type(img_path.name)[0] or "image/png"
    encoded = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_manual_review_html(review_json: str | Path, output_html: str | Path) -> dict:
    review_path = Path(review_json)
    rows = json.loads(review_path.read_text(encoding="utf-8"))
    mismatches = [row for row in rows if row.get("match") is False]
    matches = [row for row in rows if row.get("match") is True]

    cards = []
    for row in mismatches:
        img_data = _image_to_data_uri(row["image_path"])
        cards.append(
            f"""
            <div class="card">
              <div class="meta">
                <div><b>sample_id</b>: {html.escape(str(row.get("sample_id", "")))}</div>
                <div><b>split</b>: {html.escape(str(row.get("split", "")))}</div>
                <div><b>policy</b>: {html.escape(str(row.get("policy_name", "")))}</div>
              </div>
              <img src="{img_data}" alt="{html.escape(str(row.get('sample_id', 'sample')))}" />
              <div class="qa"><b>My answer</b>: {html.escape(str(row.get("my_answer", "")))}</div>
              <div class="qa"><b>Gold answer</b>: {html.escape(str(row.get("gold_answer", "")))}</div>
              <div class="qa"><b>Notes</b>: {html.escape(str(row.get("notes", "")))}</div>
              <details>
                <summary>Question</summary>
                <pre>{html.escape(_shorten(str(row.get("question", ""))))}</pre>
              </details>
            </div>
            """
        )

    summary = {
        "reviewed": len(rows),
        "matched": len(matches),
        "mismatches": len(mismatches),
        "accuracy": (len(matches) / len(rows)) if rows else 0.0,
    }
    html_text = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>manual review mismatches</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        .summary {{ margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 18px; }}
        .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; background: #fff; }}
        img {{ width: 100%; max-height: 360px; object-fit: contain; background: #f5f5f5; }}
        .meta, .qa {{ margin: 8px 0; }}
        pre {{ white-space: pre-wrap; word-break: break-word; font-size: 12px; }}
      </style>
    </head>
    <body>
      <div class="summary">
        <h1>Manual Review Mismatches</h1>
        <div>reviewed: {summary['reviewed']}</div>
        <div>matched: {summary['matched']}</div>
        <div>mismatches: {summary['mismatches']}</div>
        <div>accuracy: {summary['accuracy']:.4f}</div>
      </div>
      <div class="grid">
        {''.join(cards) if cards else '<div>No mismatches yet.</div>'}
      </div>
    </body>
    </html>
    """
    output_path = Path(output_html)
    output_path.write_text(html_text, encoding="utf-8")
    return {"output_html": str(output_path), **summary}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build HTML mismatch report from manual review JSON")
    parser.add_argument("--review-json", required=True)
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args(argv)
    summary = build_manual_review_html(args.review_json, args.output_html)
    print(summary)


if __name__ == "__main__":
    main()
