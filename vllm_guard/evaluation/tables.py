from pathlib import Path
from typing import Any


_SECTION_SHORT = {
    1: "Nudity \\& Sexual",
    2: "Violence \\& Hate",
    3: "Regulated Goods",
    4: "IP \\& Brand",
    5: "Cultural \\& Religious",
    6: "Privacy \\& PII",
    7: "Text-in-Image",
}


def _short(sid: int, fallback_title: str = "") -> str:
    if sid in _SECTION_SHORT:
        return _SECTION_SHORT[sid]
    title = fallback_title.split("(")[0].strip()
    if len(title) > 25:
        title = title[:22] + "..."
    return title.replace("&", "\\&")


def _pct(value: float) -> str:
    return f"{value * 100:.1f}"


def generate_main_table(metrics: dict[str, Any]) -> str:
    overall = metrics["overall"]
    per_section = metrics["per_section"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Adaptive Policy Benchmark --- Per-Section Results}",
        r"\label{tab:section-results}",
        r"\small",
        r"\begin{tabular}{l r c c c c c}",
        r"\toprule",
        r"\textbf{Section} & \textbf{N} & \textbf{Acc} & \textbf{PCA$_{\text{dis}}$} & \textbf{PSS} & \textbf{Consist.} & \textbf{Red-Line} \\",
        r"\midrule",
    ]
    for sid in sorted(per_section):
        sec = per_section[sid]
        redline = _pct(sec["redline_recall"]) if sec["redline_n"] > 0 else "---"
        lines.append(
            f"{_short(sid, sec.get('section_title', ''))} & {sec['n_instances']} & {_pct(sec['accuracy'])} & "
            f"{_pct(sec['pca_disagree'])} & {_pct(sec['pss'])} & {_pct(sec['consistency'])} & {redline} \\\\"
        )
    overall_redline = _pct(overall["redline_recall"]) if overall["redline_n"] > 0 else "---"
    lines += [
        r"\midrule",
        r"\textbf{Overall} & "
        f"{overall['n_valid']} & {_pct(overall['accuracy'])} & {_pct(overall['pca_disagree'])} & "
        f"{_pct(overall['pss'])} & {_pct(overall['consistency'])} & {overall_redline} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def generate_summary_table(metrics: dict[str, Any]) -> str:
    overall = metrics["overall"]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Overall Performance Summary}",
        r"\label{tab:summary}",
        r"\begin{tabular}{l c}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value (\%)} \\",
        r"\midrule",
        f"Overall Accuracy & {_pct(overall['accuracy'])} \\\\",
        f"PCA (All Agree) & {_pct(overall['pca_agree'])} \\\\",
        f"PCA (Disagree) & {_pct(overall['pca_disagree'])} \\\\",
        f"PSS & {_pct(overall['pss'])} \\\\",
        f"Consistency & {_pct(overall['consistency'])} \\\\",
    ]
    if overall["redline_n"] > 0:
        lines.append(f"Red-Line Recall & {_pct(overall['redline_recall'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_confusion_table(metrics: dict[str, Any]) -> str:
    cm = metrics["overall"]["confusion_matrix"]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Confusion Matrix}",
        r"\label{tab:confusion}",
        r"\begin{tabular}{l c c}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Ground Truth}} \\",
        r"\cmidrule(lr){2-3}",
        r"\textbf{Predicted} & Block & Pass \\",
        r"\midrule",
        f"Block & {cm['tp']} & {cm['fp']} \\\\",
        f"Pass & {cm['fn']} & {cm['tn']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def save_tables(metrics: dict[str, Any], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_main.tex").write_text(generate_main_table(metrics), encoding="utf-8")
    (out / "table_summary.tex").write_text(generate_summary_table(metrics), encoding="utf-8")
    (out / "table_confusion.tex").write_text(generate_confusion_table(metrics), encoding="utf-8")
