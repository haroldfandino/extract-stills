"""Portable, local candidate reports."""
from pathlib import Path
import csv
import html
import json
import os


def write_reports(analysis, output_dir=None):
    directory = Path(output_dir or analysis["analysis_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    selected = set(analysis["selected_ids"])
    rows = []
    cards = []
    for candidate in analysis["candidates"]:
        row = {"frame_index": candidate["frame_index"], "timestamp": candidate["timestamp"],
               "scene_id": candidate["scene_id"], "score": candidate["score"],
               "tier": candidate["tier"], "selected": candidate["id"] in selected,
               "eligible": candidate["eligible"], "duplicate_of": candidate["duplicate_of"],
               "text_state_id": candidate.get("text_state_id"),
               "recognized_text": json.dumps(candidate.get("text_contents", []), ensure_ascii=False),
               "text_ready": candidate.get("text_ready", True),
               "native_focus": json.dumps(candidate["metrics"].get("native_focus", {})),
               "reasons": "; ".join(candidate["reasons"]),
               "metrics": json.dumps(candidate["metrics"])}
        rows.append(row)
        preview = Path(analysis["analysis_dir"]) / candidate["preview"]
        link = os.path.relpath(preview, directory).replace("\\", "/")
        e = html.escape
        cards.append(
            f'<article class="{candidate["tier"].lower()}">'
            f'<a href="{e(link, quote=True)}"><img loading="lazy" src="{e(link, quote=True)}" alt="Frame {candidate["id"]}"></a>'
            f'<strong>Frame {candidate["id"]} · {candidate["timestamp"]:.3f}s</strong>'
            f'<p>Scene {candidate["scene_id"] + 1} · {candidate["tier"]} · {candidate["score"]:.1f}/100'
            f'{" · SELECTED" if row["selected"] else ""}</p><p>{e(row["reasons"])}</p>'
            f'<p>{e(" · ".join(candidate.get("text_contents", [])))}</p>'
            f'<p>{e(candidate.get("text_reason", ""))}</p></article>')
    with (directory / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [
            "frame_index", "timestamp", "scene_id", "score", "tier", "selected", "eligible",
            "duplicate_of", "reasons", "metrics"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: analysis[key] for key in
               ("schema_version", "scoring_version", "source", "target", "base_target",
                "selected_ids", "end_card_id", "warnings")}
    summary["candidates"] = rows
    summary["text_states"] = analysis.get("text_states", [])
    (directory / "scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in analysis["warnings"])
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Extract Stills scores</title>
<style>body{{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1500px;margin:2rem auto;padding:1rem}}
h1{{font-size:1.8rem}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:1rem}}
article{{border:1px solid #374151;border-radius:12px;overflow:hidden;padding:12px;background:#1f2937}}
img{{width:100%;height:180px;object-fit:contain;background:#000}}p{{font-size:14px;color:#cbd5e1}}
.high{{border-top:4px solid #34d399}}.mid{{border-top:4px solid #fbbf24}}.low{{border-top:4px solid #f87171}}
</style><h1>{html.escape(Path(analysis["source"]).name)}</h1>
<p>{len(selected)} selected · target {analysis["target"]} · {len(analysis["scenes"])} detected scenes</p>
<p>Scores are scene-relative heuristics. High ≥75, Mid 50–74, Low &lt;50.
Text stability estimates settled position/contrast; it does not measure original opacity.
Source frame indexes are zero-based.</p><ul>{warnings}</ul><div class="grid">{"".join(cards)}</div></html>"""
    (directory / "scores.html").write_text(document, encoding="utf-8")
