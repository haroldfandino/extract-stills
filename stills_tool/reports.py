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
    shots = analysis.get("shots", analysis.get("scenes", []))
    rows = []
    cards = {}
    covered = set()
    for candidate in analysis["candidates"]:
        shot_id = candidate.get("shot_id", candidate.get("scene_id", 0))
        row = {"frame_index": candidate["frame_index"], "timestamp": candidate["timestamp"],
               "shot_id": shot_id, "score": candidate["score"],
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
        if row["selected"]:
            covered.add(shot_id)
        cards.setdefault(shot_id, []).append((row["selected"], candidate["timestamp"],
            f'<article class="{candidate["tier"].lower()}{" selected" if row["selected"] else ""}">'
            f'<a href="{e(link, quote=True)}"><img loading="lazy" src="{e(link, quote=True)}" alt="Frame {candidate["id"]}"></a>'
            f'<strong>Frame {candidate["id"]} · {candidate["timestamp"]:.3f}s</strong>'
            f'<p>Shot {shot_id + 1} · {candidate["tier"]} · {candidate["score"]:.1f}/100'
            f'{" · SELECTED" if row["selected"] else ""}</p><p>{e(row["reasons"])}</p>'
            f'<p>{e(" · ".join(candidate.get("text_contents", [])))}</p>'
            f'<p>{e(candidate.get("text_reason", ""))}</p></article>'))
    with (directory / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [
            "frame_index", "timestamp", "shot_id", "score", "tier", "selected", "eligible",
            "duplicate_of", "reasons", "metrics"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: analysis[key] for key in
               ("schema_version", "scoring_version", "source", "target", "base_target",
                "selection_profile", "selected_ids", "end_card_id", "warnings") if key in analysis}
    summary["shots"] = shots
    summary["shot_count"] = len(shots)
    summary["covered_shot_ids"] = sorted(covered)
    summary["candidates"] = rows
    summary["text_states"] = analysis.get("text_states", [])
    (directory / "scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in analysis["warnings"])
    shot_map = {shot["id"]: shot for shot in shots}
    sections = []
    for shot_id in sorted(set(shot_map) | set(cards)):
        shot = shot_map.get(shot_id, {})
        timings = (f' · {shot["start"]:.3f}–{shot["end"]:.3f}s'
                   if "start" in shot and "end" in shot else "")
        entries = cards.get(shot_id, [])
        chosen_count = sum(entry[0] for entry in entries)
        caption = f"{chosen_count} selected" if chosen_count else "No still selected"
        shot_cards = "".join(entry[2] for entry in sorted(entries, key=lambda entry: (not entry[0], entry[1])))
        sections.append(f'<section><h2>Shot {shot_id + 1}{timings}</h2><p>{caption}</p>'
                        f'<div class="grid">{shot_cards}</div></section>')
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Extract Stills scores</title>
<style>body{{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1500px;margin:2rem auto;padding:1rem}}
h1{{font-size:1.8rem}}h2{{font-size:1.3rem}}section{{margin-top:2.5rem}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:1rem}}
article{{border:1px solid #374151;border-radius:12px;overflow:hidden;padding:12px;background:#1f2937}}
img{{width:100%;height:180px;object-fit:contain;background:#000}}p{{font-size:14px;color:#cbd5e1}}
.high{{border-top:4px solid #34d399}}.mid{{border-top:4px solid #fbbf24}}.low{{border-top:4px solid #f87171}}
.selected{{outline:2px solid #60a5fa}}
</style><h1>{html.escape(Path(analysis["source"]).name)}</h1>
<p>{len(selected)} selected · {len(covered)} of {len(shots)} {'shot' if len(shots) == 1 else 'shots'} covered</p>
<p>Automatic selection chooses one best still from each detected shot. Selected stills appear first in each group.</p>
<p>Scores are shot-relative heuristics. High ≥75, Mid 50–74, Low &lt;50.
Text stability estimates settled position/contrast; it does not measure original opacity.
Source frame indexes are zero-based.</p><ul>{warnings}</ul>{"".join(sections)}</html>"""
    (directory / "scores.html").write_text(document, encoding="utf-8")
