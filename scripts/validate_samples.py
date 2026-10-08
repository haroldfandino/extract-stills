"""Run local acceptance media without uploading it or changing source videos."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stills_tool.core import analyze_video, discover_inputs, export_analysis, fingerprint
from stills_tool.types import Options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("validation/samples"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    options = Options(output=str(args.output), report=True)
    for path in discover_inputs([args.input]):
        started = time.monotonic()
        print(f"START {path.name}", flush=True)
        def progress(stage, done, total):
            if done == total:
                print(f"  {stage}: {done}/{total}", flush=True)
        try:
            before = fingerprint(path)
            analysis = analyze_video(path, options, progress=progress)
            result = export_analysis(analysis, options, progress=progress)
            if fingerprint(path) != before:
                raise RuntimeError("Source identity changed.")
            record = {"video": path.name, "status": "success", "selected": result["selected_count"],
                      "target": result["target"], "scenes": len(analysis["scenes"]),
                      "base_target": analysis["base_target"],
                      "end_card_time": next(c["timestamp"] for c in analysis["candidates"]
                                            if c["id"] == analysis["end_card_id"]),
                      "source_unchanged": True, "warnings": result["warnings"],
                      "output_dir": result["output_dir"]}
        except Exception as error:
            record = {"video": path.name, "status": "failed", "error": str(error)}
        record["seconds"] = round(time.monotonic() - started, 2)
        records.append(record)
        (args.output / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
    return int(not records or any(r["status"] != "success" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
