from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .core import analyze_video, discover_inputs, export_analysis, extract_legacy
from .types import CancelledError, Options


def _settings(parser, inputs=True, mode=False, count=False):
    if inputs:
        parser.add_argument("inputs", nargs="+", help="Video files or folders")
        parser.add_argument("--recursive", action="store_true")
    if mode:
        parser.add_argument("--mode", choices=["smart", "legacy"], default="smart")
    if count:
        parser.add_argument("--count", type=int, help="Maximum count; automatic count grows for scene coverage")
    parser.add_argument("--output", type=Path, help="Output root; default: stills beside each source")
    parser.add_argument("--format", choices=["png", "tiff", "jpeg"], default="png")
    parser.add_argument("--bit-depth", type=int, choices=[8, 16], default=8)
    parser.add_argument("--color", choices=["srgb", "source"], default="srgb")
    parser.add_argument("--report", action="store_true", help="Write HTML/JSON/CSV candidate score reports")
    parser.add_argument("--json", action="store_true", help="Machine-readable result on stdout")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--legacy-worker":
        from .legacy_original import extract_stills
        extract_stills(argv[1])
        return 0
    if argv and argv[0] not in {"extract", "analyze", "export", "gui", "--version", "-h", "--help"}:
        argv.insert(0, "extract")
    parser = argparse.ArgumentParser(prog="extract-stills", description="Offline scene-aware still extraction")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    _settings(commands.add_parser("extract", help="Analyze and export automatically"), mode=True, count=True)
    _settings(commands.add_parser("analyze", help="Save candidate analysis without final exports"), count=True)
    exporter = commands.add_parser("export", help="Export an analysis using explicit source-frame IDs")
    exporter.add_argument("analysis", type=Path, help="analysis.json or its containing directory")
    exporter.add_argument("--frames", help="Comma-separated zero-based source frame IDs; default uses automatic selection")
    _settings(exporter, inputs=False)
    gui = commands.add_parser("gui", help="Launch desktop application")
    gui.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == "gui":
        from .gui import main as gui_main
        if args.smoke_test:
            import os
            os.environ["EXTRACT_STILLS_GUI_SMOKE_TEST"] = "1"
        return gui_main()
    options = Options(mode=getattr(args, "mode", "smart"), count=getattr(args, "count", None),
                      format=args.format, bit_depth=args.bit_depth, color=args.color,
                      output=args.output, report=args.report, recursive=getattr(args, "recursive", False))
    results, errors = [], []
    def progress(stage, done, total):
        if not args.json and (done == total or done in {0, 1} or done % 24 == 0):
            print(f"{stage}: {done}/{total}", file=sys.stderr, flush=True)
    try:
        options.validate()
        if args.command == "export":
            frames = None if args.frames is None else [int(v.strip()) for v in args.frames.split(",")]
            results.append(export_analysis(args.analysis, options, frames, progress))
        else:
            files = []
            for value in args.inputs:
                try:
                    discovered = discover_inputs([value], args.recursive)
                    if not discovered:
                        raise ValueError("No supported video files found.")
                    files.extend(discovered)
                except Exception as error:
                    errors.append({"source": str(value), "error": str(error)})
            for path in sorted(set(files)):
                try:
                    if options.mode == "legacy":
                        result = extract_legacy(path, progress)
                    else:
                        analysis = analyze_video(path, options, progress)
                        result = ({"source": analysis["source"], "analysis_dir": analysis["analysis_dir"],
                                   "analysis": str(Path(analysis["analysis_dir"]) / "analysis.json"),
                                   "target": analysis["target"], "selected_ids": analysis["selected_ids"],
                                   "warnings": analysis["warnings"]} if args.command == "analyze" else
                                  export_analysis(analysis, options, progress=progress))
                    results.append(result)
                except Exception as error:
                    errors.append({"source": str(path), "error": str(error)})
    except KeyboardInterrupt:
        errors.append({"error": "Interrupted."})
    except Exception as error:
        errors.append({"error": str(error)})
    payload = {"version": __version__, "status": "failed" if errors else "success",
               "results": results, "errors": errors}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for result in results:
            print(f"{result['source']}: {result.get('selected_count', len(result.get('selected_ids', [])))} stills"
                  f" → {result.get('output_dir', result.get('analysis_dir'))}")
            for warning in result.get("warnings", []):
                print(f"  {warning}")
        for error in errors:
            print(f"Error: {error.get('source', '')}: {error['error']}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
