"""Smoke-test a native bundle using synthetic media and a restricted PATH.

No user videos are read or uploaded. This verifies offline assets and tool lookup;
repeat on a machine with networking disabled for a full offline release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import struct
import sys
import tempfile
import zlib


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_png(path: Path) -> None:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">IIB", data[16:25]) != (160, 90, 8):
        raise ValueError(f"Incorrect original resolution or default PNG bit depth: {path.name}")
    offset = 8
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        if kind == b"iCCP":
            payload = data[offset + 8:offset + 8 + length]
            _, compressed = payload.split(b"\0", 1)
            profile = zlib.decompress(compressed[1:])
            if profile[36:40] != b"acsp":
                raise ValueError(f"Invalid embedded ICC profile: {path.name}")
            return
        offset += length + 12
    raise ValueError(f"Default sRGB PNG has no embedded ICC profile: {path.name}")


def validate_tiff(path: Path, expected_size: tuple[int, int] = (160, 90)) -> None:
    data = path.read_bytes()
    endian = "<" if data[:2] == b"II" else ">" if data[:2] == b"MM" else None
    if not endian or struct.unpack(endian + "H", data[2:4])[0] != 42:
        raise ValueError(f"Invalid classic TIFF header: {path.name}")
    offset = struct.unpack(endian + "I", data[4:8])[0]
    count = struct.unpack(endian + "H", data[offset:offset + 2])[0]
    tags = {}
    for index in range(count):
        entry = offset + 2 + index * 12
        tag, dtype, length = struct.unpack(endian + "HHI", data[entry:entry + 8])
        if tag not in (256, 257, 258, 34675):
            continue
        size = {1: 1, 3: 2, 4: 4, 7: 1}.get(dtype)
        if not size:
            raise ValueError(f"Unsupported TIFF verification tag type: {dtype}")
        start = entry + 8 if size * length <= 4 else struct.unpack(endian + "I", data[entry + 8:entry + 12])[0]
        raw = data[start:start + size * length]
        tags[tag] = raw if dtype == 7 else list(struct.unpack(endian + {1: "B", 3: "H", 4: "I"}[dtype] * length, raw))
    if tags.get(256) != [expected_size[0]] or tags.get(257) != [expected_size[1]] or tags.get(258) != [16, 16, 16]:
        raise ValueError(f"Incorrect original resolution or 16-bit RGB TIFF samples: {path.name}")
    if not isinstance(tags.get(34675), bytes) or tags[34675][36:40] != b"acsp":
        raise ValueError(f"TIFF has no valid embedded ICC profile: {path.name}")


def run(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command}\n{result.stdout}\n{result.stderr}")
    return result


def bundle_resources(executable: Path) -> Path:
    candidates = [executable.parent / "_internal", executable.parent.parent / "Resources", executable.parent]
    for candidate in candidates:
        if (candidate / "models" / "manifest.json").is_file():
            return candidate
    raise ValueError(f"Cannot locate bundled models beside {executable}")


def validate_assets(resources: Path) -> None:
    manifest = json.loads((resources / "models" / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("models", [])
    if isinstance(entries, dict):
        entries = list(entries.values())
    if not entries:
        raise ValueError("Bundle has no declared offline models.")
    for entry in entries:
        name = entry.get("filename") or entry.get("file") or entry.get("path")
        path = (resources / "models" / name).resolve()
        if not path.is_relative_to((resources / "models").resolve()) or not path.is_file():
            raise ValueError(f"Missing bundled model: {name}")
        if digest(path) != entry["sha256"].lower():
            raise ValueError(f"Bundled model checksum mismatch: {name}")
    for notice in ("EXTRACT-STILLS-LICENSE.txt", "FFMPEG-BINARIES.json", "PYTHON-PACKAGES.json", "MODEL-MANIFEST.json"):
        if not (resources / "licenses" / notice).is_file():
            raise ValueError(f"Missing bundled notice: {notice}")


def macos_audit(executables: list[Path]) -> dict:
    """Inspect bundled Mach-O minimum versions without claiming an OS13 test."""
    records = []
    seen = set()
    for executable in executables:
        root = executable.parents[2] if executable.parent.name == "MacOS" else executable.parent
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            with real.open("rb") as stream:
                magic = stream.read(4)
            if magic not in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca"):
                continue
            command = subprocess.run(["/usr/bin/otool", "-l", str(real)], capture_output=True, text=True, check=True)
            minimums = re.findall(r"\bminos\s+(\d+(?:\.\d+){0,2})", command.stdout)
            minimums += re.findall(r"cmd LC_VERSION_MIN_MACOSX\s+cmdsize\s+\d+\s+version\s+(\d+(?:\.\d+){0,2})", command.stdout)
            minimum = max(minimums, key=lambda value: tuple(int(part) for part in value.split("."))) if minimums else None
            records.append({"binary": str(path.relative_to(root)), "minimum_macos": minimum})
    known = [record["minimum_macos"] for record in records if record["minimum_macos"]]
    highest = max(known, key=lambda value: tuple(int(part) for part in value.split("."))) if known else None
    incompatible = [record for record in records if record["minimum_macos"] and tuple(int(part) for part in record["minimum_macos"].split(".")) > (13, 0, 0)]
    return {"tested_host_macos": platform.mac_ver()[0], "architecture": platform.machine(),
            "mach_o_binaries_scanned": len(records), "highest_declared_minimum_macos": highest,
            "dependencies_above_macos_13": incompatible,
            "macos_13_status": "Requires compatible dependencies and native macOS 13 validation" if incompatible else "Requires native macOS 13 validation",
            "binary_minimums": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path, help="Absolute path to the frozen console executable")
    parser.add_argument("--gui", type=Path, help="Optional frozen GUI executable to smoke-test")
    parser.add_argument("--result-json", type=Path, help="Save smoke-test results and native macOS dependency audit")
    args = parser.parse_args()
    cli = args.cli.resolve()
    try:
        resources = bundle_resources(cli)
        validate_assets(resources)
        ffmpeg = resources / "binaries" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        ffprobe = resources / "binaries" / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if not ffmpeg.is_file() or not ffprobe.is_file():
            raise ValueError("Bundle must include both FFmpeg and FFprobe.")
        env = os.environ.copy()
        env["PATH"] = str(Path(env.get("SystemRoot", r"C:\Windows")) / "System32") if os.name == "nt" else "/usr/bin:/bin"
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "QT_PLUGIN_PATH", "QML2_IMPORT_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
            env.pop(key, None)
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "NO_PROXY": "", "no_proxy": "",
                    "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
                    "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9"})
        with tempfile.TemporaryDirectory(prefix="extract-stills-packaged-") as directory:
            cwd = Path(directory)
            fixture = cwd / "synthetic.mkv"
            run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                 "testsrc2=size=160x90:rate=12:duration=3", "-c:v", "ffv1", str(fixture)], cwd, env)
            original = digest(fixture)
            version = run([str(cli), "--version"], cwd, env).stdout.strip()
            extraction = run([str(cli), "extract", str(fixture), "--json", "--output", str(cwd / "output"), "--report"], cwd, env)
            extracted = json.loads(extraction.stdout)
            images = list((cwd / "output").rglob("*.png"))
            if not images:
                raise ValueError("Packaged extraction produced no PNG images.")
            for image in images:
                validate_png(image)
            result = extracted["results"][0]
            tiff_export = run([str(cli), "export", result["analysis"], "--frames", str(result["files"][0]["frame_index"]),
                               "--format", "tiff", "--bit-depth", "16", "--output", str(cwd / "output_tiff"), "--json"], cwd, env)
            tiff_result = json.loads(tiff_export.stdout)["results"][0]
            validate_tiff(Path(tiff_result["files"][0]["path"]))
            if digest(fixture) != original:
                raise ValueError("Extraction modified its source fixture.")
            if args.gui:
                gui = args.gui.resolve()
                validate_assets(bundle_resources(gui))
                smoke_env = {**env, "QT_QPA_PLATFORM": "offscreen", "EXTRACT_STILLS_GUI_SMOKE_TEST": "1"}
                run([str(gui), "--smoke-test"], cwd, smoke_env)
                # GUI legacy extraction reuses the frozen desktop executable as
                # its worker; exercise that route to prevent recursive windows.
                run([str(gui), "--legacy-worker", str(fixture)], cwd, env)
                for index in (0, 24, 35):
                    if not (cwd / "stills" / f"synthetic_frame_{index}.png").is_file():
                        raise ValueError(f"Frozen GUI legacy worker did not export frame {index}.")
                if digest(fixture) != original:
                    raise ValueError("The legacy worker modified its source fixture.")
            payload = {"status": "passed", "version": version, "images": len(images),
                              "restricted_path": True, "source_preserved": True,
                              "original_resolution": True, "png_8bit_icc_verified": True,
                              "tiff_16bit_icc_verified": True,
                              "offline_assets_verified": True, "gui_smoke_test": bool(args.gui)}
            if sys.platform == "darwin":
                payload["macos_dependency_audit"] = macos_audit([cli] + ([args.gui.resolve()] if args.gui else []))
            if args.result_json:
                args.result_json.parent.mkdir(parents=True, exist_ok=True)
                args.result_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(payload, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError, struct.error, zlib.error) as error:
        print(f"Packaged verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
