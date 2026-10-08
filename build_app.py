"""Build native one-folder desktop and console apps; never cross-compile.

Run with the project's Python 3.12 environment after installing requirements-dev.
FFmpeg is resolved from ./ffmpeg first, then PATH; models are always project-local.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return (result.stdout + result.stderr).strip()


def resolve_ffmpeg(directory: Path) -> dict[str, Path]:
    suffix = ".exe" if os.name == "nt" else ""
    binaries = {}
    for name in ("ffmpeg", "ffprobe"):
        local = directory / (name + suffix)
        nested = directory / "bin" / (name + suffix)
        found = local if local.is_file() else nested if nested.is_file() else shutil.which(name)
        if not found:
            raise ValueError(f"Missing {name}: put it in {directory} or on PATH.")
        resolved = Path(found).resolve()
        # Scoop executables on PATH are small launchers requiring a .shim file.
        # Resolve the real executable instead of freezing the launcher alone.
        shim = resolved.with_suffix(".shim")
        if shim.is_file():
            match = re.search(r'^path\s*=\s*"([^"\r\n]+)"', shim.read_text(encoding="utf-8"), re.MULTILINE)
            if not match or not Path(match.group(1)).is_file():
                raise ValueError(f"Cannot resolve the actual executable behind {resolved}")
            resolved = Path(match.group(1)).resolve()
        binaries[name] = resolved
    return binaries


def validate_models() -> dict:
    manifest_path = ROOT / "models" / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing offline model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("models", [])
    if isinstance(entries, dict):
        entries = list(entries.values())
    if not entries:
        raise ValueError("The model manifest has no models.")
    required = {"face_landmarker", "text_detector", "text_recognizer", "text_dictionary"}
    if not required.issubset({entry.get("id") for entry in entries}):
        raise ValueError("The model manifest must include face detection, text detection, Latin recognition, and its exact dictionary.")
    for entry in entries:
        filename = entry.get("filename") or entry.get("file") or entry.get("path")
        if not filename:
            raise ValueError("Each model requires a filename and sha256 in the manifest.")
        candidate = (ROOT / "models" / filename).resolve()
        if not candidate.is_relative_to(ROOT / "models") or not candidate.is_file():
            raise ValueError(f"Missing or unsafe model path: {filename}")
        expected = entry.get("sha256")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if not expected or actual != expected.lower():
            raise ValueError(f"Model checksum does not match manifest: {filename}")
        if candidate.stat().st_size != entry.get("size_bytes"):
            raise ValueError(f"Model size does not match manifest: {filename}")
    return manifest


def collect_notices(directory: Path, ffmpeg: dict[str, Path], manifest: dict, source_directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "LICENSE", directory / "EXTRACT-STILLS-LICENSE.txt")
    if (ROOT / "THIRD_PARTY_NOTICES.md").is_file():
        shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", directory / "THIRD_PARTY_NOTICES.md")
    records = {}
    for name, binary in ffmpeg.items():
        record = {
            "filename": binary.name,
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "version": run_output([str(binary), "-version"]),
            "buildconf": run_output([str(binary), "-buildconf"]),
            "license": run_output([str(binary), "-L"]),
        }
        records[name] = record
        (directory / f"{name.upper()}-BUILD.txt").write_text(
            "\n\n".join(record[key] for key in ("version", "buildconf", "license")) + "\n", encoding="utf-8"
        )
    (directory / "FFMPEG-BINARIES.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (directory / "MODEL-MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    distributor_roots = {source_directory}
    for binary in ffmpeg.values():
        if binary.parent.name == "bin" and (binary.parent.parent / "LICENSE").is_file():
            distributor_roots.add(binary.parent.parent)
    for folder in (ROOT / "licenses", ROOT / "models" / "licenses", *sorted(distributor_roots)):
        if not folder.is_dir():
            continue
        if folder.name == "licenses":
            shutil.copytree(folder, directory / ("model-notices" if folder.parent.name == "models" else "project-notices"), dirs_exist_ok=True)
        else:
            for path in folder.rglob("*"):
                if path.is_file() and (path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE", "SOURCE", "README"))):
                    destination = directory / "ffmpeg-distributor" / path.relative_to(folder)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "unknown")
        versions[name] = distribution.version
        for item in distribution.files or []:
            license_name = any(marker in item.name.upper() for marker in ("LICENSE", "LICENCE", "COPYING", "NOTICE"))
            license_folder = any(part.casefold() in ("license", "licenses", "licence", "licences", "notices") for part in item.parts[:-1])
            if not (license_name or license_folder):
                continue
            path = distribution.locate_file(item)
            if path.is_file():
                safe_parts = [part for part in item.parts if part not in ("..", ".", "/", "\\")]
                destination = directory / "python-packages" / name / Path(*safe_parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        shutil.copy2(python_license, directory / "PYTHON-LICENSE.txt")
    (directory / "PYTHON-PACKAGES.json").write_text(json.dumps(versions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / "BUILD-PLATFORM.json").write_text(json.dumps({
        "python": sys.version, "platform": platform.platform(), "architecture": platform.machine(),
        "macos_deployment_target": "13.0" if sys.platform == "darwin" else None,
        "distribution": "development bundle; validate source redistribution and native release gates before publishing",
    }, indent=2) + "\n", encoding="utf-8")


def write_spec(kind: str, directory: Path, binaries: dict[str, Path], notices: Path, signing_identity: str | None) -> Path:
    gui = kind == "gui"
    name = "Extract Stills" if gui else "extract-stills"
    data = [(str(ROOT / "models"), "models"), (str(notices), "licenses")]
    if (ROOT / "assets").is_dir():
        data.append((str(ROOT / "assets"), "assets"))
    binary_data = [(str(path), "binaries") for path in binaries.values()]
    icon = ROOT / "assets" / ("icon.icns" if sys.platform == "darwin" else "icon.ico")
    icon_value = str(icon) if icon.is_file() else None
    spec = f'''# Generated by build_app.py; rebuild through that helper.
from PyInstaller.utils.hooks import collect_all, collect_submodules
datas = {data!r}
binaries = {binary_data!r}
hiddenimports = collect_submodules("stills_tool")
for package in ("mediapipe", "onnxruntime"):
    package_datas, package_binaries, package_imports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports
a = Analysis(
    [{str(ROOT / ("gui_entry.py" if gui else "cli_entry.py"))!r}],
    pathex=[{str(ROOT)!r}], binaries=binaries, datas=datas,
    hiddenimports=hiddenimports, hookspath=[],
    runtime_hooks=[{str(ROOT / "pyi_qt_runtime_hook.py")!r}],
    excludes=["tensorflow", "torch", "IPython", "notebook"], noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name={name!r},
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console={not gui!r}, disable_windowed_traceback=False, icon={icon_value!r},
    target_arch={"arm64" if sys.platform == "darwin" else None!r},
    codesign_identity={signing_identity!r}, contents_directory="_internal",
)
collection = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name={name!r})
'''
    if gui and sys.platform == "darwin":
        spec += f'''app = BUNDLE(
    collection, name="Extract Stills.app", icon={icon_value!r},
    bundle_identifier="com.filmkraft.extract-stills",
    info_plist={{"LSMinimumSystemVersion": "13.0", "NSHighResolutionCapable": True}},
)
'''
    path = directory / (kind + ".spec")
    path.write_text(spec, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("all", "gui", "cli"), default="all")
    parser.add_argument("--ffmpeg-dir", type=Path, default=ROOT / "ffmpeg")
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--codesign-identity", help="Apple Developer ID identity; omission uses ad-hoc signing")
    parser.add_argument("--dry-run", action="store_true", help="Validate prerequisites without building or writing files")
    args = parser.parse_args()
    try:
        if sys.version_info[:2] != (3, 12):
            raise ValueError("Build with Python 3.12 from the project environment.")
        if sys.platform not in ("win32", "darwin"):
            raise ValueError("Supported build hosts are Windows x64 and Apple Silicon macOS.")
        expected = ("AMD64", "x86_64") if os.name == "nt" else ("arm64", "aarch64")
        if platform.machine() not in expected:
            raise ValueError(f"Unsupported native build architecture: {platform.machine()}")
        importlib.metadata.version("pyinstaller")
        ffmpeg = resolve_ffmpeg(args.ffmpeg_dir.resolve())
        for path in ffmpeg.values():
            config = run_output([str(path), "-buildconf"])
            if "--enable-nonfree" in config:
                raise ValueError("FFmpeg configured with --enable-nonfree cannot be used for this distribution.")
        filters = run_output([str(ffmpeg["ffmpeg"]), "-hide_banner", "-filters"])
        missing = [name for name in ("zscale", "tonemap", "iccgen") if not any(name == word for word in filters.split())]
        if missing:
            raise ValueError("FFmpeg must include the color filters: " + ", ".join(missing))
        manifest = validate_models()
        print(f"Native build: {sys.platform} / {platform.machine()} / Python {platform.python_version()}")
        for name, path in ffmpeg.items():
            print(f"Bundled {name}: {path}")
        if args.dry_run:
            print("Preflight passed. No build files were written.")
            return 0
        directory = ROOT / "build" / "native"
        directory.mkdir(parents=True, exist_ok=True)
        notices = directory / "notices"
        collect_notices(notices, ffmpeg, manifest, args.ffmpeg_dir.resolve())
        env = os.environ.copy()
        for key in ("QT_PLUGIN_PATH", "QML2_IMPORT_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
            env.pop(key, None)
        if os.name == "nt":
            # Unrelated programs on PATH can supply incompatible Qt/ICU DLLs
            # during dependency discovery. Python and system DLLs suffice here.
            windows = Path(env.get("SystemRoot", r"C:\Windows"))
            env["PATH"] = os.pathsep.join(str(path) for path in (
                Path(sys.prefix) / "Scripts", Path(sys.base_prefix),
                Path(sys.base_prefix) / "DLLs", windows / "System32", windows,
            ))
        if sys.platform == "darwin":
            env["MACOSX_DEPLOYMENT_TARGET"] = "13.0"
        for kind in (("cli", "gui") if args.target == "all" else (args.target,)):
            spec = write_spec(kind, directory, ffmpeg, notices, args.codesign_identity)
            subprocess.run([
                sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                "--distpath", str(args.dist_dir.resolve() / kind),
                "--workpath", str(directory / kind), str(spec),
            ], cwd=ROOT, env=env, check=True)
        print(f"Build complete: {args.dist_dir.resolve()}")
        print("Run scripts/verify_packaged.py before distributing; see docs/PACKAGING.md.")
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError, importlib.metadata.PackageNotFoundError) as error:
        print(f"Build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
