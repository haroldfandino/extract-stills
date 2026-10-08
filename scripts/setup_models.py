"""Provision pinned local models. This developer/build command needs the internet.

Runtime extraction never calls this script or downloads models. No conversion or
Paddle framework is needed: the manifest pins PaddlePaddle's official ONNX file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from urllib.request import Request, urlopen


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path: Path, entry: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == entry["size_bytes"]
        and file_hash(path) == entry["sha256"]
    )


def provision(model_dir: Path, manifest_path: Path, verify_only: bool = False) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported model manifest version")
    if not verify_only:
        model_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest["models"]:
        filename = entry["filename"]
        if Path(filename).name != filename or not entry["url"].startswith("https://"):
            raise ValueError("Model manifest requires plain filenames and HTTPS URLs")
        destination = model_dir / filename
        if verified(destination, entry):
            print(f"Verified {filename} ({entry['license']})")
            continue
        if verify_only:
            raise RuntimeError(f"Missing or invalid model: {destination}")
        print(f"Downloading {entry['name']} from its pinned official source...")
        request = Request(entry["url"], headers={"User-Agent": "extract-stills-model-setup/2.0"})
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=model_dir, suffix=".download", delete=False) as output:
                temporary = Path(output.name)
                with urlopen(request, timeout=60) as response:
                    total = 0
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        total += len(block)
                        if total > entry["size_bytes"]:
                            raise RuntimeError(f"Download larger than pinned size: {filename}")
                        output.write(block)
            if not verified(temporary, entry):
                raise RuntimeError(f"SHA256 or size verification failed: {filename}")
            temporary.replace(destination)
            print(f"Verified {filename}: {entry['sha256']}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    installed_manifest = model_dir / "manifest.json"
    if installed_manifest.resolve() != manifest_path.resolve():
        installed_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=repository / "models")
    parser.add_argument("--manifest", type=Path, default=repository / "models" / "manifest.json")
    parser.add_argument("--verify", action="store_true", help="Verify local artifacts without network access")
    args = parser.parse_args(argv)
    try:
        provision(args.model_dir.resolve(), args.manifest.resolve(), args.verify)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
