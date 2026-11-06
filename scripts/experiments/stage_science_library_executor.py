#!/usr/bin/env python3
"""Stage an already installed SciPy into an ABI-matched, offline CPU image context.

No package download, installation hook, model operation or Docker action is run.
Run this with the trusted source scientific Python; supply captured base-image
Python/library metadata. The Dockerfile uses only FROM, COPY and USER.
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
import stat
import sys
import sysconfig


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stamp(path: Path) -> tuple:
    item = path.stat()
    return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "soabi": sysconfig.get_config_var("SOABI"),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "numpy": importlib.metadata.version("numpy"),
        "sympy": importlib.metadata.version("sympy"),
    }


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--base-metadata-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if os.getuid() == 0:
        raise ValueError("run staging as a non-root user")
    if (
        not args.output.is_absolute()
        or args.output.is_symlink()
        or os.path.lexists(args.output)
    ):
        raise ValueError("output must be an absolute fresh directory")
    metadata_bytes = args.base_metadata.read_bytes()
    if digest(metadata_bytes) != args.base_metadata_sha256:
        raise ValueError("base metadata hash mismatch")
    metadata = json.loads(metadata_bytes)
    if not re.fullmatch("sha256:[0-9a-f]{64}", metadata["image"]):
        raise ValueError("immutable existing base image required")
    actual = environment()
    if actual["soabi"] != "cpython-312-x86_64-linux-gnu":
        raise ValueError("Dockerfile supports CPython 3.12 x86_64 only")
    if any(metadata[key] != value for key, value in actual.items()):
        raise ValueError("source/base Python ABI or library version mismatch")
    # Use installed distribution metadata, not arbitrary directory names.
    distribution = importlib.metadata.distribution("scipy")
    site = Path(str(distribution.locate_file(""))).resolve()
    roots = [
        site / "scipy",
        site / "scipy.libs",
        site / f"scipy-{distribution.version}.dist-info",
    ]
    dockerfile = Path(__file__).with_name("science_library_executor.Dockerfile")
    canonical_output = args.output.resolve()
    pinned = [
        args.base_metadata.resolve(),
        Path(__file__).resolve(),
        dockerfile.resolve(),
    ]
    if (
        canonical_output != args.output
        or any(
            canonical_output.is_relative_to(root)
            or root.is_relative_to(canonical_output)
            for root in roots
        )
        or any(path.is_relative_to(canonical_output) for path in pinned)
    ):
        raise ValueError("output overlaps a source or uses a symlink parent")
    template = dockerfile.read_bytes()
    own_path = Path(__file__).resolve()
    own_hash = digest(own_path.read_bytes())
    files = []
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("expected regular installed SciPy tree")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("installed package symlinks are not supported")
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_dir():
                continue
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("nonregular installed package file")
            files.append((path, path.relative_to(site), stamp(path)))
    report = {
        "schema_version": 1,
        "status": "preview",
        "base_image": metadata["image"],
        "base_metadata_sha256": args.base_metadata_sha256,
        "environment": actual,
        "scipy_version": distribution.version,
        "source_python": sys.executable,
        "source_roots": [str(p) for p in roots],
        "file_count": len(files),
        "bytes": sum(row[2][2] for row in files),
        "excluded": ["__pycache__ directories", "*.pyc"],
        "helper_sha256": own_hash,
        "dockerfile_sha256": digest(template),
        "scope": "Existing-package byte copy only; image compatibility requires installed-image CPU execution.",
    }
    if not args.apply:
        return report
    args.output.mkdir(parents=True)
    try:
        rows = []
        for source, relative, before in files:
            target = args.output / "site-packages" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = source.read_bytes()
            if stamp(source) != before or len(data) != before[2]:
                raise ValueError("package changed while staging")
            with target.open("xb") as stream:
                stream.write(data)
            os.chmod(target, 0o644)
            if digest(target.read_bytes()) != digest(data):
                raise ValueError("copied package hash mismatch")
            rows.append(
                {"path": str(relative), "size": len(data), "sha256": digest(data)}
            )
        if any(stamp(source) != before for source, _, before in files):
            raise ValueError("source package changed")
        if digest(args.base_metadata.read_bytes()) != args.base_metadata_sha256:
            raise ValueError("base metadata changed")
        if (
            digest(own_path.read_bytes()) != own_hash
            or dockerfile.read_bytes() != template
        ):
            raise ValueError("staging implementation changed")
        with (args.output / "Dockerfile").open("xb") as stream:
            stream.write(template)
        report.update(status="staged", files=rows)
        with (args.output / "stage-receipt.json").open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        return report
    except BaseException as error:
        # Only our successfully created directory may receive failure evidence.
        with (args.output / "failure.json").open("x") as stream:
            json.dump({"status": "failed", "error_type": type(error).__name__}, stream)
        raise


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "file_count", "bytes", "scipy_version")
            }
        )
    )
