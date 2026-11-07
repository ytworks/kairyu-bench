"""CPU-only CLI for checkpoint observations. Model files are opened read-only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, help="write JSON atomically; default stdout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-hash", action="store_true", help="record file metadata without full content hashes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kairyu-bench wsp weights")
    commands = parser.add_subparsers(dest="operation", required=True)
    inspect = commands.add_parser("inspect", help="stream tensor statistics and storage inventory")
    inspect.add_argument("checkpoint", type=Path)
    diff = commands.add_parser("diff", help="measure stored and numerical changes between checkpoints")
    diff.add_argument("base", type=Path)
    diff.add_argument("candidate", type=Path)
    diff.add_argument("--allow-partial", action="store_true", help="compare the compatible subset explicitly")
    for command in (inspect, diff):
        _common(command)
        command.add_argument("--pattern", help="select tensor names by regular expression")
        command.add_argument("--group-pattern", help="group names by regex match or first capture")
        command.add_argument("--chunk-mib", type=_positive, default=8)
        command.add_argument("--sample-size", type=_positive, default=4096)
        command.add_argument("--top-k", type=_positive, default=8)
        command.add_argument("--axis", action="store_true", help="include rank-two row and column profiles")
    spectrum = commands.add_parser("spectrum", help="inspect singular values of reproducible submatrices")
    spectrum.add_argument("checkpoint", type=Path)
    spectrum.add_argument("--tensor", action="append", required=True)
    spectrum.add_argument("--sample-rows", type=_positive, default=128)
    spectrum.add_argument("--sample-columns", type=_positive, default=128)
    _common(spectrum)
    align = commands.add_parser("align", help="compare explicitly corresponding weight channel vectors")
    align.add_argument("checkpoint", type=Path)
    align.add_argument("--left", required=True)
    align.add_argument("--right", required=True)
    align.add_argument("--left-axis", choices=("rows", "columns"), default="rows")
    align.add_argument("--right-axis", choices=("rows", "columns"), default="rows")
    align.add_argument("--channels", type=_positive, default=128)
    align.add_argument("--features", type=_positive, default=128)
    _common(align)
    return parser


def _validate_output(path: Path | None, protected: Sequence[Path], roots: Sequence[Path]) -> None:
    if path is None:
        return
    from kairyu_bench.wsp.weights.checkpoint import CONFIG_FILES

    resolved = path.resolve()
    for source in protected:
        if resolved == source.resolve() or (path.exists() and os.path.samefile(path, source)):
            raise ValueError(f"output would overwrite checkpoint input: {path}")
    # Creating a previously absent config/index can change how a model loads.
    # Also protect unreferenced weight files without claiming them in the hash.
    reserved = (resolved.name in CONFIG_FILES or resolved.name.endswith(".safetensors")
                or resolved.name.endswith(".safetensors.index.json"))
    if reserved and any(resolved.is_relative_to(root.resolve()) for root in roots):
        raise ValueError(f"output would alter checkpoint files: {path}")


def _output(path: Path | None, value: dict[str, Any], protected: Sequence[Path],
            roots: Sequence[Path] = ()) -> None:
    _validate_output(path, protected, roots)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(str(path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    try:
        from kairyu_bench.wsp.weights.checkpoint import Checkpoint
        from kairyu_bench.wsp.weights.matrix import alignment, spectra
        from kairyu_bench.wsp.weights.statistics import diff_checkpoints, inspect_checkpoint

        checkpoints = [Checkpoint(args.base), Checkpoint(args.candidate)] if args.operation == "diff" else [
            Checkpoint(args.checkpoint)
        ]
        protected = [path for checkpoint in checkpoints for path in checkpoint.files]
        roots = [checkpoint.root for checkpoint in checkpoints]
        _validate_output(args.output, protected, roots)
        common = {"seed": args.seed, "hash_files": not args.no_hash}
        if args.operation in {"inspect", "diff"}:
            options = {
                **common,
                "pattern": args.pattern,
                "group_pattern": args.group_pattern,
                "chunk_bytes": args.chunk_mib * 1024 * 1024,
                "sample_size": args.sample_size,
                "top_k": args.top_k,
                "axis": args.axis,
            }
            result = (
                diff_checkpoints(*checkpoints, allow_partial=args.allow_partial, **options)
                if args.operation == "diff"
                else inspect_checkpoint(checkpoints[0], **options)
            )
        elif args.operation == "spectrum":
            result = spectra(
                checkpoints[0], args.tensor, sample_rows=args.sample_rows,
                sample_columns=args.sample_columns, **common,
            )
        else:
            result = alignment(
                checkpoints[0], args.left, args.right, left_axis=args.left_axis,
                right_axis=args.right_axis, channels=args.channels, features=args.features, **common,
            )
        _output(args.output, result, protected, roots)
        return 0
    except ImportError as error:
        parser.exit(2, f"weights dependencies unavailable ({error}); install kairyu-bench with pip install -e .\n")
    except (ValueError, OSError) as error:
        parser.exit(2, f"kairyu-bench wsp weights: {error}\n")
