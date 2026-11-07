"""Tokenize a pinned authored training cohort without loading a model."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--cases-sha256', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    data = args.cases.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.cases_sha256:
        raise ValueError('cases SHA mismatch')
    groups = json.loads(data)
    rows = [row for folds in groups.values() for fold in folds for row in fold]
    if (not rows or len(rows) > 256 or any(row['split'] != 'train' for row in rows)
            or len({row['id'] for row in rows}) != len(rows)
            or any(len(fold) != 1 for folds in groups.values() for fold in folds)):
        raise ValueError('unique singleton training rows required')
    if list(Path('/dev').glob('nvidia*')):
        raise ValueError('tokenizer-only container must not expose GPUs')
    from transformers import AutoTokenizer
    from kairyu_bench.wsp.weights.finite_replay import prepare_frames, validate_frame_rows

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    prepared, shared = prepare_frames(tokenizer, groups, max_frame_tokens=None)
    frames = [row for folds in prepared.values() for fold in folds for row in fold]
    lengths = [{'id': row[0], 'prefix': len(row[1]), 'correct': len(row[2]), 'foil': len(row[3]),
                'correct_complete': len(row[1]) + len(row[2]), 'foil_complete': len(row[1]) + len(row[3])}
               for row in frames]
    evidence = {**shared, 'lengths': lengths, 'expected_count': len(rows),
                'over_limit_ids': [row['id'] for row in lengths if max(row['correct_complete'], row['foil_complete']) > 1024],
                'maximum': max(max(row['correct_complete'], row['foil_complete']) for row in lengths),
                'input_sha256': {'train-cases.json': args.cases_sha256}, 'model_forwards': 0,
                'gradient_calls': 0, 'generation_calls': 0, 'max_frame_tokens': 1024,
                'elapsed_seconds': time.monotonic() - started,
                'versions': {name: version(name) for name in ('transformers', 'tokenizers', 'numpy', 'torch')}}
    args.output.mkdir(exist_ok=True)
    for name, value in [('frame-evidence.json', evidence), ('train-frame-evidence.json', shared)]:
        with (args.output / name).open('x') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
    validate_frame_rows(frames, 1024)
    if args.cases.read_bytes() != data:
        raise ValueError('cases changed during tokenization')
    print(json.dumps({key: evidence[key] for key in ('expected_count', 'maximum', 'model_forwards', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
