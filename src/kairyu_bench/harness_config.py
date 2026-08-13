from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def write_hle_config(path: Path, model_id: str, endpoint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "kairyu": {
                    "model": f"openai/{model_id}",
                    "generation_config": {
                        "api_key_env": "OPENAI_API_KEY",
                        "api_base_url": endpoint,
                    },
                }
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kairyu_bench.harness_config")
    parser.add_argument("kind", choices=["hle"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    model = os.environ.get("KAIRYU_MODEL")
    endpoint = os.environ.get("KAIRYU_ENDPOINT")
    if not model or not endpoint:
        print("kairyu-bench config: adapter environment is incomplete", file=sys.stderr)
        return 2
    write_hle_config(args.output, model, endpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
