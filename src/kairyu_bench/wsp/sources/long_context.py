"""Long context: MRCR v2 and LongBench v2 with kairyu-bench's existing prompts
and scorers (``kairyu_bench.direct``), and RULER's synthetic tasks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from kairyu_bench.direct import extract_choice, longbench_item, mrcr_score, token_bin
from kairyu_bench.wsp.catalog import SOURCE_BY_NAME
from kairyu_bench.wsp.manifest import Item
from kairyu_bench.wsp.sources import Adapter, Context, Loader, Outcome, Row, register, run_chat, score_each


def _load_mrcr(loader: Loader) -> dict[str, Row]:
    import tiktoken

    spec = SOURCE_BY_NAME["mrcr-v2"]
    encoder = tiktoken.get_encoding("o200k_base")
    rows: dict[str, Row] = {}
    for index, row in enumerate(loader.dataset(spec.dataset or "", split="train", revision=spec.revision)):
        if row.get("n_needles") != 8:
            continue
        messages = json.loads(row["prompt"])
        answer = str(row["answer"])
        total = sum(len(encoder.encode(str(m.get("content", "")))) for m in messages) + len(encoder.encode(answer))
        upper = token_bin(total)
        if upper is None:
            continue
        rows[f"mrcr-v2:{index:04d}"] = {
            "messages": messages,
            "answer": answer,
            "prepend": str(row["random_string_to_prepend"]),
            "token_bin": upper,
        }
    return rows


def _execute_mrcr(context: Context, item: Item, row: Row) -> Outcome:
    return run_chat(context, item, row["messages"])


register(
    "mrcr-v2",
    Adapter(
        load=_load_mrcr,
        stratum=lambda row: str(row["token_bin"]),
        execute=_execute_mrcr,
        score=score_each(
            lambda row, sample: (mrcr_score(sample["content"], row["answer"], row["prepend"]), {})
        ),
    ),
)


def _load_longbench(loader: Loader) -> dict[str, Row]:
    spec = SOURCE_BY_NAME["longbench-v2"]
    rows: dict[str, Row] = {}
    for row in loader.dataset(spec.dataset or "", split="train", revision=spec.revision):
        # "long" contexts (128K-2M words) exceed the reference's 262,144 tokens.
        if row["length"] not in {"short", "medium"}:
            continue
        built = longbench_item(dict(row))
        rows[f"longbench-v2:{built['id']}"] = {
            "messages": built["messages"],
            "expected": built["expected"],
            "difficulty": row["difficulty"],
            "length": row["length"],
        }
    return rows


register(
    "longbench-v2",
    Adapter(
        load=_load_longbench,
        stratum=lambda row: f"{row['difficulty']}/{row['length']}",
        execute=lambda context, item, row: run_chat(context, item, row["messages"]),
        score=score_each(
            lambda row, sample: (
                float(extract_choice(sample["content"]) == row["expected"]),
                {"extracted": extract_choice(sample["content"])},
            )
        ),
    ),
)


# RULER (NVIDIA/RULER db43f92ff634bbce1c33d1fb15858aec4bba509a): the six
# synthetic tasks that need no downloaded corpus, generated once by
# `wsp prepare` with the reference tokenizer and a fixed seed. The answer
# prefix (a completion-model prefill) is not sent; scoring is RULER's
# string_match_all: the share of expected strings present in the reply.
RULER_REVISION = "db43f92ff634bbce1c33d1fb15858aec4bba509a"
RULER_TASKS = ("niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt", "cwe", "fwe")
RULER_LENGTHS = (32768, 131072)
RULER_SAMPLES = 20
RULER_SEED = 42
TOKENIZER = ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "config.json")


def _ruler_dir(loader: Loader) -> Path:
    if loader.data_dir is None:
        raise RuntimeError("RULER data directory is not configured (KAIRYU_BENCH_WSP_DATA)")
    return loader.data_dir / f"ruler-{TOKENIZER[1][:12]}-seed{RULER_SEED}"


def _ruler_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*/*/validation.jsonl")):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _materialize_lfs(path: Path) -> None:
    """Replace a Git LFS pointer in the pinned checkout by its verified content."""

    from urllib.request import urlopen

    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return
    oid = next(line.split(":", 1)[1] for line in text.splitlines() if line.startswith("oid sha256:"))
    relative = path.relative_to(path.parents[4]).as_posix()  # scripts/data/synthetic/json/<file>
    url = f"https://media.githubusercontent.com/media/NVIDIA/RULER/{RULER_REVISION}/{relative}"
    with urlopen(url, timeout=300) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != oid:
        raise RuntimeError(f"{url} does not match its LFS pointer {oid}")
    path.write_bytes(content)


def generate_ruler(loader: Loader) -> str:
    from huggingface_hub import hf_hub_download

    directory = _ruler_dir(loader)
    tokenizer = directory / "tokenizer"
    for name in TOKENIZER_FILES:
        hf_hub_download(TOKENIZER[0], name, revision=TOKENIZER[1], local_dir=tokenizer)
    scripts = loader.source("ruler") / "scripts" / "data"
    # cwe reads english_words.json, an LFS file in the repository.
    _materialize_lfs(scripts / "synthetic" / "json" / "english_words.json")
    # prepare.py runs its generators with `python` from PATH.
    environment = dict(os.environ, PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}")
    for length in RULER_LENGTHS:
        for task in RULER_TASKS:
            subprocess.run(
                [
                    sys.executable, "prepare.py",
                    "--save_dir", str(directory / str(length)),
                    "--benchmark", "synthetic",
                    "--task", task,
                    "--tokenizer_path", str(tokenizer),
                    "--tokenizer_type", "hf",
                    "--max_seq_length", str(length),
                    "--num_samples", str(RULER_SAMPLES),
                    "--random_seed", str(RULER_SEED),
                    "--model_template_type", "base",
                ],
                cwd=scripts,
                env=environment,
                check=True,
            )
            # prepare.py reports a failing generator but still exits 0.
            output = directory / str(length) / task / "validation.jsonl"
            lines = output.read_text(encoding="utf-8").count("\n") if output.is_file() else 0
            if lines != RULER_SAMPLES:
                raise RuntimeError(f"RULER {task} at {length} produced {lines} of {RULER_SAMPLES} samples")
    return _ruler_digest(directory)


def _load_ruler(loader: Loader) -> dict[str, Row]:
    directory = _ruler_dir(loader)
    rows: dict[str, Row] = {}
    for length in RULER_LENGTHS:
        for task in RULER_TASKS:
            path = directory / str(length) / task / "validation.jsonl"
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    rows[f"ruler:{task}:{length}:{record['index']:03d}"] = {
                        "prompt": record["input"],
                        "outputs": list(record["outputs"]),
                        "stratum": f"{task}/{length}",
                    }
    return rows


def _ruler_rule(row: Row, sample: dict) -> tuple[float, dict]:
    # RULER scripts/eval/evaluate.py postprocess_pred + string_match_all.
    prediction = (sample.get("content") or "").strip()
    prediction = "".join("\n" if ord(ch) < 0x20 else ch for ch in prediction)
    found = [reference.lower() in prediction.lower() for reference in row["outputs"]]
    return sum(found) / len(found), {"found": sum(found), "references": len(found)}


register(
    "ruler",
    Adapter(
        load=_load_ruler,
        stratum=lambda row: str(row["stratum"]),
        execute=lambda context, item, row: run_chat(context, item, [{"role": "user", "content": row["prompt"]}]),
        score=score_each(_ruler_rule),
        equal_strata=True,
    ),
)
