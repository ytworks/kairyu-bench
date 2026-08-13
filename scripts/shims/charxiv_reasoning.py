from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

from kairyu_bench.adapter_context import read_context
from kairyu_bench.target import Endpoint, TargetClient


def _extract_images(archive: Path, destination: Path) -> None:
    marker = destination / ".complete"
    if marker.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError("CharXiv images archive contains an unsafe path")
        zipped.extractall(destination)
    marker.write_text("ok\n", encoding="utf-8")


def _json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("CharXiv judge response is not an object")
    return value


def main() -> int:
    context_path = os.environ.get("KAIRYU_BENCH_CONTEXT")
    raw_directory = os.environ.get("KAIRYU_BENCH_RAW_DIR")
    source_path = os.environ.get("KAIRYU_BENCH_SOURCE_PATH")
    if not context_path or not raw_directory or not source_path:
        raise RuntimeError("adapter environment is incomplete")
    context = read_context(Path(context_path))

    from get_stats import get_reasoning_scores, get_stats
    from huggingface_hub import hf_hub_download
    from reasoning_utils import build_reasoning_grading_queries, build_reasoning_queries

    dataset = context["dataset"]
    archive = Path(
        hf_hub_download(
            repo_id=dataset["id"],
            filename="images.zip",
            repo_type="dataset",
            revision=dataset["revision"],
            token=os.environ.get("HF_TOKEN") or None,
        )
    )
    images = Path(os.environ.get("KAIRYU_BENCH_CACHE_DIR", "/work/cache")) / "charxiv" / dataset["revision"]
    _extract_images(archive, images)
    image_root = images / "images" if (images / "images").is_dir() else images

    source = Path(source_path)
    reasoning_data = json.loads((source / "data/reasoning_val.json").read_text(encoding="utf-8"))
    descriptive_data = json.loads((source / "data/descriptive_val.json").read_text(encoding="utf-8"))
    image_metadata = json.loads((source / "data/image_metadata_val.json").read_text(encoding="utf-8"))
    queries = build_reasoning_queries(reasoning_data, str(image_root))
    selected_items = list(queries.items())
    limit = context.get("limit")
    if limit is not None:
        selected_items = selected_items[:limit]
    selected_data = {
        str(item_id): reasoning_data[str(item_id)] for item_id, _ in selected_items
    }
    selected_queries = {str(item_id): query for item_id, query in selected_items}
    if not selected_queries:
        raise RuntimeError("CharXiv selected no reasoning questions")

    client = TargetClient(
        Endpoint.parse(context["endpoint"]),
        api_key=os.environ.get("KAIRYU_API_KEY"),
        timeout=300,
    )
    responses: dict[str, dict[str, Any]] = {}
    for figure_id, query in selected_queries.items():
        image_path = Path(query["figure_path"])
        media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response = client.chat(
            context["model_id"],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query["question"]},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        responses[figure_id] = {
            "figure_id": figure_id,
            "raw_question": query["raw_question"],
            "response": response,
        }

    grading = build_reasoning_grading_queries(selected_data, responses)
    scores: dict[str, dict[str, Any]] = {}
    for figure_id, query in grading.items():
        response = client.chat(
            context["model_id"],
            [
                {
                    "role": "user",
                    "content": query["grading_query"]
                    + "\nReturn only a JSON object with extracted_answer and score.",
                }
            ],
            max_tokens=1024,
        )
        judged = _json_response(response)
        score = judged.get("score")
        if isinstance(score, bool) or score not in {0, 1}:
            score = -1
        scores[figure_id] = {
            "figure_id": figure_id,
            "extracted_answer": judged.get("extracted_answer"),
            "score": score,
        }

    stats = get_reasoning_scores(scores, descriptive_data, selected_data, image_metadata)
    stats = get_stats(stats)
    output = Path(raw_directory) / "reasoning_summary.json"
    output.write_text(
        json.dumps(
            {"responses": responses, "scores": scores, "stats": stats},
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
