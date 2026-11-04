"""MMMU-Pro (standard 10 options and vision): MMMU-Benchmark/MMMU
268471d0d488258990025331c7528359c324aa25 mmmu-pro/prompts.yaml (CoT),
infer/infer_gpt.py prompt and image order, evaluate.py answer parsing."""

from __future__ import annotations

import ast
import base64
import io
import re
from typing import Any

from kairyu_bench.wsp.catalog import SOURCE_BY_NAME
from kairyu_bench.wsp.manifest import Item
from kairyu_bench.wsp.sources import Adapter, Context, Loader, Outcome, Row, register, run_chat, score_each

COT_PROMPTS = {
    "vision": (
        "Write out the multiple-choice question in the image and then solve it. The last line of your response "
        "should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. "
        "Think step by step before answering."
    ),
    "standard": (
        "Answer the preceding multiple choice question. The last line of your response should be of the following "
        "format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before "
        "answering."
    ),
}
CONFIGS = {"standard": "standard (10 options)", "vision": "vision"}


def _load(loader: Loader) -> dict[str, Row]:
    spec = SOURCE_BY_NAME["mmmu-pro"]
    loaded: dict[str, Row] = {}
    for setting, config in CONFIGS.items():
        dataset = loader.dataset(spec.dataset or "", config=config, split="test", revision=spec.revision)
        # Images decode only when an item runs; rows keep the Arrow dataset.
        metadata = dataset.select_columns(["id", "subject", "answer", "options"])
        for index, row in enumerate(metadata):
            loaded[f"mmmu-pro:{setting}:{row['id']}"] = {
                "setting": setting,
                "id": row["id"],
                "subject": row["subject"],
                "answer": row["answer"],
                "options": ast.literal_eval(str(row["options"])),
                "index": index,
                "dataset": dataset,
            }
    return loaded


def _image_part(image: Any) -> dict[str, Any]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def _messages(row: Row) -> list[dict[str, Any]]:
    record = row["dataset"][row["index"]]
    if row["setting"] == "vision":
        return [{"role": "user", "content": [{"type": "text", "text": COT_PROMPTS["vision"]}, _image_part(record["image"])]}]
    choices = "\n".join(f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(row["options"]))
    text = f"{record['question']}\n{choices}\n{COT_PROMPTS['standard']}"
    order = [int(number) for number in re.findall(r"<image\s+(\d+)>", text)]
    text = re.sub(r"<image\s+\d+>", "<image>", text)
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": text}, *(_image_part(record[f"image_{i}"]) for i in order)],
        }
    ]


def _execute(context: Context, item: Item, row: Row) -> Outcome:
    return run_chat(context, item, _messages(row))


def parse_multi_choice_response(response: str, all_choices: list[str], index2ans: dict[str, str]) -> str | None:
    """evaluate.py's parser; where it would pick a random choice, None (wrong)."""

    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:") :].strip()
        matching_options = [option for option in all_choices if option in answer_str]
        if len(matching_options) == 1:
            return matching_options[0]
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "
    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False
    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                start_indexes = [response.rfind(f"({can})") for can in candidates]
            else:
                start_indexes = [response.rfind(f" {can} ") for can in candidates]
        else:
            start_indexes = [response.lower().rfind(index2ans[can].lower()) for can in candidates]
        return candidates[max(range(len(start_indexes)), key=start_indexes.__getitem__)]
    return candidates[0]


def _rule(row: Row, sample: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    letters = [chr(ord("A") + i) for i in range(len(row["options"]))]
    predicted = parse_multi_choice_response(
        sample["content"], letters, {letter: str(option) for letter, option in zip(letters, row["options"])}
    )
    return float(predicted == row["answer"]), {"extracted": predicted}


register(
    "mmmu-pro",
    Adapter(
        load=_load,
        stratum=lambda row: f"{row['setting']}/{row['subject']}",
        # The vision setting shows the same question as a screenshot.
        group=lambda row: f"mmmu:{row['id']}",
        execute=_execute,
        score=score_each(_rule),
    ),
)
