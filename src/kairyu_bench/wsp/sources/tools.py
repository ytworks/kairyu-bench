"""Function calling: BFCL v4 (bfcl-eval 2026.3.23) through the OpenAI tools
API, with the official FC handler's request/response handling, multi-turn
loop and checkers. Agentic web-search and memory categories are excluded
(they need SerpAPI and model downloads)."""

from __future__ import annotations

import copy
import functools
import json
import os
import threading
import uuid
from typing import Any

from kairyu_bench.wsp.manifest import Item
from kairyu_bench.wsp.sources import Adapter, Context, Loader, Outcome, Row, Scored, register, reply_record, seed

CATEGORIES = {
    "non_live": ("simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple"),
    "live": ("live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_relevance"),
    "irrelevance": ("irrelevance", "live_irrelevance"),
    "multi_turn": ("multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"),
}
# A registered OpenAI FC configuration with underscore_to_dot: tools sent to
# the model have "." replaced by "_", so the checker maps expected names back.
CHECKER_MODEL = "openbmb_MiniCPM-SALA-FC"
_lock = threading.Lock()


@functools.lru_cache(maxsize=1)
def _bfcl() -> dict[str, Any]:
    os.environ.setdefault("OPENAI_API_KEY", "not-used")  # the handler builds a client it never calls
    os.environ.setdefault("BFCL_PROJECT_ROOT", os.path.join(os.environ.get("KAIRYU_BENCH_CACHE_DIR", "/tmp"), "bfcl"))
    with _lock:
        from bfcl_eval.constants.default_prompts import (
            DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
            MAXIMUM_STEP_LIMIT,
        )
        from bfcl_eval.constants.enums import Language, ModelStyle, ReturnFormat
        from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
        from bfcl_eval.eval_checker.eval_runner import (
            _evaluate_single_ast_entry,
            _evaluate_single_multi_turn_entry,
            _evaluate_single_relevance_entry,
        )
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
            is_empty_execute_response,
        )
        from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
        from bfcl_eval.model_handler.utils import convert_to_tool
        from bfcl_eval.utils import is_java, is_js, load_dataset_entry, load_ground_truth_entry

        handler = OpenAICompletionsHandler(
            model_name=CHECKER_MODEL.replace("_", "/"),
            temperature=None,
            registry_name=CHECKER_MODEL.replace("_", "/"),
            is_fc_model=True,
        )
    return {
        "DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
        "MAXIMUM_STEP_LIMIT": MAXIMUM_STEP_LIMIT,
        "Language": Language,
        "ModelStyle": ModelStyle,
        "ReturnFormat": ReturnFormat,
        "GORILLA_TO_OPENAPI": GORILLA_TO_OPENAPI,
        "_evaluate_single_ast_entry": _evaluate_single_ast_entry,
        "_evaluate_single_multi_turn_entry": _evaluate_single_multi_turn_entry,
        "_evaluate_single_relevance_entry": _evaluate_single_relevance_entry,
        "execute_multi_turn_func_call": execute_multi_turn_func_call,
        "is_empty_execute_response": is_empty_execute_response,
        "convert_to_tool": convert_to_tool,
        "is_java": is_java,
        "is_js": is_js,
        "load_dataset_entry": load_dataset_entry,
        "load_ground_truth_entry": load_ground_truth_entry,
        "handler": handler,
    }


def _load(loader: Loader) -> dict[str, Row]:
    bfcl = _bfcl()
    rows: dict[str, Row] = {}
    for group, categories in CATEGORIES.items():
        for category in categories:
            prompts = bfcl["load_dataset_entry"](category, include_language_specific_hint=True)
            plain = {entry["id"]: entry for entry in bfcl["load_dataset_entry"](category, include_language_specific_hint=False)}
            answers = (
                {}
                if category in CATEGORIES["irrelevance"] or category == "live_relevance"
                else {entry["id"]: entry["ground_truth"] for entry in bfcl["load_ground_truth_entry"](category)}
            )
            for entry in prompts:
                rows[f"bfcl:{entry['id']}"] = {
                    "group": group,
                    "category": category,
                    "entry": entry,
                    "scoring_entry": plain[entry["id"]],
                    "answer": answers.get(entry["id"]),
                }
    return rows


def _tools(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bfcl = _bfcl()
    return bfcl["convert_to_tool"](copy.deepcopy(functions), bfcl["GORILLA_TO_OPENAPI"], bfcl["ModelStyle"].OPENAI_COMPLETIONS)


def _model_responses(reply: Any) -> Any:
    # openai_completion.py _parse_query_response_FC: tool calls as
    # [{name: arguments-json}], otherwise the text content.
    if reply.tool_calls:
        return [{call["function"]["name"]: call["function"]["arguments"]} for call in reply.tool_calls]
    return reply.content


def _assistant_message(reply: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": reply.content or None}
    if reply.tool_calls:
        message["tool_calls"] = reply.tool_calls
    return message


def _single_turn(context: Context, item: Item, row: Row, sample: int) -> dict[str, Any]:
    assert context.chat is not None
    entry = row["entry"]
    reply = context.chat.chat(
        list(entry["question"][0]), max_tokens=item.max_tokens, seed=seed(item.id, sample), tools=_tools(entry["function"])
    )
    record = reply_record(reply)
    record["model_responses"] = _model_responses(reply)
    return record


def _multi_turn(context: Context, item: Item, row: Row, sample: int) -> dict[str, Any]:
    # base_handler.py inference_multi_turn_FC with the FC handler's history.
    assert context.chat is not None
    bfcl = _bfcl()
    entry = copy.deepcopy(row["entry"])
    category = row["category"]
    executor_name = f"wsp_{uuid.uuid4().hex}"
    long_context = "long_context" in category or "composite" in category
    holdout = entry.get("missed_function", {})
    functions = list(entry["function"])
    messages: list[dict[str, Any]] = []
    responses: list[list[Any]] = []
    calls: list[dict[str, Any]] = []
    force_quit = False
    for turn, turn_messages in enumerate(entry["question"]):
        if str(turn) in holdout:
            functions.extend(holdout[str(turn)])
            turn_messages = [{"role": "user", "content": bfcl["DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC"]}]
        messages.extend(turn_messages)
        turn_responses: list[Any] = []
        steps = 0
        while True:
            reply = context.chat.chat(
                messages,
                max_tokens=item.max_tokens,
                seed=seed(f"{item.id}:{turn}:{steps}", sample),
                tools=_tools(functions),
            )
            calls.append(reply_record(reply))
            model_responses = _model_responses(reply)
            turn_responses.append(model_responses)
            messages.append(_assistant_message(reply))
            try:
                decoded = bfcl["handler"].decode_execute(model_responses, has_tool_call_tag=False)
                if bfcl["is_empty_execute_response"](decoded):
                    break
            except Exception:  # noqa: BLE001 - the official loop ends the turn on any decode error
                break
            results, _ = bfcl["execute_multi_turn_func_call"](
                decoded, entry.get("initial_config", {}), entry["involved_classes"], executor_name, entry["id"],
                long_context=long_context, is_evaL_run=False,
            )
            for result, call in zip(results, reply.tool_calls, strict=False):
                messages.append({"role": "tool", "content": result, "tool_call_id": call["id"]})
            steps += 1
            if steps > bfcl["MAXIMUM_STEP_LIMIT"]:
                force_quit = True
                break
        responses.append(turn_responses)
        if force_quit:
            break
    return {
        "calls": calls,
        "model_responses": responses,
        "completion_tokens": sum(call["completion_tokens"] for call in calls),
        "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
        "finish_reason": "length" if any(call["finish_reason"] == "length" for call in calls) else "stop",
        "content": "",
    }


def _execute(context: Context, item: Item, row: Row) -> Outcome:
    run = _multi_turn if row["group"] == "multi_turn" else _single_turn
    return {"samples": [run(context, item, row, sample) for sample in range(item.samples)]}


def _verdict(row: Row, sample: dict[str, Any]) -> dict[str, Any]:
    bfcl = _bfcl()
    category, entry = row["category"], copy.deepcopy(row["scoring_entry"])
    if row["group"] == "multi_turn":
        return bfcl["_evaluate_single_multi_turn_entry"](
            bfcl["handler"], entry["id"], sample["model_responses"], row["answer"], entry,
            f"wsp_{uuid.uuid4().hex}", category,
        )
    if category in CATEGORIES["irrelevance"] or category == "live_relevance":
        return bfcl["_evaluate_single_relevance_entry"](
            bfcl["handler"], entry["id"], sample["model_responses"], entry, CHECKER_MODEL, category
        )
    if bfcl["is_java"](category):
        language, return_format = bfcl["Language"].JAVA, bfcl["ReturnFormat"].JAVA
    elif bfcl["is_js"](category):
        language, return_format = bfcl["Language"].JAVASCRIPT, bfcl["ReturnFormat"].JAVASCRIPT
    else:
        language, return_format = bfcl["Language"].PYTHON, bfcl["ReturnFormat"].PYTHON
    return bfcl["_evaluate_single_ast_entry"](
        bfcl["handler"], entry["id"], sample["model_responses"], row["answer"], entry, CHECKER_MODEL,
        category, language, return_format,
    )


def _score(entries: list[tuple[Item, Row, Outcome]], loader: Loader) -> list[Scored]:
    results = []
    for _, row, outcome in entries:
        values: list[float] = []
        details: list[dict[str, Any]] = []
        for sample in outcome["samples"]:
            if sample.get("finish_reason") == "length":
                values.append(0.0)
                details.append({"truncated": True})
                continue
            verdict = _verdict(row, sample)
            values.append(float(bool(verdict["valid"])))
            details.append({"error_type": verdict.get("error_type") or (verdict.get("error") or {}).get("error_type")} if not verdict["valid"] else {})
        results.append(Scored(sum(values) / len(values), tuple(values), {"samples": json.loads(json.dumps(details, default=str))}))
    return results


register(
    "bfcl",
    Adapter(load=_load, stratum=lambda row: str(row["group"]), execute=_execute, score=_score, equal_strata=True),
)
