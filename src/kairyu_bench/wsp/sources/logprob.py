"""Log-likelihood multiple choice (lm-evaluation-harness definitions) and
next-token distribution probes. Both read prompt logprobs, not generations."""

from __future__ import annotations

import math
import re
import threading
from typing import Any

from kairyu_bench.wsp.catalog import SOURCE_BY_NAME
from kairyu_bench.wsp.manifest import Item
from kairyu_bench.wsp.sources import Adapter, Context, Loader, Outcome, Row, Scored, register

# lm-evaluation-harness d6de81643928d653435c431bae19945d41d32520:
# hellaswag/utils.py, arc/arc_easy.yaml, winogrande/preprocess_winogrande.py.
LM_EVAL_REVISION = "d6de81643928d653435c431bae19945d41d32520"
TARGET_DELIMITER = " "
PROBE_TOP_K = 20
PROBE_TOKENS = 512
BOS_TEXT = "<bos>"

_bos_lock = threading.Lock()
_bos: dict[int, list[int]] = {}


def _bos_ids(context: Context) -> list[int]:
    # The reference tokenizer does not add BOS itself; lm-eval prepends it.
    assert context.logprob is not None
    with _bos_lock:
        if not _bos:
            ids = context.logprob.tokenize(BOS_TEXT, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"{BOS_TEXT!r} is not one token: {ids}")
            _bos[0] = ids
        return _bos[0]


def _hellaswag_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def _requests(source: str, row: Row) -> tuple[list[tuple[str, str]], int]:
    """``(context, continuation)`` per choice and the gold index."""

    if source == "hellaswag":
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        query = _hellaswag_preprocess(row["activity_label"] + ": " + ctx)
        choices = [_hellaswag_preprocess(ending) for ending in row["endings"]]
        return [(query, TARGET_DELIMITER + choice) for choice in choices], int(row["label"])
    if source == "arc-challenge":
        query = f"Question: {row['question']}\nAnswer:"
        choices = list(row["choices"]["text"])
        return [(query, TARGET_DELIMITER + choice) for choice in choices], list(
            row["choices"]["label"]
        ).index(row["answerKey"])
    if source == "winogrande":
        index = row["sentence"].index("_")
        target = TARGET_DELIMITER + row["sentence"][index + 1 :].strip()
        contexts = [row["sentence"][:index] + option for option in (row["option1"], row["option2"])]
        return [(ctx, target) for ctx in contexts], {"1": 0, "2": 1}[row["answer"]]
    raise ValueError(source)


def _continuation_logprob(context: Context, prefix: str, continuation: str) -> float:
    assert context.logprob is not None
    # lm-eval _encode_pair: trailing context whitespace moves to the continuation.
    spaces = len(prefix) - len(prefix.rstrip())
    if spaces:
        continuation = prefix[-spaces:] + continuation
        prefix = prefix[:-spaces]
    whole = context.logprob.tokenize(prefix + continuation, add_special_tokens=False)
    head = context.logprob.tokenize(prefix, add_special_tokens=False)
    tokens = _bos_ids(context) + whole
    start = len(_bos_ids(context)) + len(head)
    positions = context.logprob.prompt_logprobs(tokens, top_k=1)
    total = 0.0
    for index in range(start, len(tokens)):
        entry = positions[index]
        assert entry is not None
        total += entry[tokens[index]][0]
    return total


def _execute_choice(context: Context, item: Item, row: Row) -> Outcome:
    pairs, gold = _requests(item.source, row)
    logliks = [_continuation_logprob(context, prefix, continuation) for prefix, continuation in pairs]
    lengths = [len(continuation.encode("utf-8")) for _, continuation in pairs]
    return {"logliks": logliks, "lengths": lengths, "gold": gold}


# acc_norm for HellaSwag and ARC (the commonly reported metric), acc for WinoGrande.
_NORMALIZED = {"hellaswag": True, "arc-challenge": True, "winogrande": False}


def _score_choice(entries: list[tuple[Item, Row, Outcome]], loader: Loader) -> list[Scored]:
    results = []
    for item, _, outcome in entries:
        logliks, lengths, gold = outcome["logliks"], outcome["lengths"], outcome["gold"]
        values = [ll / n for ll, n in zip(logliks, lengths, strict=True)] if _NORMALIZED[item.source] else logliks
        predicted = max(range(len(values)), key=values.__getitem__)
        peak = max(logliks)
        weights = [math.exp(ll - peak) for ll in logliks]
        raw = max(range(len(logliks)), key=logliks.__getitem__)
        correct = float(predicted == gold)
        results.append(
            Scored(
                correct,
                (correct,),
                {
                    "predicted": predicted,
                    "gold": gold,
                    "confidence": weights[raw] / sum(weights),
                    "confidence_correct": float(raw == gold),
                },
            )
        )
    return results


def _load_choice(source: str):  # type: ignore[no-untyped-def]
    spec = SOURCE_BY_NAME[source]

    def load(loader: Loader) -> dict[str, Row]:
        if source == "hellaswag":
            rows = loader.dataset(spec.dataset or "", split="validation", revision=spec.revision)
            return {f"hellaswag:{row['ind']}": dict(row) for row in rows}
        if source == "arc-challenge":
            rows = loader.dataset(spec.dataset or "", config="ARC-Challenge", split="test", revision=spec.revision)
            return {f"arc-challenge:{row['id']}": dict(row) for row in rows}
        rows = loader.dataset(spec.dataset or "", config="winogrande_xl", split="validation", revision=spec.revision)
        return {f"winogrande:{index:04d}": dict(row) for index, row in enumerate(rows)}

    return load


_STRATA = {
    "hellaswag": lambda row: str(row["source_id"]).split("~")[0],  # activitynet / wikihow
    "arc-challenge": lambda row: str(len(row["choices"]["text"])),
    "winogrande": lambda row: "all",
}

for _source in ("hellaswag", "arc-challenge", "winogrande"):
    register(
        _source,
        Adapter(load=_load_choice(_source), stratum=_STRATA[_source], execute=_execute_choice, score=_score_choice),
    )


def _probe_text_rows(source: str, loader: Loader) -> dict[str, Row]:
    spec = SOURCE_BY_NAME[source]
    if source == "probe-wikitext":
        rows = loader.dataset(spec.dataset or "", config="wikitext-2-raw-v1", split="test", revision=spec.revision)
        articles: list[list[str]] = []
        for row in rows:
            line = row["text"]
            if re.fullmatch(r" = [^=].* = \n", line):
                articles.append([line])
            elif articles:
                articles[-1].append(line)
        return {f"probe-wikitext:{index:03d}": {"text": "".join(lines).strip()} for index, lines in enumerate(articles)}
    if source == "probe-wikipedia-ja":
        # One pinned shard (176 MB) instead of the whole 3.7 GB dump.
        rows = loader.dataset(
            spec.dataset or "",
            data_files="20231101.ja/train-00014-of-00015.parquet",
            split="train",
            revision=spec.revision,
        )
        return {
            f"probe-wikipedia-ja:{row['id']}": {"text": row["text"]}
            for row in rows
            if len(row["text"]) >= 1500
        }
    rows = loader.dataset(spec.dataset or "", split="test", revision=spec.revision)
    return {f"probe-code:{row['task_id']}": {"text": row["code"].strip()} for row in rows}


def _execute_probe(context: Context, item: Item, row: Row) -> Outcome:
    assert context.logprob is not None
    text_ids = context.logprob.tokenize(row["text"], add_special_tokens=False)[:PROBE_TOKENS]
    tokens = _bos_ids(context) + text_ids
    positions = context.logprob.prompt_logprobs(tokens, top_k=PROBE_TOP_K)
    records: list[dict[str, Any]] = []
    for index in range(1, len(tokens)):
        entry = positions[index]
        assert entry is not None
        records.append(
            {
                "token": tokens[index],
                "logprob": entry[tokens[index]][0],
                "top": [[token, value] for token, (value, rank) in sorted(entry.items(), key=lambda kv: kv[1][1]) if rank <= PROBE_TOP_K],
            }
        )
    return {"positions": records}


def _score_probe(entries: list[tuple[Item, Row, Outcome]], loader: Loader) -> list[Scored]:
    # A probe's "score" is its mean negative log-likelihood (nats per token).
    results = []
    for _, _, outcome in entries:
        positions = outcome["positions"]
        nll = -sum(entry["logprob"] for entry in positions) / len(positions)
        results.append(Scored(nll, (nll,), {"positions": len(positions)}))
    return results


def _probe_loader(source: str):  # type: ignore[no-untyped-def]
    return lambda loader: _probe_text_rows(source, loader)


for _source in ("probe-wikitext", "probe-wikipedia-ja", "probe-code"):
    register(
        _source,
        Adapter(load=_probe_loader(_source), stratum=lambda row: "all", execute=_execute_probe, score=_score_probe),
    )
