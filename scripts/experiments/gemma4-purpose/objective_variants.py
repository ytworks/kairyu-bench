"""Explicit authored-answer objective variants; legacy shared scorer is unchanged."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import recipe_common as common


def correct_logprob_objective(torch, net, head, tokenizer, batch, mode, device):
    """Same good+foil forward shape as margin; only good means enter this loss."""
    if mode != "chat-public-answer":
        raise ValueError("correct-logprob requires chat-public-answer frames")
    cap = float(net.config.final_logit_softcapping)
    seq, lengths, targets = [], [], []
    for _, prefix, good, bad in batch:
        for answer in (good, bad):
            seq.append(prefix + answer)
            lengths.append(len(prefix))
            targets.append(answer)
    width = max(map(len, seq))
    ids = torch.tensor([row + [tokenizer.pad_token_id] * (width - len(row)) for row in seq], device=device)
    mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in seq], device=device)
    hidden = net(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
    selected = torch.cat([hidden[i, n - 1:len(seq[i]) - 1] for i, n in enumerate(lengths)])
    logits = head(selected).float()
    if cap:
        logits = cap * torch.tanh(logits / cap)
    answer = torch.tensor([token for row in targets for token in row], device=device)
    logp = torch.log_softmax(logits, -1).gather(1, answer[:, None]).flatten()
    means, at = [], 0
    for row in targets:
        means.append(logp[at:at + len(row)].mean())
        at += len(row)
    return torch.stack(means)[::2]


def single_answer_logprob_objective(torch, net, head, tokenizer, row, answer_kind,
                                    mode, device, padded_width):
    """One answer graph, retaining the caller's original paired-pack width."""
    if mode != "chat-public-answer" or answer_kind not in {"correct", "foil"}:
        raise ValueError("single answer requires explicit chat-public-answer selection")
    _, prefix, good, bad = row
    answer = good if answer_kind == "correct" else bad
    if type(padded_width) is not int or padded_width < len(prefix) + max(len(good), len(bad)):
        raise ValueError("single-answer width must preserve the original paired padding")
    sequence = prefix + answer
    ids = torch.tensor([sequence + [tokenizer.pad_token_id] * (padded_width - len(sequence))], device=device)
    mask = torch.tensor([[1] * len(sequence) + [0] * (padded_width - len(sequence))], device=device)
    hidden = net(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
    selected = hidden[0, len(prefix) - 1:len(sequence) - 1]
    logits = head(selected).float()
    cap = float(net.config.final_logit_softcapping)
    if cap:
        logits = cap * torch.tanh(logits / cap)
    target = torch.tensor(answer, device=device)
    values = torch.log_softmax(logits, -1).gather(1, target[:, None]).flatten()
    return values.mean().reshape(1)


def objective_references(cases, measured, tolerance, objective):
    if objective == "margin":
        return common.check_references(cases, measured, tolerance)
    values = []
    for row in cases:
        if "expected_correct_logprob" in row:
            value = row["expected_correct_logprob"]
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("finite expected correct log probability required")
            values.append(abs(measured[row["id"]] - value))
    maximum = max(values, default=None)
    if maximum is not None and maximum > tolerance:
        raise ValueError("stored reference correct log probability differs")
    return {"reference_cases": len(values), "max_reference_error": maximum,
            "reference_objective": "correct-logprob"}


def pin_objective(provenance, mode, objective):
    if objective not in {"margin", "correct-logprob"} or objective == "correct-logprob" and mode != "chat-public-answer":
        raise ValueError("unsupported objective/frame combination")
    path = Path(__file__).resolve()
    before = common.input_state(path)
    digest = common.sha(path)
    if common.input_state(path) != before:
        raise ValueError("objective implementation changed while pinning")
    provenance["input_stat"][str(path)] = before
    provenance["input_sha256"][str(path)] = digest
    provenance.update(objective_kind=objective,
                      objective_forward_layout="same good+foil batch/frame/padding; one objective VJP per forward",
                      objective_units="mean correct-answer token log probability (maximize)" if objective == "correct-logprob" else "legacy correct-minus-foil margin")
    return common.objective if objective == "margin" else correct_logprob_objective

def attach_correct_references(path, cases, provenance, objective):
    """Recover only optional reference metadata from the exact preflight bytes."""
    if objective != "correct-logprob":
        return
    path = path.resolve()
    name = str(path)
    before = common.input_state(path)
    content = path.read_bytes()
    if (before != provenance["input_stat"].get(name)
            or common.input_state(path) != before
            or hashlib.sha256(content).hexdigest() != provenance["input_sha256"].get(name)):
        raise ValueError("case bytes/stat changed after preflight")
    if common.read_cases(path, content=content) != cases:
        raise ValueError("normalized cases differ from pinned input")
    document = json.loads(content)
    groups = document.get("cases", document)
    for purpose, folds in groups.items():
        for index, rows in enumerate(folds):
            for position, row in enumerate(rows):
                if "expected_correct_logprob" in row:
                    value = row["expected_correct_logprob"]
                    if type(value) not in (float, int) or not math.isfinite(value):
                        raise ValueError("finite expected correct log probability required")
                    cases[purpose][index][position]["expected_correct_logprob"] = value
