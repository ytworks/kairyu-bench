#!/usr/bin/env python3
"""Bounded numeric oracle for official Gemma4 logits and public-answer boundaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import recipe_common as common


def select_cases(cases, identifiers):
    if not 1 <= len(identifiers) <= 3 or len(set(identifiers)) != len(identifiers):
        raise ValueError("one to three unique explicit authored case IDs required")
    found = {}
    for purpose, folds in cases.items():
        for fold, rows in enumerate(folds):
            for row in rows:
                if row["id"] in identifiers:
                    if row["id"] in found:
                        raise ValueError("selected case ID is not globally unique")
                    found[row["id"]] = (purpose, fold, row)
    if set(found) != set(identifiers):
        raise ValueError("selected IDs must exist in the explicit cases")
    return [found[name] for name in identifiers]


def legacy_frame(tokenizer, case):
    """Historical assistant-start condition, with no invented thought contents."""
    user = {"role": "user", "content": case["prompt"].strip()}
    marker = "KAIRYU_PUBLIC_ANSWER_BOUNDARY_6e4af21d"
    rendered = tokenizer.apply_chat_template(
        [user, {"role": "assistant", "content": marker}],
        tokenize=False, enable_thinking=True)
    if rendered.count(marker) != 1:
        raise ValueError("legacy public marker is ambiguous")
    prefix, _ = rendered.split(marker)
    if not prefix.endswith("<|turn>model\n"):
        raise ValueError("this diagnostic requires the documented legacy assistant start")
    answer = case["correct"].strip()
    pids = tokenizer.encode(prefix, add_special_tokens=False)
    full = tokenizer.encode(prefix + answer, add_special_tokens=False)
    if full[:len(pids)] != pids or len(full) == len(pids):
        raise ValueError("legacy token boundary differs")
    return pids, full[len(pids):]


def cap_logits(torch, raw, cap, *, upcast):
    values = raw.float() if upcast else raw
    if cap is not None:
        values = values / cap
        values = torch.tanh(values)
        values = values * cap
    return values.float()


def numeric_comparison(torch, official, native, fp32, targets, channel_token_id=None):
    result = {}
    for name, logits in (("official_full_model", official), ("custom_native_softcap", native),
                         ("custom_fp32_softcap", fp32)):
        logp = torch.log_softmax(logits.float(), -1)
        target_logp = logp.gather(1, targets[:, None]).flatten()
        result[name] = {
            "correct_mean_logprob": float(target_logp.mean()),
            "first_target_logprob": float(target_logp[0]),
            "minimum_target_logprob": float(target_logp.min()),
            "maximum_target_logprob": float(target_logp.max()),
            "first_target_rank": int((logits[0] > logits[0, targets[0]]).sum()) + 1,
            "first_channel_token_logprob": (
                float(logp[0, channel_token_id]) if channel_token_id is not None else None),
        }
    for name, logits in (("custom_native_softcap", native), ("custom_fp32_softcap", fp32)):
        result[name]["max_abs_logit_difference_from_official"] = float((logits - official).abs().max())
        result[name]["mean_logprob_difference_from_official"] = (
            result[name]["correct_mean_logprob"] - result["official_full_model"]["correct_mean_logprob"])
    result["answer_tokens"] = int(targets.numel())
    result["logits_saved"] = False
    return result


def run(args, checkpoint, cases, provenance):
    selected = select_cases(cases, args.case_id)
    start = time.monotonic()
    subset = {"diagnostic": [[row for _, _, row in selected]]}
    torch, tokenizer, model = common.load_model(args, cases=subset)
    net, head = model.model.language_model, model.get_output_embeddings()
    prepared, frame_proof = common.prepare(tokenizer, subset, "chat-public-answer", max_frame_tokens=common.frame_limit(args))
    new_rows = {row[0]: row for row in prepared["diagnostic"][0]}
    channel = tokenizer.encode("<|channel>", add_special_tokens=False)
    channel_id = channel[0] if len(channel) == 1 else None
    source = __import__("inspect").getsourcefile(type(model))
    provenance.update(common.observation_runtime(args, provenance))
    provenance["official_modeling_source"] = {
        "path": source, "sha256": common.sha(Path(source)) if source else None}
    provenance["transformers_version"] = common.transformers_version()
    provenance["torch_version"] = torch.__version__
    provenance["frame_evidence"] = frame_proof
    provenance["softcap_arithmetic"] = {
        "official": "head output dtype -> divide -> tanh -> multiply; logsoftmax upcast",
        "legacy_custom": "head output upcast FP32 -> divide -> tanh -> multiply -> logsoftmax",
        "exact_bit_parity_required_between_dtypes": False}
    rows = []
    with torch.inference_mode():
        for purpose, fold, case in selected:
            old_prefix, old_answer = legacy_frame(tokenizer, case)
            _, new_prefix, new_answer, _ = new_rows[case["id"]]
            if old_prefix == new_prefix:
                raise ValueError("new shared frame has not changed the assistant-start boundary")
            if old_answer != new_answer:
                raise ValueError("boundary diagnostic must retain identical public answer tokens")
            for label, prefix in (("legacy_assistant_start", old_prefix),
                                  ("shared_public_final_boundary", new_prefix)):
                sequence = prefix + old_answer
                common.finite_replay.validate_frame_rows([(case["id"], prefix, old_answer, old_answer)], common.frame_limit(args, 512))
                for padding in (0, 8):
                    if time.monotonic() - start > args.max_seconds:
                        raise TimeoutError("diagnostic dispatch budget exceeded")
                    ids = torch.tensor([sequence + [tokenizer.pad_token_id] * padding], device=args.device)
                    mask = torch.tensor([[1] * len(sequence) + [0] * padding], device=args.device)
                    at, stop = len(prefix) - 1, len(sequence) - 1
                    official_output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                    official = official_output.logits[0, at:stop].float()
                    del official_output
                    hidden = net(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
                    raw = head(hidden[0, at:stop])
                    cap = net.config.final_logit_softcapping
                    native = cap_logits(torch, raw, cap, upcast=False)
                    fp32 = cap_logits(torch, raw, cap, upcast=True)
                    target = torch.tensor(old_answer, device=args.device)
                    result = numeric_comparison(torch, official, native, fp32, target, channel_id)
                    result.update(id=case["id"], purpose=purpose, fold=fold, frame=label,
                                  padding=padding, prefix_tokens=len(prefix),
                                  prefix_token_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
                                  head_dtype=str(raw.dtype), final_logit_softcapping=cap)
                    rows.append(result)
                    del ids, mask, official, hidden, raw, native, fp32, target
                    common.save(args.output / "partial.json", {"status": "running", "rows": rows})
    common.assert_inputs(checkpoint, provenance)
    result = {"schema_version": 1, "status": "completed", "rows": rows,
              "seconds_including_model_load": time.monotonic() - start,
              "actual_model_forwards": len(rows) * 2,
              "cases": len(selected), "new_checkpoints": 0, "generated_tokens": 0,
              "source_stat_continuity": True, "provenance": provenance,
              "scope": "Authored teacher-forced numeric diagnostic; no capability/adoption claim.",
              "limits": ["Shared forced public boundary is not a sampled thinking trajectory.",
                         "Wall budget checked between calls, not a hard interrupt of loading or a forward.",
                         "Official BF16 versus FP32 softcap is intentionally measured, not assumed bit-identical.",
                         "Padding and whole-model/submodel differences must be interpreted separately."]}
    common.save(args.output / "report.json", result)
    return result


def main(argv=None):
    parser = common.parser(__doc__, chat=True)
    parser.set_defaults(_intrinsic_serial=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--max-seconds", type=common.positive_int, default=120)
    args = parser.parse_args(argv)
    if args.mode != "chat-public-answer":
        parser.error("this diagnostic accepts only chat-public-answer")
    checkpoint, cases, provenance = common.preflight(args, Path(__file__))
    selected = select_cases(cases, args.case_id)
    if not args.run:
        print(json.dumps({"status": "preview", "cases": [row["id"] for _, _, row in selected],
                          "model_forwards": 0, "output_created": False,
                          "provenance": provenance}))
        return 0
    args.output.mkdir(parents=True)
    common.save(args.output / "provenance.json", provenance)
    try:
        report = run(args, checkpoint, cases, provenance)
    except BaseException as error:
        common.save(args.output / "failure.json",
                    {"status": "incomplete", "error_type": type(error).__name__,
                     "source_stat_continuity": checkpoint.assert_stable() is None})
        raise
    print(json.dumps({"status": report["status"], "cases": report["cases"],
                      "model_forwards": report["actual_model_forwards"]}))
    return 0


if __name__ == "__main__":
    main()
