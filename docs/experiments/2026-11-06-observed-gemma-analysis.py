"""Reproduce the 2026-11-06 observed baseline's exploratory CPU analysis.

Uses only recorded public score/usage metadata and activation summaries.
No prompts, final code, hidden reasoning, model execution, or benchmark replay.
Contrasts are observational associations; they do not identify edit directions.
Run with PYTHONPATH=src and the four explicit path options below.
"""

from pathlib import Path
import argparse
import json
import re
import math
import hashlib
import collections
import datetime
import numpy as np
from kairyu_bench.wsp.observe.core import json_lines

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--capture", type=Path, required=True)
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--snapshot", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
capture, results, output = (
    args.capture.resolve(),
    args.results.resolve(),
    args.output_dir.resolve(),
)
if (
    output == capture
    or capture in output.parents
    or output == results
    or results in output.parents
):
    raise ValueError("analysis output must be outside immutable capture/results")
if output.exists() and any(output.iterdir()):
    raise ValueError("analysis output must be new or empty")
cfg = json.loads((capture / "capture.json").read_text())
snapshot = json.loads(args.snapshot.read_text())
assert snapshot["capture_path"] == str(capture) and snapshot["run_id"] == results.name
output.mkdir(parents=True, exist_ok=True)
for f in snapshot["files"]:
    p = capture / f["path"]
    s = p.stat()
    assert s.st_size == f["size"] and s.st_mtime_ns == f["mtime_ns"], str(p)
items = {
    x["id"]: {k: x[k] for k in ("id", "source", "area", "score")}
    for x in json_lines(results / "items.jsonl")
}
traces = {x["trace_id"]: x for x in json_lines(capture / "requests-client.jsonl")}
accept = {}
for p in sorted(capture.glob("workers/*/acceptance.jsonl")):
    for x in json_lines(p):
        key = (x["replica"], x["step"], x["request_id"])
        assert key not in accept or accept[key] == x
        accept[key] = x
pattern = re.compile(r"observe-([0-9a-f]{32})-([0-9a-f]{32})")
reqs = {}
byrequest = {}
rowcount = collections.Counter()
primary_sites = None
paired = {}
tpstat = collections.Counter()
max_tp_diff = 0.0
selected = {i for i, x in items.items() if x["area"] in ("code", "tools")}
for worker in sorted((capture / "workers").iterdir()):
    if not worker.is_dir():
        continue
    meta = json.loads((worker / "sites.json").read_text())
    rank = (
        meta["rank"]
        if "rank" in meta
        else json.loads((worker / "status.json").read_text())["rank"]
    )
    sites = meta["sites"]
    site_ids = [s["site_id"] for s in sites]
    if primary_sites is None:
        primary_sites = sites
    assert [s["site_id"] for s in primary_sites] == site_ids
    for b in json_lines(worker / "observations.jsonl.gz"):
        assert b["capture_id"] == cfg["capture_id"] and b["site_ids"] == site_ids
        assert b["metric_names"] == [
            "mean",
            "rms",
            "max_abs",
            "nonfinite_count",
            "std",
            "zero_fraction",
            "outlier_fraction",
        ]
        rows = b["rows"]
        values = None
        for j, row in enumerate(rows):
            rid, pos, phase, spec, ok = row
            m = pattern.search(rid)
            t = traces.get(m.group(1)) if m else None
            if t is None:
                rowcount["unlinked_rank" + str(rank)] += 1
                continue
            if t.get("run_id") != results.name:
                rowcount["other_run_rank" + str(rank)] += 1
                continue
            key = (b["replica"], b["step"], rid)
            if ok is None and spec is not None and key in accept:
                ar = accept[key]
                assert 0 <= spec < ar["scheduled_draft_tokens"]
                ok = spec < ar["accepted_draft_tokens"]
            item = t["item_id"]
            usage = t.get("usage") or {}
            plen = usage.get("prompt_tokens")
            offset = pos - plen if phase == "decode" and type(plen) is int else None
            rowcount[(f"rank{rank}", phase, str(ok))] += 1
            if rank == 0:
                rq = reqs.setdefault(
                    rid,
                    {
                        "item_id": item,
                        "source": items[item]["source"],
                        "operation": t.get("operation"),
                        "request_scope": t.get("request_scope"),
                        "prompt_tokens": plen,
                        "completion_tokens": usage.get("completion_tokens"),
                        "prefill": set(),
                        "decode_accepted": set(),
                        "decode_rejected": set(),
                        "decode_unknown": set(),
                        "decode_absolute": set(),
                        "trace_id": t["trace_id"],
                    },
                )
                if phase == "prefill" and ok is True:
                    rq["prefill"].add(pos)
                if phase == "decode":
                    rq["decode_absolute"].add(pos)
                    target = (
                        "decode_accepted"
                        if ok is True
                        else "decode_rejected"
                        if ok is False
                        else "decode_unknown"
                    )
                    rq[target].add(offset if offset is not None else pos)
            if item not in selected:
                continue
            if values is None:
                values = np.asarray(b["values"], dtype=np.float64)
            # Audit same native full-width role values across TP ranks; never sum them.
            tk = (b["replica"], rid, phase, pos)
            vec = values[:, j, :]
            if rank == 0:
                paired[tk] = vec.astype(np.float32)
            elif tk in paired:
                v0 = paired.pop(tk).astype(np.float64)
                fin = np.isfinite(v0) & np.isfinite(vec)
                max_tp_diff = max(
                    max_tp_diff, float(np.max(np.abs(v0[fin] - vec[fin]), initial=0))
                )
                tpstat["paired_rows"] += 1
                tpstat["nonfinite_pattern_mismatch"] += int(
                    not np.array_equal(np.isfinite(v0), np.isfinite(vec))
                )
                tpstat["numeric_same_rows"] += int(
                    np.array_equal(v0, vec, equal_nan=True)
                )
            else:
                tpstat["rank1_without_rank0"] += 1
            if rank != 0 or ok is not True:
                continue
            if phase == "prefill":
                band = "prefill_first32"
            elif offset is None:
                band = "decode_unknown_offset"
            elif offset <= 16:
                band = "decode_0_16"
            elif offset <= 256:
                band = "decode_32_256"
            else:
                band = "decode_384_plus"
            k = (item, rid, band)
            if k not in byrequest:
                byrequest[k] = [np.zeros_like(vec), np.zeros_like(vec, dtype=np.int32)]
            acc, cnt = byrequest[k]
            finite = np.isfinite(vec)
            acc += np.where(finite, vec, 0)
            cnt += finite
    print("streamed", worker.name, flush=True)
# Equal requests within each task; each row within request band first averaged.
taskacc = {}
for (item, rid, band), (s, c) in byrequest.items():
    v = np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)
    k = (item, band)
    if k not in taskacc:
        taskacc[k] = [np.zeros_like(v), np.zeros_like(c)]
    a, n = taskacc[k]
    fin = np.isfinite(v)
    a += np.where(fin, v, 0)
    n += fin
taskmeans = {
    k: np.divide(a, n, out=np.full_like(a, np.nan), where=n > 0)
    for k, (a, n) in taskacc.items()
}


def cat(item):
    if not item.startswith("bfcl:"):
        return items[item]["source"]
    sub = item.split(":")[1]
    if "irrelevance" in sub:
        return "bfcl:irrelevance"
    if sub.startswith("multi_turn"):
        return "bfcl:multi_turn"
    if sub.startswith("live_"):
        return "bfcl:live"
    return "bfcl:non_live"


features = {}
for item in selected:
    rr = [
        x
        for x in reqs.values()
        if x["item_id"] == item
        and x["prompt_tokens"] is not None
        and x["completion_tokens"] is not None
    ]
    if rr:
        features[item] = np.array(
            [
                np.median([math.log2(1 + x[f]) for x in rr])
                for f in ("prompt_tokens", "completion_tokens")
            ]
        )
comparisons = []
for item in sorted(selected):
    if items[item]["score"] == 1 or item not in features:
        continue
    controls = sorted(
        [
            j
            for j in selected
            if items[j]["score"] == 1 and cat(j) == cat(item) and j in features
        ],
        key=lambda j: (float(np.linalg.norm(features[j] - features[item])), j),
    )[:3]
    sitescores = collections.defaultdict(list)
    candidates = []
    for band in ("prefill_first32", "decode_0_16", "decode_32_256", "decode_384_plus"):
        if (item, band) not in taskmeans:
            continue
        ctr = [j for j in controls if (j, band) in taskmeans]
        if len(ctr) < 2:
            continue
        a = taskmeans[item, band]
        control = np.stack([taskmeans[j, band] for j in ctr])
        med = np.median(control, axis=0)
        ratios = np.log2(np.maximum(a[:, 1], 1e-20) / np.maximum(med[:, 1], 1e-20))
        for k, site in enumerate(primary_sites):
            if not np.isfinite(ratios[k]):
                continue
            sitescores[site["site_id"]].append(float(ratios[k]))
        top = np.argsort(-np.abs(ratios))[:8]
        candidates.append(
            {
                "band": band,
                "control_count": len(ctr),
                "top_rms_ratios": [
                    {
                        "site": site_ids[k],
                        "log2_ratio": float(ratios[k]),
                        "failed_rms": float(a[k, 1]),
                        "controls_median_rms": float(med[k, 1]),
                        "failed_outlier_fraction": float(a[k, 6]),
                        "controls_outlier_fraction": float(med[k, 6]),
                    }
                    for k in top
                ],
            }
        )
    consensus = sorted(
        [
            {
                "site": site,
                "bands": len(v),
                "median_log2_ratio": float(np.median(v)),
                "sign_consistent": all(x >= 0 for x in v) or all(x <= 0 for x in v),
            }
            for site, v in sitescores.items()
            if len(v) >= 2
        ],
        key=lambda x: (-x["sign_consistent"], -abs(x["median_log2_ratio"])),
    )[:12]
    comparisons.append(
        {
            "item_id": item,
            "score": items[item]["score"],
            "source_group": cat(item),
            "controls": [
                {
                    "item_id": j,
                    "log2_length_distance": float(
                        np.linalg.norm(features[j] - features[item])
                    ),
                }
                for j in controls
            ],
            "contrasts": candidates,
            "consistent_sites": consensus,
        }
    )
coverage = {}
for source in sorted({x["source"] for x in items.values()}):
    rr = [x for x in reqs.values() if x["source"] == source]
    known = [
        x
        for x in rr
        if x["prompt_tokens"] is not None and x["completion_tokens"] is not None
    ]
    cv = {
        "items_observed": len({x["item_id"] for x in rr}),
        "items_total": sum(x["source"] == source for x in items.values()),
        "engine_requests": len(rr),
        "known_usage_requests": len(known),
        "requests_with_prefill": sum(bool(x["prefill"]) for x in rr),
        "requests_with_decode": sum(bool(x["decode_absolute"]) for x in rr),
        "requested_offsets": {},
    }
    for off in cfg["decode_offsets"]:
        # Strict interior input position: last generated token need not be executed.
        eligible = [x for x in known if x["completion_tokens"] > off + 1]
        cv["requested_offsets"][str(off)] = {
            "strict_interior_reached": len(eligible),
            "accepted_observed": sum(off in x["decode_accepted"] for x in eligible),
            "first_execution_rejected": sum(
                off in x["decode_rejected"] for x in eligible
            ),
            "acceptance_unknown": sum(off in x["decode_unknown"] for x in eligible),
        }
    coverage[source] = cv
savedreq = []
for rid, r in reqs.items():
    savedreq.append(
        {
            "request_id": rid,
            **{k: sorted(v) if isinstance(v, set) else v for k, v in r.items()},
        }
    )
meta = {
    "schema_version": 1,
    "run_id": results.name,
    "capture_id": cfg["capture_id"],
    "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "primary_rank": 0,
    "aggregation": "mean tokens per request/phase band, then equal request mean per item; controls are three nearest successful items in same source/BFCL group by log2(prompt+1),log2(completion+1) median lengths",
    "limitations": [
        "observational associations, not capability or causality; output length is itself an outcome",
        "few failures; no statistical-significance claim or edit direction",
        "LCB arc191_c fixed-output grading may reject valid constructive answers; do not use its contrast as capability failure",
        "all semantic thinking/code boundaries unknown",
        "no channel/head capture; layer/role only",
        "prefix-cache and first-execution rejected MTP positions are missing; no zero imputation",
        "strict interior reached uses API completion_tokens and may not correspond exactly to token stop handling",
        "closed acknowledgement retained per worker in snapshot; all queues drained and gzip EOF validated",
    ],
    "row_counts": {
        "|".join(k) if isinstance(k, tuple) else k: v for k, v in rowcount.items()
    },
    "tp_audit": dict(
        tpstat,
        rank0_unmatched_rows=len(paired),
        max_absolute_metric_difference=max_tp_diff,
        equality_semantics="decoded numerical equality with equal_nan=True; not signed-zero/NaN bit-pattern identity",
    ),
    "coverage": coverage,
    "failure_contrasts": comparisons,
}
(output / "activation-analysis.json").write_text(
    json.dumps(meta, indent=2, allow_nan=False) + "\n"
)
(output / "observed-request-coverage.json").write_text(
    json.dumps(savedreq, indent=2, allow_nan=False) + "\n"
)
np.savez_compressed(
    output / "code-tools-task-phase-metrics.npz",
    keys=np.array(["|".join(k) for k in taskmeans]),
    values=np.stack(list(taskmeans.values())),
    site_ids=np.array(site_ids),
    metrics=np.array(b["metric_names"]),
)
print(
    "DONE",
    json.dumps(
        {
            "requests": len(reqs),
            "observed_items": len({x["item_id"] for x in reqs.values()}),
            "tp_audit": meta["tp_audit"],
            "source_coverage": {
                k: {
                    "items": v["items_observed"],
                    "total": v["items_total"],
                    "requests": v["engine_requests"],
                }
                for k, v in coverage.items()
            },
        }
    ),
    flush=True,
)
