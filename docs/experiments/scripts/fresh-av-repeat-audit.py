#!/usr/bin/env python3
"""Audit same-AV completed dev10 runs without treating tied models as repeats."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, repo = args.project, args.repository
    index_path = repo/"docs/experiments/evidence-2026-11-07-fresh-audit/runs.json"
    index = read(index_path)
    assert len(index) == 151
    av = "5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf"
    fresh_id = "gemma4-fresh-av-observed-dev10-20261107-v1"
    candidates = []
    scanned = []
    for entry in index:
        choices = [base/entry["run_id"] for base in (root/"results", Path("/home/y-takagi/kairyu-bench/results"))]
        matches = [p for p in choices if (p/"run.json").exists() and sha(p/"run.json")==entry["inputs_sha256"]["run.json"]]
        assert matches, entry["run_id"]
        path = matches[0]
        run = read(path/"run.json")
        scanned.append(dict(run_id=entry["run_id"],label=run["label"],tier=run["tier"],split=run["split"],status=run["status"],run_sha256=sha(path/"run.json")))
        if run["label"] == av and run["tier"] == 10 and run["split"] == "dev" and run["status"]=="completed":
            candidates.append(path)
    candidates.append(root/"results"/fresh_id)
    assert len(candidates)==2
    runs = []
    code = {}
    inputs = {str(index_path):sha(index_path)}
    cohorts = []
    for path in candidates:
        run, report = read(path/"run.json"), read(path/"report.json")
        items = [json.loads(line) for line in (path/"items.jsonl").read_text().splitlines()]
        assert run["label"]==av and run["tier"]==10 and run["split"]=="dev" and run["status"]=="completed"
        assert len(items)==len({x["id"] for x in items})==run["items"]==778 and not run["unscored"]
        assert all(x["score"] is not None for x in items)
        cohorts.append({x["id"]: {k:x[k] for k in ("source","area","kind","max_tokens","samples")} for x in items})
        code[path.name]={x["id"]:x for x in items if x["area"]=="code"}
        assert len(code[path.name])==26
        sources={}
        for source in ("humaneval-plus","mbpp-plus","livecodebench"):
            rows=[x for x in code[path.name].values() if x["source"]==source]
            sources[source]=dict(count=len(rows),score=sum(x["score"] for x in rows)/len(rows))
        mean=sum(x["score"] for x in sources.values())/3
        assert abs(mean-report["areas"]["code"])<1e-12
        runs.append(dict(run_id=path.name,status=run["status"],items=778,scored=778,unscored=0,
                         started_at=run["started_at"],finished_at=run["finished_at"],sources=sources,
                         code_source_mean=mean,overall=report["overall"],timing=run["timing"],
                         manifest_sha256=run["manifest_sha256"],capture_id=run["observation"]["capture_id"]))
        inputs.update({str(path/name):sha(path/name) for name in ("run.json","report.json","items.jsonl")})
    assert cohorts[0]==cohorts[1] and runs[0]["manifest_sha256"]==runs[1]["manifest_sha256"]
    old_dir=root/"analysis/code90-resume-20261106/current-ad-av-retention-full14-v1"
    fresh_dir=root/"analysis/fresh-purpose-20261107-av-live10-v1"
    condition_paths=[old_dir/"full14-pair-launch.json",old_dir/"full14-actual-runner-oci.json",
                     old_dir/"plan.json",old_dir/"observers/av/capture.json",
                     fresh_dir/"plan.json",fresh_dir/"capture/capture.json",
                     fresh_dir/"runner-image-source.json",fresh_dir/"wsp-command.json"]
    inputs.update({str(p):sha(p) for p in condition_paths})
    launch,old_runner,old_plan,old_capture,new_plan,new_capture,new_runner,new_command=map(read,condition_paths)
    job=next(job for job in launch["jobs"] if job["role"]=="av")
    assert job["tree"]==av==new_capture["checkpoint"]["sha256"]
    assert job["candidate"]==new_capture["checkpoint"]["path"]==new_plan["original_container"]["checkpoint"]
    old_runner=old_runner["rows"]["av"]
    old_source=old_runner["installed_source_audit"]["source_files"]
    new_source={k.removeprefix("src/kairyu_bench/"):v for k,v in new_runner["source_files_sha256"].items() if k.startswith("src/kairyu_bench/")}
    changes={k:dict(old=old_source[k],new=new_source[k]) for k in sorted(old_source.keys()&new_source.keys()) if old_source[k]!=new_source[k]}
    relevant_prefixes=("wsp/run.py","wsp/client.py","wsp/scorers/","wsp/sources/","wsp/select.py","wsp/manifest.py","wsp/report.py","data/wsp/")
    assert not any(k.startswith(relevant_prefixes) for k in changes)
    def option(command,name):
        return command[command.index(name)+1]
    old_concurrency={k:int(option(job["command"],"--"+k+"-concurrency")) for k in ("chat","logprob","agent")}
    new_concurrency={k:int(option(new_command["command"],"--"+k+"-concurrency")) for k in ("chat","logprob","agent")}
    assert old_concurrency==new_concurrency==dict(chat=32,logprob=8,agent=1)
    item_rows=[]
    for item_id,old in sorted(code[runs[0]["run_id"]].items()):
        new=code[runs[1]["run_id"]][item_id]
        item_rows.append(dict(id=item_id,source=old["source"],old_score=old["score"],fresh_score=new["score"],
                              delta=new["score"]-old["score"],old_detail=old["detail"],fresh_detail=new["detail"],
                              old_completion_tokens=sum(x["completion_tokens"] for x in old["outcome"]["samples"]),
                              fresh_completion_tokens=sum(x["completion_tokens"] for x in new["outcome"]["samples"])))
    output=dict(status="complete_read_only_audit",historical_run_count=151,fresh_run_count=1,
        same_checkpoint_completed_dev10_count=2,checkpoint_tree=av,
        exact_current_condition_prior_repeats=0,exact_current_condition_repeat_statement="none",
        reason="Older AV uses request tracing only and a different runner image; fresh AV adds actual serving activation capture. The two runs are not an exact-condition repeat pair.",
        tied_other_models_are_not_repeats=True,all_778_selection_and_generation_budgets_equal=True,
        runs=runs,all_26_code_items=item_rows,code_changed_items=[x for x in item_rows if x["delta"]!=0],
        conditions=dict(old=dict(runner_image=old_runner["image_id"],source_head=launch["source_head"],
                                serving_image=job["snapshot"]["image_id"],tp=2,replicas=1,
                                concurrency=old_concurrency,capture_kind="completed HTTP request traces only, no worker files",
                                capture_limits=old_capture["limits"],overlap="Concurrent full AD WSP on GPUs2/3"),
                        fresh=dict(runner_image=new_runner["docker_image_id"],serving_image=new_plan["original_container"]["image"],
                                   topology=new_plan["topology"],concurrency=new_concurrency,
                                   capture_kind="live residual/attention/MLP on all 60 layers, TP0/1",
                                   capture_limits=new_capture["limits"],sampling=new_capture["sampling"],
                                   overlap="F01 finite probe on GPU6 during part of WSP, no overlapping second canonical WSP"),
                        package_source_changed=changes,package_source_added=sorted(new_source.keys()-old_source.keys()),
                        benchmark_selection_client_sources_and_scorers_sha_equal=True),
        scanned_historical_runs=scanned,inputs_sha256=inputs,script_sha256=sha(Path(__file__)),
        model_calls=0,new_benchmark_runs=0)
    with args.output.open("x") as handle:
        json.dump(output,handle,indent=2,allow_nan=False)
        handle.write(chr(10))
    print(json.dumps(dict(same_checkpoint_completed_dev10_count=2,exact_current_condition_prior_repeats=0,
                         code_scores=[r["code_source_mean"] for r in runs],
                         changed_code_items=len(output["code_changed_items"]),source_changed=list(changes))))


if __name__=="__main__":
    main()
