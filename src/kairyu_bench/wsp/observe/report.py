"""Offline observation reports. Rendering does not load weights or contact a model."""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _validate(data: Mapping[str, Any]) -> None:
    if data.get("schema_version") != 1:
        raise ValueError("observation report requires schema_version=1")
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("observation report requires at least one run")
    seen_runs: set[str] = set()
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("id"), str) or not run["id"]:
            raise ValueError("each run requires a nonempty string id")
        if run["id"] in seen_runs:
            raise ValueError(f"duplicate run id: {run['id']}")
        seen_runs.add(run["id"])
        if not isinstance(run.get("tasks"), list) or not isinstance(run.get("sites", []), list):
            raise ValueError("run tasks and sites must be lists")
        site_ids = [site.get("site_id") for site in run.get("sites", []) if isinstance(site, dict)]
        if len(site_ids) != len(run.get("sites", [])) or any(not isinstance(x, str) for x in site_ids):
            raise ValueError("sites require string site_id")
        if len(set(site_ids)) != len(site_ids):
            raise ValueError("duplicate site_id in run")
        seen_tasks: set[tuple[str, str]] = set()
        for task in run["tasks"]:
            if not isinstance(task, dict) or not isinstance(task.get("id"), str):
                raise ValueError("each task requires a string id")
            key = (task.get("source", ""), task["id"])
            if key in seen_tasks:
                raise ValueError(f"duplicate task in run: {key}")
            seen_tasks.add(key)
            rows = task.get("observations", [])
            if not isinstance(rows, list):
                raise ValueError("task observations must be a list")
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("metrics"), dict):
                    raise ValueError("observation requires a metrics object")
                if not isinstance(row.get("request_id"), str) or not row["request_id"]:
                    raise ValueError("observation requires a nonempty request_id")
                if row.get("accepted") is not None and not isinstance(row["accepted"], bool):
                    raise ValueError("accepted must be boolean or null")
                if "sample_count" in row and (isinstance(row["sample_count"], bool) or not isinstance(row["sample_count"], int) or row["sample_count"] < 1):
                    raise ValueError("sample_count must be a positive integer")
                for value in row["metrics"].values():
                    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                        raise ValueError("metrics must be numeric or null")


def build_report(data: Mapping[str, Any]) -> str:
    """Return self-contained HTML for the version-1 runs/tasks/observations schema.

    No raw text is interpolated into markup or executable JavaScript. JSON contains
    only escaped text and dynamic UI nodes are populated with textContent.
    """
    _validate(data)
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for old, new in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                     ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(old, new)
    return _HTML.replace("__OBSERVATION_DATA__", payload).replace("__OBSERVATION_JS__", _JS)


def write_report(data: Mapping[str, Any], output: str | Path) -> Path:
    """Atomically write a local report; no web server is required to view it."""
    html = build_report(data)
    path = Path(output)
    if path.is_symlink():
        raise ValueError("refusing to replace a symlink report")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(html)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return path


_HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>WSP 重み・活性・問題比較</title>
<style>
:root{color-scheme:light;--ink:#192b38;--muted:#536576;--line:#c9d3dc;--bg:#f1f5f7;--accent:#126a83}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,sans-serif}
header,main{max-width:1500px;margin:auto;padding:24px}header{padding-bottom:0}h1{font-size:25px;margin:0}h2{font-size:19px;margin:0 0 12px}h3{font-size:16px}
p{margin:8px 0}.muted,small{color:var(--muted)}.panel{background:#fff;border:1px solid var(--line);border-radius:8px;padding:18px;margin-bottom:18px}
.controls{display:flex;gap:14px;flex-wrap:wrap}.controls label{display:flex;flex-direction:column;gap:4px;min-width:150px;flex:1}
select,input{font:inherit;max-width:100%;padding:7px;border:1px solid #93a8b8;border-radius:4px;background:white;color:var(--ink)}
input[type=checkbox]{margin-right:8px}label.check{display:block;flex:none;align-self:flex-end;padding:8px}
.badge{display:inline-block;padding:3px 9px;margin:3px 6px 3px 0;border:1px solid #9cb8c7;border-radius:5px;font-size:13px;background:#eef5f8}
.warning{background:#fff6df;border-left:4px solid #a66c00;padding:10px}.statline{margin:12px 0;font-variant-numeric:tabular-nums}
.scroll{overflow:auto;max-height:560px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;border-bottom:1px solid var(--line);padding:7px 9px;vertical-align:top}th{background:#f3f7fa;position:sticky;top:0;z-index:1;white-space:nowrap}
button{font:inherit;cursor:pointer;color:#075c79;background:white;border:1px solid #8eacbd;border-radius:4px;padding:4px 8px}
button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #cf8610;outline-offset:2px}
#heatmap{width:max-content;min-width:100%;font-size:12px}#heatmap th:first-child{left:0;z-index:2;max-width:220px;white-space:normal}
#heatmap td{padding:1px;border:1px solid #d5dfe6}#heatmap button{display:block;width:46px;height:30px;padding:0;border:0;border-radius:0;color:#142736;font-size:11px}
.empty{color:#687b89;text-align:center}.columns{display:grid;grid-template-columns:1fr 1fr;gap:16px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:10px;font-size:12px;max-height:350px;overflow:auto}
details{margin:10px 0}summary{cursor:pointer}#task-table tr.selected{background:#e8f5fa}
@media(max-width:800px){header,main{padding:14px}.columns{grid-template-columns:1fr}.controls label{min-width:130px}h1{font-size:21px}}
</style></head><body>
<header><h1>WSP 重み・活性・問題比較</h1>
<p class="muted">保存した観測をローカルで表示します。外部通信・モデル呼出しはありません。</p>
<div id="data-kind"></div></header>
<main>
<section class="panel" aria-label="実験と問題の選択">
<div class="controls">
<label>基準の実験<select id="base"></select></label><label>比較する実験<select id="candidate"></select></label>
<label>能力領域<select id="area"></select></label><label>出典<select id="source"></select></label>
<label>問題検索<input id="search" type="search" placeholder="問題ID"></label>
</div>
<div class="statline" id="outcome-summary" aria-live="polite"></div>
<p class="muted">問題ごとの公式スコアを表示します。未採点を0に補完せず、異なる出典を合算した公式overallは作りません。</p>
<div class="scroll"><table id="task-table"><thead><tr><th>問題</th><th>領域・出典</th><th>基準</th><th>比較</th><th>差</th><th>活性記録 基準/比較</th></tr></thead><tbody id="task-body"></tbody></table></div>
<p id="task-limit" class="muted"></p>
</section>
<section class="panel">
<h2>層 × token位置</h2>
<div class="controls">
<label>対象問題<select id="task"></select></label><label>基準のrequest / call / trial<select id="base-request"></select></label><label>比較のrequest / call / trial<select id="candidate-request"></select></label><label>値<select id="view"><option value="baseline">基準</option><option value="candidate">比較</option><option value="diff">対応あり差分</option></select></label>
<label>指標<select id="metric"></select></label><label>TP rank<select id="rank"></select></label><label>観測段階<select id="phase"></select></label><label>構造の種類<select id="role"></select></label><label>観測site<select id="site"></select></label>
<label>位置の下限<input id="start" type="number" min="0" placeholder="制限なし"></label><label>位置の上限<input id="end" type="number" min="0" placeholder="制限なし"></label>
<label class="check"><input id="accepted-only" type="checkbox">受理確認済みのみ</label>
</div>
<p id="request-scope" class="muted" style="overflow-wrap:anywhere"></p><div id="coverage" class="statline" aria-live="polite"></div>
<p id="pairing" class="warning"></p>
<p class="muted">各セルは選択問題・選択requestの採取範囲です。異なるcall/trialの同位置を平均しません。RMSはsample数で重み付けした二乗平均平方根、meanは加重平均、max_absは最大、nonfinite_countは合計。その他はbin値の加重平均です。差分は対応bin差の加重平均です。TP rankは別の行です。活性量や重みnormだけから能力への因果効果を断定できません。</p>
<div class="scroll"><table id="heatmap" aria-label="activation heatmap"></table></div>
<p id="heatmap-limit" class="muted"></p>
</section>
<section class="panel">
<h2>選択セルと重みの関係</h2>
<p id="cell-summary">セルを選ぶと、観測site・採取範囲・明示された重みの対応を表示します。</p>
<div class="columns"><div><h3>観測</h3><pre id="cell-detail"></pre></div><div><h3>対応する重み</h3><p class="muted">siteに明示されたtensor名のみを結びます。未登録の対応を推測しません。</p><div id="weight-detail"></div></div></div>
</section>
<section class="panel"><h2>採取範囲・実行条件</h2><div id="provenance"></div></section>
</main><script id="observation-data" type="application/json">__OBSERVATION_DATA__</script>
<script>__OBSERVATION_JS__</script></body></html>"""

_JS = r"""
"use strict";
const DATA = JSON.parse(document.getElementById("observation-data").textContent);
const byId = id => document.getElementById(id);
const finite = x => typeof x === "number" && Number.isFinite(x);
const str = x => x === null || x === undefined ? "未記録" : String(x);
const fmt = x => finite(x) ? (x === 0 ? "0" : Math.abs(x) < .001 || Math.abs(x) >= 10000 ? x.toExponential(3) : Number(x.toFixed(5)).toString()) : "未採点/欠測";
const stable = x => JSON.stringify(x, (key, value) => value && typeof value === "object" && !Array.isArray(value) ? Object.fromEntries(Object.keys(value).sort().map(k => [k,value[k]])) : value);
const taskKey = task => JSON.stringify([task.source || "",task.id]);
const node = (tag, text, className) => {const e=document.createElement(tag); if(text !== undefined)e.textContent=text; if(className)e.className=className; return e;};
function options(id, values, desired) {
  const el=byId(id); el.replaceChildren();
  for(const [value,label] of values){const o=node("option",label);o.value=value;el.append(o);}
  el.value=values.some(v=>v[0]===desired) ? desired : (values[0]?.[0] || "");
}
function selectedRuns(){return [DATA.runs.find(r=>r.id===byId("base").value),DATA.runs.find(r=>r.id===byId("candidate").value)];}
function allTaskRows(base,candidate){
  const m=new Map();
  for(const [side,run] of [["baseline",base],["candidate",candidate]])for(const task of run?.tasks || []){
    const key=taskKey(task); if(!m.has(key))m.set(key,{key,id:task.id,source:task.source||"",area:task.area||"未記録"});
    m.get(key)[side]=task;
  }
  return [...m.values()].sort((a,b)=>(a.source+"\0"+a.id).localeCompare(b.source+"\0"+b.id));
}
function visibleTasks(rows, area, source, search){
  return rows.filter(r=>(!area||r.area===area)&&(!source||r.source===source)&&(!search||(r.id+" "+r.source).toLowerCase().includes(search.toLowerCase())));
}
function observationScope(r){return JSON.stringify([r.request_id??null,r.call_id??null,r.trial_id??null,r.request_scope??null,r.replica??null]);}
function requestOptions(records){
  const scopes=new Map();
  for(const r of records)if(!scopes.has(observationScope(r)))scopes.set(observationScope(r),r);
  return [...scopes].map(([key,r],index)=>[key,(index+1)+": "+str(r.request_id)+" · call "+str(r.call_id)+" · trial "+str(r.trial_id)+" · replica "+str(r.replica)]);
}
function siteFor(run, record){return (run?.sites || []).find(s=>s.site_id===record.site_id);}
function pairReason(a,b,ra,rb,ta,tb){
  if(!a.pair_key || a.pair_key!==b.pair_key)return "対応input keyなし";
  if(ta.id!==tb.id || ta.source!==tb.source)return "問題/出典が不一致";
  if(!ra.observation_mode || ra.observation_mode!==rb.observation_mode)return "観測modeが不一致/未記録";
  if(a.accepted!==true || b.accepted!==true)return "出力受理が未確認";
  const fields=["site_id","layer","role","phase","token_kind","token_position","rank"];
  for(const key of fields)if(a[key]===undefined || b[key]===undefined || a[key]!==b[key])return key+"が不一致/未記録";
  if(sampleCount(a)!==sampleCount(b) || (a.token_end??a.token_position)!==(b.token_end??b.token_position))return "token bin範囲/採取数が不一致";
  if((a.request_scope||"request")!==(b.request_scope||"request"))return "request scopeが不一致";
  const sa=siteFor(ra,a),sb=siteFor(rb,b);
  if(!sa?.tp_layout || !sb?.tp_layout || stable(sa.tp_layout)!==stable(sb.tp_layout))return "TP/軸対応が不一致/未記録";
  for(const key of ["parent_site_id","detail_kind","channel_index","head_local_index","head_global_index","head_dim","activation_axis","checkpoint_axes"]){
    if((key in sa || key in sb)&&stable(sa[key])!==stable(sb[key]))return "site."+key+"が不一致";
  }
  for(const key of ["adapter","vllm_version","tp_size","mtp_tokens"]){
    if(ra.runtime?.[key]===undefined || rb.runtime?.[key]===undefined || stable(ra.runtime[key])!==stable(rb.runtime[key]))return "runtime."+key+"が不一致/未記録";
  }
  for(const key of ["dtype","quantization","chat_template_sha256","generation_config_sha256","adapter_version"]){
    if((key in (ra.runtime||{}) || key in (rb.runtime||{})) && stable(ra.runtime?.[key])!==stable(rb.runtime?.[key]))return "runtime."+key+"が不一致";
  }
  return null;
}
function sampleCount(r,metric){
  const n=metric===undefined?undefined:r.metric_sample_counts?.[metric];
  if(Number.isInteger(n)&&n>=0)return n;
  return Number.isInteger(r.sample_count)&&r.sample_count>0?r.sample_count:1;
}
function recordAllowed(r,filters){
  return r.accepted!==false && (!filters.acceptedOnly || r.accepted===true) &&
    (!filters.rank || String(r.rank)===filters.rank) &&
    (!filters.phase || r.phase===filters.phase) && (!filters.role || r.role===filters.role) && (!filters.site || r.site_id===filters.site) &&
    (filters.start===null || (finite(r.token_position)&&r.token_position>=filters.start)) &&
    (filters.end===null || (finite(r.token_position)&&r.token_position<=filters.end));
}
function pairIndex(r){return JSON.stringify([r.pair_key,r.site_id,r.layer,r.role,r.phase,r.token_kind,r.token_position,r.rank]);}
function collectRecords(tasks,base,candidate,side,metric,filters){
  const out=[], reasons=new Map();let total=0,rejected=0,unknown=0,missing=0;
  const note=reason=>reasons.set(reason,(reasons.get(reason)||0)+1);
  for(const task of tasks){
    const left=(task.baseline?.observations||[]).filter(r=>filters.baseRequest===undefined||observationScope(r)===filters.baseRequest);
    const right=(task.candidate?.observations||[]).filter(r=>filters.candidateRequest===undefined||observationScope(r)===filters.candidateRequest);
    if(side!=="diff"){
      const records=side==="candidate"?right:left,run=side==="candidate"?candidate:base;
      for(const record of records){
        total++; if(record.accepted===false)rejected++;else if(record.accepted!==true)unknown++;
        if(!recordAllowed(record,filters))continue;
        if(!finite(record.metrics[metric])||sampleCount(record,metric)===0){missing++;continue;}
        out.push({record,run,task,value:record.metrics[metric],baseline:side==="baseline"?record:null,candidate:side==="candidate"?record:null});
      }
    }else{
      const map=new Map(),leftCounts=new Map();
      for(const r of left)if(r.pair_key)leftCounts.set(pairIndex(r),(leftCounts.get(pairIndex(r))||0)+1);
      for(const r of right)if(r.pair_key){if(!map.has(pairIndex(r)))map.set(pairIndex(r),[]);map.get(pairIndex(r)).push(r);}
      for(const a of left){
        total++;if(!recordAllowed(a,filters))continue;
        if(!a.pair_key){note("明示pair_keyなし");continue;}
        const candidates=map.get(pairIndex(a))||[];
        if(leftCounts.get(pairIndex(a))!==1 || candidates.length>1){note("pair_keyが重複");continue;}
        if(!candidates.length){note("比較側の対応record欠測");continue;}
        const b=candidates[0],reason=pairReason(a,b,base,candidate,task.baseline,task.candidate);
        if(reason){note(reason);continue;}
        if(!recordAllowed(b,filters)){note("比較側がfilter対象外");continue;}
        if(!finite(a.metrics[metric])||!finite(b.metrics[metric])||sampleCount(a,metric)===0||sampleCount(b,metric)===0){missing++;continue;}
        if(sampleCount(a,metric)!==sampleCount(b,metric)){note("指標ごとの採取数が不一致");continue;}
        out.push({record:b,run:candidate,task,value:b.metrics[metric]-a.metrics[metric],baseline:a,candidate:b});
      }
      const keys=new Set(left.filter(r=>r.pair_key).map(pairIndex));
      for(const b of right)if(recordAllowed(b,filters)&&(!b.pair_key||!keys.has(pairIndex(b))))note("基準側の対応record欠測");
    }
  }
  return {records:out,total,rejected,unknown,missing,reasons:[...reasons.entries()]};
}
function makeHeatmap(records,metric="rms",mode="baseline"){
  const cells=new Map(),rowLabels=new Map(),colLabels=new Map();
  const multipleScopes=new Set(records.map(x=>observationScope(x.record))).size>1;
  for(const item of records){
    const r=item.record;
    const rowKey=JSON.stringify([item.task?.key||item.task?.id||null,observationScope(r),r.layer,r.site_id,r.rank,r.role,r.accepted]);
    const colKey=JSON.stringify([r.phase,r.token_position,r.token_end??r.token_position,r.token_kind]);
    const metadata=siteFor(item.run,r)||{};
    const channel=r.channel_index??metadata.channel_index,head=r.head_global_index??metadata.head_global_index;
    const detail=(channel!==undefined?" channel "+str(channel):head!==undefined?" head "+str(head):"")+(multipleScopes?" · req "+str(r.request_id):"");
    rowLabels.set(rowKey,{label:"L"+str(r.layer)+" · "+str(r.role)+detail+" · rank"+str(r.rank)+(r.accepted===true?" · 受理":" · 受理未確認"),layer:r.layer,site:r.site_id,rank:r.rank});
    colLabels.set(colKey,{label:str(r.phase)+":"+str(r.token_position)+(r.token_end!==undefined&&r.token_end!==r.token_position?"–"+str(r.token_end):"")+" "+str(r.token_kind),phase:r.phase,position:r.token_position});
    const key=JSON.stringify([rowKey,colKey]);
    if(!cells.has(key))cells.set(key,{sum:0,squares:0,samples:0,count:0,min:Infinity,max:-Infinity,items:[],rowKey,colKey});
    const cell=cells.get(key);const n=sampleCount(r,metric);cell.sum+=metric==="nonfinite_count"&&mode!=="diff"?item.value:item.value*n;cell.squares+=item.value*item.value*n;cell.samples+=n;cell.count++;cell.min=Math.min(cell.min,item.value);cell.max=Math.max(cell.max,item.value);cell.items.push(item);
  }
  for(const cell of cells.values())cell.value=mode==="diff"?cell.sum/cell.samples:metric==="rms"?Math.sqrt(cell.squares/cell.samples):metric==="max_abs"?cell.max:metric==="nonfinite_count"?cell.sum:cell.sum/cell.samples;
  const rows=[...rowLabels.keys()].sort((a,b)=>Number(rowLabels.get(a).layer)-Number(rowLabels.get(b).layer)||a.localeCompare(b));
  const columns=[...colLabels.keys()].sort((a,b)=>str(colLabels.get(a).phase).localeCompare(str(colLabels.get(b).phase))||Number(colLabels.get(a).position)-Number(colLabels.get(b).position)||a.localeCompare(b));
  return {cells,rowLabels,colLabels,rows,columns};
}
function weightAssociation(weights,base,candidate,observedRun){
  const hash=x=>typeof x?.sha256==="string"&&x.sha256.length?x.sha256:null;
  if(!weights)return "静的重みレポートは未添付";
  if(weights.kind==="checkpoint_diff"){
    if(!hash(weights.base)||!hash(weights.candidate)||!hash(base?.checkpoint)||!hash(candidate?.checkpoint))return "重み差と実験の内容hashが未確認";
    if(hash(weights.base)!==hash(base.checkpoint)||hash(weights.candidate)!==hash(candidate.checkpoint))return "重み差の基準/比較checkpointが選択実験と不一致";
    return null;
  }
  if(weights.kind==="checkpoint_inspection"){
    if(!hash(weights.checkpoint)||!hash(observedRun?.checkpoint))return "静的重みと実験の内容hashが未確認";
    return hash(weights.checkpoint)===hash(observedRun.checkpoint)?null:"静的重みcheckpointが選択実験と不一致";
  }
  return "重みレポートの種類は未対応";
}
function renderCell(cell,metric){
  const first=cell.items[0],run=first.run,site=siteFor(run,first.record);
  byId("cell-summary").textContent="指標 "+metric+" · 集計 "+fmt(cell.value)+" · sample n="+cell.samples+" / bin records="+cell.count+" · bin min "+fmt(cell.min)+" / max "+fmt(cell.max);
  const details={site:site||{missing:"site mapping未記録"},records:cell.items.slice(0,20).map(x=>({
    task:x.task.id,source:x.task.source,request_id:x.record.request_id,call_id:x.record.call_id,trial_id:x.record.trial_id,request_scope:x.record.request_scope,replica:x.record.replica,rank:x.record.rank,
    phase:x.record.phase,token_position:x.record.token_position,token_end:x.record.token_end,sample_count:sampleCount(x.record),metric_sample_count:sampleCount(x.record,metric),token_kind:x.record.token_kind,accepted:x.record.accepted,
    detail_kind:x.record.detail_kind,channel_index:x.record.channel_index,head_local_index:x.record.head_local_index,head_global_index:x.record.head_global_index,
    value:x.value,baseline_metrics:x.baseline?.metrics,candidate_metrics:x.candidate?.metrics,sample:x.record.sample
  })),shown_records:Math.min(cell.count,20),total_records:cell.count};
  byId("cell-detail").textContent=JSON.stringify(details,null,2);
  const dest=byId("weight-detail");dest.replaceChildren();
  const names=new Set();
  for(const item of cell.items){
    for(const linked of [siteFor(item.run,item.record),siteFor(selectedRuns()[0],item.baseline||{})]){
      for(const name of linked?.weight_names||[])names.add(name);
    }
  }
  if(!names.size){dest.append(node("p","重み対応は未登録です。"));return;}
  const weights=DATA.weights,association=weightAssociation(weights,...selectedRuns(),run);
  if(association){dest.append(node("p",association+"。重みの数値をこの活性の根拠として表示しません。","warning"));dest.append(node("pre",JSON.stringify({weight_names:[...names],weight_kind:weights?.kind,base:weights?.base?.sha256,candidate:weights?.candidate?.sha256,checkpoint:weights?.checkpoint?.sha256},null,2)));return;}
  const entries=Array.isArray(weights?.tensors)?weights.tensors:[];
  for(const name of names){
    const detail=node("details"),summary=node("summary",name);detail.append(summary);
    const weight=entries.find(w=>w.name===name);
    detail.append(node("pre",JSON.stringify(weight||{name,status:"静的重みレポートに該当tensorなし"},null,2)));
    dest.append(detail);
  }
  if(weights)dest.append(node("pre",JSON.stringify({kind:weights.kind,selection:weights.selection,global:weights.global},null,2)));
}
function renderProvenance(base,candidate){
  const dest=byId("provenance");dest.replaceChildren();
  for(const run of [base,candidate].filter(Boolean)){
    const detail=node("details");detail.append(node("summary",str(run.label||run.id)+" · "+str(run.observation_mode)));
    detail.append(node("pre",JSON.stringify({id:run.id,data_kind:run.data_kind||DATA.data_kind||"未申告",observation_mode:run.observation_mode,checkpoint:run.checkpoint,runtime:run.runtime,capture:run.capture,provenance:run.provenance},null,2)));
    dest.append(detail);
  }
}
function render(){
  const [base,candidate]=selectedRuns(),all=allTaskRows(base,candidate);
  options("area",[["","全領域"],...[...new Set(all.map(t=>t.area))].sort().map(x=>[x,x])],byId("area").value);
  options("source",[["","全出典"],...[...new Set(all.map(t=>t.source))].sort().map(x=>[x,x])],byId("source").value);
  const tasks=visibleTasks(all,byId("area").value,byId("source").value,byId("search").value);
  const firstSampled=tasks.find(t=>(t.baseline?.observations?.length||t.candidate?.observations?.length));
  options("task",tasks.map(t=>[t.key,t.source+" / "+t.id]),tasks.some(t=>t.key===byId("task").value)?byId("task").value:firstSampled?.key);
  const focus=tasks.filter(t=>t.key===byId("task").value),body=byId("task-body");body.replaceChildren();
  let gain=0,loss=0,equal=0,unpaired=0;
  for(const t of tasks){
    const a=t.baseline?.score,b=t.candidate?.score;
    if(finite(a)&&finite(b)){if(b>a)gain++;else if(b<a)loss++;else equal++;}else unpaired++;
  }
  for(const t of tasks.slice(0,300)){
    const tr=node("tr");if(byId("task").value===t.key)tr.className="selected";
    const td=node("td"),button=node("button",t.id);button.addEventListener("click",()=>{byId("task").value=t.key;render();});td.append(button);tr.append(td);
    tr.append(node("td",t.area+" / "+t.source));
    for(const task of [t.baseline,t.candidate])tr.append(node("td",fmt(task?.score)+" · "+str(task?.status)));
    tr.append(node("td",finite(t.baseline?.score)&&finite(t.candidate?.score)?fmt(t.candidate.score-t.baseline.score):"対応スコアなし"));
    tr.append(node("td",(t.baseline?.observations?.length||0)+" / "+(t.candidate?.observations?.length||0)));body.append(tr);
  }
  byId("outcome-summary").textContent=tasks.length+"問題 · 得点増 "+gain+" / 減 "+loss+" / 同 "+equal+" / 対応スコアなし "+unpaired;
  byId("task-limit").textContent=tasks.length>300?"問題表は先頭300件。検索または領域で絞り込めます。":"";
  const baselineObservations=focus.flatMap(t=>t.baseline?.observations||[]),candidateObservations=focus.flatMap(t=>t.candidate?.observations||[]);
  options("base-request",requestOptions(baselineObservations),byId("base-request").value);
  options("candidate-request",requestOptions(candidateObservations),byId("candidate-request").value);
  const observations=[...baselineObservations.filter(r=>observationScope(r)===byId("base-request").value),...candidateObservations.filter(r=>observationScope(r)===byId("candidate-request").value)];
  byId("request-scope").textContent="基準 "+(byId("base-request").value||"観測なし")+" / 比較 "+(byId("candidate-request").value||"観測なし")+"。各配列はrequest ID / call ID / trial ID / scope / replica。trial単位traceでもbackend requestは分離します。";
  const metrics=[...new Set(observations.flatMap(r=>Object.keys(r.metrics)))].sort();
  options("metric",metrics.map(x=>[x,x]),byId("metric").value||"rms");
  options("phase",[["","全段階"],...[...new Set(observations.map(r=>r.phase).filter(x=>x!==undefined))].sort().map(x=>[x,x])],byId("phase").value);
  options("role",[["","全構造"],...[...new Set(observations.map(r=>r.role).filter(x=>x!==undefined))].sort().map(x=>[x,x])],byId("role").value);
  options("site",[["","全site"],...[...new Set(observations.filter(r=>!byId("role").value||r.role===byId("role").value).map(r=>r.site_id).filter(x=>x!==undefined))].sort().map(x=>[x,x])],byId("site").value);
  const ranks=[...new Set(observations.map(r=>String(r.rank)))].sort((a,b)=>Number(a)-Number(b));
  options("rank",[...ranks.map(x=>[x,"rank "+x]),["","全rank（行を分離）"]],byId("rank").dataset.initialized?byId("rank").value:"0");
  byId("rank").dataset.initialized="true";
  const filters={baseRequest:byId("base-request").value,candidateRequest:byId("candidate-request").value,rank:byId("rank").value,role:byId("role").value,phase:byId("phase").value,site:byId("site").value,acceptedOnly:byId("accepted-only").checked,start:byId("start").value===""?null:Number(byId("start").value),end:byId("end").value===""?null:Number(byId("end").value)};
  const view=byId("view").value,metric=byId("metric").value,result=collectRecords(focus,base,candidate,view,metric,filters);
  const sampled=focus.filter(t=>(view==="candidate"?t.candidate:view==="baseline"?t.baseline:t.baseline)?.observations?.length).length;
  byId("coverage").textContent="観測あり "+sampled+"/"+focus.length+"問題 · 表示 "+result.records.length+" bin record · metric sample n="+result.records.reduce((n,x)=>n+sampleCount(x.record,metric),0)+" / 対象record "+result.total+" · 受理未確認 "+result.unknown+" · 不受理除外 "+result.rejected+" · 指標欠測 "+result.missing;
  byId("pairing").textContent=view==="diff" ? "差分は明示pair_key＋同条件＋受理確認済みrecordのみ。"+(result.reasons.length?" 除外: "+result.reasons.map(([r,n])=>r+" "+n).join(" / "):" 対応record "+result.records.length) : "観測mode: "+str((view==="candidate"?candidate:base)?.observation_mode)+"。生成分岐後の同じtoken位置は同じ内容とは限りません。受理未確認のrecordは計算された活性であり、採用された生成tokenの活性とは未確認です。MTP受理はsampler上の採用を表し、公開応答でのEOS以後の打切りとは異なります。";
  const map=makeHeatmap(result.records,metric,view),table=byId("heatmap");table.replaceChildren();
  const rows=map.rows.slice(0,96),columns=map.columns.slice(0,128),thead=node("thead"),head=node("tr");head.append(node("th","層 / site / TP rank"));
  for(const col of columns)head.append(node("th",map.colLabels.get(col).label));thead.append(head);table.append(thead);
  const tbody=node("tbody");let magnitude=0;
  for(const cell of map.cells.values())magnitude=Math.max(magnitude,Math.abs(cell.value));
  for(const row of rows){
    const tr=node("tr"),label=map.rowLabels.get(row);const th=node("th",label.label);th.title=label.site;tr.append(th);
    for(const col of columns){
      const td=node("td"),cell=map.cells.get(JSON.stringify([row,col]));
      if(cell){const mean=cell.value,button=node("button",fmt(mean));button.title="site "+label.site+" · samples="+cell.samples+" · records="+cell.count+" · bin min="+fmt(cell.min)+" / max="+fmt(cell.max);
        button.dataset.cell=JSON.stringify([row,col]);const level=magnitude?Math.min(1,Math.abs(mean)/magnitude):0;
        button.style.backgroundColor=view==="diff"&&mean<0?"hsl(22 76% "+(96-40*level)+"%)":"hsl(194 65% "+(96-40*level)+"%)";
        button.addEventListener("click",()=>renderCell(cell,metric));td.append(button);
      }else{td.textContent="—";td.className="empty";td.title="未採取または対応不能（0ではありません）";}
      tr.append(td);
    }tbody.append(tr);
  }
  table.append(tbody);
  byId("heatmap-limit").textContent=rows.length?"色は現在の表示対象に対する相対値。"+(map.rows.length>96||map.columns.length>128?" 先頭96行×128列を表示。site/位置filterで絞り込めます。":"")+" 全"+map.rows.length+"行 / "+map.columns.length+"列。":"この条件の活性はありません。未採取・不受理・対応不能・指標欠測を確認してください。";
  byId("cell-summary").textContent="セルを選ぶと、観測site・採取範囲・明示された重みの対応を表示します。";
  byId("cell-detail").textContent="";byId("weight-detail").replaceChildren();
  renderProvenance(base,candidate);
}
const labels=DATA.runs.map(r=>[r.id,(r.label||r.id)+" ["+str(r.observation_mode)+"]"]);
options("base",labels,DATA.runs[0].id);
options("candidate",[["","比較なし"],...labels],DATA.runs[1]?.id||"");
const kinds=[...new Set(DATA.runs.map(r=>r.data_kind||DATA.data_kind||"undeclared"))];
byId("data-kind").append(node("span",kinds.includes("synthetic")?"合成データを含む — モデルの実測成績ではありません":kinds.some(kind=>kind!=="recorded")?"データ由来は未申告 — 実測とは確認していません":"記録済み観測（出所は下部の実行条件を確認）","badge"));
if(DATA.runs.some(r=>r.provenance?.input_kind==="synthetic_calibration"))byId("data-kind").append(node("span","合成入力のツール検証 — 能力評価ではありません","badge"));
for(const id of ["base","candidate","area","source","task","base-request","candidate-request","view","metric","rank","phase","role","site","start","end","accepted-only"])byId(id).addEventListener("change",render);
byId("search").addEventListener("input",render);
window.KairyuObserve=Object.freeze({observationScope,requestOptions,allTaskRows,visibleTasks,pairReason,collectRecords,makeHeatmap,weightAssociation,render});
render();
"""
