from __future__ import annotations

import base64
import copy
import gzip
import json
import shutil
import subprocess
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from kairyu_bench.wsp.observe.report import _JS, build_report, write_report


def fixture() -> dict:
    runtime = {"adapter": "synthetic", "vllm_version": "not-applicable", "tp_size": 2, "mtp_tokens": 0}
    site = {"site_id": "residual.0", "module": "model.layers.0", "layer": 0,
            "role": "residual", "weight_names": ["model.layers.0.norm.weight"], "tp_layout": {"axis": "replicated"}}
    observation = {"site_id": "residual.0", "layer": 0, "role": "residual", "request_id": "request",
                   "token_position": 0, "token_end": 63, "sample_count": 2, "phase": "prefill",
                   "token_kind": "target", "accepted": True, "replica": 0, "rank": 0,
                   "metrics": {"rms": 1.0, "mean": .25, "max_abs": 2.0, "nonfinite_count": 0},
                   "pair_key": "fixed-input-hash"}
    base = {"id": "base", "label": "基準", "observation_mode": "replay",
            "checkpoint": {"sha256": "a" * 64}, "runtime": runtime, "capture": {"status": "closed"},
            "sites": [site], "tasks": [
                {"id": "problem-1", "source": "unit", "area": "code", "score": 0., "status": "scored",
                 "observations": [observation]},
                {"id": "problem-2", "source": "other", "area": "tools", "score": None,
                 "status": "unscored", "observations": []}]}
    candidate = copy.deepcopy(base)
    candidate["id"] = "candidate"
    candidate["label"] = "編集"
    candidate["checkpoint"]["sha256"] = "b" * 64
    candidate["tasks"][0]["score"] = 1.
    candidate["tasks"][0]["observations"][0]["metrics"]["rms"] = 2.
    candidate["tasks"][0]["observations"][0]["metrics"]["max_abs"] = 3.
    return {"schema_version": 1, "data_kind": "synthetic", "runs": [base, candidate],
            "weights": {"kind": "checkpoint_diff", "base": {"sha256": "a" * 64}, "candidate": {"sha256": "b" * 64}, "selection": {"scope": "selected tensors"},
                        "tensors": [{"name": "model.layers.0.norm.weight", "relative_delta_l2": .01}]}}


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.resources: list[str] = []
        self.data = ""
        self.in_data = False
        self.active_payload: str | None = None
        self.payloads: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"])
        for key in ("src", "href"):
            if values.get(key):
                self.resources.append(values[key])
        self.in_data = tag == "script" and values.get("id") == "observation-data"
        self.active_payload = values.get("id") if tag == "script" and values.get("id", "").startswith("observation-payload-") else None
        if self.active_payload:
            self.payloads[self.active_payload] = ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.in_data = False
            self.active_payload = None

    def handle_data(self, data: str) -> None:
        if self.in_data:
            self.data += data
        if self.active_payload:
            self.payloads[self.active_payload] += data


_NODE = r"""
const fs=require("fs"),vm=require("vm"),assert=require("assert");
const input=JSON.parse(fs.readFileSync(0,"utf8"));
class Element {
 constructor(tag){this.tagName=tag;this.children=[];this.textContent="";this.value="";this.checked=false;this.dataset={};this.style={};this.listeners={};}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=[...nodes];}
 addEventListener(event, fn){(this.listeners[event]??=[]).push(fn);}
 fire(event){for(const fn of this.listeners[event]||[])fn();}
}
const elements={};for(const id of input.ids)elements[id]=new Element("div");
elements["observation-data"].textContent=JSON.stringify(input.data);
elements.view.value="baseline";
const document={getElementById:id=>elements[id],createElement:tag=>new Element(tag)};
const context={document,window:{},console};
vm.createContext(context);vm.runInContext(input.js,context);
const api=context.window.KairyuObserve;
const descendants=e=>[e,...e.children.flatMap(descendants)];
const cellButtons=()=>descendants(elements.heatmap).filter(x=>x.tagName==="button");
const data=input.data, a=data.runs[0],b=data.runs[1],ta=a.tasks[0],tb=b.tasks[0],ra=ta.observations[0],rb=tb.observations[0];
assert(elements["data-kind"].children[0].textContent.includes("合成"));
assert(elements["outcome-summary"].textContent.includes("得点増 1"));
assert.equal(elements.rank.value,"0");
assert.equal(cellButtons().length,1);
assert.equal(cellButtons()[0].textContent,"1");
elements.view.value="diff";elements.view.fire("change");
assert.equal(cellButtons().length,1);assert.equal(cellButtons()[0].textContent,"1");
cellButtons()[0].fire("click");
assert(elements["cell-summary"].textContent.includes("sample n=2"));
assert(elements["weight-detail"].children.length>0);
elements.area.value="tools";elements.area.fire("change");
assert.equal(cellButtons().length,0);
assert(elements["outcome-summary"].textContent.includes("対応スコアなし 1"));
elements.area.value="";elements.area.fire("change");
elements.search.value="problem-1";elements.search.fire("input");
assert(elements["outcome-summary"].textContent.startsWith("1問題"));
elements.search.value="";elements.search.fire("input");
assert.equal(api.pairReason(ra,rb,a,b,ta,tb),null);
for(const [kind,mutate,expected] of [
 ["mode",x=>x.observation_mode="live","mode"],
 ["runtime",x=>x.runtime.tp_size=4,"tp_size"],
 ["sampling",x=>x.capture.sampling={prefill_policy:"first_execution_of_last_prompt_positions"},"prefill"]
]){
 const changed=JSON.parse(JSON.stringify(b));mutate(changed);
 assert(api.pairReason(ra,rb,a,changed,ta,tb).includes(expected));
}
const noPair={...rb,pair_key:null};
assert(api.pairReason(ra,noPair,a,b,ta,tb).includes("key"));
assert(api.pairReason(ra,{...rb,accepted:null},a,b,ta,tb).includes("受理"));
assert(api.pairReason(ra,{...rb,sample_count:1},a,b,ta,tb).includes("採取数"));
const rows=api.allTaskRows(a,b),filters={rank:"0",phase:"",site:"",acceptedOnly:false,start:null,end:null};
const paired=api.collectRecords(rows,a,b,"diff","rms",filters);
assert.equal(paired.records.length,1);
const duplicate=JSON.parse(JSON.stringify(b));duplicate.tasks[0].observations.push(duplicate.tasks[0].observations[0]);
const dup=api.collectRecords(api.allTaskRows(a,duplicate),a,duplicate,"diff","rms",filters);
assert.equal(dup.records.length,0);assert(dup.reasons.some(x=>x[0].includes("重複")));
const live=JSON.parse(JSON.stringify(a));delete live.tasks[0].observations[0].pair_key;live.observation_mode="live";
const liveResult=api.collectRecords(api.allTaskRows(live,b),live,b,"diff","rms",filters);
assert.equal(liveResult.records.length,0);
const second={...ra,token_position:64,token_end:127};
const a2=JSON.parse(JSON.stringify(a)),b2=JSON.parse(JSON.stringify(b));
a2.tasks[0].observations.push(second);b2.tasks[0].observations.push({...second,metrics:{rms:4}});
assert.equal(api.collectRecords(api.allTaskRows(a2,b2),a2,b2,"diff","rms",filters).records.length,2,"same input hash can cover multiple positions");
const records=[{record:{...ra,sample_count:1},value:1},{record:{...ra,sample_count:3},value:3}];
const heat=api.makeHeatmap(records,"rms","baseline");
assert.equal([...heat.cells.values()][0].value,Math.sqrt(7));
const diff=api.makeHeatmap(records,"rms","diff");
assert.equal([...diff.cells.values()][0].value,2.5);
const sum=api.makeHeatmap(records,"nonfinite_count","baseline");
assert.equal([...sum.cells.values()][0].value,4);
const counted=records.map(x=>({...x,record:{...x.record,metric_sample_counts:{rms:1}}}));
assert.equal([...api.makeHeatmap(counted,"rms","baseline").cells.values()][0].value,Math.sqrt(5));
assert.equal(api.weightAssociation(data.weights,a,b,b),null);
assert(api.weightAssociation(data.weights,b,a,b).includes("不一致"));
assert(api.weightAssociation({...data.weights,base:{sha256:null}},a,b,b).includes("未確認"));
const details=api.makeHeatmap([{record:{...ra,role:"channel",channel_index:19},value:2}],"value");
assert([...details.rowLabels.values()][0].label.includes("channel 19"));
const heads=api.makeHeatmap([{record:{...ra,role:"attention_head",head_global_index:17},value:2}],"rms");
assert([...heads.rowLabels.values()][0].label.includes("head 17"));
const other={...ra,request_id:"request-2",call_id:"call-2",trial_id:"trial-2",metrics:{rms:9},pair_key:null};
const split=api.makeHeatmap([{record:ra,value:1},{record:other,value:9}]);
assert.equal(split.cells.size,2,"different requests must never be averaged at the same layer/position");
const otherTrial={...ra,trial_id:"other-trial"};
assert.equal(api.makeHeatmap([{record:ra,value:1},{record:otherTrial,value:9}]).cells.size,2);
const scopedA={...a,tasks:[{...ta,observations:[ra,other]}]};
const scoped=api.collectRecords(api.allTaskRows(scopedA,b),scopedA,b,"baseline","rms",{...filters,baseRequest:api.observationScope(other)});
assert.equal(scoped.records.length,1);assert.equal(scoped.records[0].value,9);
assert.equal(api.requestOptions([ra,other]).length,2);
assert.equal(api.collectRecords(rows,a,b,"baseline","rms",{...filters,role:"mlp"}).records.length,0);
assert.equal(api.collectRecords(rows,a,b,"baseline","rms",{...filters,role:"residual"}).records.length,1);
console.log("DOM interactions and pairing checks passed");
"""


class ObservationReportTest(unittest.TestCase):
    def test_embedded_data_is_inert_and_roundtrips(self) -> None:
        data = fixture()
        attack = '</script><script>alert("x")</script>&\u2028'
        data["runs"][0]["label"] = attack
        data["runs"][0]["tasks"][0]["id"] = attack
        html = build_report(data)
        self.assertNotIn(attack, html)
        self.assertNotIn('alert("x")</script>', html)
        self.assertNotIn("innerHTML", html)
        parser = _Parser()
        parser.feed(html)
        decoded = json.loads(parser.data)
        self.assertEqual(decoded.pop("report_storage")["encoding"], "task-gzip-base64")
        for run in decoded["runs"]:
            for task in run["tasks"]:
                payload = task.pop("observations_payload", None)
                if payload:
                    raw = gzip.decompress(base64.b64decode(parser.payloads[payload["element_id"]]))
                    self.assertEqual(len(raw), payload["raw_bytes"])
                    task["observations"] = json.loads(raw)
                    self.assertEqual(len(task["observations"]), payload["count"])
        self.assertEqual(decoded, data)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        self.assertEqual(parser.resources, [])
        self.assertIn("connect-src 'none'", html)

    def test_single_run_and_atomic_output(self) -> None:
        data = fixture()
        data["runs"] = data["runs"][:1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested/report.html"
            self.assertEqual(write_report(data, path), path)
            self.assertTrue(path.read_text().startswith("<!doctype html>"))
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            symlink = path.parent / "link.html"
            symlink.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "symlink"):
                write_report(data, symlink)

    def test_invalid_schema_and_nonfinite_values(self) -> None:
        bad_values = [{"schema_version": 2, "runs": []}, {"schema_version": 1, "runs": []}]
        duplicate = fixture()
        duplicate["runs"].append(copy.deepcopy(duplicate["runs"][0]))
        bad_values.append(duplicate)
        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_report(value)
        for invalid in (float("nan"), float("inf"), True, "1"):
            value = fixture()
            value["runs"][0]["tasks"][0]["observations"][0]["metrics"]["rms"] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    build_report(value)
        value = fixture()
        value["runs"][0]["tasks"][0]["observations"][0]["request_id"] = None
        with self.assertRaisesRegex(ValueError, "request_id"):
            build_report(value)
        value = fixture()
        value["runs"][0]["tasks"][0]["observations"][0]["sample_count"] = 0
        with self.assertRaisesRegex(ValueError, "sample_count"):
            build_report(value)

    @unittest.skipUnless(shutil.which("node"), "Node is needed for gzip/lazy interaction checks")
    def test_lazy_task_loading_races_lru_and_errors(self) -> None:
        data = fixture()
        for run in data["runs"]:
            first = run["tasks"][0]
            for index in range(3, 8):
                task = copy.deepcopy(first)
                task["id"] = f"problem-{index}"
                task["observations"][0]["request_id"] = f"request-{index}"
                run["tasks"].append(task)
        parser = _Parser()
        parser.feed(build_report(data))
        script = _NODE[:_NODE.index('const data=input.data')]
        script = script.replace('const context={document,window:{},console};',
            'for(const [id,text] of Object.entries(input.payloads))elements[id].textContent=text;'
            'const context={document,window:{},console,DecompressionStream,Blob,TextDecoder,Uint8Array,atob};')
        script += r"""
(async()=>{
 await api.render();
 assert.equal(elements["load-state"].dataset.state,"ready");
 assert.equal(cellButtons().length,1);
 assert.equal(api.cacheInfo().tasks,2);
 elements.view.value="diff";await api.render();
 assert.equal(cellButtons()[0].textContent,"1");
 elements.task.value=JSON.stringify(["unit","problem-3"]);
 const stale=api.render();
 elements.task.value=JSON.stringify(["unit","problem-4"]);
 const latest=api.render();
 await Promise.all([stale,latest]);
 assert(elements["request-scope"].textContent.includes("request-4"));
 for(const id of ["problem-5","problem-6","problem-7"]){
  elements.task.value=JSON.stringify(["unit",id]);await api.render();
  assert(api.cacheInfo().tasks<=4);
  assert(api.cacheInfo().raw_bytes<=api.cacheInfo().max_raw_bytes);
 }
 const original=context.DecompressionStream;
 context.DecompressionStream=undefined;
 elements.task.value=JSON.stringify(["unit","problem-1"]);await api.render();
 assert.equal(elements["load-state"].dataset.state,"error");
 assert(elements["load-state"].textContent.includes("DecompressionStream"));
 assert.equal(cellButtons().length,0);
 assert(elements["coverage"].textContent.includes("未表示"));
 context.DecompressionStream=original;
 await api.render();
 assert.equal(elements["load-state"].dataset.state,"ready");
 elements.task.value=JSON.stringify(["unit","problem-3"]);
 const payload=input.data.runs[0].tasks.find(t=>t.id==="problem-3").observations_payload;
 vm.runInContext('DATA.runs[0].tasks.find(t=>t.id==="problem-3").observations_payload.raw_bytes=1',context);
 await api.render();
 assert.equal(elements["load-state"].dataset.state,"error");
 assert(elements["load-state"].textContent.includes("実展開容量"));
 vm.runInContext('DATA.runs[0].tasks.find(t=>t.id==="problem-3").observations_payload.raw_bytes='+payload.raw_bytes,context);
 elements[payload.element_id].textContent="corrupt";
 await api.render();
 assert.equal(elements["load-state"].dataset.state,"error");
 assert.equal(cellButtons().length,0);
 elements.area.value="tools";await api.render();
 assert.equal(elements["load-state"].dataset.state,"ready");
 assert.equal(cellButtons().length,0);
 assert(elements["outcome-summary"].textContent.includes("対応スコアなし 1"));
 console.log("Lazy gzip, epoch, LRU, missing feature and corrupt payload checks passed");
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run(
            ["node", "-e", script],
            input=json.dumps({"data": json.loads(parser.data), "payloads": parser.payloads,
                              "ids": parser.ids, "js": _JS}),
            capture_output=True, text=True, timeout=20, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("checks passed", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node is needed for offline DOM interaction checks")
    def test_offline_interactions_and_pairing_guards(self) -> None:
        data = fixture()
        parser = _Parser()
        parser.feed(build_report(data))
        result = subprocess.run(
            ["node", "-e", _NODE], input=json.dumps({"data": data, "ids": parser.ids, "js": _JS}),
            capture_output=True, text=True, timeout=15, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("passed", result.stdout)
        unknown = fixture()
        unknown["data_kind"] = "unknown"
        unknown_js = _NODE.replace(
            'assert(elements["data-kind"].children[0].textContent.includes("合成"));',
            'assert(elements["data-kind"].children[0].textContent.includes("未申告"));',
        )
        result = subprocess.run(
            ["node", "-e", unknown_js], input=json.dumps({"data": unknown, "ids": parser.ids, "js": _JS}),
            capture_output=True, text=True, timeout=15, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
