#!/usr/bin/env python3
"""One explicit remote AV observer/WSP operation; all lifecycle uses existing tools.

This host-bound recipe preserves all official tier10/dev questions. It measures
four short deterministic HTTP requests before/after observer attachment, keeps
only public-answer hashes and numeric usage, and restores the original CID.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid

R = Path('/mnt/nvme/kairyu/wsp/gemma4-coding-20261104')
REPO = R / 'worktrees/kairyu-bench-weight-tools'
PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('--artifact-dir', type=Path, required=True)
PARSER.add_argument('--preview', action='store_true', help='validate plan/source identities only; do not hash checkpoint or operate serving')
ARGS = PARSER.parse_args()
N = ARGS.artifact_dir.resolve()
assert N.is_relative_to(R), 'artifact directory must stay beneath the NVMe experiment root'
PYTHON = R / 'infra/weight-tools-ci/bin/python'
PLAN = json.loads((N / 'plan.json').read_text())
NAME = PLAN['original_container']['name']
ORIGINAL_ID = PLAN['original_container']['id']
CAPTURE = Path(PLAN['capture']['output'])
SOURCE = Path(PLAN['capture']['source'])
RUN_ID = PLAN['wsp']['run_id']
START = time.time()
STATE = {'status': 'starting', 'started_epoch': START, 'pid': os.getpid()}


def save(name, value):
    path = N / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def phase(value, **extra):
    STATE.update(phase=value, updated_epoch=time.time(), elapsed_seconds=time.time()-START, **extra)
    save('execution.json', STATE)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect(name=NAME):
    return json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]


def address():
    item = inspect()
    return 'http://' + item['NetworkSettings']['Networks']['gemma-4-31b-8gpu_default']['IPAddress'] + ':8000'


def idle():
    text = urllib.request.urlopen(address() + '/metrics', timeout=10).read().decode()
    rows = [line for line in text.splitlines() if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
    assert len(rows) == 2 and all(float(line.rsplit(' ', 1)[1]) == 0 for line in rows)
    return rows


def run(command, filename, env=None):
    with (N / filename).open('w') as stream:
        process = subprocess.run([str(x) for x in command], cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError('command_failed_' + filename + '_' + str(process.returncode))


def numeric_probe(label):
    from kairyu_bench.wsp.observe.tracing import RequestRecorder, RequestTrace
    prompts = ['Return the integer result of 17 + 26.', 'Reply with the Python expression for an empty list.', 'Return only the sorted list of 9, 2, 5.', 'Reply with the number of characters in abcdef.']
    result = {'label': label, 'kind': 'synthetic_overhead_not_WSP', 'samples': [], 'thinking': False, 'temperature': 0, 'top_p': 1, 'max_tokens': 64}
    recorder = RequestRecorder(CAPTURE / 'capture.json', run_id=RUN_ID+'-overhead') if label == 'on' else None
    for index, prompt in enumerate(prompts):
        payload = {'model': 'gemma-4-31b', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'top_p': 1, 'seed': 1700+index, 'max_tokens': 64, 'chat_template_kwargs': {'enable_thinking': False}}
        url = address() + '/v1/chat/completions'
        trace = RequestTrace(recorder, 'synthetic-overhead:'+str(index), payload, url, uuid.uuid4().hex, 0) if recorder else None
        headers = {'Content-Type': 'application/json', **(trace.headers if trace else {})}
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        before = time.monotonic()
        try:
            response = json.load(urllib.request.urlopen(request, timeout=180))
            if trace:
                trace.finish(response=response)
            choice = response['choices'][0]
            public = choice['message'].get('content') or ''
            result['samples'].append({'index': index, 'seconds': time.monotonic()-before, 'public_sha256': hashlib.sha256(public.encode()).hexdigest(), 'public_characters': len(public), 'finish_reason': choice.get('finish_reason'), 'usage': response.get('usage'), 'request_id': response.get('id')})
        except Exception as exc:
            if trace:
                trace.finish(error=type(exc).__name__)
            result['samples'].append({'index': index, 'error_type': type(exc).__name__, 'seconds': time.monotonic()-before})
            save('overhead-'+label+'.json', result)
            raise
        save('overhead-'+label+'.json', result)
    return result


def worker_status():
    return {p.parent.name: json.loads(p.read_text()) for p in CAPTURE.glob('workers/*/status.json')}


def drained():
    idle()
    rows = worker_status()
    return len(rows) == 2 and all(s['writer_queue']['depth'] == 0 and s['writer_queue']['pending_bytes'] == 0 for s in rows.values())


def validate_plan():
    assert PLAN['topology'] == {'tp': 2, 'replicas': 1, 'mtp_tokens': 4}
    for key in ['output', 'source', 'source_template', 'runtime_cache', 'backup']:
        assert Path(PLAN['capture'][key]).resolve().is_relative_to(R), 'capture paths must stay on NVMe'
    assert PLAN['wsp']['tier'] == 10 and PLAN['wsp']['split'] == 'dev'
    assert PLAN['wsp']['expected_items'] == 778 and PLAN['wsp']['expected_areas'] == 14
    assert PLAN['wsp']['only'] is None and not PLAN['wsp']['excluded_items']
    assert [PLAN['wsp'][key] for key in ['chat_concurrency', 'logprob_concurrency', 'agent_concurrency']] == [32, 8, 1]
    assert PLAN['wsp']['logprob_endpoints'] == [PLAN['wsp']['endpoint']]
    assert PLAN['capture']['prefill_positions'] == 8
    assert PLAN['capture']['storage_bytes'] == 2 * 1024**3
    assert PLAN['capture']['decode_offsets'] == [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    reference = Path(PLAN['baseline_evidence_dir'])
    expected_runner = json.loads((reference / 'runner-image-source.json').read_text())['docker_image_id']
    assert PLAN['environment']['KAIRYU_BENCH_RUNNER_IMAGE_ID'] == expected_runner
    template = Path(PLAN['capture']['source_template'])
    pins = {str(p.relative_to(template)): sha(p) for p in sorted(template.rglob('*')) if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts}
    expected_pins = json.loads((reference / 'source-dependencies.json').read_text())['source']
    assert pins == expected_pins, 'observer template differs from baseline source'
    assert not CAPTURE.exists() and not SOURCE.exists()
    assert not (R / 'results' / RUN_ID).exists()
    return {'mode': 'preview' if ARGS.preview else 'execution', 'status': PLAN['status'], 'run_id': RUN_ID, 'checkpoint_planned': PLAN['original_container']['checkpoint'], 'runner_image_id': expected_runner, 'observer_files_verified': len(pins), 'checkpoint_payload_hashed': False, 'serving_mutated': False, 'benchmark_started': False}


def main():
    preview = validate_plan()
    if ARGS.preview:
        print(json.dumps(preview, indent=2))
        return
    assert PLAN['status'] == 'approved_for_execution', 'planned candidate cannot execute'
    assert re.fullmatch(r'[0-9a-f]{64}', ORIGINAL_ID), 'target serving CID must be resolved'
    assert Path(PLAN['original_container']['checkpoint']).is_dir()
    assert os.getuid() == 1003
    lock = (R / 'candidate-evaluation.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert inspect()['Id'] == ORIGINAL_ID
    idle()
    assert not (R/'results'/RUN_ID).exists()
    phase('snapshot_and_identity')
    shutil.copytree(Path(PLAN['capture']['source_template']), SOURCE, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    pins = {str(p.relative_to(SOURCE)): sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()}
    dependencies = {str(p.relative_to(REPO)): sha(p) for p in [REPO/'pyproject.toml', REPO/'Dockerfile', *sorted((REPO/'scripts/locks').glob('*'))] if p.is_file()}
    save('source-dependencies.json', {'source': pins, 'dependencies': dependencies, 'operation_recipe_sha256': sha(Path(__file__)), 'plan_sha256': sha(N/'plan.json'), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()})
    sys.path.insert(0, str(SOURCE))
    env = {**os.environ, 'KAIRYU_BENCH_PYTHON': str(PYTHON), 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(SOURCE)}
    run([PYTHON, '-m', 'kairyu_bench', 'wsp', 'observe', 'init', PLAN['original_container']['checkpoint'], '--output-dir', CAPTURE, '--model', 'gemma-4-31b', '--tp-size', '2', '--replicas', '1', '--mtp-tokens', '4', '--mode', 'live', '--decode-offsets', ','.join(map(str,PLAN['capture']['decode_offsets']))], 'capture-init.log', env)
    from kairyu_bench.wsp.observe.collector import sampling_metadata, CaptureConfig
    config = json.loads((CAPTURE/'capture.json').read_text())
    if PLAN.get('expected_checkpoint_sha256'):
        assert config['checkpoint']['sha256'] == PLAN['expected_checkpoint_sha256'], 'fresh checkpoint hash differs from frozen saved candidate'
    config['limits']['prefill_positions'] = 8
    config['sampling'] = sampling_metadata(8,16,tuple(config['decode_offsets']))
    config['runtime_source_dir'] = str(SOURCE)
    config['runtime_cache_dir'] = PLAN['capture']['runtime_cache']
    (CAPTURE/'capture.json').write_text(json.dumps(config,indent=2)+'\n')
    CaptureConfig.read(CAPTURE/'capture.json')
    base = json.loads((R/'observations/base-dev10-20261106/capture.json').read_text())['checkpoint']
    continuity = []
    for row in base['files']:
        stat = (Path(base['path'])/row['path']).stat()
        match = stat.st_size == row['size'] and all(getattr(stat,'st_'+k) == v for k,v in row['stat'].items() if k in ['mtime_ns','ctime_ns']) and stat.st_ino == row['stat']['inode'] and stat.st_dev == row['stat']['device']
        continuity.append({'path':row['path'],'prior_hash_stat_continuity':match})
    assert all(x['prior_hash_stat_continuity'] for x in continuity)
    save('checkpoint-identities.json', {'current':config['checkpoint'], 'original_base':base, 'original_base_proof':'prior full file SHA plus exact stat continuity; not freshly rehashed', 'base_stat_checks':continuity, 'capture_sha256':sha(CAPTURE/'capture.json')})
    # The prior 778-item capture wrote 806 MiB at 32+32 positions, TP2x4.
    # Records scale with requests, not replicas. 8+16 positions is a 0.375
    # nominal ratio; reserve 4x = 1.21 GiB, below the hard 2 GiB allocation.
    save('capture-budget.json', {'historical_capture_bytes':806*1024**2,'nominal_position_ratio':24/64,'fourfold_conservative_estimate_bytes':int(806*1024**2*24/64*4),'storage_limit_bytes':2*1024**3,'estimate_not_guarantee':True,'missing_policy':'Drop/quota/unobserved offsets remain explicit; never delete benchmark items or zero-fill missing activations'})
    phase('baseline_four_requests')
    numeric_probe('off')
    attach = [PYTHON, REPO/'scripts/observe-docker.py', 'attach', '--container', NAME, '--capture', CAPTURE/'capture.json', '--source', SOURCE, '--runtime-cache', PLAN['capture']['runtime_cache'], '--backup-dir', PLAN['capture']['backup'], '--replica', 'gemma-0']
    run(attach, 'attach-preview.json', env)
    attached = False
    try:
        phase('attaching_observer')
        idle()
        run(attach+['--apply'], 'attach.json', env)
        attached = True
        phase('observed_four_requests')
        numeric_probe('on')
        deadline=time.time()+120
        while not drained() and time.time()<deadline:
            time.sleep(1)
        assert drained()
        status=worker_status()
        assert all(x['records_written'] > 0 and not x['errors'] and not x['dropped_records'] for x in status.values())
        save('worker-preflight.json',status)
        phase('official_full778_run')
        wspenv={**env, **PLAN['environment'], 'KAIRYU_BENCH_IMAGE':PLAN['wsp']['runner_tag']}
        command=[REPO/'kairyu-bench','wsp','run',PLAN['wsp']['endpoint'],'--results-dir',R/'results','--tier','10','--split','dev','--label',config['checkpoint']['sha256'],'--run-id',RUN_ID,'--chat-concurrency','32','--logprob-concurrency','8','--agent-concurrency','1','--logprob-endpoint',PLAN['wsp']['endpoint'],'--observe-config',CAPTURE/'capture.json']
        save('wsp-command.json',{'command':[str(x) for x in command],'environment':PLAN['environment'],'image_tag':wspenv['KAIRYU_BENCH_IMAGE']})
        run(command,'wsp.log',wspenv)
        phase('draining_observer')
        deadline=time.time()+120
        while not drained() and time.time()<deadline:
            time.sleep(1)
        assert drained()
        save('worker-drained.json',worker_status())
    finally:
        if attached:
            phase('restoring_original')
            idle()
            run([PYTHON,REPO/'scripts/observe-docker.py','restore','--container',NAME,'--backup-dir',PLAN['capture']['backup'],'--apply'],'restore.json',env)
            assert inspect()['Id']==ORIGINAL_ID and inspect()['State']['Health']['Status']=='healthy'
            save('worker-closed.json',worker_status())
    phase('merging_capture')
    run([PYTHON,'-m','kairyu_bench','wsp','observe','merge',CAPTURE,'--results',R/'results'/RUN_ID,'--token-bin-size','64','--output',N/'merged.json'],'merge.log',env)
    phase('completed',status='completed',finished_epoch=time.time())


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        if not ARGS.preview:
            phase('failed',status='failed',error_type=type(error).__name__,finished_epoch=time.time())
        else:
            print(json.dumps({'preview_error_type': type(error).__name__}), file=sys.stderr)
        sys.exit(1)
