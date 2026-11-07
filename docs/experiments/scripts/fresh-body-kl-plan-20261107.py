#!/usr/bin/env python3
"""Bind one completed winner and fixed body/KL training job; CPU preview only."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time

R = Path('/mnt/nvme/kairyu/wsp/gemma4-coding-20261104')
B = R / 'worktrees/kairyu-bench-weight-tools'
PY = R / 'infra/weight-tools-ci/bin/python'
PROTOCOL = dict(kind='body_first_token_kl_v1', **{'lambda': 1.0},
                teacher_frame='identical_complete_frame', teacher_dtype='F32')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pin(p):
    return dict(path=str(p), sha256=sha(p))


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output-root', 'selection', 'frame-evidence', 'teacher-attestation', 'served-attestation'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('owner-sha256', 'trainer-sha256', 'body-helper-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    n = args.output_root
    assert n.resolve().is_relative_to(R / 'analysis') and not n.exists()
    choice = read(args.selection)
    assert choice['status'] == 'highest_complete_coding_parent_selected'
    assert choice['selected_label'] in ('F04', 'F06')
    parent_attestation = choice['selected_attestation']
    assert sha(parent_attestation['path']) == parent_attestation['sha256']
    parent = read(parent_attestation['path'])['checkpoint']
    served_attestation = pin(args.served_attestation)
    served = read(args.served_attestation)['checkpoint']
    teacher_attestation = pin(args.teacher_attestation)
    teacher = read(args.teacher_attestation)['checkpoint']
    for run in choice['completed_runs']:
        assert sha(run['report']['path']) == run['report']['sha256']
        assert read(run['run_json']['path'])['status'] == 'completed'
        assert sha(run['run_json']['path']) == run['run_json']['sha256']
    owner = B / 'docs/experiments/scripts/fresh-av-native-gpu4-20261107.py'
    guard_path = owner.with_name('fresh-f01-finite.py')
    recipe = B / 'scripts/experiments/gemma4-purpose'
    for path, digest in ((owner, args.owner_sha256), (recipe / 'train_lora.py', args.trainer_sha256),
                         (recipe / 'train_lora_body_kl.py', args.body_helper_sha256)):
        assert sha(path) == digest
    owner_module = load('body_plan_owner', owner)
    guard = load('body_plan_guard', guard_path)
    owner_module.full_identity(parent, Path(parent['path']))
    owner_module.full_identity(served, Path(served['path']))
    owner_module.full_identity(teacher, Path(teacher['path']))
    frame = read(args.frame_evidence)
    assert frame['loss_protocol'] == PROTOCOL and not frame['ineligible_ids'] and not frame['unsupported_ids']
    names = guard.docker('ps', '-a', '--format', '{{.Names}}').splitlines()
    assert len(names) == len(set(names)) == 27
    target_name = owner_module.TARGET
    target = guard.snapshot(target_name)
    protected = {name: guard.snapshot(name) for name in names if name != target_name}
    assert len(protected) == 26 and target['running'] and target['health'] == 'healthy'
    details = guard.inspect(target_name)
    mounts = [v for v in details['Mounts'] if v['Source'] == served['path']]
    assert len(mounts) == 1 and mounts[0]['RW'] is False
    metrics = guard.target_metrics(target['id'])
    assert all(value == 0 for group in metrics['queues'].values() for value in group)
    n.mkdir()
    shutil.copyfile(__file__, n / 'prepare-source.py.txt')
    save(n / 'current-serving-proof.json', dict(kind='read_only_current_serving_binding_no_redeployment',
         current_model='F06', best_parent=choice['selected_label'],
         checkpoint_all9_fullstat_matched_manifest=True,
         replicas={'gemma-2': dict(cid=target['id'], tree=served['sha256'],
                   checkpoint_mount=served['path'], checkpoint_readonly=True)},
         served_attestation=served_attestation, parent_selection=pin(args.selection),
         snapshot_epoch=time.time(), payload_rehash=False, model_calls=0, service_changes=0))
    save(n / 'safe27-snapshot.json', dict(epoch=time.time(), target=target, protected=protected,
         target_metrics=metrics, container_count=27, service_mutations=0))
    source = n / 'source'
    for original in (B / 'src').rglob('*.py'):
        destination = source / original.relative_to(B)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    for name in ('native.py', 'recipe_common.py', 'objective_variants.py', 'train_lora.py', 'train_lora_body_kl.py'):
        destination = source / 'scripts/experiments/gemma4-purpose' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(recipe / name, destination)
    pins = {str(p): sha(p) for p in source.rglob('*.py')}
    pins.update({str(p): sha(p) for p in (owner, guard_path)})
    save(n / 'source-pins.json', pins)
    data_root = R / 'analysis/fresh-purpose-20261107-mbpp-train-preparation-v1/filter-v1'
    quality_root = R / 'analysis/fresh-purpose-20261107-mbpp-reference-quality-v1'
    quality_identity = R / 'analysis/fresh-purpose-20261107-f04-mbpp-lora-training-v1/input-quality-identity.json'
    split = read(data_root / 'summary.json')
    task = dict(schema_version=1, kind='gemma4_lora_sft_v1',
        checkpoint=dict(path=parent['path'], tree_sha256=parent['sha256'], attestation=parent_attestation),
        teacher_checkpoint=dict(path=teacher['path'], tree_sha256=teacher['sha256'], attestation=teacher_attestation),
        loss_protocol=PROTOCOL,
        data=dict(**pin(data_root / 'train.jsonl'), split='train', ids=split['training_ids'],
                  split_summary=pin(data_root / 'summary.json')),
        input_quality=pin(quality_root / 'summary.json'), frame_evidence=pin(args.frame_evidence),
        hyperparameters=dict(rank=8, alpha=8, learning_rate=.0001, weight_decay=0., grad_clip=1.,
             accumulation=4, seed=0, max_frame_tokens=1024, relative_delta_cap=.005, pilot_steps=4, total_steps=64),
        maximum_seconds=2400, continuation_wait_seconds=300)
    save(n / 'trainer-plan.json', task)
    plan = dict(schema_version=1, mode='train-lora', approved_for_execution=False,
        checkpoint=parent['path'], checkpoint_tree_sha256=parent['sha256'], attestation=parent_attestation,
        source_root=str(source), source_pins=pins, output_root=str(n), container_prefix='fresh-body-kl-f07-v1',
        maximum_training_seconds=3000, training_startup_allowance_seconds=300,
        target=target, protected=protected,
        training_binding=dict(editing_parent_label=choice['selected_label'], served_model_label='F06',
             served_checkpoint=dict(path=served['path'], tree_sha256=served['sha256'], attestation=served_attestation)),
        training_forward_contract=dict(teacher_once_per_train_case=True, student_microbatches_per_step=4, holdout_forwards=0),
        training=dict(plan=pin(n / 'trainer-plan.json'), input_quality=pin(quality_identity),
             holdout=pin(data_root / 'holdout.jsonl'), restoration_proof=pin(n / 'current-serving-proof.json')),
        parent_selection=pin(args.selection), snapshot_evidence=pin(n / 'safe27-snapshot.json'),
        authorization='CPU preview only; exact plan must be root-reviewed before --run',
        continuation='root must bind the exact four-step pilot receipt to continue the same process to64')
    arguments = [str(source / 'scripts/experiments/gemma4-purpose/train_lora.py'), '--plan',
                 str(n / 'trainer-plan.json'), '--output', str(n / 'work/train-lora')]
    env = dict(os.environ, PYTHONPATH=str(source / 'src'), TMPDIR=str(R / 'infra/tmp/r13'), PYTHONDONTWRITEBYTECODE='1')
    started = time.time()
    result = subprocess.run([str(PY), '-B', *arguments], capture_output=True, text=True, env=env)
    (n / 'trainer-preview.stdout').write_text(result.stdout)
    (n / 'trainer-preview.stderr').write_text(result.stderr)
    save(n / 'trainer-preview.json', dict(command=[str(PY), '-B', *arguments], trainer_arguments=arguments,
         returncode=result.returncode, elapsed_seconds=time.time()-started, model_calls=0, service_changes=0))
    assert result.returncode == 0, result.stderr
    plan['training']['cpu_preview'] = pin(n / 'trainer-preview.json')
    save(n / 'owner-plan.json', plan)
    started = time.time()
    command = [str(PY), '-B', str(owner), '--plan', str(n / 'owner-plan.json')]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    (n / 'owner-preview.stdout').write_text(result.stdout)
    (n / 'owner-preview.stderr').write_text(result.stderr)
    save(n / 'owner-preview.json', dict(command=command, returncode=result.returncode,
         elapsed_seconds=time.time()-started, model_calls=0, service_changes=0,
         source_unchanged=all(sha(path)==digest for path,digest in pins.items())))
    assert result.returncode == 0, result.stderr
    print(json.dumps(dict(plan=pin(n / 'owner-plan.json'), trainer_plan=pin(n / 'trainer-plan.json'),
          preview=pin(n / 'owner-preview.json'), mode='preview_no_execution')))


if __name__ == '__main__':
    main()
