# Load-time sparse BF16 overlay experiment

This optional, separately installed vLLM plugin avoids creating a new full checkpoint for a sparse trial. It is a CPU-tested experiment, not an enabled serving path. It does not change an already loaded model. Existing saved-checkpoint creation, verification, and deployment remain the defaults.

The implementation uses vLLM's formal vllm.general_plugins entrypoint and register_model_loader extension. SparseOverlayLoader subclasses DefaultModelLoader; it delegates file discovery, tensor loading, model construction, TP loading, and postprocessing. It wraps the ordinary safetensors iterator and clones only each edited complete CPU BF16 tensor before replacing the explicitly selected raw words. No FP32 conversion of a whole tensor, new merge kernel, model class replacement, or monkeypatch is involved.

## Supported boundary

The initial target is dense BF16 Gemma4ForConditionalGeneration, TP1 or TP2, PP1/DP1, without runtime LoRA, quantization, expert parallelism, or secondary checkpoint sources. Only model.language_model.layers.N.self_attn.o_proj.weight and the three MLP gate/up/down matrices can be edited. Existing editor validation enforces exact BF16 values, coordinates, non-noop edits, and coordinate budgets (512 per tensor; the explicit sampled-matrix batch contract for larger totals). Native vectors, Q/K/V, embeddings, vision tensors, and all other tensors pass through unchanged.

The spec must declare the exact saved logical parent path and full-file tree. This version does not accept an unmaterialized overlay as its next parent. A future promotion would first materialize and verify the accepted candidate using the existing editor.

Target and MTP draft are distinct, explicitly pinned local checkpoint directories. The draft path is an unchanged iterator bypass, with its own prior full-file hash/stat proof and per-rank bypass receipt. An unknown checkpoint invocation fails. The plugin retains the ordinary draft model loading and embedding-sharing behavior; it does not apply the target overlay to MTP.

## Plan and identity

The owner supplies --load-format kairyu_sparse_bf16_v1 and a --model-loader-extra-config JSON object containing only overlay_plan (absolute path) and overlay_plan_sha256. The small plan/spec/proof files are SHA-pinned and checked again after loading. All paths must be canonical and free of symlinks.

Container paths must use the same canonical declarations as the plan, proofs, and spec. The current serving example's /model aliases cannot be substituted for host checkpoint paths without changing this binding. A future owner should mount the source, draft, plan, and proof files at their identical absolute paths and pass matching model arguments, or design a separately verified path projection. This plugin does not silently translate aliases.

The plan schema is:

- schema_version: 1, kind: "kairyu_sparse_bf16_v1".
- physical_source and draft_bypass: {path, tree_sha256, proof: {path, sha256}}.
- Each proof uses the existing freeze projection: checkpoint_path, control_tree_sha256, status beginning with verified, sorted all_checkpoint_files rows {path,size,sha256}, and candidate_inventory_stat keyed by relative filename. Required stat values are device, inode, size, mtime_ns, and ctime_ns. Additional provenance in an existing freeze is retained in that pinned file.
- logical_parent: {path, tree_sha256}, equal to this saved physical source.
- spec: {path, sha256}: existing sparse schema-1 edits with explicit coordinates and set_values, plus source_checkpoint and base_tree_sha256. An explicitly empty edits list is the baseline mode.
- tensor_parallel_size: 1 or 2.
- receipt_directory: a fresh owner-created directory outside both checkpoints.
- implementation_sha256: exactly core.implementation_identity() (plugin files plus existing reader/editor).
- vllm_source_sha256: exactly the installed files listed by core.VLLM_FILES. The loader rejects any source drift. The CPU test used the audited vLLM 0.33.0 installation; this is not a version-agnostic mapping claim.

The owner must derive the proof projection from actual prior full-hash verification, pin the current incumbent state, and confirm its saved path/tree before launch. The plugin recomputes the small inventory tree from those prior file hashes and checks all current file stats; it does **not** rehash the large shards. Source proof is explicitly prior_full_file_sha256_plus_exact_current_stat.

logical_overlay_sha256 hashes the logical parent, exact spec, plugin implementation, and contract version. It is separate from physical_checkpoint_tree_sha256, which continues to describe the unchanged saved source. No receipt calls the physical base the saved overlay candidate or claims a fresh full-model hash. Launch/deploy attribution must carry this logical identity rather than reuse the base tree as the candidate identity. No existing deployment helper has been adapted automatically.

## Post-TP proof

Raw iterator hashes alone are insufficient. After the normal loader and weight postprocessing complete, every rank reads the selected raw BF16 words from the actual model parameters:

- Gate/up source rows map to the rank's packed gate_up_proj rows, including the distinct gate/up offsets.
- Down/o source columns map to the corresponding row-parallel rank's local columns.
- Exact local parameter names, shapes, BF16 dtype, and divisibility are checked.

The readback transfers selected words once per tensor. All ranks exchange their proof records through the existing TP CPU process group. Every rank checks the same complete rank set, plan/logical identity, unique coordinate ownership, coverage, and exact expected uint16 values before returning the model. A local readback error is carried into that collective so other ranks reject it too.

Fresh target-rank-N.json files contain the all-rank proof and draft_bypass-rank-N.json files record the explicit bypass. Existing files are never overwritten. The owner must require every expected target and draft receipt, matching the command/image/plan and successful engine readiness. A source/iteration failure before the collective remains an ordinary worker startup failure; the owner must terminate the failed launch, not retry within the same receipt directory.

The proof covers selected coordinates, not an independent complete-model GPU hash. Empty-overlay proofs contain zero selected coordinates and must not be represented as full-model parity evidence.

## CPU validation and future rollout

One E2E fixture writes tiny real BF16 safetensors, uses the installed DefaultModelLoader iterator through the custom subclass, simulates the reviewed TP2 gate/up packing and down/o partition boundary with Torch tensors, and validates all-rank readback. It checks source bytes/stat preservation, signed-zero preservation in untouched native/Q tensors, exact private target words, corrupted post-TP word rejection, missing/duplicate coverage rejection, pinned draft bypass, empty-overlay object identity, and wrong logical parent rejection. This is not a full vLLM engine or GPU test.

The test requires the existing audited vLLM/Torch CPU environment; it skips in the lightweight repository environment. Run it without GPU devices or network:

    python -B -m unittest discover -s tests -p test_vllm_sparse_overlay.py -v

Before any real candidate trial, a separately authorized rollout must:

1. Install this package and the pinned benchmark reader/editor package into a new image using --no-deps; do not modify a running image/container. Enable only the intended general plugin.
2. Compare ordinary saved-source loading with an explicitly empty overlay under identical image/runtime/model/MTP/TP/sampling settings. Pin the ordinary loader to safetensors with lazy strategy, matching the plugin’s internal strategy; do not leave an auto-format difference unrecorded. Check the all-rank startup records, selected diagnostic word mapping, and fixed authored numeric output parity; record the limits of any nondeterministic output comparison.
3. For one small nonempty overlay, verify every selected word after TP, full receipt/source continuity, and ordinary readiness. Run both comparison arms using the same plugin/image (empty versus nonempty).
4. Retain the saved-checkpoint fallback. Successful real model startup, loaded-word validation, and ordinary generation remain required; the first attempted rollout described below did not reach readiness.

The potential saving is the full candidate copy and its subsequent verification scan. Base checkpoint loading and GPU allocation still occur. Each edited tensor gets one private clone (approximately 220.5 MiB for the largest current Gemma4 MLP matrix). A consuming loop may retain the previous clone while the generator allocates the next, so two adjacent clones per rank can overlap (approximately 882 MiB across TP2 for two largest clones per rank), in addition to ordinary loader/model memory. Actual loader peak memory and any speed improvement remain unmeasured. Matrix changes remain incompatible with the native-only faststore generation described separately.

## Offline image staging

The optional [staging helper](../scripts/experiments/vllm_sparse_overlay/stage_image.py) performs no Docker or model actions. Preview with an existing local Python environment that provides pip and modern setuptools; the current audited af9 image provides both, while the lightweight managed test environment does not provide pip:

    python -B scripts/experiments/vllm_sparse_overlay/stage_image.py --output /owned/nvme/fresh-stage

After reviewing the reported input bundle SHA, the same command with --expected-input-sha256 SHA --apply builds exactly two universal pure-Python wheels with existing local pip and setuptools (including its bundled bdist_wheel command) (--no-index, --no-deps, --no-build-isolation) and installs them into its owned context. It refuses root execution, existing outputs, source drift, non-pure wheels, and installed Python byte changes. A failure preserves its owned staging directory; no hidden retry occurs.

The support wheel copies only the actual three package initializers plus checkpoint.py/editor.py from this revision, byte for byte. It is a deliberately limited serving dependency, not an installed complete benchmark runner. No benchmark datasets or generated answers enter the context. Before building, verify that the base image has no conflicting kairyu_bench package or reconcile that namespace explicitly; never let an accidental import fallback hide the limited dependency.

The [Dockerfile](../scripts/experiments/vllm_sparse_overlay/Dockerfile) has no RUN instruction. It only copies the preinstalled Python tree and the existing tiny smoke fixture, preserves the base entrypoint/Cmd, and sets USER 1003:1003. The owner must first verify its BASE_IMAGE reference equals the reviewed immutable base image ID and later record the resulting new image ID. Existing images and containers remain unchanged.

The built-image CPU smoke reuses the existing E2E with KAIRYU_OVERLAY_INSTALLED_ROOT=/opt/kairyu-sparse-overlay, no repository/PYTHONPATH mount, no network, and no GPU devices:

    python -B -m unittest discover -s /opt/kairyu-sparse-overlay-smoke -p test_vllm_sparse_overlay.py -v

In this mode plugin-source fallback is disabled and the imported plugin/reader/editor must all resolve inside the staged installed root. Record actual source hashes, entrypoint metadata, package versions, and the iterator/TP-boundary result. Passing the source-tree fixture alone does not establish installed-wheel compatibility. Staging, image building, this isolated smoke, and actual model rollout remain separate explicitly authorized actions.

## Recorded installed-image and first rollout evidence

The offline staged image sha256:33025e08d05ed9f79ba0f4fa34a62bb005e6404c3aa70af20ade5e73c1e88602
preserved the pinned af9 base layers and ran as UID 1003. The installed-only CPU
E2E passed once in 9.571 seconds; all plugin/reader/editor imports resolved to
/opt/kairyu-sparse-overlay. Repository CI passed 513 collected tests
(460 passed, 53 optional skips), with unchanged checked source pins. The
optional installed-image test is reported separately from those skipped tests.

The first authorized four-arm rollout on 2026-11-06 UTC failed while loading
the first ordinary saved-S arm, before readiness or generation. The owner
recorded only ValueError and removed its temporary container, so the original
startup cause is unknown. It restored the same original O2 container and
checked the protected services and source state. No selected-word proof or
functional parity result was produced; the failure does not establish an
overlay numerical mismatch. The current accepted AC checkpoint was unchanged.
The preserved receipt is
analysis/code90-resume-20261106/vllm-sparse-overlay-validation-v1/gpu-completion.json
(SHA-256 994cfe8e0cc6847b92fb95076a3e7fd241299321b41db1f561827d5b07c99c95).

A separate CPU-only check of the installed vLLM config writer reproduced
PermissionError for UID 1003 at /.config/vllm/usage_stats.json; writing under
an owned NVMe config directory succeeded. This establishes a writable-config
requirement, not the cause of the discarded startup failure. A prepared,
unexecuted owner revision sets XDG_CONFIG_HOME and VLLM_CONFIG_ROOT to that
dedicated directory for every arm. It also saves a bounded private startup log
before cleanup and publishes only fixed assertion messages, error classes,
flags, and container exit status. No plugin, numerical kernel, or image change
was made, and no GPU retry is implied by these diagnostic changes.
