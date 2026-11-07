# F01 lossless retention and storage audit — 2026-11-07

Subsequent state: the explicitly authorized two-file representation change completed after independent prechecks. See [the exact operation and receipts](2026-11-07-f01-representation-change.md). F01 is now a dependent archive, not a loadable saved checkpoint. The creation-time statements below are preserved as historical evidence.

F01 is not deployed: its 120-case finite comparison completed base/candidate/restored passes with reported restoration error zero, but referent fold-1 LL and margin changes exceeded the predeclared −0.03 retention allowance. Its official WSP score is unmeasured. The full failed trial remains in the experiment tree. This audit changes no benchmark score or selection.

The user requires retention of every failed experiment. A dependent lossless archive can retain F01 exactly while allowing the owner to consider releasing only this session's generated duplicate payload. This archival operation did not delete or modify any checkpoint or change a service. AV remains a mandatory retained dependency. No other candidate or cache is a release target.

## Disk and existing alternatives

At inspection the NVMe is /dev/md2, ext4, with approximately 73 GiB available. No reflink clone was attempted; ext4 is not a usable assumed reflink path. The existing editor makes a private copy of each edited shard. All 410 F01 matrices touch both shards, so an unchanged-shard pool does not avoid these copies. The existing native_faststore only isolates scalar/vector edits and requires an initial repack; it cannot avoid copies for this matrix edit. The existing sparse vLLM loader excludes direct V edits in its present tensor-family validation, and the old overlay screen runs a code-only dev90 selection. Neither is a drop-in substitute for this 410-matrix direction or the required full 778-item WSP. No edit family or benchmark item was removed to fit these tools.

The saved F01 checkpoint is 62,578,656,419 bytes across nine files. Its two private weight shards occupy 62,546,338,264 bytes. The seven nonweight files occupy 32,318,155 bytes. A second full materialization with the previous 50 GiB reserve cannot fit in the present free space. Releasing only an independently retained F01 representation would recover approximately 58.25 GiB before a later accepted candidate is materialized. The accepted candidate can serve all four final replicas from one read-only checkpoint; four replicas do not require four checkpoint copies. A later eligible HF upload can read those same accepted files and need not duplicate the payload into another release directory.

## Proof and limitations

The existing [editor.verify](../../src/kairyu_bench/wsp/weights/editor.py) at lines 543–585 performs full-file checks. It validates the manifest against the fixed spec, recomputes source-header coordinates and before/after bytes, checks exact file inventory and sizes, freshly hashes every base file, hashes every candidate file, and hashes the candidate with declared edited spans replaced by original bytes. It rejects any undeclared byte difference, edited shared inode, changed file or missing recorded hash. Thus the historical successful verify proves that unchanged bytes equal AV and each changed span equals the manifest after_hex.

Historical F01 verify.json is true, with all nine file hashes and legacy_full_hashes_missing=false. The manifest records 201426 two-byte edits across 410 matrices, L2 0.9915271042485978. It contains every offset, original byte pair and new byte pair. F01 has no .cache content or .kairyu-model-attestation.json, so the verifier's documented exclusions omit nothing from this actual checkpoint. Filesystem inode/ctime identity is provenance, not a promised reconstruction output; file bytes and relative names are the reconstruction target.

The archive additionally re-planned all 201426 spans using the frozen existing editor and current AV bytes. Every offset, before/after byte pair, shape, dtype and changed-index list matched the historical manifest. This took 8.57 seconds CPU and zero model calls. AV full-hash evidence from the current baseline capture matches all nine source hashes and exact device/inode/size/mtime/ctime continuity. F01's earlier finite owner recorded only device/inode/size/mtime; the current audit matches that partial stat evidence and newly records ctime, mode and link count without calling it retroactive full-stat continuity. No fresh full candidate payload hash and no physical full reconstruction were performed during archival.

## Preserved artifact

Under the experiment root, analysis/fresh-purpose-20261107-f01/lossless-archive-v1 contains:

- Original-byte gzip copies of proposal/fresh-code-joint-f01.json and materialization/edit-manifest.json, each decompressed and SHA checked against its original.
- All seven nonweight files: chat template, config, generation config, safetensors index, processor config, tokenizer and tokenizer config. Each copied file was freshly hashed against both historical source and candidate SHA.
- Original create/preview/verify, proposal report, finite owner plan, completion and numeric analysis receipts. All original experiment files, failed attempts and review are also retained at their existing paths.
- All 89 package source files from the finite run's frozen source snapshot, with per-file hashes. The editor and checkpoint reader hashes match both finite owner pins and current source. CPU reconstruction uses the recorded Python/NumPy versions, not Torch or GPUs.
- archive.json, archive-validation.json, BASE-SHA256SUMS, CANDIDATE-SHA256SUMS, whole-archive SHA256SUMS, and a README containing existing-editor reconstruction commands into new paths.

The archive was initially 67,307,900 bytes before the small validation, checksum and README additions. The archive.json SHA256 is fb228636f67f4287151b20332e06b34def82f9d0a6f9af1b07c9ea62913c1409; its status describes creation, when payload still existed. Any later duplicate release requires its own receipt and does not rewrite that history.

The mandatory AV tree is 5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf. Original spec SHA is 12f4e111a9d31539c6e2d3a94e8270c34b4dbc8a1c4f145fcdcf21ef936eb9d5; original manifest SHA is d84737dacb9fc79000b074ba8a33be0c877c007c06b5a7f8da305928e91c4701; original verify SHA is 80f4bedc3f32a37eb00138bf294a75563b60f05fe628889acda8a9c6df8a1990.

## Commands and one-off processing

All archival commands ran through SSH with the existing CPU Python. No separate common archive tool was installed. The one-off Python operations were:

1. Refuse an existing archive directory. Load original spec, manifest, verify and finite owner plan. Require true verify, no legacy missing hashes, all nine verify file hashes equal the manifest, exact spec JSON and SHA match, and manifest/verify SHA match the finite owner pins.
2. Enumerate F01 regular files and require exactly the nine manifest names. For each, match finite owner device/inode/size/mtime and record current full stat. Record AV current full stat separately.
3. Create metadata and records subdirectories. Stream each original spec and manifest through gzip.GzipFile(filename="", mode="wb", mtime=0) into a newly created .gz, then stream-decompress and verify SHA against the untouched original. This preserves original bytes, including whitespace and trailing newline; it does not reserialize the manifest.
4. Use shutil.copy2 for each nonweight file and selected receipt to its new archive path. Hash both source and copy, and require every nonweight hash equal the historical source and candidate hash. No hard links are created.
5. Use shutil.copytree for the finite run's frozen source, excluding only __pycache__ and *.pyc. Record all 89 file SHAs. Require editor.py and checkpoint.py equal finite source pins and current source. Record Python and NumPy versions.
6. Write archive.json with SHA/size inventory, both checkpoint stats, original paths, mandatory AV dependency, prior verification evidence, explicit stat limitations and no-deletion status. Recheck every current AV/F01 file stat against step 2.
7. Run the existing library validation below, save its result as archive-validation.json, and write the two expected complete-checkpoint checksum lists directly from the manifest.
8. Add the README with existing editor create preview, create --apply, verify, and complete sha256sum -c CANDIDATE-SHA256SUMS reconstruction commands. Finally hash every archive file except SHA256SUMS itself, write that checksum list, and verify it in full.

The exact core operations were the following Python statements, under PYTHONDONTWRITEBYTECODE=1. The surrounding code only constructs paths, records the stated inventory and asserts the conditions above.

    with original.open("rb") as inp, target.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
            shutil.copyfileobj(inp, out, 8 * 1024**2)
    h = hashlib.sha256()
    with gzip.open(target, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    assert h.hexdigest() == digest(original)

    sys.path.insert(0, str(archive / "source"))
    from kairyu_bench.wsp.weights.editor import validate_manifest, plan_edits
    with gzip.open(archive / "edit-manifest.json.gz", "rt") as stream:
        manifest = json.load(stream)
    summary = validate_manifest(manifest)
    expected = plan_edits(Path(manifest["source"]), manifest["spec"])
    fields = ("tensor", "shard", "offset", "size", "dtype", "shape",
              "before_hex", "after_hex", "selected_indices", "changed_indices", "changed_count")
    assert len(expected) == len(manifest["edits"])
    assert all(all(a[k] == b[k] for k in fields)
               for a, b in zip(expected, manifest["edits"], strict=True))

Reconstruction must first verify the whole archive and all mandatory AV hashes, use a fresh output and manifest path, decompress the original spec, use the archived existing editor to materialize and fully verify, then compare all nine reconstructed hashes to the archived candidate list. The original full manifest has original paths; a new creation manifest has new paths and therefore need not have the same JSON hash. It must describe identical checkpoint bytes. Keeping only a spec without its AV dependency and hashes would not satisfy this archival method.

The executable reconstruction recipe is stored in the archive README. Its central existing-tool commands are:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$F01_ARCHIVE/source" "$F01_PYTHON" \
      -m kairyu_bench.wsp.weights.editor create "$F01_BASE" "$F01_RESTORE" \
      --spec "$F01_META/spec.json" --manifest "$F01_META/edit-manifest.json" \
      --reserve-bytes 53687091200 --apply > "$F01_META/create.json"
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$F01_ARCHIVE/source" "$F01_PYTHON" \
      -m kairyu_bench.wsp.weights.editor verify "$F01_META/edit-manifest.json" \
      > "$F01_META/verify.json"
    (cd "$F01_RESTORE" && sha256sum -c "$F01_ARCHIVE/CANDIDATE-SHA256SUMS")

One failed documentation-writing invocation used malformed outer shell quoting around an apostrophe. It left the repository note at five lines; the archive payload and its completed validation were unaffected. The command text escaped to the local shell and emitted command-not-found errors, so this was an operational failure of the SSH wrapper construction. It displayed no credentials and made no model or service calls. Local git status remained clean. The corrected writer quotes the complete remote script as a single shell argument. All archive calculations and file operations occurred remotely. The incomplete note was completed; the failed attempt is retained here.

## Fixed full-778 runner handoff

The observed AV runner's immutable image ID is sha256:f91bdcf997af6d7b194bec229812fd018277158f6795de7d91e58448936e7ad8, distinct from config digest sha256:9ab4fbffb5a957cb96f4a8afd4afd31d4cb5649c25251dbc1d5cf3b83e954ac7. Its frozen runtime/fresh-purpose-20261107-av-live10-v1/runner-app has 224 source/resource hashes, all rechecked here. The candidate must reuse the frozen runner and the same observer source/config sampling semantics with fresh IDs/outputs. Conditions are tier10/dev, all 778 items and 14 areas, chat concurrency32, logprob8, agent1, TP2, one replica, MTP4, and chat/logprob to the same edited checkpoint. Cache and dedicated Docker socket/network remain the baseline values.

The original wrapper always built before running, so setting the ordinary image-tag environment variable alone did not establish reuse. The parent is adding an explicit immutable-image opt-in to the official wrapper and tests; that change is reviewed separately. The active baseline runner is unchanged. No candidate execution was launched by this audit.
