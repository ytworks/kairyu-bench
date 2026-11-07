# Body-start suffix CE + explicit teacher first-token KL: CPU implementation

This is an opt-in implementation awaiting the completed F06 evaluation, parent selection and a separate GPU authorization. It does not select a candidate or change serving. The earlier [two-hypothesis memo](2026-11-07-f06-next-loss-hypotheses.md) remains historical: its prefix-only teacher estimate is superseded here by the required **identical complete frame** calculation.

The existing train_lora.py default is preserved. Its prepare_training_frames, answer_loss and make_adapter functions are AST-identical to the F06 source. A plan opts in with loss_protocol exactly:

    {"kind":"body_first_token_kl_v1","lambda":1.0,
     "teacher_frame":"identical_complete_frame","teacher_dtype":"F32"}

The body target is precisely the suffix from the first top-level function's first statement after its leading docstring through the end of the reference code. Introductory imports/signature/docstring, tokens crossing the starting boundary and the official turn terminator are masked. This is not the union of every function's body: 9 of the 204 references contain later function definitions whose headers remain in the suffix (IDs 671, 702, 717, 896, 905, 926, 940, 949, 962). Full IDs and full frame lengths remain unchanged. Unsupported rows are retained with reasons and make the whole training preflight ineligible; no case is silently removed.

An optional teacher_checkpoint has the same path/tree_sha256/attestation fields as the editing checkpoint. Missing means the same checkpoint, with one load. Separate means explicit frozen teacher load, one no-grad full-frame forward for each of 204 training examples, F32 full-vocabulary softcapped first-token-logit cache, teacher release/GC/CUDA cache release and allocated-byte check, then editing-base load. A nonzero remaining CUDA allocation stops before loading a second model. Both content attestations and stat continuity are checked; tokenizer/template/config/generation/processor hashes must match. The editing base is never replaced by the teacher. Thus a higher-scoring F06 can remain the editing base while an explicitly chosen F04 provides the reference initial mode.

Teacher and student use identical complete token IDs and lengths, not a prefix-only teacher versus full-frame student. Both initial logits use the same single-row head shape at prefix_tokens−1. There is no padding or truncation. For the existing train204 these full frames total 22,922 tokens (maximum 336), rather than the earlier 6,602 prefix tokens. Full-vocabulary F32 cache files and per-example frame/source identities are retained; the 51 holdout items never enter this computation.

Student loss is mean suffix CE plus lambda1 KL(p_teacher || p_student) at the first assistant token. Teacher logits are detached. Logs distinguish body CE, initial-token KL, total objective, body target tokens, teacher forward attempts/completions, student attempts/completions and their sum. Completed64 uses teacher204 + student256 = total460 forwards; pilot4 uses teacher204 + student16 = total220. A failed partial teacher cache preserves its completed rows and error. This is training computation, not a baseline benchmark or capability score.

CPU verification used the existing immutable Torch image without network, GPU exposure or writable repository mount. All 7 tiny tests passed in 1.027 s (3.513 s container wall), including full-length teacher versus student, detached teacher, mask/unsupported-row retention and existing LoRA numerical contracts. Host tests passed4/skipped3 optional Torch tests; Ruff passed. A test-only scalar-conversion warning is retained in the log. The later log-field correction changes no numerical path. Real Gemma training, separate-teacher GPU memory release and resulting capabilities remain unverified. The final trim/owner changes passed 28 owner tests plus 4 host trainer tests (3 Torch tests skipped on the host), and focused Ruff. The earlier seven-test real Torch result remains the numerical evidence; the trim amendment does not change the loss math. The final all204 tokenizer proof passed separately.

The existing GPU owner now supports a separate editing-parent and currently served-target binding. This permits editing the selected F04 parent while stopping and restoring the exact currently served F06 CID; it does not require an extra F04 deployment. Its original default F04 binding remains intact. Add these top-level owner fields to opt in:

    "training_binding": {
      "editing_parent_label": "F04",
      "served_model_label": "F06",
      "served_checkpoint": {"path":"<absolute-checkpoint>",
        "tree_sha256":"<content-tree>",
        "attestation":{"path":"<full-hash-and-stat-receipt>","sha256":"<receipt-sha>"}}
    },
    "training_forward_contract": {
      "teacher_once_per_train_case":true,
      "student_microbatches_per_step":4,
      "holdout_forwards":0
    }

The existing five training pins are unchanged. The served proof binds the current exact CID/read-only mount/full identity independently of the editing parent. The latter remains bound by the owner and trainer checkpoint/attestation equality and the selected-parent label. The optional KL contract requires the new helper source pin and checks completed/attempted teacher, student and total forwards, all teacher cache file hashes, same full-frame IDs/lengths and exact teacher/base metadata. Original stop/start, late-create cleanup and protected-container logic are unchanged. The earlier informal assertion that the old owner already had a fixed256 forward guard was incorrect; this work adds the explicit opt-in accounting.

The first real204 tokenizer check retained 71 unsupported cases: the pinned official strip_thinking macro trims whitespace, whereas the initial mask helper expected raw reference text verbatim. Its failed v1 proof is preserved. The minimal fix accepts only official boundary whitespace trim, rejects channel-bearing references, checks the complete rendered string exactly, and records both trim amounts and raw/body offsets. The raw input rows and original complete token IDs are unchanged. V2 passed all204: 22,922 full tokens, maximum336, 12,621 selected body tokens, per-case range3–281, no unsupported/ineligible case, no prompt or turn-terminator target. The 9 later-function cases remain explicit suffix supervision. Container9.543s/tokenizer2.762s; teacher/model/gradient calls0. See the [failed and successful tokenizer records](2026-11-07-mbpp204-body-frame-proof.md).

Final live source identities: trainer c9406e597d015ec4f3cf1d626b0208d26fd6de7d673c6efca580af7001756374; body helper290365237443e0138079e106452c88dcf64779e0dc6834447f29e92666c90ef8; owner8d48cd69f9fe4109dec8fe190dde665483b24c6edb92e59dde4118859de6b2da. Source freeze and exact checks are in final-optin-source-freeze.json. Both mask/identity fixes protect demonstrated failures; there is no new general training framework. No GPU model run or capability gain is claimed.

[Source snapshots, exact CPU command/logs, default-path AST proof and SHA index](records/2026-11-07-body-kl-implementation/) distinguish CPU evidence from unexecuted GPU behavior. Normal runtime thinking, finished-token semantics and full canonical WSP scoring remain the evaluation conditions. A KL value or reduced CE is not evidence of an improvement in coding scores.
