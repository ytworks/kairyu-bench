# Full UP-output unit gain observations

[up_unit_gains.py](../scripts/experiments/gemma4-purpose/up_unit_gains.py) collects local sensitivities for every hidden unit of each MLP UP projection. It uses the same complete authored answer frames, objective definitions, serial answer execution and observation policy as the existing collectors. This is observation only; it does not select units, solve an edit or change checkpoint weights.

For UP output u_j, the virtual coordinate is u_j(1+g_j) at g=0. The stored derivative is sum(u_j * dL/du_j), without cost normalization. The separate cost is the Euclidean norm of the BF16 source UP weight row. Selector tools must state how they use these costs and objective groups.

All model parameters stay frozen. The embedding output is the only model input leaf; hooks contract the UP output with its VJP in chunks. Detached UP outputs share their existing storage; no dense down-projection input is retained by this collector. This does not establish that the real model fits a particular GPU budget. The first collected answer checks four units in the first and last observed UP projection against explicit zero-additive gain probes. Parameter versions, absence of parameter gradients, source hashes and checkpoint metadata continuity are checked on completion. This is not a fresh full GPU weight hash.

## Output contract

- `coverage.json`: `units_method: "up_output_gain"`, `entries` containing `tensor`, full UP `shape`, unit `count`, `layer` and concatenation `start`, plus `observed_units`.
- `source-up-row-norms.npz`: full UP weight names mapped to float64 row norm vectors.
- `gradient-{purpose}-{fold}.npz`: the same names mapped to float32 raw gain derivatives. Answers and folds accumulate in float64; each fold uses its original example count.
- Standard provenance, frame evidence, progress and summary files. Fold and case records retain canonical source order even when execution is longest first.
- Native parameters are frozen and unobserved. There are no native gradient arrays and no cross-family native parity measurement.

The default command is an inert metadata preview. Actual collection requires the existing explicit `--run`, checkpoint attestation and exclusive GPU authorization. An example opt-in policy is:

```sh
python up_unit_gains.py \
  --checkpoint /absolute/checkpoint --checkpoint-attestation /absolute/attestation.json \
  --attestation-sha256 SHA256 --cases /absolute/authored-cases.json \
  --output /absolute/new-output --objective correct-logprob \
  --answer-batching serial --max-cases-per-microbatch 1 \
  --text-attention-backend sdpa --max-frame-tokens 2048 \
  --cuda-memory-cap-gib 88 --execution-order longest_frame_first --device cuda:0
```

Correct-logprob and margin use separate outputs. The explicit policy dispatches text through the Transformers SDPA API while retaining eager vision/audio policies; absent subconfigs are recorded. PyTorch can select a math kernel. There is no automatic attention API fallback, shorter-frame retry or truncation.

## Validation status

The existing CPU derivative fixture checks the full-row gain identity against an independent FP32 weight gradient, then runs both objectives through a BF16 gated MLP, saves actual gradient/cost NPZ files, verifies canonical order and rejects a wrong reference without changing weights or leaving hooks. It passed in the installed serving image on CPU. This CPU check makes no free-generation or benchmark improvement claim.

See [long-context observations](wsp-long-context-observation.md) for frame and allocator policy limits, and [matrix-only observations](wsp-matrix-only-observation.md) for the accepted native-frozen provenance route.

## First real collection

On the accepted saved AV checkpoint, both objectives completed all 120 cases and 38 folds, covering 60 UP matrices and 1,290,240 unit gains. Correct-logprob took 65.419 seconds and margin 126.954 seconds inside the collectors. Each measured peak allocation was 84.68965 GiB under the declared 88 GiB allocator cap; peak reservation was 87.97266 and 87.91211 GiB. The longest actual complete frame was 1,268 tokens. The configured 2,048-token limit does not establish that every frame of that length fits.

The first/last observed projection zero-additive checks differed by at most 8.73e-11 and 9.09e-13. Source row costs were exactly equal across the two objectives. Against the previous complete matrix observations of the same AV checkpoint and frames, per-case correct-logprob difference was zero and maximum margin difference was 8.94e-7. This is baseline consistency, not a claim that either derivative family measures generation quality.

The same O2 container was restored healthy at 2026-11-07 02:30:34.499906 UTC; AV, AD, AR and embedding services were protected. Evidence is under `analysis/code90-resume-20261106/current-av-up-unit-gains-observation-v1`: completion `bc75e6b7334f5e94b74d8a9f36d955e16d3a22613048b22622ff25ed07c28c3d`, collector output hashes `b07f4cfe3d0cb8fbf9bcd957d6b8c0976c7ff13295bbb16ef97d4638a06667e8`, and postchecks `51143b9d89337305283c877b19b810d6c2280450949f51462219df6260bcc257`. The exercised collector source was `67d4e19d41b88a3d6dd96970d0f123a3cf6dfa8545b1d4ad3a8cc9cc64eca57b`. No checkpoint weights were edited by this collection.
