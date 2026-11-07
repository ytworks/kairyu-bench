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

The existing CPU derivative fixture checks the full-row gain identity against an independent FP32 weight gradient, then runs both objectives through a BF16 gated MLP, saves actual gradient/cost NPZ files, verifies canonical order and rejects a wrong reference without changing weights or leaving hooks. It passed in the installed serving image on CPU. Real 31B GPU collection, memory fit and subsequent unit selection have not yet been validated. This check makes no free-generation or benchmark improvement claim.

See [long-context observations](wsp-long-context-observation.md) for frame and allocator policy limits, and [matrix-only observations](wsp-matrix-only-observation.md) for the accepted native-frozen provenance route.
