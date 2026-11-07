# Helper-stage capped functional targets (AT–AY)

The accepted source remains cumulative AD. This experiment retains the original 160 authored cases but fits 112 cases in 17 purposes and 34 folds. Six legacy purposes (tools, instruction, general, reasoning, lookup, Japanese) are excluded prospectively because some authored answers are continuation suffixes or ambiguous answer units. Their observations remain available as separate diagnostics; official benchmark coverage is unchanged. Plain code is a retention objective, not a primary objective in this batch.

The four new training chains already passed helper, oracle-helper downstream, and generated-public-helper downstream execution: 96/96 trial checks from 12 native-thinking requests. The two unused transfer chains passed their helper stages but failed downstream output-shape checks with both oracle and generated helpers. A separate, explicitly exploratory prompt-copy contrast stating shape `(1,)` and float64 passed all 48 checks in six requests. Original cases, tests, and scores were retained. This is consistent with output-contract ambiguity; it does not establish a dependency-propagation deficit or a learned improvement. No generated answer bodies or hidden reasoning were inspected.

## Fixed methods and CPU results

For each primary correct-LL fold, the target is `min(0.05 * remaining_headroom, 0.05)` nat/token. Primary margin targets are independently +0.02 nat/token; other included rows permit −0.01. All six use Euclidean cost, frozen native tensors, a declared raw BF16 delta-L2 cap of 5, one continuous calibration, one BF16 proposal, and no amplitude sweep. Science includes the six existing science purposes plus helper-stage and composition-stage.

| Candidate | Eligible matrices and primary purposes | Actual changed coordinates / tensors | Raw delta L2 | Predicted misses |
|---|---|---:|---:|---:|
| AT | O + MLP; science + implementation/algorithm/workflow | 117,977 / 240 | 2.407130 | 0 |
| AU | MLP; same purposes | 90,235 / 180 | 3.179997 | 0 |
| AV | Q/K/O + MLP; same purposes | 171,925 / 360 | 1.813708 | 0 |
| AW | Q/K/O + MLP; helper/composition, implementation/algorithm, referent/fact retention | 167,095 / 360 | 1.331330 | 0 |
| AX | O + MLP; science + implementation/algorithm/workflow + complete tool-call purposes | 118,067 / 240 | 2.431513 | 1 |
| AY | Q/K/O + MLP; science/core fold-agreement layer selection | 53,624 / 108 | 3.386045 | 0 |

AX's predicted shortfall is 1.82e−8. BF16 realization exceeds the continuous coordinate box at one coordinate each for AT, AX, and AY (9.80e−5, 1.74e−6, and 2.02e−5). These are reported without rescaling. All six meet the prospective raw norm cap and have nonzero exact-word edits.

The box uses `rho = max(abs(delta_continuous) / max(abs(base), 1e-6)) + 1/128` and `absolute = max(1e-6, rho * 1e-6)`. The buffer is not a universal BF16 ULP guarantee. Actual rho ranges from about 6,231 to 21,008, with absolute floors 0.00623–0.02101: near-zero source cells make this a broad containing box, not evidence of a small relative functional perturbation. Raw norm also does not guarantee local linear accuracy.

The first batch completed AT/AU/AX. AV/AW/AY stopped in metadata validation before either solver because an inherited wrapper allowed only O/MLP. Their original failures were preserved. A versioned runtime extended only the family allowlist to Q/K/O/MLP, and the three first solves completed in 38.85 seconds. Completed candidates were not recalculated; original public runtime paths remained fixed for provenance.

The versioned runtime inherited a reporting-only `qkv_edit_columns=0` literal. An additive correction preserves the reports and distinguishes eligible Q/K columns (61,440 / 61,440 / 18,432 for AV/AW/AY) from changed Q/K coordinates (56,646 / 54,953 / 17,473). Saved layouts, gradients, exact specifications, and independent realized-cost checks already include those columns.

## Coupled K/V interpretation

In tied K/V attention blocks, a K projection edit changes both attention scores and the value representation. No explicit V tensor is edited, but the value function is not frozen. These candidates are outcome comparisons, not a routing-versus-content ablation. The collector differentiates the complete objective through projection outputs; the derived box uses those saved gradients, without a routing-only sensitivity assumption.

The actual Opus 5.5 methods-only addendum used no tools, took 46.52 seconds, and cost $0.100032. It endorsed this candidate-level interpretation and did not require an extra Q-only candidate. We retained the fixed policies, measured finite responses, and ordinary generation evidence; optional counterfactual forwards, new collections, and universal zero-loss gates were not added. Only AY uses fold-agreement selection. Proposal workers are CPU workers; finite replay is a separate GPU batch.

All six were selected for one same-parent finite replay over full 160-case diagnostics (46 folds), while fit-only statistics retain their 112-case/34-fold identity. No candidate has been adopted on the basis of analytic predictions.

## Reproduction and immutable evidence

The [versioned batch recipe](scripts/run-versioned-derived-matrix-batch.py) reuses the existing bounded worker functions and accepts explicit repository, alternate-runtime, roster, hashes, and fresh-output arguments. The [compatibility patch](scripts/derived-radius-qk-v1.patch) reproduces the exercised Q/K runtime and the existing single end-to-end fixture variant. Apply it only to a separate copy when original paths remain pinned; never rewrite historical plans or reports.

The fixture checks Q selection through continuous calibration, BF16 proposal, saved bytes, and numeric postcheck, while preserving excluded O/V/native bytes and rejecting a reference-mask mismatch. It passed in 0.708 seconds. The exact exercised wrapper passed Ruff and inert metadata preview. No new test method was added.

Private evidence is retained under `current-ad-helper-capped-six-v1`: fixed policy `85caa68a…`, combined postcheck `1eee6477…`, finite roster `a00f3b2b…`, reporting correction `d9c2bd80…`. The K/V review receipt is `b7dc4a38…`, disposition `2fd76a7b…`; the helper output-contract contrast is `356ff617…`. The reproduction patch maps original derive source `2e66fef3…` to exercised version `271b2bab…`; it deliberately preserves the historical reporting literal, which is corrected by the accompanying sidecar. After finite closure and confirmation that the next official owner did not pin ancestor math sources, the public derive source was updated to `70a1922e…`, computing the count from the selected layout. The same existing Q-selected end-to-end fixture passed in 0.607 seconds, with Ruff and mypy also passing; no actual candidate was recalculated. Original source `2e66fef3…`, versioned runtime `271b2bab…`, and all historical reports/specs remain archived unchanged.

Any later accepted parent requires fresh anchoring. Absolute `set_values` from this source must not overwrite accumulated changes on a future incumbent.

## Measured full-160 finite response

The six candidates completed one same-parent replay. Independent arithmetic from saved numeric outputs agrees with the replay summary; all 360 selected tensors / 179,842 union coordinates match expected BF16 words in all eight phases (base, six candidates, restored base). Restoration margin, correct-NLL, and KL errors are zero. These proofs cover selected words, not a new full-checkpoint hash.

Values below are equal-purpose means of two folds, in nat/token. Correct LL is negative NLL; foil LL is correct LL minus margin.

| Candidate | Old science6 LL / margin | New helper LL / margin | New composition LL / margin | Implementation/algorithm/workflow LL / margin | Plain-code LL / margin | Actual fit misses / 68 |
|---|---:|---:|---:|---:|---:|---:|
| AT | +.04450 / +.01397 | +.09729 / +.03165 | +.07218 / +.02574 | +.04673 / +.01259 | −.17305 / −.52467 | 29 |
| AU | +.04686 / +.01350 | +.09352 / +.03709 | +.07100 / +.01835 | +.05048 / +.01213 | −.05202 / −.49051 | 26 |
| AV | +.04660 / +.01017 | +.09336 / +.04210 | +.07151 / +.01242 | +.04020 / +.00571 | +.11866 / +.01736 | 36 |
| AW | +.01213 / −.00135 | +.06079 / +.04214 | +.06063 / +.02197 | +.04941 / +.02957 | +.12962 / +.16997 | 13 |
| AX | +.04551 / +.01503 | +.09427 / +.02906 | +.07711 / +.02798 | +.04248 / +.01139 | +.13179 / −.04117 | 31 |
| AY | +.04213 / +.01597 | +.07500 / +.03174 | +.07462 / +.01893 | +.03621 / +.01660 | +.02998 / −.50056 | 36 |

The independent prioritization is AX, AV, AW for an ordinary generation comparison, not adoption. AX has positive old-science LL and margin in all 12 folds, plus complete-tool/referent/fact mean LL/margin +.05833/+.01951. AV keeps positive old-science LL in all folds and positive plain-code LL/margin, but two science margins and five repair margins are negative (repair worst LL −.05255, margin −.07185). AW provides a complementary repair/core tradeoff and the smallest raw norm; its old-science margin is weaker (8/12 negative), so it is not a science winner. Science8 mean KL is .00139/.00089/.00294 for AX/AV/AW; maximum science6 fold KL is .00996/.01105/.04235.

AT/AU/AY's plain-code margin losses are substantial relative to their LL changes. Targets and losses remain disclosed; not meeting every local bound is not converted into a new universal rejection gate. Helper training chains were already solved, so new-stage proxy gains are retention/dependency evidence rather than demonstrated repair of a training failure. Ordinary native-thinking scores remain necessary to assess transfer; no benchmark accuracy gain is claimed here.

Independent aggregate receipt: `4654c4f6…`; selected-word/tail receipt: `19dabb3a5b87d96e2a74c839f6ce67d6979b17eeb3fd0ad4a90b6d36eaf1c9b7`. Current incumbent remains AD.
