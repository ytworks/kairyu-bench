# Helper-chain calibration and prospective purpose selection

Status: prospective optimization policy; accepted model remains AD. This note changes authored calibration selection, not official benchmark prompts, sampling, scoring, or all-area retention coverage.

## Why another small additive edit is insufficient evidence

The ordinary saved-checkpoint comparison completed all 90 items per arm:
AD 74.74206349, AN 72.43650794, AR 73.07539683 on the equal four-source code mean.
The apparently positive overlay screens did not carry through to this fresh saved-model comparison. Neither AN nor AR was adopted; their full-area follow-up was skipped. All three ordinary runs had zero unknown, truncated, empty, retry and API-error counts, with 119 formal requests each. The accepted P → S → AC → AD ancestry is preserved.

Evidence: `analysis/code90-resume-20261106/current-ad-an-ar-dev90-v1/final-summary.json`,
SHA256 `2f3a50cdf8610c42f2c753d588c190f64ba7dae71d2405f075b90170a1951b86`.
These are development results, not a new held-out claim.

## Complete-answer units and legacy continuation probes

Inspection was limited to our authored prompts/references and previously saved numeric calibration observations. No generated benchmark answer or hidden reasoning was inspected.

Some legacy tools prompts end the user turn with an incomplete command, while their reference assistant answers contain only its suffix. For example, a prompt ending in `git` scores ` commit` instead of a complete command. The measured tools-fold correct LL around -21 nat/token therefore cannot be read as a tool-skill deficit. Its large LL headroom previously produced disproportionate functional targets. Two first-letter instruction prompts also admit a transformed whole-word reading rather than only the reference letter.

Other general, reasoning, lookup and Japanese probes include continuation-style wording. Leading whitespace alone does not invalidate an answer, and this audit does not establish that every excluded row is wrong. We conservatively select complete-answer units for the next optimization instead of attributing their exact-string probabilities to the intended capabilities.

The six legacy groups `tools`, `instruction`, `general`, `reasoning`, `lookup` and `japanese` (48 rows) remain in the preserved observation cohort but are omitted explicitly from the prospective optimization view. Their observations are not rewritten or silently discarded. Complete implementation/algorithm programs, shell workflows, explicit tool-call/no-call/argument JSON, referent and factual-reference tasks remain. The existing official all-area benchmark still measures the excluded capability names; this calibration selection does not waive their outcome tradeoffs.

## Add only the missing dependency observations

Four independent training families contribute a helper stage and a composition stage, eight new rows in two separate purposes. The family stays in one fold; two families occupy each fold. Two separate evaluation chains distinguish helper correctness, downstream correctness with an authored correct helper, and downstream correctness with the model's extracted public helper. Reference programs and distinct foils are CPU-executed; a designated single foil per stage enters derivatives, so additional foil tests do not multiply the correct-answer LL weight.

AD did not change, so its previous 152 observations remain usable. Only the eight new rows were collected. The full cohort has 160 rows / 23 purposes / 46 folds. The explicit optimization subset has 112 rows / 17 purposes / 34 folds. Composition preserves original fold numerics and gradient bytes and records both segment receipts; it must never be represented as one newly executed 160-case model run.

The new-eight owner completed in 178.79 seconds including service restoration. Actual native LL/margin collection took 6.228/7.152 seconds and matrix LL/margin took 3.016/5.857 seconds. Completion receipt:
`analysis/code90-resume-20261106/current-ad-helper-chain-observation-v1/completion.json`,
SHA256 `4563af78459f4b133dd9ff9eafbbb806012e1e46295857f6836d131f6dff15d3`.

## Bound functional targets separately from displacement

The existing candidate builder now optionally accepts `ll_target_cap` for functional targets only:

```text
correct-LL target = min(ll_headroom_fraction * (-baseline_mean_LL), ll_target_cap)
```

Omitting the cap preserves the previous behavior. A zero headroom still gives a zero LL target. The independent margin delta and retention bounds are unchanged by this option. The cap bounds the requested diagnostic improvement, not the actual weight norm or the model's measured response; finite BF16 outcomes must still be measured.

The existing saved-byte end-to-end test makes the uncapped target require twice the capped displacement and verifies the resulting solve, postcheck and checkpoint bytes. The test count is unchanged. Focused E2E and Ruff passed; no production solver or collector was modified by this option.

Six prospective structural contrasts and one fixed stronger capped target policy were sent for an actual Opus 5.5 methods-only review. No private weights, cases, benchmark scores or generated outputs were sent. The actual Opus 5.5 review completed with zero tool uses in 96.094 seconds. Receipt SHA256: `89bcd104637b5cc337e0fec99c07f0efde8f302f1b44f835c9dc3eb650d77943`; disposition: `1c961fb6d4a1b90a624af75ee699456dfa6178f717db2f9119aed301e610c52d`. The fixed AT–AY policy is `prospective-helper-capped-six-v1/method-policy.json`, SHA256 `85caa68a4b93a129090cbb0de6d0d9fb4d49d353891d3e4bcd208a4db545cd8f`. It uses LL target `min(0.05 * H, 0.05)`, margin delta `+0.02`, other selected-purpose delta lower bounds `-0.01`, Euclidean cost and a realized raw-L2 exploration budget of 5. These are calibration goals and a search budget, not accuracy or locality guarantees. There is no amplitude grid or extra full-benchmark approval gate.

The structural contrasts are O+MLP, MLP-only, Q/K/O+MLP, focused chain/context purposes, explicit tool-execution support, and descriptive fold-agreement layer selection. Native values and V matrices remain frozen in this round. A positive pre-existing correct/foil margin does not satisfy a positive *delta* request; isolated Q/K weight changes do not themselves violate RoPE rotation consistency; BF16 relative ULP spacing varies approximately between 1/256 and 1/128. The review's stronger causal readings and optional extra research gates were not adopted.
