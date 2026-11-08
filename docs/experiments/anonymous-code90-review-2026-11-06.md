# Code90: anonymous Opus 5.5 method review

This is a review of eight public-method hypotheses, not a review of measured candidate coordinates or private benchmark answers. The dispatch verified claude-opus-5-5, no tool use, and no raw reasoning persistence. Duration: 102.42 seconds. The requested final response follows; its recommendations are evidence to consider, not automatic adoption rules.

## A

Promising target, but under-specified. Numerical invariants, array contracts and cross-stage state are three different failure modes, and they may sit in different places in the model. Editing MLP transformations and attention readout directions together means a gain or loss cannot be attributed to either one. Independently authored scientific tasks are the right source, but they become a synthetic family if one author, template or library style dominates. Executable grading needs fixed numerical tolerances and fixed array-shape and dtype contracts that are set before any edit, so a loosened checker cannot create apparent gains. Expect transfer only where the base model's failures actually come from these three modes; that has not been established yet.

**Cheapest falsification:** Before making any edit, run the base model several times per task on the development scientific tasks and label each failure by stage: invariant violation, shape or contract error, lost cross-stage state, interface or API misuse, wrong algorithm, or harness/timeout. If the three targeted modes are a small share of failures, A's ceiling is small and it should be deprioritized. If they dominate, realize two or three BF16 proposals from the original base, each split into MLP-only, attention-only and joint variants. Compare each against the base on held-out task families using paired per-task, multi-sample comparisons. A is falsified if no variant improves execution success beyond sampling variance while staying within the regression floors.

**Change of mechanism:** Switch mechanisms if: (1) the attention-only variant is neutral or harmful while MLP-only carries the gain, so drop the joint readout component; (2) cross-stage state errors grow with context or stage distance, which points to E and long-context state; (3) residual failures are mostly misused identifiers or signatures, which points to B; (4) margins and NLL improve but execution success does not, which suggests redistribution is the wrong tool and points to H or a different locus. A smaller or larger version of the same edit is not a mechanism change.

## B

Coupling regression preservation to the improvement objective is correct; a post-hoc average penalty hides concentrated breakage. Two risks. First, strengthening exact identifier and signature reference can overcorrect into copying stale names or refusing to introduce new ones, so tests should include cases where renaming or adapting is correct. Second, 'unaffected behavior' has to be a broad protection set frozen before development, scored per capability with explicit floors, not as a single mean. B has two separable roles. As a constraint it is always needed. As an improvement target it is only justified if interface errors are a real failure share.

**Cheapest falsification:** As an improvement target: if the base failure taxonomy shows identifier, signature and contract mismatches are rare among executable failures, drop B as a target and keep it only as a constraint. As a constraint method: if adding B's coupled term does not reduce measured per-capability regressions compared with the same proposal without it, over realized BF16 variants and paired protection-set scoring, the coupling mechanism adds nothing beyond plain regression floors.

**Change of mechanism:** Switch from joint strengthening to constraint or projection if regressions concentrate in specific capabilities tied to specific groups. Options: exclude those groups, or remove the component of the edit that measurably moves protected outputs. Switch to learned preservation, such as replay or KL in H, if regressions are diffuse across many capabilities rather than localized. Switch away from strengthening references if new copying errors appear, such as reusing stale identifiers.

## C

Metamorphic tests and counterexamples are a strong way to separate look-alike relations such as off-by-one, strict versus non-strict comparison, or stable versus unstable ordering. They are also cheap to verify by execution. The main risk is overfitting to the generator: counterexamples from one generator share its distribution, and directions derived from them can encode generator artifacts. 'Unseen problem families' must be defined before derivation, with a family split rather than an instance split. Directions derived jointly across matrices and native scalars are hard to attribute, so realize the matrix-only and native-only parts separately as well.

**Cheapest falsification:** Derive directions from a small number of families and evaluate the realized BF16 proposal on one or two held-out families plus the protection set. Gains on source families with no held-out gain beyond variance indicate synthetic-family overfitting and falsify the transfer claim. A shift in teacher-forced counterexample margins with no change in sampled program pass rate on held-out families indicates teacher-forcing mismatch rather than improved behavior.

**Change of mechanism:** Switch if: (1) native scalar or normalization components alone reproduce the gain, so prefer the simpler native mechanism; (2) gains appear only in sampled generations that reason through more test cases, suggesting a behavioral or workflow mechanism (F or H) rather than relation representation; (3) held-out failures are mostly interface errors (B) or lost state (E). Shrinking or growing the same direction does not count.

## D

Judging simulator end states rather than tool-token probability is the correct choice. The four sub-targets differ, though. Call versus no-call is a threshold or routing decision. Function selection and argument types are format and schema binding. Optional values require reading context. State transfer across calls is a long-context memory problem. A single joint direction risks trading one against another, for example raising call rate while over-calling. The simulator's schemas, tool names and argument conventions must be disjoint from any evaluation collection already viewed. Report over-calling and under-calling separately, never as a single tool-use score.

**Cheapest falsification:** Label base simulator failures by sub-target. Then compare one realized proposal against the base on a held-out schema family, using end-state success split into call/no-call precision and recall, selection, argument validity and multi-call state. D is falsified if end-state success does not improve beyond variance, or if a gain in one sub-target is offset by a loss in another, such as call recall rising while no-call precision falls. A tool-token logprob gain with no end-state gain counts as teacher-forcing mismatch, not a success.

**Change of mechanism:** Switch if: (1) errors are mostly over-calling or under-calling, where a native scalar or threshold-like late edit may suffice and a matrix direction is unnecessary; (2) multi-call state loss dominates, so solve jointly with E; (3) argument errors come from misreading the schema, so couple with B; (4) gains appear only on schemas resembling the development set, which indicates synthetic overfitting and points to H trained across diverse schemas.

## E

A pinned public donor is reasonable as a direction source, but four checks come first. Confirm the donor was trained on the exact same base weights, not merely the same model family; otherwise the delta is not a meaningful direction. Check the donor's license. Note that donor deltas often carry chat-template, formatting or style changes unrelated to long-context state. Bounded per-group ablations used for selection are noisy, so taking the best groups invites a winner's curse, and selected groups need a confirmation run separate from the one used to select them. Low-rank factors merged into BF16 can lose small components, so measure the realized delta, not the adapter product.

**Cheapest falsification:** First verify base identity, for example by checksum comparison of the shared base tensors; a mismatch ends E. Then realize the whole donor at two or three bounded scales from the original base. Evaluate on development long-context state tasks such as multi-file reference tracking and multi-turn state, plus the protection set. If no scale improves the target beyond variance without breaching regression floors, the donor contains no usable direction for this objective and group selection is unlikely to rescue it. Only if the whole donor shows signal is group-level ablation worth its cost.

**Change of mechanism:** Switch if: (1) the gain survives only when nearly all groups are included, so the effect is not selectable and should be treated as whole-donor or replaced by H; (2) the gain comes mainly from normalization or scalar groups, so use native scalar edits; (3) the gain is mostly format or template adherence, which belongs to B or F rather than long-context state; (4) per-group ablation rankings flip between independent seeds or subsets, so selection is noise-driven and should not proceed.

## F

Judging actual repository and artifact state is correct; verbal completion claims are unreliable. The assumption that the mechanism is late routing or readout is untested. The failure could instead be early stopping or end-of-sequence tendency, budget truncation, harness timeouts, never running the tests, or running tests and misreading the result. Each implies a different locus. Small independent repositories risk a shared template, so vary build systems, test runners and submission conventions. Separate 'correct code produced but not verified or submitted' from 'code incorrect', because F can only fix the first.

**Cheapest falsification:** On the base model, measure the share of runs where a correct solution existed in the working tree at some point but the final state was unverified, unsubmitted or overwritten. If that share is small, F's ceiling is small. If it is large, a realized late-layer proposal that fails to raise verified-and-submitted end-state rates on held-out repository styles, beyond variance, falsifies the late-routing mechanism. As a diagnostic only, never a deliverable, an instruction-only counterfactual estimates how much of the gap is behavioral rather than a capability gap.

**Change of mechanism:** Switch if: (1) failures track length or turn count, suggesting end-of-sequence or stopping behavior and a native scalar or readout edit rather than routing; (2) the model claims verification without running tests, a policy-level behavior where H with execution-grounded trajectories is better suited; (3) failures come from losing track of files or state across steps, which points to E; (4) a harness timeout or budget limit dominates, which is not a weight problem and must be excluded before attributing anything to weights.

## G

Necessary measurement infrastructure rather than an improvement hypothesis, and it should be built before other edits. Costs must be computed on the realized BF16 delta, meaning the stored rounded new weight minus the stored base weight, never on the intended continuous delta. A pre-normalization RMS change does not equal the perturbation after normalization: the effect depends on whether the edited tensor writes into the residual stream before or after a norm, on the norm's gain parameterization, and on the residual norm at that layer. A diagonal energy approximation is a heuristic, not Fisher information, and must not be described as such. No cost metric is known in advance to predict damage; that has to be measured.

**Cheapest falsification:** Over a modest set of realized proposals of varied type and size, compute absolute norm, role-relative norm and measured branch output perturbation at the actual action point on fixed bounded inputs. Rank-correlate each with measured protection-set regression from paired generation and execution, not NLL. If no metric beats absolute norm, or none correlates beyond chance, drop functional cost as a selection or budget criterion and rely on direct regression measurement.

**Change of mechanism:** Switch budgeting if: (1) measured branch perturbation predicts regression but norms do not, so budget edits by functional cost instead of norm; (2) predictive power differs by edit type, so use type-specific budgets; (3) many proposals show near-zero realized delta after BF16 rounding, so the proposed mechanism is not being realized, and either restructure the edit or accept that a different mechanism is under test.

## H

An appropriate fallback with the right guards. Remaining risks: (1) passing execution checks does not mean good programs, since trivial, overfit or test-specific code can pass, so checks need adversarial or metamorphic tests; (2) generated training programs share a generator style, so family separation must also cover generator and template, not only topic; (3) preserving prior behavior needs an explicit term such as KL to the base on broad prompts or replay, plus measurement; (4) a fp32 adapter merged into BF16 can lose components, so the inspected object is the merged BF16 delta; (5) the learned direction may mainly encode style or verbosity. Mark its gains as provisional until the fresh reserve is run.

**Cheapest falsification:** Run a small-rank pilot on a few training families. Evaluate the merged BF16 model, not the adapter, on held-out families, the protection set and a target-specific execution set, against both the base and the best redistribution candidate. Reject H if held-out execution gains do not exceed the redistribution candidate beyond variance, if gains vanish after merging, or if regressions breach floors.

**Change of mechanism:** Switch if: (1) the merged BF16 delta differs materially in effect from the unmerged adapter, so restructure scaling, keep an approach that survives rounding, or treat the merged version as the actual mechanism; (2) the learned delta concentrates in a few layers or roles, so convert it to a structured edit and re-test; (3) gains appear only on training families, which is synthetic overfitting, so diversify generators rather than raise rank or steps; (4) gains track output length or style rather than correctness.

## Suggested sequence

- 1. Runtime verification before any edit. Check: the normalization type and whether gain is stored as w or as (1+w); pre-norm, post-norm or sandwich placement; tied or untied embeddings; logit or attention softcapping; attention scaling and grouped-query head layout; fused QKV or gate/up layouts and their ordering; RoPE convention; the low-rank scaling convention (alpha/r) and merge order; accumulation dtype in matmuls. Confirm that realizing a proposal from the original base is deterministic and reproducible, and that an identity edit reproduces base outputs exactly.
- 2. Freeze the protection set, per-capability regression floors, decision rules, sampling settings and number of samples per task. Treat any evaluation collection already viewed as a known reference. Define the fresh reserve's composition now and do not look at it.
- 3. Measure base variance and the base failure taxonomy across scientific, algorithmic, tool, workflow and long-context tasks. This decides which hypotheses have a real ceiling.
- 4. Build G's cost accounting on realized BF16 deltas, and run B as a standing constraint on every proposal.
- 5. Solve A and C together, coupled to B. Both target executable code correctness and share the protection constraint; realize matrix-only, native-only and joint variants separately.
- 6. Solve D. Pair it with E only if the taxonomy shows multi-call or long-context state loss is a major share; run E's base-identity check and whole-donor test before any group selection.
- 7. Solve F after the code-correctness edits, because verification-and-submission gains are only meaningful once correct code is produced more often. Exclude harness and budget failures first.
- 8. Run H only if redistribution (steps 5 to 7) leaves a meaningful gap on held-out families. Compare it against the best redistribution candidate, not only the base.
- 9. Accumulate greedily. Realize each combination once from the original base, not by stacking stored deltas, and count shared groups once. Accept a combination only if paired multi-sample results improve the target over both the original and the parent, and every protection capability stays above its floor. Record losses per capability, not as an average.
- 10. Prune with leave-one-group-out ablations on the realized final combination, each re-realized from the base. Remove groups whose removal does not reduce target performance beyond variance, or whose removal reduces regressions. Re-realize the pruned set and re-verify against original and parent. Activation correlations may suggest which groups to try removing; only realized ablations decide.
- 11. Run the fresh reserve once, under the pre-registered rule, without further tuning. Report target gains, per-capability regressions and failed hypotheses.

## Limits

- Teacher-forcing mismatch: margin or NLL gains on reference continuations do not show that sampled programs pass. Sampled generation plus execution is the deciding evidence.
- Synthetic-family overfitting: gains on generator-produced or development-style families must replicate on held-out families split by family, author or generator style, and ultimately on the fresh reserve.
- BF16 mechanism loss: proposals are evaluated as stored after rounding. If the realized delta differs materially from the intended one, the hypothesis actually tested is the realized delta, and conclusions apply only to it.
- Actual generation or execution failures must be separated from harness artifacts such as timeouts, sandbox errors, budget truncation and nondeterministic tests before they are attributed to weights.
- Static weight and activation summaries are observational. They can prioritize candidates but cannot establish causal localization; only realized interventions with ablations support causal claims, and only for the tested inputs.
- Selection on noisy development measurements inflates apparent gains (winner's curse). Selected candidates need confirmation on data not used to select them.
- Paired per-task, multi-sample comparisons with variance estimates are required; one sample per task cannot separate small gains from noise.
- Expected improvement is not evidence. No numeric accuracy target is guaranteed reachable, and a hypothesis can fail even when its mechanism appears plausible.
- Average scores can hide serious damage to a single capability. Regressions must be reported per capability against frozen floors.
- Runtime conventions (normalization parameterization and placement, softcapping, fused layouts, adapter scaling, accumulation dtype) must be verified in the actual runtime, not assumed from documentation or other model families.

## Confirmation

- Build a new frozen confirmation collection after development, with task families, authors and generator styles disjoint from all development and already-viewed collections. Do not inspect it before the single confirmation run.
- Cover every claimed target: scientific multistage, algorithmic boundary cases, tool end states, repository verification and submission, and long-context state if E is adopted. Include a broad protection set covering capabilities outside the coding objective.
- Pre-register the comparison (final candidate versus original base, plus the pruned parent if relevant), sample counts, sampling settings, statistical decision rule and per-capability regression floors.
- Use a single confirmation run with no tuning, no candidate swapping and no threshold changes based on its outcomes. If it fails, report the failure; any new development needs another, newer reserve.
- Report confirmation gains and regressions separately from development results, including hypotheses that were tested and rejected.

The owner prioritizes scientific and algorithmic execution with regression preservation. The known64 reference and prospective108 confirmation are explicitly historically exposed; they are not relabeled unseen. Actual runtime behavior and BF16 realization take precedence over assumptions in the review. [Current plan](2026-11-06-code90-plan.md).
