# Fresh audit of three AV code failures, 2026-11-07

Run `gemma4-ad-av-av-c32-dev10-full14-20261107-v1` records three failed code items:
`livecodebench:arc191_c`, `mbpp-plus:Mbpp/410` and `mbpp-plus:Mbpp/576`.
This diagnosis uses the final public generated code, public problem statements,
stored official verdicts, and the pinned official scoring/reference contracts.
It does not change scores, remove problems, or supply evaluation questions,
reference answers or tests to calibration/weight editing. Generated code was not
executed on the host, and no new benchmark run was performed for this audit.

## Evidence and classification

| Item | Stored official verdict | Independently established mismatch | Interpretation |
| --- | --- | --- | --- |
| arc191_c | `passed: false` | Problem allows any valid construction; pinned LCB compares output values against one reference construction | Official failure cannot be equated with an invalid mathematical construction |
| Mbpp/410 | base pass, plus fail | Reference filters integers; generated implementation includes both integers and floats | Type-domain contract mismatch, not missing helper code |
| Mbpp/576 | base pass, plus fail | Reference checks an order-preserving subsequence; generated implementation checks contiguous slices | Relation-definition mismatch, not missing helper code |

The two MBPP public prompts do not disambiguate the relevant contract. This
supports a diagnosis of interpretation sensitivity; it does not authorize changing
EvalPlus's reference contract or upgrading the official score.

## arc191_c: arbitrary-valid-output task under exact-output scoring

The pinned public task asks for positive integers A and M at most 10^18 such that
the multiplicative order condition has minimum positive exponent N. It explicitly
accepts any valid solution, with N at most 10^9 and up to 10,000 cases.

The final public code handles N=1 by outputting `2 1`. This pair is valid: every
integer is divisible by 1, so the minimum positive exponent is 1. The public
sample instead contains `20250126 1` for that case. These are different valid
constructions.

The pinned official LCB source is
[`testing_util.py`, revision 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24](https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/evaluation/testing_util.py#L397).
The retrieved file SHA-256 is
`b7cb6a8a69807bb868150a61742e25d7bb5328bbe01d514471b6ec43c9fa9ed2`.
Lines 397–422 accept exact stripped-line equality or exact Decimal-list equality;
otherwise they return wrong answer. There is no task-specific construction
validator in this path. Thus this known-valid public output is rejected by the
pinned comparison. This conclusion does not require executing model code.

The final code otherwise searches for a prime modulus of the form kN+1 and
searches for an element whose order is N using modular exponentiation and checks
at N/q for every prime factor q of N. Those checks are structurally relevant to
the stated mathematical condition. However, the unbounded prime/element searches
have not been timed here, nor has the entire output been independently validated
across large cases. Do not claim the whole implementation is correct or fast.

The repository wrapper
[`livecodebench_score.py`](../../src/kairyu_bench/wsp/scorers/livecodebench_score.py)
retains only a `passed` boolean from `codegen_metrics`; it discards the underlying
failure metadata. Therefore the saved result does not identify the first failed
case or distinguish an additional timeout from a wrong-reference-output failure.
The official zero is preserved. Any later independent validity audit is a
separate diagnostic metric and must not replace the official source average.

## Mbpp/410: integer versus numeric domain

The public prompt requests the minimum in a heterogeneous list and provides one
mixed string/integer example. The pinned MBPP+ v0.2.0 canonical function includes
only `isinstance(value, int)`. The generated function includes
`isinstance(value, (int, float))` and independently chooses `None` when no numeric
values remain. Its main function is complete and requires no helper.

The stored verdict is base pass and plus fail. The pinned dataset has three base
inputs and 104 plus inputs for this task; eight plus inputs contain float values.
The type policy differs demonstrably. The stored verdict does not preserve the
first failing input, so this audit does not assert that a specific float or the
empty-result branch caused the recorded failure. The public specification is
underspecified relative to the integer-only reference policy.

## Mbpp/576: subsequence versus contiguous sublist

The public prompt asks whether one list is a sublist of another and gives a
negative example. The pinned canonical implementation advances through the main
list while matching the next requested element, thereby accepting noncontiguous
order-preserving subsequences. The generated implementation checks equal
contiguous slices. Both handle an empty requested list as true, so the empty-list
branch itself does not distinguish the two implementations.

The stored verdict is base pass and plus fail. The pinned task has three base and
110 plus inputs. No exact failing case is retained in the WSP verdict. The
implementation's missing behavior is noncontiguous matching under the reference
contract; the public wording alone does not explain that interpretation. There
is no observed syntax, missing helper or incomplete-function failure in the
final public code.

## Consequences for the new editing cycle

Keep all three official outcomes and the full source composition in the WSP
score. Avoid directing edits at a particular reference construction or memorized
contract. The evidence motivates studying general specification interpretation,
type/relational distinctions and algorithmic construction as separate phenomena.
It does not identify any tensor or activation channel as their causal owner.
A claim that one small weight edit repairs these mechanisms requires independent
runtime observations and new full-manifest evaluation, not this diagnosis alone.
