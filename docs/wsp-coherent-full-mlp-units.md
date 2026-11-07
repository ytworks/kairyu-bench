# Coherent full MLP unit observations

This experiment separates support selection from editing. Native parameters and
explicit V tensors remain frozen; tied K/V layers can still change functional
value computation when K is edited. Unit salience is not a semantic circuit
label or evidence of generated-code improvement.

`up_unit_gains.py` observes the raw derivative of each UP output multiplied by
`(1 + gain)` at gain zero. It aliases the UP activation already used by the gated
MLP, contracts it with its backward gradient, and saves full per-unit gain
arrays plus source UP-row L2 costs. It does not compute native weight gradients
or retain a second full set of down-projection inputs.

`select_mlp_unit_support.py --plan PLAN --plan-sha256 SHA --output NEW_DIR --run`
selects 32 distinct units, at most two per layer. The prospective plan names the
science/helper, code/workflow and tool/reference groups and their 12/12/8
quotas. For selection only, each objective/fold gain vector is divided by its
source UP-row L2 cost and normalized to unit L2; group scores are RMS across
their declared rows. Zero vectors remain zero. Groups are filled in declared
order, skipping selected units and full layers; ties use layer then unit index.
No quota silently shrinks. Signed group means and selected gate/up-row/down-column
norms and gate/up cosines are reported. Selection is not a solver call.

The reference coordinate candidate is consumed as completed, pinned metadata
and attention-sample evidence. Its old live collector source paths are not
revalidated as executable inputs. No old gradients are reused for the new
full-unit candidate. New unit sensitivities must have the exact same checkpoint,
cases, frame and numerical policy, with explicit baseline tolerance.

`matrix_full_units.py --unit-support support.json --unit-support-sha256 SHA`
uses the usual collector arguments and explicit `--run`. It observes complete
gate/up rows and corresponding down columns for the fixed support. Q/K/V/O
diagnostic sample coordinates are checked against the unchanged reference
attention sample. Candidate eligibility keeps V frozen. Serial complete answers
use an input-embedding VJP and frozen weights; products accumulate in FP64 from
FP32 products in bounded 64-token tiles. Full gate/up input captures move to
CPU without changing their dtype and return once for their projection VJP; each
capture is released immediately afterward. Attention/down captures remain
device-local. Storage and release behavior are recorded in provenance/runtime.
No dense parameter gradient is saved.

The schema-2 adapter accepts `sampling_mode: mlp-full-units` only with explicit
matrix-only observations. It regenerates the full MLP coordinate layout from
the pinned support, checks attention continuity, and passes the same selected
columns through analytic radius, quantized proposal, numeric postcheck and the
explicit full-unit editor declaration. A realized sparse BF16 edit may omit
unchanged cells; declaring a unit does not force every cell to change.

Legacy 512-cell sampling and its editor bounds remain unchanged. Full-unit mode
uses a separate opt-in declaration and bounded export path. Its derivative and
support APIs have CPU fixtures; production memory use and functional response
must be measured. Longer authored contexts and teacher-forced likelihood gains
alone do not establish broader coding ability.
