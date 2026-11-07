# Matrix-only objective observations

A schema-2 coupled-objective plan may explicitly set
`observed_blocks: ["matrix"]` together with `eligible_blocks: ["matrix"]`.
Its `collections` must contain only the matrix `correct-logprob` and `margin`
directories. `native_tensors` is forbidden. Omission preserves the historical
native-plus-matrix observation contract; schema 1 is unchanged.

Both matrix collections must be complete on the exact same checkpoint, cases,
frame, sampling layout, and numerical observation policy. No old eager
collection can fill a missing SDPA result. Failed partial collections remain
failed evidence. Native checkpoint tensors stay bound by the full checkpoint
identity and are never solver variables; no native gradient arrays are created
or read.

The science/code target builder uses matrix correct-LL baselines for headroom.
The existing analytic-radius and quantized solvers operate on the same selected
matrix columns. Reports expose zero observed native columns, `frozen_not_observed`
native parameters, and `not_measured` cross-family baseline parity with null
differences. This means no native/matrix baseline comparison was made, not that
the two collectors agreed.

The numeric postcheck reconstructs actual BF16 changes from the recorded matrix
sample words, enforces the declared parent, scope and cost, and needs no native
source NPZ. Existing E2Es remove native observation directories before running
target preparation / analytic calibration / quantized proposal / postcheck and
verify that physical saved native bytes remain unchanged. Historical default
and broader prior-AA reference behavior remain covered in the same fixtures.
No optimizer, coordinate budget, attention backend, or target policy changes
are introduced by this mode.
