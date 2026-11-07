# Coherent MLP unit edit envelope

The native sparse editor accepts a bounded opt-in envelope for a selected set of
MLP hidden units. It preserves the ordinary 512-coordinate-per-tensor and
262,144-total matrix-batch limits when this envelope is absent.

```json
{
  "matrix_batch": true,
  "matrix_sample_plan_sha256": "<64 lowercase hex characters>",
  "matrix_full_units": {
    "schema_version": 1,
    "support_plan_sha256": "<64 lowercase hex characters>",
    "layer_prefix": "model.language_model.layers.",
    "units": [{"layer": 12, "indices": [41, 908]}]
  },
  "edits": []
}
```

The example only illustrates metadata; an executable candidate needs nonempty,
exact changed BF16 coordinates and values. The support plan is selected and hashed
before full-coordinate observation and fitting.

This mode permits up to 32 declared hidden units, at most two in one layer, and at
most 1,000,000 changed matrix coordinates. Gate and up edits must be in the declared
rows; down edits must be in the matching columns. Gate/up shapes must match and down
must have their transposed dimensions. Other allowed matrices are sampled Q/K/O
projections in the same layer namespace, still limited to 512 coordinates each.
Native scalars/vectors and explicit V projections are frozen. On Gemma full-attention
layers, changed K weights can also change the functional V path through tied loading;
explicit V exclusion does not claim functional V is frozen.

The emitted edit may contain only a subset of the observed rows/columns because
unchanged BF16 cells are omitted. The declaration is a support constraint, not a claim
that every declared cell changed or that a selected unit implements a named ability.

`joint.editor_compatibility(..., matrix_full_units=declaration)`, native checkpoint
creation/verification, and finite RAM replay enforce the envelope. The ordinary
batch opt-in is still required. The proposer additionally verifies support-plan and
sample-plan identity; a digest in the editor metadata alone does not establish how
the support was selected.

Full-unit metadata may exceed the ordinary 256 MiB JSON acceptance limit. Only a
valid explicit full-unit specification, or a manifest containing that specification,
can exceed it; the reader has a separate 2 GiB hard ceiling. Actual memory requirements
depend on the decoded manifest, not just file size.

Manifest coordinate validation now builds one coordinate-to-position index per
tensor. Later patches must retain the same dtype and shape. This removes repeated
full coordinate-list scans and linear searches inside the per-cell loop. File hashing,
patch verification, source preservation, and complete declared-edit coverage remain
required. No performance speedup or GPU-memory fit is inferred from the code change.

The first planned experiment compares the fixed coordinate control from accepted
AV with a coherent-unit candidate using the same objectives and sampled Q/K/O
coordinates. Unit salience is observed separately from final full-row/column
derivatives. Both finite response and generated/official outcomes remain necessary
to assess whether the edit helps.
