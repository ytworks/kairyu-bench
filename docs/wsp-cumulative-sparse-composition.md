# Carry a complete sparse edit onto a newer parent

`scripts/experiments/gemma4-purpose/rebase_sparse_composition.py` prepares a
metadata-only sparse specification for a newer checkpoint. It retains the
current parent's unrelated increments. It does not merge floating-point deltas,
write a checkpoint, call a model, or establish that the edits work well together.

Every selected source BF16 word must equal the corresponding current-parent
word, including the sign of zero. Any conflict rejects the complete earlier edit;
the tool never drops conflicting coordinates or retains only part of that edit.
The resulting specification retains its original sparse coordinates and target
values, including any sampled-matrix plan binding.

## Inputs and proof

Supply the earlier sparse specification and both checkpoint freezes, each with
its SHA-256. The earlier specification must identify the frozen earlier source.
Each freeze must contain `checkpoint_path`, `all_checkpoint_files`,
`candidate_inventory_stat`, and `control_tree_sha256`.

The tool checks the complete file inventory against the frozen sizes and stat
identities, then reads only selected coordinates. It inherits the freezes'
previous content hashes through exact stat continuity; it does not claim a
fresh full-payload hash. The caller must supply trusted content-verified freezes.
The evidence is rechecked before publication.

Only explicit sparse `set_values` edits on equal-shaped BF16 tensors are
supported. Declared numeric values must survive conversion to FP32 exactly and
be finite exact BF16 values. Integer bit expansion and contraction preserve
signed zero. Matrix coordinates retain the existing 512-cell per-tensor bound.
The existing editor remains responsible for the complete materialization
contract, including aggregate sparse limits and saved-checkpoint verification.

For current selected values `w` and carried targets `t`, the reported costs are:

- Raw weight distance: `sqrt(sum((t-w)^2))`.
- Relative distance: `sqrt(sum(((t-w)/max(abs(w), floor))^2))`.

The floor and cap must be finite and positive. A no-change result or relative
distance above the supplied cap is rejected. There is no clipping, rescaling,
new optimization, or per-coordinate salvage.

## CLI

Run from the repository's Python environment:

```sh
python scripts/experiments/gemma4-purpose/rebase_sparse_composition.py \
  --spec /absolute/earlier-spec.json --spec-sha EARLIER_SPEC_SHA256 \
  --source-freeze /absolute/earlier-freeze.json --source-freeze-sha SOURCE_FREEZE_SHA256 \
  --parent-freeze /absolute/current-freeze.json --parent-freeze-sha PARENT_FREEZE_SHA256 \
  --candidate-id cumulative-candidate \
  --relative-floor 1e-6 --weighted-cap 1 \
  --output /absolute/fresh-metadata-directory
```

The default prints a compact diagnostic and writes nothing. Add `--write-spec`
to publish `spec.json` and `report.json` in a fresh directory. An existing
destination, destination symlink, or overlap with either input checkpoint is
rejected. These two files are preparation evidence, not a deployment freeze or
an atomic checkpoint publication. Use the existing editor's create/verify path
and separately measure actual model response before selecting a saved result.

## Recorded AD + AE preparation

The first application carried the complete earlier AE sparse matrix set from
its original AC source onto accepted AD, retaining AD's unrelated native
increment. The selected source words matched exactly. The resulting
`analysis/code90-resume-20261106/current-ad-ae-composition-v1/spec.json`
has SHA-256
`b18b102793e5daba96d465ebfa506968492d789f67e7ac109cc6ee879153e495`.
Its report (`7b92a08e76903db47fd19396006a16a4382a155e35880ab7f3d632e8c8c7e3fb`)
records 7,856 changed values across 231 tensors, raw L2
0.017724598923659067 and relative L2 0.6667720154428621 at floor
`1e-6` and cap 1. These are the incremental carried edits relative to AD, not
the cumulative distance from the original model. Functional interaction was
unmeasured by this preparation.

The executed source was SHA-256
`0f6a5785e6282281f5c4564b0fecad2d30191a4b029fecc3aa5825c4c50b5f9d`,
preserved as
`runtime/code90-resume-20261106/rebase_sparse_composition-executed-v1.py`.
The published source
`18ce1814441d9ec202aeb783438f91bf23eb32177b3b869c400c353f655da7a7`
adds rejection of input-overlapping output directories and declared values
that would silently round during FP32 conversion. The original spec/report
remain unchanged; `execution-source.json` records this distinction.

One additional E2E method in `tests/test_compose_verified_checkpoints.py` uses
tiny checkpoints to check the complete carry through editor create/verify,
unrelated-parent preservation, exact signed-zero bytes, whole-edit rejection
on a source-word conflict, the cost cap, nonexact values, and output isolation.
The composition fixture count increased from six to seven. No GPU is required.

The combined CPU suite after this change ran 515 tests: 462 passed and 53
optional-dependency cases skipped. Ruff, mypy, data and shell checks passed;
source hashes were unchanged throughout the checks. Evidence is
`analysis/code90-resume-20261106/cpu-integration-native-rebase-v1/ci-results.json`
(SHA-256 `39967803082faaacfca730af9c0323aa423970aeb043fea5dd1d761e8a31f29d`).
