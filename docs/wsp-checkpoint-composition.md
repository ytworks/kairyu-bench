# Compose verified checkpoint tensors

`scripts/experiments/compose-verified-checkpoints.py` combines **disjoint complete
BF16 tensors** from already verified native schema-1 and dense LoRA schema-2
checkpoints sharing one original. It copies stored bytes, without loading model
code, factor GEMM, scalar edits, or weight rounding. It does not combine
overlapping coordinates, partial-tensor matrix patches, or different layouts.

```sh
PYTHONPATH=src python scripts/experiments/compose-verified-checkpoints.py \
  --plan composition.json --plan-sha256 SHA256
# Default preview: metadata/header reads only; no payload hash/candidate write.
# Explicit --apply performs the full CPU copy and verification.
```

Plan shape (paths are explicit; SHA values refer to complete JSON bytes):

```json
{
  "schema_version": 1,
  "kind": "verified_tensor_byte_composition_v1",
  "id": "native-plus-donor",
  "source": {"path": "/checkpoints/original", "tree_sha256": "ORIGINAL_TREE_SHA"},
  "components": [
    {
      "path": "/checkpoints/donor",
      "manifest": {"path": "/proofs/donor-manifest.json", "sha256": "MANIFEST_SHA"},
      "freeze": {"path": "/proofs/donor-freeze.json", "sha256": "FREEZE_SHA"},
      "input_verification": "frozen_stat"
    },
    {
      "path": "/checkpoints/native",
      "manifest": {"path": "/proofs/native-manifest.json", "sha256": "MANIFEST_SHA"},
      "input_verification": "fresh_hash"
    }
  ],
  "candidate": "/checkpoints/new-composition",
  "manifest": "/proofs/composition-manifest.json",
  "freeze": "/proofs/composition-freeze.json",
  "reserve_bytes": 0
}
```

Optional `attestations` is a list of `{path, sha256}` JSON pins for caller review
receipts; they are hash-bound but their substantive conclusions are not inferred.
All output parents must exist. All outputs must be new nonsymlink paths, separate
from input checkpoints and pinned JSON. Components are ordered: the first supplies
the full private copy; only later components' declared full tensors are replaced.
The original and all components must have identical complete file inventories,
tensor names, headers/layout, sizes, shapes and dtypes. Edit tensor overlap is
rejected, including nominally zero edits, rather than silently choosing precedence.

The original is always freshly full-file hashed during apply. `fresh_hash`
(default) also hashes every component file against its pinned manifest. An old
native manifest without a freeze therefore does not acquire a false historical
stat claim. `frozen_stat` requires a pinned recognized verification freeze,
exact whole-inventory stats, matching full-file hashes/tree and manifest SHA.
It reuses prior hashes with stat continuity; it is not a new independent full
component hash. The first component is nevertheless fully hashed while copying.
Input inventories, metadata bytes and implementation hashes are checked again
after the operation. The exclusion of `.cache` and
`.kairyu-model-attestation.json` matches existing checkpoint/deployment proofs.

Verification streams at most 8 MiB per block. It checks every chosen tensor's
original and resulting SHA, freshly hashes every output file, and virtually
replaces the union of edited spans with original bytes. That restored full-file
hash must equal the pinned original, covering all untouched weights, header bytes
and nonweight files. Additional hash states retain only each component's edits
and restore all other edits: each full-file hash must equal that component. For
two components this verifies all→original, native-reverted→donor, and
donor-reverted→native. They consume the same candidate/original blocks, with
no additional payload I/O or GEMM. Every output file has a private owned inode and link count
one. Inputs are opened read-only. Changed counts and L2 are inherited from the
verified component manifests, bound to those exact tensor hashes, and combined
only because tensor sets and the original are identical/disjoint. L2 uses the
hypotenuse of component actual stored-change norms, not a predicted LoRA cost;
the composer does not independently recompute numerical norms from every word.

Publication is exclusive. Before publishing its own verified manifest, failure
cleans only files whose inodes this invocation created. A foreign manifest is
never overwritten. Once its own manifest is published, the complete candidate
is retained even if freeze publication later fails, so evidence is not destroyed.
Do not blindly retry creation after a partial publication; inspect the retained
manifest and resolve the output conflict first.

The freeze contains the existing redeployment fields
`checkpoint_path`, `candidate_inventory_stat`, `all_checkpoint_files`,
`control_tree_sha256` and manifest/input hashes. Its distinct status is
`verified_tensor_byte_splice`: this is a stored-byte composition proof,
**not** a RAM-logit comparison, regenerated dense recipe, or capability result.
The existing redeployment identity helper consumes it without model/scorer
changes. A caller must still control deployment and evaluation separately.

Focused CPU fixtures use handcrafted tiny safetensors and cover exact disjoint
composition (including negative zero), existing redeployment compatibility,
fresh output/symlink and provenance failures, corrupt inputs, undeclared output
changes, partial-write cleanup, and exclusive publication ownership. They do not
create a real checkpoint or make model/GPU calls.
