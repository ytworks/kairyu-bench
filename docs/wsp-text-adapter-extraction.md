# Lossless text adapter extraction

`scripts/experiments/gemma4_lora/extract_text_factors.py` exports the exact
820 factor tensors for Gemma 4's 410 text projections from a local ordinary
LoRA safetensors file. It copies raw F32 or BF16 bytes; it does not round,
scale, merge, load a model, run a benchmark, or write a full checkpoint.
Every excluded factor must be explicitly named and recognized as a matching
vision projection. Unknown tensors and incomplete A/B pairs are rejected.

```sh
PYTHONPATH=src python scripts/experiments/gemma4_lora/extract_text_factors.py \
  --plan selection.json --plan-sha256 SHA256
# Preview is the default; --apply creates only a fresh adapter directory.
```

The JSON plan uses `schema_version: 1`, `kind:
"gemma4_text_factor_selection_v1"`, `repo`, a 40-character `revision`,
`base` (local original checkpoint), and `output` (fresh directory).
`adapter` is `{path, sha256, header_sha256}`; `config` is
`{path, sha256}`. `selected` and `excluded` are explicit lists of
**every** input factor name, with no overlap or automatic omissions.
Header SHA covers the raw safetensors JSON bytes, including padding.
Preview verifies the complete adapter SHA, configuration and shapes but
writes nothing. All 410 base text targets and all declared vision pairs must
match rank, orientation, dtype and dimensions; the original weights are
read only as headers, so no full original-weight identity is claimed.

Application writes `adapter_model.safetensors`, a byte-identical
`adapter_config.json`, and `extraction.json`. The receipt distinguishes
the original repository payload from the derived text-only payload. It binds
the plan, implementation, original full/header/config hashes and file states,
selected and excluded names/shapes/extents, each selected factor's source
and derived byte SHA, output full/header SHA, and base header/index evidence.
No tensor values are printed or interpreted; numeric finiteness remains the
downstream merge validator's responsibility. Input state changes fail closed.
Outputs are fresh, with exclusive receipt publication; pre-publication
failure removes only files created by this operation. A completed published
receipt and its outputs survive a later exception.

Use the derived adapter's **new** full and header hashes with the existing
[standalone dense factor-format v2](../scripts/experiments/gemma4_lora/README.md#explicit-factor-format-v2-standalone-dense-only).
Keep the extraction receipt pinned alongside the recipe. The adapter config
retains its original declared training base; layout agreement does not prove
that base's weights match the target checkpoint, capability transfer, or
absence of evaluation overlap. A text-only export is not a claim to reproduce
the original multimodal adapter or its training runtime.
