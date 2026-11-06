# Public donor metadata reconnaissance, 2026-11-06

This bounded Hugging Face API/card search found two additional inspectable
adapters, but **no verified adapter-specific coding improvement**. It did not
download weights, execute repository code, read training examples, or send any
local benchmark/calibration content. Results are leads for later analysis,
not selected models. Quantizations, abliterations and style-only models were
excluded from consideration. The search was capped and is not exhaustive.

| Repository / pinned revision | Base and adapter metadata | Dataset / evaluation evidence | Assessment |
|---|---|---|---|
| [kai-os/gemma4-31b-Opus-4.6-reasoning](https://huggingface.co/kai-os/gemma4-31b-Opus-4.6-reasoning/tree/e78fa262a61573cf076883a7cd8100f17ab4f7f5) | Declares google/gemma-4-31B-it, NF4 QLoRA; rank64, alpha128, ordinary scaling2; seven projection families with `.linear` suffix | Card names Crownelius/Opus-4.6-Reasoning-2100x-formatted; reports 2,025 retained rows (1,899 math,126 code), 1,924 train/101 validation. Eval loss3.6018/PPL36.66. Its benchmark table explicitly belongs to the original Google model, not this adapter. | Most relevant new math/reasoning lead. Exact training base revision and evaluation overlap unknown. Factor headers/shapes, suffix mapping and numerical compatibility remain unverified. |
| [Lambent/Fabled-Gemma4-31B-Adapter](https://huggingface.co/Lambent/Fabled-Gemma4-31B-Adapter/tree/13bd7092991b5a50636f4ebe2fb5190a72a53b49) | Declares google/gemma-4-31B-it, QLoRA; rank256, alpha256, scaling1; regex restricts text MLP/attention projections | Names fable-conversations-memory-gemma-8k and fable-distilled-memory-gemma-8k. Public training config is present. Model-index results and training-results section contain no evaluation outcomes. | Inspectable provenance, but no evidence of a coding or science gain. Larger rank makes a direction collector more expensive. Exact base revision and data overlap unknown. |
| [DavidrPatton/Gemma-4-31B-Fable-Coder](https://huggingface.co/DavidrPatton/Gemma-4-31B-Fable-Coder/tree/a9e358f75bfc85e7eb289cd80120cf1fc5a61181) | Card declares an Unsloth four-bit base and a merged16-bit model; repository lists full model index/config and no adapter config/payload | Names Fable agentic traces, OmniACT, WebLINX and VisualToolBench. Card contains capability claims but no measured coding evaluation. | Full-model lead, not an available small LoRA direction under the current recipe. No payload download justified by this metadata alone. |

These are publisher claims, not independently reproduced training/evaluation
facts. None establishes compatibility with the experiment's exact original
BF16 tree. In particular, a broad model-name match does not establish the
training revision, and benchmark names or copied base scores do not establish
adapter performance. Benchmark contamination is unknown for every lead.

The reasoning adapter's rank64/alpha128 and Lambent's rank256/alpha256 differ
from the currently pinned rank16/alpha32 donor. The existing deterministic
recipe deliberately rejects those alternatives; it must not silently reinterpret
them as the current donor. Any future use would need explicit versioned mapping,
header and source validation, unit-direction norms, zero-gate invariance,
actual BF16 finite response, and official evaluation.

## Inspectable sources

Only public metadata was fetched over SSH. Exact request URLs, downloaded card
and config bytes, SHA-256 values and query times are retained in the experiment
artifact directory `public-donor-recon-20261106/`, with `receipt.json`.

| Source file | SHA-256 |
|---|---|
| kai-os README | `24aceb4c54d25d0c37712e1a83a7c68d2e22860873d26b259fbdd52978360f40` |
| kai-os adapter_config.json | `e21508b3c85b1cc39d892dccace56121ea2a20f56a35d311e4d3665de761ada6` |
| Lambent README | `09df0856ce814924a374ba9a1fc1b5cb222f455ce0679ce356f45e5a7be2444d` |
| Lambent adapter_config.json | `ae9e7034b8902d44d3d1ceb15d508521cde3881b21616ace205aec43caaeafa3` |
| DavidrPatton README | `8964b1c9297b077f7bffe5ddb02640c18f8618f19f57c804aeebb0967fa8b01e` |

The public API queries used model-name searches plus the `peft` filter.
Pinned files used the repository's returned revision, for example
`https://huggingface.co/<repo>/resolve/<revision>/adapter_config.json`.
No private authentication token was required or recorded.
