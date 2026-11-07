# Public TRAIN reference and official ON-frame validation

The unchanged filtered input contains 204 training and 51 holdout examples, all from original MBPP TRAIN. Source/filter provenance remains in the separate `mbpp-train-preparation-v1/filter-v1` records; this audit did not revisit benchmark gold or alter that split.

| Check | Train | Holdout | Result |
|---|---:|---:|---|
| Original reference + setup + all three original asserts | 204/204 | 51/51 | 765/765 assertions pass |
| Reference unknown / timeout / failed cases | 0 | 0 | No reference exclusions |
| Independent official ON tokenizer frame | 204/204 | 51/51 | Every token prefix matches |
| Actual trainer helper against independent token rows | 204/204 | 51/51 | Exact all-token agreement |
| Maximum complete frame length | 336 | 366 | All <=1024; no truncation |

No challenge tests were present. Passing three original assertions is evidence about those assertions, not certification of the entire natural-language contract. Holdout reference verification did not train on or evaluate a model against holdout answers.

The reference check used the existing `science_sandbox.Docker.arguments`, `guard`, `check_image`, `cleanup`, and `bounded_command` boundaries unchanged: pinned c4ab image, network none, runc, UID1003, read-only root, cap-drop ALL, no-new-privileges, CPU1, 512MiB memory/swap, pids64, tmpfs16MiB, no host mounts or GPU. One bounded container ran 255 separate child processes, each with a ten-second limit, preserving every original assert result, exception class and duration. Each scope starts from reference code followed by `test_setup_code`; challenge is separately reported. The host never executes dataset code. Total reference validation was 10.171445 seconds, with normal cleanup and all27 pre/post containers unchanged.

The F04 tokenizer checks used the existing af9 image with CPU2/4GiB, network none, runtime runc, no GPU devices, read-only input/checkpoint/source mounts and one NVMe output mount. `AutoTokenizer` was loaded locally with remote code disabled. No model weights were loaded for inference or differentiation. The independent run took 10.337271 container seconds (3.188181 tokenization); the trainer helper check took 9.325117 seconds (2.397480 helper work). All27 containers were unchanged after each cleanup.

The exact frame contract is raw `row["text"]` as one user message and raw `row["code"]` as one assistant message. Both renderings use `enable_thinking=True`; the prefix uses `add_generation_prompt=True`, the complete dialog uses `False`, and both encode with `add_special_tokens=False`. No test text, Markdown fences, invented reasoning, forced empty thought, manual EOS, padding or truncation is added. Every complete token sequence starts with the exact prefix sequence. The loss target is the entire remaining official assistant suffix, including the official turn terminator and trailing newline.

A concrete pretraining bug was found: the draft trainer required configured global EOS id1 in the official assistant answer. The actual template ends in `<turn|>\n`, token IDs `[106,107]`; all255 frames contain zero EOS1 tokens. That check would reject every valid official frame. The trainer author changed the check to exact official rendered/text/token suffix without inserting EOS. The immutable extracted `prepare_training_frames` plus `FRAME_PROTOCOL` was then evaluated for all255 and train204 separately; all token rows matched independent evidence, the train list was an exact subset, and holdout ID overlap was zero. The full trainer source was still being edited elsewhere; the extracted helper AST SHA, historical whole-file SHA and exact executed source are recorded, rather than claiming that subsequent trainer code was validated here.

Reproducible reference command (CPU only):

```sh
/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/infra/weight-tools-ci/bin/python \
  scripts/experiments/gemma4-purpose/mbpp_reference_quality.py \
  --inputs /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/analysis/fresh-purpose-20261107-mbpp-train-preparation-v1/filter-v1 \
  --output /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/analysis/fresh-purpose-20261107-mbpp-reference-quality-v1
```

That output already exists and must not be overwritten. The complete Docker argv, container names/ownership labels, source/input pins, all case/test rows, all exact prefix/full/suffix token arrays, closure and timing are retained in [records](records/2026-11-07-mbpp-reference-quality-v1/). Large exact records are losslessly gzip-compressed; `artifact-index.json` binds raw and compressed hashes. Runtime roots are `fresh-purpose-20261107-mbpp-reference-quality-v1`, `fresh-purpose-20261107-mbpp255-frames-v1`, and `fresh-purpose-20261107-mbpp255-trainer-frames-v1`.

Key identities:

- train204 input: `d501b58dd3860a5c1b666a9318f3f8f70c405d2369afce3aea13bd0f379223f6`
- holdout51 input: `2397ed986d06fa36f5a0dbfb80d4f08c8ba771eda0153288f0110bd20c26db2c`
- all255 quality ledger: `ec34b47365772df17fb1698627ec7aad244e06ead2ac3979d1da0b1b2345e22e`
- train204 actual trainer frame proof: `41fc78e1f8d26c52b1f5ce85120aba3049e6f2797c792232d59a9785b6c85cf7`
- extracted helper AST: `66e51697ce014c342a2c91259523b8d8cfa709068712d6f16c78d083d2437c3d`

All three CPU executions completed; no model forward, gradient, model API, serving mutation, benchmark run or original-input edit occurred. Syntax and focused Ruff passed. Initial unused-import lint findings were corrected before reference execution; the tokenizer helper's executed bytes were preserved before formatting, and its helper AST remains identical. A local tool-wrapper syntax error while preparing this Markdown caused no remote execution and was corrected. No broad test suite was rerun for these experiment-only wrappers.
