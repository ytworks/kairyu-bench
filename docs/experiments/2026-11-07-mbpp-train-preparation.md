# MBPP original TRAIN: fixed contamination filtering

**204 training rows and 51 separate holdout rows are ready for a reference-quality check. Training and model calls are zero.** These 255 rows pass the predeclared split, ID exclusion, exact/near-duplicate and syntax checks. They are not claimed to have correct executable references or proven semantic independence.

The official Google Research [README](https://github.com/google-research/google-research/blob/0eebfd403ba438bfd14cb70b9b24a7745c997083/mbpp/README.md) assigns task IDs 601–974 to TRAIN, 511–600 to validation, 11–510 to test, and 1–10 to prompting. The repository [Apache-2.0 license](https://github.com/google-research/google-research/blob/0eebfd403ba438bfd14cb70b9b24a7745c997083/LICENSE) was downloaded and inspected before the data; the pinned MBPP directory listing had no overriding license. Attribution, the full license and original metadata are preserved. Commit `0eebfd403ba438bfd14cb70b9b24a7745c997083` pins every upstream URL. The only dataset download was [original mbpp.jsonl](https://github.com/google-research/google-research/blob/0eebfd403ba438bfd14cb70b9b24a7745c997083/mbpp/mbpp.jsonl), 563,743 bytes in 0.254 seconds, SHA `ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f`. No repository clone or authenticated request was made.

| Stage | Rows | Meaning |
|---|---:|---|
| Original release | 974 | Kept as source provenance; non-TRAIN rows never enter training/holdout |
| Official original TRAIN | 374 | All IDs601–974, each in the complete exclusion ledger |
| Excluded by complete MBPP+378 pool IDs | 108 | Includes all73 original-TRAIN IDs present in the union of official WSP code/probe selections |
| Additional exact/near exclusions | 11 | Nine match original non-TRAIN problems; two duplicate earlier retained TRAIN problems |
| Retained unique rows | 255 | Syntax-valid, contamination-filter eligible; executable quality unknown |
| Training / separate holdout | 204 / 51 | Fixed hash ordering, 20% holdout, no intersection |

Before downloading the JSONL, the filter policy was frozen (SHA `f332c7277a561ff47d9fb57683737b2865a7014f4e00501b3992c265bdb73e65`). It excludes the full10/30/60 dev/test MBPP+ and probe-code ID union228, then the complete MBPP+378 pool. It also compares against all600 original non-TRAIN MBPP rows, full HumanEval+164, SciCode test/validation main and step references (421 entries), and LCB342 question texts. The benchmark adapter, manifest and reference inputs are identified in the output; no evaluation condition, source or item was reduced.

Exact prompts use Unicode NFKC/casefold and word tokens. Exact code uses AST structure after docstring removal, preserving names and literals. Near prompts use3-token shingles with at least12 tokens and Jaccard≥0.80 or shorter containment≥0.95. Near code uses5-token shingles of normalized code, at least20 tokens, and Jaccard≥0.90 or containment≥0.98. Earlier retained TRAIN examples are checked with the same rules, so exact/near duplicates cannot be split between train and holdout. All matched reference IDs, scores and reasons remain; thresholds were not changed after observing counts. These lexical/structural rules can miss semantic rewrites and conservatively exclude harmless similarities.

The extra excluded IDs are625,634,653,699,704,841,842,872,928,930,965. For example625 duplicates the AST of non-TRAIN591,704 duplicates non-TRAIN248, and930 duplicates retainedTRAIN719. HumanEval/SciCode/LCB comparisons found no additional surviving-row exclusions beyond this policy's MBPP exclusions. LCB's approximately692MB Arrow files were memory-mapped and only question ID/content columns converted; public/private tests were neither decoded nor hashed. The selected Arrow projections totaled1,567,466 bytes across LCB/SciCode, pinned by projection SHA plus before/after file stats; these are not whole-file hashes.

Holdout assignment is independent of scores: ascending SHA256(`mbpp-original-train-dedup-20261107-v1:<task_id>`), first51 held out. All original row fields were preserved unchanged. All255 have exactly3 supplied tests, no challenge tests, and one has nonempty setup. Code/setup/test syntax parsed, but no dataset program was executed. The next gate must run supplied references in an isolated CPU sandbox and preserve failures before treating rows as reliable supervision; three tests alone do not establish general semantic correctness. Do not consume holdout labels for optimizer updates.

The dedicated [recipe](scripts/fresh-mbpp-train-prepare-20261107.py) uses existing pyarrow and the fixed policy; it is not a new trainer. Ruff passed. The single full filter completed in5.695 seconds (process5.894), with one CPU thread, nice15 and idle I/O. A preliminary schema read used system Python without pyarrow and failed before reading data; the corrected existing CPU environment was used for the actual filter. Its32 SyntaxWarning lines were preserved. GPU, model, optimizer, container creation and serving changes all remain0.

Output root: `/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/analysis/fresh-purpose-20261107-mbpp-train-preparation-v1/filter-v1`.

- `train.jsonl` (204): `d501b58dd3860a5c1b666a9318f3f8f70c405d2369afce3aea13bd0f379223f6`
- `holdout.jsonl` (51): `2397ed986d06fa36f5a0dbfb80d4f08c8ba771eda0153288f0110bd20c26db2c`
- `all374-ledger.json`: `231ad3d4d18a464e6977df7c8c30346b64d18a985ad3eb12d5dee45cad7cd981`
- `summary.json`: `e3022119b905b00bd7f9f8f3574f6d83fcd84e9df6fb5c3afb8e1da01cd2dc11`

[All downloads, URLs, license, timings, policy, actual command/source, complete ledger and output rows](records/2026-11-07-mbpp-train-preparation/retention-index.json) are retained in the PR, losslessly gzip-compressed where useful. Previous cache-only inspection remains a historical report; this newly authorized public download supplies the missing original TRAIN.
