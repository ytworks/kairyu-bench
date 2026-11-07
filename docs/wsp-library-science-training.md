# Library-composition training extension

The standalone library_train_cases.py authors 16 training examples from eight families. Each family has two interface or composition variants, one per fold. These are eight families, not 16 independent families. The source contains every prompt, correct implementation, plausible foil, and independent reference generator; no benchmark task or reference implementation is imported.

The extension is appended to the existing training JSON with its exact SHA supplied by the operator. Every inherited case object remains equal, in its original purpose/fold order. The existing 136-case artifact is never written. The actual bundle contains 152 examples, the same 21 purposes and 42 folds.

| Family | Existing purpose | New examples per fold |
|---|---|---:|
| Sparse boundary/banded solve | science_numerics | 1 |
| FFT linear convolution/periodic correlation | science_numerics | 1 |
| Residual–Jacobian composition | science_advanced_state | 1 |
| Bracketed roots/coordinate transforms | science_advanced_state | 1 |
| Piecewise/transformed quadrature | science_advanced_simulation | 1 |
| Symbolic variable order/chain rule | science_contracts | 1 |
| Optimizer equality/feasibility | science_contracts | 1 |
| Distribution shape/scale interfaces | science_advanced_probability | 1 |

Science counting receives no new example. Before/after counts are explicit in metadata; new-case results should be reported separately from inherited cases.

## CPU authoring and export

Use an environment with NumPy, SciPy and SymPy. No model is loaded.

    PYTHONPATH=src python scripts/experiments/gemma4-purpose/library_train_cases.py --trials 8
    PYTHONPATH=src python scripts/experiments/gemma4-purpose/library_train_cases.py \
      --write --output /fresh/output \
      --base /existing/cases.json --base-sha256 <exact-sha256> --trials 8

Output is fresh-only: train-extension.json contains the 16 examples, train-cases.json contains the append-only merged cohort, and metadata.json records the mapping, versions, old-object equality and per-case reference/foil outcomes. Omitting the base exports the standalone extension. Transfer data is owned by the separate library_transfer_cases.py module and is never included here.

The validator executes only trusted programs embedded in this source. Arbitrary generated programs must use science_sandbox.py with its explicit CPU sandbox. The library-train-v1 sandbox catalog accepts the complete standalone train-extension.json, not the merged152 cohort. No reference values are supplied to the worker.

References use analytic identities, independent dense solves, direct convolution/correlation loops, an independent simplex projection, and the standard-library inverse normal CDF. They check shape, float64 type, finite values and unchanged inputs. A valid correct implementation must pass every sampled check, and each foil must fail at least one. Numerical trials validate one implementation; they do not create independent language tasks.

SciPy may use subnormal sentinels internally. This library catalog allows underflow while raising overflow, divide and invalid errors and rejecting nonfinite outputs. Existing NumPy catalogs retain their prior policy.

## Actual validation

The existing CPU environment used NumPy2.2.6, SciPy1.13.1 and SymPy1.14.0. All16 correct implementations passed eight trials; all16 foils were detected. One E2E fixture at a distinct seed passed in1.433s. It covers reference/foil behavior through actual collector-reader export, original-case preservation and existing-output refusal. Ruff and targeted mypy passed. The ordinary af9 image lacks SciPy; this validation does not claim the generated-code sandbox image is ready for these libraries.

The exported bundle is A/library-science-training-v1:
- train-cases.json: 475b1b3a8c88ca8602c9f3078d635aa0f867aa9669166fdbc5a278f6579b2016
- train-extension.json: 116ed7b97d095dc8f842fe87c6d3f256bf06aaeddf4bab4aab650b8bc8b5ec8e
- metadata.json: 2fd6edf53a38fc99ef0607fe8fc9e421ef5747790069375fbdb84aa70a29d925
- tokenizer-validation.json: 9a7a4bced60c06174403ca79ed5275ee6b147227476f868138618b94a97a3ad4

A local-tokenizer-only check reused the existing forced-empty-thought public-answer frame. New examples require at most311 total tokens, inherited examples at most509; all fit512. This is a teacher-forced diagnostic frame, not ordinary thinking-enabled generation. No model forward or benchmark call was made by authoring/validation.

This is a small training extension, not a broad competence certificate. It introduces library contracts while retaining short examples; it does not establish long-context, safety, or unseen scientific-code performance. The eight separate new transfer families must remain excluded from fitting, and repeated transfer use must be reported as known calibration.

The combined library/group-tool CPU validation passed 518 collected tests (463 passed, 55 optional skips), whole-tree Ruff, mypy74, data checks and 31 shell syntax checks, with unchanged source hashes. The managed suite lacks the optional scientific libraries; the train/transfer reference and foil E2E checks also passed separately in the recorded SciPy/SymPy environment, and the offline executor passed its actual CPU sandbox checks. Receipt: A/cpu-integration-library-science-v1/ci-results.json, SHA256 94822c7a5dbb131df8e1949235d5604f6a3b6a71d29b774327a43ed80ab488ac. CPU validation overlapped the separately owned expanded-cohort collection; no speed comparison is claimed.
