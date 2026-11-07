# Authored library transfer cases

`scripts/experiments/gemma4-purpose/library_transfer_cases.py` exports eight new
transfer-only scientific programming families. They use independently authored
problems and reference calculations; no official benchmark questions, solutions,
or tests were used. Every exported case has `gradient_allowed: false`; the
existing training exporter rejects these cases.

| Family | Reference check | Metamorphic relation | Deliberate foil |
|---|---|---|---|
| Event-terminated ODE plus integral | Analytic exponential and event time | Inverse time/rate scaling | Wrong event direction |
| Generalized symmetric eigensystem | Constructed eigenpairs in an SPD metric | Scale operator and metric | Ignore metric |
| PCHIP with boundary policy | Independent Hermite coefficients and slope rules | Positive affine value transform | Extrapolation flag/derivative policy |
| IIR state and frequency response | Direct recurrence and complex polynomial ratio | Common coefficient scaling | Drop initial state |
| Labeled directed shortest paths | Bellman–Ford relaxation | Relabel output order and reverse edge order | Last duplicate overwrites minimum |
| Symbolic Jacobian axis order | Explicit tensor-index sums | Permute latent axes | Transpose symbolic flattening |
| Bootstrap means and covariance | Explicit sums over supplied resamples | Affine data transform | Population covariance denominator |
| Composite uncertainty propagation | Analytic chain rule | Scale input covariance | Drop cross-covariances |

The ODE samples include both event and no-event branches. The graph has a
disconnected vertex and duplicate directed edges. Interpolation queries include
both endpoints and both outside regions. Bootstrap randomness is supplied by an
explicit integer index table. These checks cover finite inputs; they do not prove
correctness for every valid input.

## Author validation and packaging

Use a CPU Python with NumPy, SciPy and SymPy already installed:

```sh
python scripts/experiments/gemma4-purpose/library_transfer_cases.py \
  --output /absolute/fresh/library-transfer-v1 --write --trials 8
```

The command runs only its trusted embedded reference and foil programs, validates
them against independent oracles and metamorphic relations, then writes
`transfer-cases.json`, an empty `train-cases.json`, and `metadata.json`.
Metadata records source hashes and actual library versions. It refuses an
existing output. No model or tokenization is performed.

The initial CPU validation used NumPy 2.2.6, SciPy 1.13.1 and SymPy 1.14.0 in an
existing NVMe environment as UID 1003. The one end-to-end test checks bundle →
existing sandbox catalog → serialized worker inputs → numeric comparisons for
all correct/foil programs. It skips explicitly when optional libraries are absent;
that skip is not evidence of execution.

## Generated public answers

Use the existing `science_generate.py` on the transfer envelope and
`science_sandbox.py --catalog library-transfer-v1` for execution. The sandbox
must be given an immutable CPU image that actually contains these libraries.
The existing serving image af9 contains NumPy/SymPy but lacks SciPy; it is not
a suitable executor for this catalog. The code90 benchmark wrapper image and
the local coding-v1 executor also lacked the necessary full library set in the
checked default Python. No image build or generated-answer run is implied here.

The sibling `library-train-v1` catalog supports the separately authored train
extension. Only library cases declare benign floating-point underflow acceptable;
overflow, invalid operations, division errors and nonfinite output remain
rejected. This permits SciPy's `solve_ivp` internal `nextafter(0, inf)`.
Existing basic/advanced catalogs retain their prior error policy. The public
worker receives sampled inputs and code, never reference outputs.

Keep this transfer envelope out of gradient collection, coefficient fitting and
weight training. Repeated evaluation or selection makes it known transfer
calibration. Separate families share libraries and mathematical primitives, so
their names do not establish statistical independence. Results are authored
execution checks, not WSP scores or a claim of unchanged model accuracy.


## Offline immutable executor

The reusable `scripts/experiments/stage_science_library_executor.py` stages the
installed SciPy package, its bundled shared libraries and distribution metadata.
It requires non-root execution, a fresh canonical output outside every source
root, exact supplied base-metadata SHA, and equal source/base Python version,
SOABI, architecture, libc, NumPy and SymPy versions. This Dockerfile supports
CPython 3.12 on x86-64. Package files are copied and independently hashed; source
stamps are rechecked. Bytecode caches are omitted. No install hooks run.

```sh
/path/to/existing/scientific/python -B \
  scripts/experiments/stage_science_library_executor.py \
  --base-metadata /absolute/base-python.json \
  --base-metadata-sha256 <verified-sha256> \
  --output /absolute/fresh/context --apply

DOCKER_BUILDKIT=0 docker build --network none --pull=false \
  --build-arg BASE_IMAGE=<verified-local-sha256-image-id> \
  --iidfile /absolute/fresh-image-id.txt /absolute/fresh/context
```

Capture `base-python.json` from a CPU-only, network-disabled, read-only invocation
of the chosen immutable base. Its fields are `image`, `python` (version),
`soabi`, `machine`, `libc` (two-element list), `numpy`, and `sympy`.
Staging compares them to the trusted source Python. The companion
`science_library_executor.Dockerfile` has no RUN instruction: it adds only the
staged package files and sets USER 1003:1003. The legacy local builder is explicit
because the initial BuildKit attempt treated a local image ID as a registry
name and failed before building; that failure was retained.

The exercised executor image is
`sha256:c4ab50223f03fca17bf871e4d05f77001e76525a1c574b6c6ee7024db06cfdc2`.
It preserves every af9 base layer and adds SciPy 1.13.1 (1,331 files,
119,022,185 bytes); NumPy 2.2.6 and SymPy 1.14.0 are inherited. The existing
Docker sandbox ran all eight correct and eight foil implementations, eight
trials each, in 22.954 seconds: correct 64/64, all eight foil families detected
(60/64 foil trials rejected), unknown zero. The four non-rejected foil trials
are the ODE no-event branch where the direction error has no effect.
This is trusted-reference executor validation, not model generation or an
accuracy result. No GPU, model load, package download or service change occurred.

Evidence is under `analysis/code90-resume-20261106/science-library-executor-v1`;
`validation.json` SHA
`0367debeefbc14739a83859ee8b0bd316270eba825f8a4e6a0243a216f9d17d6`
pins the actual image and sandbox reports. Staging used helper b21edc21; the final
c77d55d4 adds a pre-mkdir overlap check. The exercised context was already
disjoint from all package/input sources; no numerical or copied-byte change
was needed. The existing E2E additionally removes the reference's local
underflow guard to verify the library-only worker policy itself.

For `library-train-v1`, the sandbox input is the standalone 16-case extension,
not the merged 152-case collector input: the executor checks the complete exact
authored catalog.

