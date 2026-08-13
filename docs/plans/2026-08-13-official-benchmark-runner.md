# kairyu-bench implementation plan

## Goal

Build a standalone Docker product that accepts a Kairyu-compatible API URL,
runs any of the 12 supported benchmarks through their official harnesses,
and writes reproducible score and comparison reports. The product must not
import or reuse the `kairyu` repository.

The only required runtime argument is the API URL. The runner discovers the
available model from `GET /v1/models`, probes the returned IDs in server order,
and records the first chat-capable model in the run metadata.

## Product contract

```sh
./kairyu-bench run https://kairyu.example/v1
./kairyu-bench run https://kairyu.example/v1 --only gpqa-diamond,hle --limit 20
./kairyu-bench compare results/run-a results/run-b
```

- The host requires only Docker and POSIX shell tools.
- Generation, benchmark execution, scoring, aggregation, and comparison run
  inside Docker.
- `--only` selects benchmark names and `--limit` selects a deterministic prefix
  after the benchmark's canonical ordering is established.
- Raw upstream artifacts, normalized results, logs, source revisions, selected
  model ID, selection IDs, and scoring metadata remain in `results/<run-id>/`.
- A benchmark never invents a fallback score. Missing official prerequisites
  produce `unsupported`; runtime errors produce `failed` or `partial`.

## Architecture

```text
host shell entrypoint
  -> privileged runner container
       -> API discovery and compatibility probe
       -> shell dispatcher
            -> official upstream harness / nested task containers
            -> benchmark adapter normalizer
       -> JSON/Markdown aggregation
       -> compatible-run comparison
```

The runner container owns Docker-in-Docker because SWE-bench and terminal task
harnesses create containers themselves. Official upstream tools are isolated
from each other in per-harness virtual environments. Shell scripts orchestrate
the workflow; small Python modules are limited to HTTP transport, deterministic
selection, schema validation, normalization, reporting, and comparison.

## Supported benchmark adapters

| Adapter | Official execution/scoring source | Special policy |
| --- | --- | --- |
| `swe-bench-pro` | SWE-bench evaluation harness with the Pro dataset | nested Docker |
| `swe-bench-verified` | SWE-bench evaluation harness | nested Docker |
| `terminal-bench` | Harbor/Terminal-Bench official harness | nested Docker |
| `livecodebench` | LiveCodeBench official generation/evaluation | official code tests |
| `livecodebench-pro` | LiveCodeBench Pro data and testcase checker | unsupported if the official checker cannot run |
| `hle` | HLE data and answer judge | same target model judges; `self_judged: true` |
| `charxiv-reasoning` | CharXiv reasoning evaluation | same target model judges; `self_judged: true` |
| `gpqa-diamond` | GPQA Diamond exact-choice scoring | deterministic exact match |
| `scicode` | SciCode official tests | official code tests |
| `tau-bench-banking` | tau2-bench banking environment | same target model simulates user; `self_simulated: true` |
| `long-context-reasoning` | LongBench v2 | explicitly reported as a LongBench v2 substitute |
| `mrcr-v2` | OpenAI MRCR v2 data/scorer | official scorer |

Every adapter contains a `run.sh` and a normalizer. Upstream source and dataset
revisions are locked in one manifest and embedded in every result.

## Result schema

Each benchmark writes `normalized/<benchmark>.json` with:

- schema version, run ID, benchmark name, and status;
- endpoint fingerprint and discovered model ID;
- upstream source and dataset revisions;
- selected problem IDs and requested/evaluated counts;
- primary score, named component metrics, and official score unit;
- scoring method plus `self_judged`/`self_simulated` flags;
- raw artifact and log paths;
- started/finished timestamps and error details.

Statuses are `completed`, `partial`, `skipped`, `failed`, or `unsupported`.
The command exits 0 only when all selected benchmarks complete, 2 for endpoint
or preflight failure, and 3 when any selected benchmark is incomplete.

Comparison is allowed only when adapter name, upstream/dataset revisions,
selection IDs, and scoring method match. Incompatible rows are displayed but
have no delta.

## Implementation batches

1. Add packaging, unit-test tooling, the host shell entrypoint, runner image,
   and a container-side command dispatcher.
2. Implement endpoint normalization, `/v1/models` discovery, ordered chat
   probes, authentication passthrough, and endpoint-safe metadata.
3. Define and validate the normalized result schema, workspace layout,
   lifecycle statuses, deterministic selection, and exit semantics.
4. Add official-source lock manifest and the 12 adapter directories. Implement
   exact-match/data-only adapters first, then code and agentic harnesses.
5. Add JSON/Markdown score aggregation and strict compatible-run comparison.
6. Add fake-server contract tests, adapter fixture tests, shell integration
   tests, Docker smoke tests, and user/operator documentation.

Each batch follows red-green-refactor, is committed separately, and is pushed
to the draft pull request after its focused tests pass.

## Verification gates

- Unit tests cover URL normalization, model selection, result validation,
  deterministic limits, aggregation, comparison, and exit codes.
- Contract tests use a local OpenAI-compatible fake server and exercise real
  HTTP requests rather than mocked client methods.
- Every adapter fixture proves both successful normalization and malformed or
  incomplete upstream output handling.
- Shell smoke tests invoke the real entrypoint with a controlled fake Docker
  executable; a final Docker smoke test builds the runner and executes the
  preflight path when Docker is available.
- `ruff`, static type checks, the complete test suite, and Dockerfile lint/smoke
  checks must pass before the PR is marked ready.

## Known comparability labels

- HLE and CharXiv self-judging results are not identical to leaderboard runs
  that use their fixed external judge models.
- tau-bench banking self-simulation is reported separately from runs that use a
  fixed external user-simulator model.
- `long-context-reasoning` is LongBench v2 and is not presented as the private
  Fugu benchmark row.
- Partial selections are comparable only when their exact selected IDs match.
