#!/usr/bin/env python3
"""Short independent science chains; host execution is source-owned code only.

Public model programs are extracted without execution and require the existing
isolated science_sandbox runner. This module never makes a model request.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit
from urllib.request import urlopen
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import science_sandbox as sandbox  # noqa: E402
import science_generate as generator  # noqa: E402


@dataclass(frozen=True)
class Stage:
    name: str
    parameters: str
    requirement: str
    imports: str
    body: str
    foils: tuple[tuple[str, str, str], ...]
    allow_underflow: bool = True

    def code(self, foil: int | None = None) -> str:
        body = self.body
        if foil is not None:
            _, old, new = self.foils[foil]
            if body.count(old) != 1:
                raise ValueError("foil mutation must select exactly one expression")
            body = body.replace(old, new)
        return (
            "import numpy as np\n"
            + self.imports
            + "\n\ndef solve("
            + self.parameters
            + "):\n"
            + "\n".join("    " + line for line in body.splitlines())
            + "\n"
        )


@dataclass(frozen=True)
class Chain:
    family: str
    split: str
    purpose: str
    fold: int
    helper: Stage
    downstream: Stage

    def prompt(self, stage: int, helper: str | None = None) -> str:
        item = self.helper if stage == 0 else self.downstream
        prefix = (
            ""
            if stage == 0
            else "Available helper (do not redefine):\n" + str(helper) + "\n"
        )
        return (
            prefix
            + f"Write solve({item.parameters}). "
            + item.requirement
            + " Return a float64 ndarray; do not mutate inputs. Include imports and the "
            "complete solve function, no prose. The helper stage is nonrecursive."
        )

    def row(self, stage: int) -> dict[str, Any]:
        item = self.helper if stage == 0 else self.downstream
        return {
            "id": f"chain-{self.family}-{stage + 1}",
            "family": self.family,
            "stage": stage + 1,
            "split": self.split,
            "gradient_allowed": self.split == "train",
            "fold": self.fold,
            "depends_on": [] if stage == 0 else [f"chain-{self.family}-1"],
            "answer_unit": "complete stage solve function and imports; helper context excluded",
            "stage_weight": 0.5,
            "family_weight": 1.0,
            "foil_id": item.foils[0][0],
            "prompt": self.prompt(
                stage, as_helper(self.helper.code()) if stage else None
            ),
            "correct": item.code(),
            "foil": item.code(0),
        }


def as_helper(public_code: str) -> str:
    """Parse public code only; never execute it on the coordinator."""
    tree = ast.parse(sandbox.extract(public_code))
    roots = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    ]
    if len(roots) != 1 or any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "helper"
        for node in tree.body
    ):
        raise ValueError(
            "one complete solve function and no reserved helper definition required"
        )
    if any(
        isinstance(node, ast.Name) and node.id == "solve" for node in ast.walk(roots[0])
    ):
        raise ValueError("helper stage must be nonrecursive")
    roots[0].name = "helper"
    return ast.unparse(tree) + "\n"


def catalog() -> list[Chain]:
    return [
        Chain(
            "whitened-fit",
            "train",
            "science_numerics",
            0,
            Stage(
                "whitened-fit-helper",
                "a, b, weight",
                "a is full-column-rank (n,p), b is (n,), positive weight is (n,). "
                "Return (n,p+1) containing sqrt(weight) times each row of [a,b].",
                "",
                "return np.column_stack((a,b))*np.sqrt(weight)[:,None]",
                (
                    ("squared-weight", "np.sqrt(weight)", "weight"),
                    (
                        "unweighted-target",
                        "np.column_stack((a,b))*np.sqrt(weight)[:,None]",
                        "np.column_stack((a*np.sqrt(weight)[:,None],b))",
                    ),
                ),
            ),
            Stage(
                "whitened-fit-downstream",
                "a, b, weight, query",
                "Using helper(a,b,weight), minimize sum(weight*(a@x-b)**2) and return query@x. "
                "query has shape(m,p).",
                "from scipy.linalg import lstsq",
                "z = helper(a,b,weight)\nx = lstsq(z[:,:-1],z[:,-1])[0]\nreturn np.asarray(query@x,dtype=np.float64)",
                (
                    ("wrong-target-column", "z[:,-1]", "z[:,0]"),
                    ("unweighted-fit", "helper(a,b,weight)", "np.column_stack((a,b))"),
                ),
            ),
        ),
        Chain(
            "boundary-response",
            "train",
            "science_numerics",
            1,
            Stage(
                "boundary-response-helper",
                "a, b, fixed, values",
                "a is SPD (n,n), b is (n,). fixed are unique valid indices, a nonempty proper subset; "
                "values correspond to fixed. In ascending free-index order, return the reduced matrix "
                "a[free,free] with RHS b[free]-a[free,fixed]@values appended as its final column.",
                "",
                "free = np.setdiff1d(np.arange(len(b)),fixed)\nrhs = b[free]-a[np.ix_(free,fixed)]@values\nreturn np.column_stack((a[np.ix_(free,free)],rhs))",
                (
                    (
                        "missing-boundary",
                        "b[free]-a[np.ix_(free,fixed)]@values",
                        "b[free]",
                    ),
                    (
                        "wrong-boundary-sign",
                        "b[free]-a[np.ix_(free,fixed)]@values",
                        "b[free]+a[np.ix_(free,fixed)]@values",
                    ),
                ),
            ),
            Stage(
                "boundary-response-downstream",
                "a, b, fixed, values",
                "Use helper(a,b,fixed,values) to return full x: x[fixed]=values and (a@x)[free]=b[free].",
                "from scipy.sparse import csc_matrix\nfrom scipy.sparse.linalg import spsolve",
                "free = np.setdiff1d(np.arange(len(b)),fixed)\nz = helper(a,b,fixed,values)\nx = np.zeros(len(b)); x[fixed] = values\nx[free] = spsolve(csc_matrix(z[:,:-1]),z[:,-1])\nreturn x",
                (
                    ("lost-prescribed-values", "x[fixed] = values", "x[fixed] = 0"),
                    ("wrong-rhs-column", "z[:,-1]", "z[:,0]"),
                ),
            ),
        ),
        Chain(
            "quadrature-mixture",
            "train",
            "science_probability",
            0,
            Stage(
                "quadrature-mixture-helper",
                "nodes, weights, density",
                "Distinct nodes, positive quadrature weights and nonnegative density are equal-length vectors; "
                "at least one density is positive. Return normalized weights*density.",
                "",
                "mass = weights*density\nreturn mass/mass.sum()",
                (
                    ("omit-density", "weights*density", "weights.copy()"),
                    ("wrong-normalizer", "mass/mass.sum()", "mass/weights.sum()"),
                ),
            ),
            Stage(
                "quadrature-mixture-downstream",
                "nodes, weights, density",
                "Use helper(nodes,weights,density) to return [weighted mean of nodes, weighted CENTRAL variance].",
                "",
                "p = helper(nodes,weights,density)\nmu = p@nodes\nreturn np.array([mu,p@((nodes-mu)**2)],dtype=np.float64)",
                (
                    ("raw-second-moment", "(nodes-mu)**2", "nodes**2"),
                    ("mean-as-variance", "p@((nodes-mu)**2)", "mu"),
                ),
            ),
        ),
        Chain(
            "log-mixture-score",
            "train",
            "science_probability",
            1,
            Stage(
                "log-mixture-score-helper",
                "log_prior, log_likelihood",
                "Finite equal-length vectors describe positive prior weights in log form and per-component log likelihood. "
                "Return normalized posterior probabilities, stably even with a common large log offset.",
                "from scipy.special import logsumexp",
                "z = log_prior+log_likelihood\nreturn np.exp(z-logsumexp(z))",
                (
                    ("prior-only", "log_prior+log_likelihood", "log_prior.copy()"),
                    (
                        "likelihood-only",
                        "log_prior+log_likelihood",
                        "log_likelihood.copy()",
                    ),
                ),
            ),
            Stage(
                "log-mixture-score-downstream",
                "log_prior, log_likelihood, score",
                "score is (components,features). Use helper(log_prior,log_likelihood) to return the posterior expected score vector.",
                "",
                "p = helper(log_prior,log_likelihood)\nreturn np.asarray(p@score,dtype=np.float64)",
                (
                    ("uniform-mixture", "p@score", "score.mean(axis=0)"),
                    ("map-component", "p@score", "score[np.argmax(p)]"),
                ),
            ),
        ),
        Chain(
            "covariance-propagation",
            "transfer",
            "science_numerics",
            0,
            Stage(
                "covariance-propagation-helper",
                "jac, variance",
                "jac is (m,n) and variance is a positive independent-input variance vector (n,). "
                "Return propagated covariance jac@diag(variance)@jac.T.",
                "",
                "return (jac*variance[None,:])@jac.T",
                (
                    (
                        "standard-deviation",
                        "variance[None,:]",
                        "np.sqrt(variance)[None,:]",
                    ),
                    ("omit-variance", "(jac*variance[None,:])@jac.T", "jac@jac.T"),
                ),
            ),
            Stage(
                "covariance-propagation-downstream",
                "jac, variance, mix",
                "mix has length m. Using helper(jac,variance), return [variance of mix@output], including covariance terms.",
                "",
                "cov = helper(jac,variance)\nreturn np.array([mix@cov@mix],dtype=np.float64)",
                (
                    ("drop-cross-covariance", "mix@cov@mix", "(mix**2)@np.diag(cov)"),
                    ("omit-mix", "mix@cov@mix", "np.trace(cov)"),
                ),
            ),
        ),
        Chain(
            "transition-reward",
            "transfer",
            "science_probability",
            1,
            Stage(
                "transition-reward-helper",
                "counts, smoothing",
                "counts is a nonnegative square matrix; smoothing is a positive scalar. "
                "Add smoothing to every cell, then row-normalize to a transition matrix.",
                "",
                "a = counts+smoothing\nreturn a/a.sum(axis=1,keepdims=True)",
                (
                    ("column-normalize", "axis=1", "axis=0"),
                    ("missing-pseudocount", "counts+smoothing", "counts.copy()"),
                ),
            ),
            Stage(
                "transition-reward-downstream",
                "counts, smoothing, initial, reward",
                "initial is a row probability vector, reward a state vector. Using helper(counts,smoothing), "
                "return [expected reward after exactly TWO transitions].",
                "",
                "p = helper(counts,smoothing)\nreturn np.array([initial@p@p@reward],dtype=np.float64)",
                (
                    ("one-step", "initial@p@p@reward", "initial@p@reward"),
                    (
                        "transpose-transition",
                        "initial@p@p@reward",
                        "initial@p.T@p.T@reward",
                    ),
                ),
            ),
        ),
    ]


def samples(chain: Chain, seed: int, trials: int):
    """Independent algebraic references; each second trial is a metamorphic pair."""
    rng = np.random.default_rng(
        seed + int(hashlib.sha256(chain.family.encode()).hexdigest()[:8], 16)
    )
    args: tuple[Any, ...]
    helper_args: tuple[Any, ...]
    previous = None
    for trial in range(trials):
        if trial % 2 == 0:
            if chain.family == "whitened-fit":
                a = rng.normal(size=(7, 3))
                b = rng.normal(size=7)
                w = rng.uniform(0.2, 3, 7)
                query = rng.normal(size=(4, 3))
                args = (a, b, w, query)
            elif chain.family == "boundary-response":
                m = rng.normal(size=(5, 5))
                a = m.T @ m + np.eye(5)
                args = (a, rng.normal(size=5), np.array([3, 0]), rng.normal(size=2))
            elif chain.family == "quadrature-mixture":
                args = (
                    np.linspace(-0.5, 2, 7),
                    rng.uniform(0.2, 2, 7),
                    rng.uniform(0.1, 3, 7),
                )
            elif chain.family == "log-mixture-score":
                args = (rng.normal(size=5), rng.normal(size=5), rng.normal(size=(5, 3)))
            elif chain.family == "covariance-propagation":
                args = (
                    rng.normal(size=(4, 3)),
                    rng.uniform(0.2, 3, 3),
                    rng.normal(size=4),
                )
            else:
                args = (
                    rng.uniform(0.1, 5, (4, 4)),
                    0.7,
                    np.array([0.1, 0.2, 0.3, 0.4]),
                    rng.normal(size=4),
                )
        else:
            args = tuple(v.copy() if isinstance(v, np.ndarray) else v for v in args)
            if chain.family in {"whitened-fit", "boundary-response"}:
                args = (
                    (args[0], args[1] * 2, *args[2:])
                    if chain.family == "whitened-fit"
                    else (args[0] * 2, args[1] * 2, *args[2:])
                )
            elif chain.family == "quadrature-mixture":
                args = (args[0] + 3, args[1], args[2])
            elif chain.family == "log-mixture-score":
                args = (args[0] + 800, args[1] - 900, args[2])
            elif chain.family == "covariance-propagation":
                args = (args[0], args[1] * 3, args[2])
            else:
                args = (args[0] * 3, args[1] * 3, *args[2:])
        if chain.family == "whitened-fit":
            a, b, w, q = args
            h = np.column_stack((a, b)) * np.sqrt(w)[:, None]
            y = q @ np.linalg.solve(a.T @ (w[:, None] * a), a.T @ (w * b))
            helper_args = args[:3]
        elif chain.family == "boundary-response":
            a, b, f, v = args
            free = np.array([i for i in range(len(b)) if i not in f])
            h = np.column_stack(
                (a[np.ix_(free, free)], b[free] - a[np.ix_(free, f)] @ v)
            )
            y = np.zeros(len(b))
            y[f] = v
            y[free] = np.linalg.solve(h[:, :-1], h[:, -1])
            helper_args = args
        elif chain.family == "quadrature-mixture":
            x, w, d = args
            h = np.array([w[i] * d[i] for i in range(len(x))])
            h /= sum(h)
            mu = sum(float(p * t) for p, t in zip(h, x, strict=True))
            y = np.array(
                [mu, sum(float(p * (t - mu) ** 2) for p, t in zip(h, x, strict=True))]
            )
            helper_args = args
        elif chain.family == "log-mixture-score":
            p, likelihood, s = args
            z = p + likelihood
            h = np.exp(z - max(z))
            h /= sum(h)
            y = np.array(
                [
                    sum(float(h[i] * s[i, j]) for i in range(len(h)))
                    for j in range(s.shape[1])
                ]
            )
            helper_args = args[:2]
        elif chain.family == "covariance-propagation":
            j, v, m = args
            h = np.sum(
                [v[k] * np.outer(j[:, k], j[:, k]) for k in range(len(v))], axis=0
            )
            y = np.array([sum(float(v[k] * (m @ j[:, k]) ** 2) for k in range(len(v)))])
            helper_args = args[:2]
        else:
            c, s, p, r = args
            h = np.array([(row + s) / sum(row + s) for row in c])
            y = np.array(
                [
                    sum(
                        float(p[i] * h[i, j] * h[j, k] * r[k])
                        for i in range(len(p))
                        for j in range(len(p))
                        for k in range(len(p))
                    )
                ]
            )
            helper_args = args[:2]
        if trial % 2:
            assert previous is not None
            expected = previous.copy()
            if chain.family == "whitened-fit":
                expected *= 2
            elif chain.family == "quadrature-mixture":
                expected[0] += 3
            elif chain.family == "covariance-propagation":
                expected *= 3
            if not np.allclose(y, expected, rtol=1e-10, atol=1e-10):
                raise ValueError("authored metamorphic reference contract failed")
        previous = np.asarray(y).copy()
        yield (
            helper_args,
            args,
            np.asarray(h, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
        )


def payload(chain: Chain, stage: int, code: str, seed: int, trials: int):
    rows = list(samples(chain, seed, trials))
    cursor = iter(rows)

    def sample(_name, _rng, _trial):
        hargs, args, h, y = next(cursor)
        return (hargs, h) if stage == 0 else (args, y)

    return sandbox.prepare(
        chain.helper if stage == 0 else chain.downstream,
        code,
        seed,
        trials,
        sample_fn=sample,
    )


def score(chain: Chain, stage: int, code: str, run: Callable, seed: int, trials: int):
    data, expected = payload(chain, stage, code, seed, trials)
    return sandbox.compare(run(data), expected)


def evaluate(
    chain: Chain,
    helper_public: str,
    downstream_public: str,
    run: Callable,
    *,
    seed: int = 64123,
    trials: int = 8,
) -> dict[str, Any]:
    """Caller supplies the existing isolated Docker.run, not host exec."""
    helper_code = sandbox.extract(helper_public)
    downstream_code = sandbox.extract(downstream_public)
    generated_helper = as_helper(helper_code)
    oracle_helper = as_helper(chain.helper.code())
    return {
        "family": chain.family,
        "gradient_allowed": chain.split == "train",
        "helper": score(chain, 0, helper_code, run, seed, trials),
        "downstream_given_oracle": score(
            chain, 1, oracle_helper + downstream_code, run, seed, trials
        ),
        "whole_chain_generated_helper": score(
            chain, 1, generated_helper + downstream_code, run, seed, trials
        ),
        "public_code_sha256": {
            "helper": digest(helper_code.encode()),
            "downstream": digest(downstream_code.encode()),
        },
        "hidden_reasoning_accepted": False,
    }


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def trusted_run(data: dict) -> dict:
    """Reuse the worker in a CPU subprocess, accepting only embedded authored code."""
    import subprocess

    allowed = set()
    for chain in catalog():
        helper = as_helper(chain.helper.code())
        for number in (None, *range(len(chain.helper.foils))):
            allowed.add(chain.helper.code(number))
            allowed.add(as_helper(chain.helper.code(number)) + chain.downstream.code())
        for number in (None, *range(len(chain.downstream.foils))):
            allowed.add(helper + chain.downstream.code(number))
    if data["code"] not in allowed:
        raise ValueError("host reference validator accepts source-owned programs only")
    result = subprocess.run(
        [sys.executable, "-I", "-c", sandbox.WORKER],
        input=json.dumps(data),
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


def evaluate_requests(
    chain: Chain,
    request: Callable,
    run: Callable,
    *,
    seed: int = 64123,
    trials: int = 8,
) -> dict:
    """At most three existing-client calls; only extracted public code enters Docker.

    The oracle and generated-helper downstream answers are separately requested.
    Answers live transiently in memory; only numeric status, usage and hashes leave.
    """
    records = {}

    def ask(label: str, prompt: str):
        try:
            answer = request({"id": f"chain-{chain.family}-{label}", "prompt": prompt})
            records[label] = {
                key: answer.get(key) for key in ("finish_reason", "usage", "seconds")
            }
            public = answer.get("public_answer")
            if not isinstance(public, str) or len(public.encode()) > 262144:
                raise ValueError("missing or oversized public answer")
            code = sandbox.extract(public)
            records[label]["public_code_sha256"] = digest(code.encode())
            return code
        except Exception as error:
            records.setdefault(label, {})["error_type"] = type(error).__name__
            return None

    def execute(stage: int, code: str | None):
        if code is None:
            return {
                "status": "unscored",
                "passed": False,
                "error_type": "missing_public_code",
            }
        try:
            return score(chain, stage, code, run, seed, trials)
        except sandbox.SafetyError:
            raise
        except Exception as error:
            return {
                "status": "unscored",
                "passed": False,
                "error_type": type(error).__name__,
            }

    helper_code = ask("helper", chain.prompt(0))
    helper_result = execute(0, helper_code)
    oracle = as_helper(chain.helper.code())
    oracle_answer = ask("oracle", chain.prompt(1, oracle))
    oracle_result = execute(1, oracle + oracle_answer if oracle_answer else None)
    generated = None
    if helper_code is not None:
        try:
            generated = as_helper(helper_code)
        except ValueError:
            records["helper"]["composition_error"] = "invalid_helper_interface"
    generated_answer = (
        ask("generated", chain.prompt(1, generated)) if generated else None
    )
    generated_result = execute(
        1, generated + generated_answer if generated and generated_answer else None
    )
    return {
        "family": chain.family,
        "split": chain.split,
        "helper_stage": helper_result,
        "oracle_helper_downstream": oracle_result,
        "generated_helper_downstream": generated_result,
        "whole_chain_passed": helper_result["passed"] and generated_result["passed"],
        "calls": records,
        "call_count": len(records),
        "hidden_reasoning_saved": False,
        "public_answers_saved": False,
    }


def validate(seed: int = 64123, trials: int = 8) -> dict:
    if trials < 4 or trials > 100 or trials % 2:
        raise ValueError("use an even 4..100 trials for paired transformations")
    rows = []
    for chain in catalog():
        correct = evaluate(
            chain,
            chain.helper.code(),
            chain.downstream.code(),
            trusted_run,
            seed=seed,
            trials=trials,
        )
        if not all(
            correct[key]["passed"]
            for key in (
                "helper",
                "downstream_given_oracle",
                "whole_chain_generated_helper",
            )
        ):
            raise ValueError(
                f"authored correct chain failed: {chain.family}: {correct}"
            )
        foils = []
        for stage, item in enumerate((chain.helper, chain.downstream)):
            for number, (label, _, _) in enumerate(item.foils):
                code = item.code(number)
                if stage:
                    code = as_helper(chain.helper.code()) + code
                outcome = score(chain, stage, code, trusted_run, seed, trials)
                if outcome["passed"]:
                    raise ValueError(
                        f"undetected authored foil: {chain.family}/{stage}/{label}"
                    )
                foils.append({"stage": stage + 1, "foil_id": label, "outcome": outcome})
        rows.append(
            {
                "family": chain.family,
                "split": chain.split,
                "correct": correct,
                "foils": foils,
            }
        )
    return {
        "seed": seed,
        "trials_per_stage": trials,
        "paired_metamorphic_trials": trials // 2,
        "libraries": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy")
        },
        "chains": rows,
    }


def write_bundle(
    output: Path,
    *,
    base: Path | None = None,
    base_sha256: str | None = None,
    seed: int = 64123,
    trials: int = 8,
) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source = Path(__file__).resolve()
    inputs = [
        source,
        Path(sandbox.__file__).resolve(),
        Path(generator.__file__).resolve(),
    ]
    inputs += [base.resolve()] if base else []
    if any(
        path.is_relative_to(output) or output.is_relative_to(path) for path in inputs
    ):
        raise ValueError("output overlaps an input")
    before = {str(path): digest(path.read_bytes()) for path in inputs}
    if base and before[str(base.resolve())] != base_sha256:
        raise ValueError("base SHA mismatch")
    combined = json.loads(base.read_text()) if base else {}
    if "cases" in combined:
        raise ValueError("base must use the existing purpose/two-fold mapping")
    extension: dict[str, list[list[dict]]] = {}
    foil_rows = []
    for chain in catalog():
        for stage, item in enumerate((chain.helper, chain.downstream)):
            if chain.split == "train":
                purpose = (
                    "science_helper_stage"
                    if stage == 0
                    else "science_composition_stage"
                )
                extension.setdefault(purpose, [[], []])[chain.fold].append(
                    chain.row(stage)
                )
            foil_rows.append(
                {
                    "task_id": f"chain-{chain.family}-{stage + 1}",
                    "family": chain.family,
                    "split": chain.split,
                    "gradient_allowed": False,
                    "foils": [
                        {"foil_id": label, "code": item.code(i)}
                        for i, (label, _, _) in enumerate(item.foils)
                    ],
                }
            )
    ids = {row["id"] for folds in combined.values() for fold in folds for row in fold}
    for purpose, folds in extension.items():
        target = combined.setdefault(purpose, [[], []])
        if len(target) != 2:
            raise ValueError("base must retain two folds")
        for fold, rows in enumerate(folds):
            if any(row["id"] in ids for row in rows):
                raise ValueError("duplicate training ID")
            target[fold].extend(rows)
    validation = validate(seed, trials)
    metadata = {
        "schema_version": 1,
        "parent_binding": None,
        "training_families": 4,
        "training_stages": 8,
        "transfer_families": 2,
        "answer_unit": "complete stage function plus imports, helper only in prompt",
        "collector_weighting": "one row per stage; two stages per family; only first foil used for gradient",
        "additional_foils": "evaluation-only; not repeated correct-LL rows",
        "source_sha256": before,
        "token_count_measured": False,
        "validation": validation,
    }
    if before != {str(path): digest(path.read_bytes()) for path in inputs}:
        raise ValueError("inputs changed")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("train-extension.json", extension),
        ("train-cases.json", combined),
        ("failure-mode-foils.json", foil_rows),
        ("transfer-stage1.json", stage_cases("transfer")),
        ("train-stage1.json", stage_cases("train")),
        ("metadata.json", metadata),
    ):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return metadata


def stage_cases(split: str, helpers: dict[str, str] | None = None) -> dict:
    stage = 0 if helpers is None else 1
    chains = [chain for chain in catalog() if chain.split == split]
    if helpers is not None and set(helpers) != {chain.family for chain in chains}:
        raise ValueError("complete same-split helper set required")
    return {
        "schema_version": 1,
        "split": split,
        "gradient_allowed": False,
        "stage": stage + 1,
        "cases": [
            {
                "id": f"chain-{chain.family}-{stage + 1}",
                "split": split,
                "family": chain.family,
                "prompt": chain.prompt(
                    stage, None if helpers is None else helpers[chain.family]
                ),
            }
            for chain in chains
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("bundle", "evaluate"), default="bundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--base-sha256")
    parser.add_argument("--seed", type=int, default=64123)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--split", choices=("train", "transfer"), default="transfer")
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--docker-socket", type=Path)
    parser.add_argument("--daemon-id")
    parser.add_argument("--image")
    parser.add_argument("--sandbox-seconds", type=float, default=30)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.action == "evaluate":
        import re

        if not 4 <= args.trials <= 100 or args.trials % 2:
            raise ValueError("an even 4..100 trials required")
        chains = [chain for chain in catalog() if chain.split == args.split]
        preview = {
            "status": "preview",
            "chains": len(chains),
            "calls_max": 3 * len(chains),
            "mode": "helper; independently generated oracle-helper and generated-helper downstream",
            "model_calls": 0,
            "hidden_reasoning_saved": False,
            "public_answers_saved": False,
        }
        if not args.run:
            print(json.dumps(preview))
            return
        url = urlsplit(args.endpoint or "")
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or not args.model
            or not re.fullmatch("[0-9a-f]{64}", args.checkpoint_sha256 or "")
        ):
            raise ValueError("explicit endpoint/model/checkpoint SHA required")
        if not (
            1 <= args.max_tokens <= 32768
            and 1 <= args.timeout <= 1200
            and 1 <= args.sandbox_seconds <= 120
        ):
            raise ValueError("invalid request/execution bounds")
        if (
            not args.docker_socket
            or not args.daemon_id
            or not re.fullmatch("sha256:[0-9a-f]{64}", args.image or "")
        ):
            raise ValueError("explicit pinned sandbox socket/daemon/image required")
        output = args.output.resolve()
        sources = [
            Path(__file__).resolve(),
            Path(sandbox.__file__).resolve(),
            Path(generator.__file__).resolve(),
        ]
        if any(
            path.is_relative_to(output) or output.is_relative_to(path)
            for path in sources
        ):
            raise ValueError("output overlaps source")
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        before = {str(path): digest(path.read_bytes()) for path in sources}
        runner = sandbox.Docker(
            args.docker_socket,
            args.daemon_id,
            args.image,
            seconds=args.sandbox_seconds,
            memory_mib=1024,
            cpus=1,
        )
        runner.check_image()
        output.mkdir(parents=True, exist_ok=False)
        report = {
            "status": "running",
            "model_id": args.model,
            "checkpoint_sha256": args.checkpoint_sha256,
            "checkpoint_identity": "caller-bound; establish serving identity externally",
            "sampling": "server config; native thinking on; seed and max_tokens explicit",
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "source_sha256": before,
            "chains": [],
        }
        try:
            for chain in chains:
                report["chains"].append(
                    evaluate_requests(
                        chain,
                        lambda row: generator.generate(row, args, urlopen),
                        runner.run,
                        seed=args.seed,
                        trials=args.trials,
                    )
                )
                if before != {str(path): digest(path.read_bytes()) for path in sources}:
                    raise ValueError("source changed")
                (output / "numeric-results.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
            report["status"] = "completed"
        except BaseException:
            report["status"] = "interrupted_or_error"
            raise
        finally:
            (output / "numeric-results.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        print(json.dumps({"status": report["status"], "chains": len(report["chains"])}))
        return
    report = write_bundle(
        args.output,
        base=args.base,
        base_sha256=args.base_sha256,
        seed=args.seed,
        trials=args.trials,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("training_families", "training_stages", "transfer_families")
            }
        )
    )


if __name__ == "__main__":
    main()
