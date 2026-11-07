#!/usr/bin/env python3
"""Eight authored library-transfer families; never gradient input.

Only trusted embedded references execute here. Public model programs belong in
science_sandbox's explicitly selected CPU container.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np

import science_cases as shared


@dataclass(frozen=True)
class Case(shared.Case):
    allow_underflow = True

    def public(self) -> dict:
        if self.split != "transfer":
            raise ValueError("library transfer cases cannot be training inputs")
        prompt = (
            "Write a Python function solve("
            + self.parameters
            + "). "
            + self.requirement
            + " NumPy, SciPy and SymPy may be used. Return a float64 ndarray. "
            "Do not mutate any input. Include imports and the function only, no prose."
        )
        correct, foil = self.code(), self.code(foil=True)
        if max(len((prompt + code).encode()) for code in (correct, foil)) > 2048:
            raise ValueError("authored prompt/continuation exceeds byte budget")
        return {
            "id": "science-" + self.name,
            "family": self.name,
            "split": "transfer",
            "gradient_allowed": False,
            "prompt": prompt,
            "correct": correct,
            "foil": foil,
        }


def catalog(revision: int = 1) -> list[shared.Case]:
    if revision != 1:
        raise ValueError("unsupported library-transfer revision")
    rows = [
        (
            "event-integral",
            "rate, initial, level, horizon",
            "For positive rate and initial>level>0, integrate y'=-rate*y and z'=y "
            "from (initial,0). Stop at the first downward y=level event or horizon. "
            "Return [stop_time,y,z,event_hit] with event_hit 0 or 1.",
            """from scipy.integrate import solve_ivp
def event(t, state):
    return state[0] - level
event.terminal = True
event.direction = -1
with np.errstate(under="ignore"):
    out = solve_ivp(lambda t, s: [-rate*s[0], s[0]], (0., horizon),
                    [initial, 0.], events=event, rtol=1e-10, atol=1e-12)
return np.array([out.t[-1], out.y[0,-1], out.y[1,-1],
                 float(len(out.t_events[0]) > 0)], dtype=np.float64)""",
            "event.direction = -1",
            "event.direction = 1",
        ),
        (
            "metric-eigenvectors",
            "a, b",
            "Given symmetric n-by-n a and positive-definite b with distinct "
            "generalized eigenvalues, solve a*v=lambda*b*v. Return shape(n+1,n), "
            "first row ascending eigenvalues, remaining rows eigenvectors as columns. "
            "Normalize v.T*b*v=I and make each column's largest-absolute entry positive.",
            """from scipy.linalg import eigh
values, vectors = eigh(a, b)
pivots = np.argmax(np.abs(vectors), axis=0)
signs = np.where(vectors[pivots, np.arange(len(values))] < 0, -1., 1.)
return np.vstack((values, vectors * signs)).astype(np.float64)""",
            "eigh(a, b)",
            "eigh(a)",
        ),
        (
            "pchip-clamped-derivative",
            "x, y, query",
            "x is strictly increasing and y is nondecreasing, both 1D length>=3. "
            "Use shape-preserving PCHIP on [x[0],x[-1]]. Return (len(query),3) "
            "columns value, first derivative, inside flag. Outside clamp value to "
            "the nearest endpoint and set derivative/flag to zero. Endpoints are inside.",
            """from scipy.interpolate import PchipInterpolator
curve = PchipInterpolator(x, y, extrapolate=False)
q = np.clip(query, x[0], x[-1])
inside = (query >= x[0]) & (query <= x[-1])
return np.column_stack((curve(q), np.where(inside, curve.derivative()(q), 0.),
                        inside.astype(float))).astype(np.float64)""",
            "(query >= x[0]) & (query <= x[-1])",
            "np.ones(query.shape, dtype=bool)",
        ),
        (
            "filter-state-response",
            "b, a, signal, previous_y, previous_x, omega",
            "b and a each have length3 with a[0]!=0 and stable denominator. "
            "Filter signal using sum_j a[j]*y[t-j]=sum_j b[j]*x[t-j]. "
            "previous_y and previous_x each store two samples, most recent first. "
            "Return concatenate(filtered_signal, real(H), imag(H)); H is the "
            "frequency response at omega radians/sample using exp(-1j*omega*j).",
            """from scipy.signal import lfiltic, lfilter, freqz
bn, an = np.asarray(b, float)/a[0], np.asarray(a, float)/a[0]
state = lfiltic(bn, an, previous_y, previous_x)
filtered, _ = lfilter(bn, an, signal, zi=state)
_, response = freqz(bn, an, worN=omega)
return np.concatenate((filtered, response.real, response.imag)).astype(np.float64)""",
            "lfiltic(bn, an, previous_y, previous_x)",
            "np.zeros(2)",
        ),
        (
            "labeled-directed-paths",
            "labels, edges, source",
            "labels contains distinct integer vertex labels in requested output order. "
            "edges is (m,3): source_label,target_label,positive_weight. For duplicate "
            "directed edges use their minimum weight. Return (len(labels),2) columns "
            "shortest distance from source and reachable flag; unreachable is [-1,0].",
            """from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
index = {int(label): i for i, label in enumerate(labels)}
adj = np.full((len(labels), len(labels)), np.inf)
for left, right, weight in edges:
    i, j = index[int(left)], index[int(right)]
    adj[i, j] = min(adj[i, j], float(weight))
distance = shortest_path(csr_matrix(adj), directed=True, indices=index[int(source)])
reachable = np.isfinite(distance)
return np.column_stack((np.where(reachable, distance, -1.),
                        reachable.astype(float))).astype(np.float64)""",
            "min(adj[i, j], float(weight))",
            "float(weight)",
        ),
        (
            "symbolic-axis-jacobian",
            "a, b, x",
            "a has shape(k,n), b shape(m,k), x shape(k,). Define "
            "C[i,j]=sum_l a[l,i]*x[l]*b[j,l]. Return shape(n,m,k+1): "
            "channel0 C, channel1+l partial C/partial x[l]. Preserve the stated axis order.",
            """import sympy as sp
variables = sp.symbols("u:" + str(len(x)))
expr = sp.Matrix(a.T) * sp.diag(*variables) * sp.Matrix(b.T)
flat = sp.Matrix(list(expr))
value = np.asarray(sp.lambdify([variables], flat, "numpy")(x), float).reshape(a.shape[1], b.shape[0])
jac = np.asarray(sp.lambdify([variables], flat.jacobian(variables), "numpy")(x), float)
return np.concatenate((value[...,None], jac.reshape(a.shape[1], b.shape[0], len(x))), axis=2).astype(np.float64)""",
            "sp.Matrix(list(expr))",
            "sp.Matrix(list(expr.T))",
        ),
        (
            "injected-bootstrap-covariance",
            "x, indices",
            "x is (n,d); indices is an integer (B,n) table, B>=2, explicitly "
            "specifying each bootstrap resample with replacement. Do not draw new "
            "random numbers. Return shape(d+1,d): mean of the B resample means "
            "followed by their sample covariance (denominator B-1).",
            """means = np.asarray(x, float)[indices].mean(axis=1)
center = means.mean(axis=0)
residual = means - center
covariance = residual.T @ residual / (len(means) - 1)
return np.vstack((center, covariance)).astype(np.float64)""",
            "(len(means) - 1)",
            "len(means)",
        ),
        (
            "composite-correlated-sensitivity",
            "x, covariance",
            "x has length3 and covariance is symmetric positive-definite (3,3). "
            "Let z=[exp(x[0])/(1+x[1]**2), x[0]*x[2]+sin(x[1])], then "
            "y=[z[0]**2+3*z[1], z[0]*z[1]]. Return (2,6) columns y, "
            "Jacobian dy/dx (3 columns), and J*covariance*J.T (2 columns), "
            "retaining all cross-covariances.",
            """def transform(v):
    z0 = np.exp(v[0])/(1+v[1]**2)
    z1 = v[0]*v[2]+np.sin(v[1])
    return np.array([z0*z0+3*z1, z0*z1])
jac = np.empty((2, 3))
for j in range(3):
    v = np.asarray(x, complex).copy()
    v[j] += 1e-30j
    jac[:,j] = transform(v).imag/1e-30
propagated = jac @ covariance @ jac.T
return np.column_stack((transform(x), jac, propagated)).astype(np.float64)""",
            "jac @ covariance @ jac.T",
            "jac @ np.diag(np.diag(covariance)) @ jac.T",
        ),
    ]
    return [
        Case(
            "library-transfer-" + name,
            "transfer",
            "library_transfer",
            parameters,
            requirement,
            body,
            old,
            new,
        )
        for name, parameters, requirement, body, old, new in rows
    ]


def _pchip_reference(x, y, query):
    h, slope = np.diff(x), np.diff(y) / np.diff(x)
    derivative = np.zeros(len(x))
    for i in range(1, len(x) - 1):
        if slope[i - 1] * slope[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            derivative[i] = (w1 + w2) / (w1 / slope[i - 1] + w2 / slope[i])
    for endpoint, h0, h1, d0, d1 in (
        (0, h[0], h[1], slope[0], slope[1]),
        (-1, h[-1], h[-2], slope[-1], slope[-2]),
    ):
        value = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(value) != np.sign(d0):
            value = 0.0
        elif np.sign(d0) != np.sign(d1) and abs(value) > 3 * abs(d0):
            value = 3 * d0
        derivative[endpoint] = value
    output = []
    for q in query:
        if q < x[0] or q > x[-1]:
            output.append([y[0] if q < x[0] else y[-1], 0.0, 0.0])
            continue
        i = min(max(np.searchsorted(x, q, side="right") - 1, 0), len(x) - 2)
        t = (q - x[i]) / h[i]
        value = (
            (2 * t**3 - 3 * t * t + 1) * y[i]
            + (t**3 - 2 * t * t + t) * h[i] * derivative[i]
            + (-2 * t**3 + 3 * t * t) * y[i + 1]
            + (t**3 - t * t) * h[i] * derivative[i + 1]
        )
        grad = (
            (6 * t * t - 6 * t) * y[i]
            + (3 * t * t - 4 * t + 1) * h[i] * derivative[i]
            + (-6 * t * t + 6 * t) * y[i + 1]
            + (3 * t * t - 2 * t) * h[i] * derivative[i + 1]
        ) / h[i]
        output.append([value, grad, 1.0])
    return np.asarray(output)


def sample(name: str, rng: np.random.Generator, trial: int) -> tuple[tuple, np.ndarray]:
    args: tuple
    ref: Any
    name = name.removeprefix("library-transfer-")
    if name == "event-integral":
        rate, initial = rng.uniform(0.3, 1.7), rng.uniform(2.0, 5.0)
        level = initial * rng.uniform(0.15, 0.65)
        event_time = np.log(initial / level) / rate
        horizon = event_time * (0.7 if trial % 2 else 1.4)
        t = min(event_time, horizon)
        y = initial * np.exp(-rate * t)
        args = (rate, initial, level, horizon)
        ref = [t, y, (initial - y) / rate, float(horizon >= event_time)]
    elif name == "metric-eigenvectors":
        n = 2 + trial % 3
        basis = np.eye(n) * 2 + rng.normal(size=(n, n)) * 0.15
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        values = np.arange(n) * 1.3 + rng.uniform(-0.2, 0.2, n)
        b = basis.T @ basis
        a = basis.T @ q @ np.diag(values) @ q.T @ basis
        vectors = np.linalg.solve(basis, q)
        vectors *= np.where(
            vectors[np.argmax(abs(vectors), axis=0), np.arange(n)] < 0, -1.0, 1.0
        )
        args, ref = (a, b), np.vstack((values, vectors))
    elif name == "pchip-clamped-derivative":
        n = 4 + trial % 4
        x = np.cumsum(rng.uniform(0.2, 1.2, n))
        increments = rng.uniform(0.1, 1.0, n)
        increments[2] = 0.0
        y = np.cumsum(increments)
        query = np.r_[x[0] - 0.4, x, rng.uniform(x[0], x[-1], 5), x[-1] + 0.6]
        args, ref = (x, y, query), _pchip_reference(x, y, query)
    elif name == "filter-state-response":
        poles = rng.uniform(0.2, 0.6, 2)
        scale = rng.uniform(1.3, 3.0)
        a = np.array([1.0, -sum(poles), np.prod(poles)]) * scale
        b = rng.uniform(0.1, 0.5, 3) * scale
        signal = rng.normal(size=5 + trial % 4)
        py, px = rng.normal(size=2), rng.normal(size=2)
        omega = np.array([0.0, 0.4, 1.7, np.pi])
        output = []
        xs, ys = list(px[::-1]), list(py[::-1])
        for value in signal:
            result = (
                b[0] * value
                + b[1] * xs[-1]
                + b[2] * xs[-2]
                - a[1] * ys[-1]
                - a[2] * ys[-2]
            ) / a[0]
            output.append(result)
            xs.append(value)
            ys.append(result)
        response = np.asarray(
            [
                sum(b[j] * np.exp(-1j * w * j) for j in range(3))
                / sum(a[j] * np.exp(-1j * w * j) for j in range(3))
                for w in omega
            ]
        )
        args = (b, a, signal, py, px, omega)
        ref = np.r_[output, response.real, response.imag]
    elif name == "labeled-directed-paths":
        labels = rng.choice(np.arange(10, 500), size=5, replace=False)
        u, v, w, z, _ = labels
        weights = rng.uniform(0.2, 2.0, 4)
        edges = np.asarray(
            [
                [u, v, weights[0]],
                [v, w, weights[1]],
                [u, w, 8.0],
                [w, z, weights[2]],
                [z, v, weights[3]],
                [u, v, weights[0] + 3.0],
            ]
        )
        distance = {int(label): np.inf for label in labels}
        distance[int(u)] = 0.0
        for _ in range(len(labels) - 1):
            for left, right, weight in edges:
                distance[int(right)] = min(
                    distance[int(right)], distance[int(left)] + weight
                )
        ref = [
            [distance[int(label)], 1.0]
            if np.isfinite(distance[int(label)])
            else [-1.0, 0.0]
            for label in labels
        ]
        args = (labels, edges, int(u))
    elif name == "symbolic-axis-jacobian":
        k, n, m = 2 + trial % 2, 2, 3
        a, b, x = rng.normal(size=(k, n)), rng.normal(size=(m, k)), rng.normal(size=k)
        ref = np.empty((n, m, k + 1))
        for i in range(n):
            for j in range(m):
                ref[i, j, 1:] = [a[axis, i] * b[j, axis] for axis in range(k)]
                ref[i, j, 0] = sum(ref[i, j, axis + 1] * x[axis] for axis in range(k))
        args = (a, b, x)
    elif name == "injected-bootstrap-covariance":
        n, d, count = 2 + trial % 4, 2 + trial % 2, 5 + trial % 3
        x = rng.normal(size=(n, d))
        indices = rng.integers(0, n, (count, n))
        means = np.asarray(
            [[sum(x[int(i), j] for i in row) / n for j in range(d)] for row in indices]
        )
        center = np.asarray([sum(means[:, j]) / count for j in range(d)])
        covariance = sum(np.outer(row - center, row - center) for row in means) / (
            count - 1
        )
        args, ref = (x, indices), np.vstack((center, covariance))
    elif name == "composite-correlated-sensitivity":
        x = rng.uniform(-0.8, 0.8, 3)
        factor = rng.normal(size=(3, 3))
        covariance = factor @ factor.T + np.eye(3) * 0.2
        z0, z1 = np.exp(x[0]) / (1 + x[1] ** 2), x[0] * x[2] + np.sin(x[1])
        first = np.array(
            [[z0, -2 * x[1] * z0 / (1 + x[1] ** 2), 0.0], [x[2], np.cos(x[1]), x[0]]]
        )
        second = np.array([[2 * z0, 3.0], [z1, z0]])
        jac = second @ first
        ref = np.column_stack(
            ([z0 * z0 + 3 * z1, z0 * z1], jac, jac @ covariance @ jac.T)
        )
        args = (x, covariance)
    else:
        raise ValueError("unknown library transfer family: " + name)
    return args, np.asarray(ref, dtype=np.float64)


def metamorphic_validate(*, seed: int, trials: int) -> list[dict]:
    """Relations use transformed inputs and do not share implementation code."""
    rows = []
    transformed: tuple
    for case in catalog():
        namespace: dict = {}
        exec(compile(case.code(), "<trusted-library-transfer>", "exec"), namespace)
        solve = namespace["solve"]
        rng = np.random.default_rng(
            seed
            + int.from_bytes(hashlib.sha256(case.name.encode()).digest()[:4], "big")
        )
        for trial in range(trials):
            args, _ = sample(case.name, rng, trial)
            original = solve(*args)
            name = case.name.removeprefix("library-transfer-")
            if name == "event-integral":
                rate, initial, level, horizon = args
                transformed = (2 * rate, initial, level, horizon / 2)
                expected = original.copy()
                expected[[0, 2]] /= 2
            elif name == "metric-eigenvectors":
                a, b = args
                transformed = (3 * a, 4 * b)
                expected = original.copy()
                expected[0] *= 0.75
                expected[1:] /= 2
            elif name == "pchip-clamped-derivative":
                x, y, query = args
                transformed = (x, 2 * y + 3, query)
                expected = original.copy()
                expected[:, 0] = 2 * expected[:, 0] + 3
                expected[:, 1] *= 2
            elif name == "filter-state-response":
                b, a, signal, py, px, omega = args
                transformed, expected = (3 * b, 3 * a, signal, py, px, omega), original
            elif name == "labeled-directed-paths":
                labels, edges, source = args
                permutation = rng.permutation(len(labels))
                transformed = (labels[permutation], edges[::-1].copy(), source)
                expected = original[permutation]
            elif name == "symbolic-axis-jacobian":
                a, b, x = args
                order = np.arange(len(x))[::-1]
                transformed = (a[order], b[:, order], x[order])
                expected = np.concatenate(
                    (original[..., :1], original[..., 1:][..., order]), axis=2
                )
            elif name == "injected-bootstrap-covariance":
                x, indices = args
                transformed = (2 * x + 3, indices)
                expected = original.copy()
                expected[0] = 2 * expected[0] + 3
                expected[1:] *= 4
            else:
                x, covariance = args
                transformed = (x, 3 * covariance)
                expected = original.copy()
                expected[:, 4:] *= 3
            np.testing.assert_allclose(
                solve(*transformed), expected, rtol=2e-6, atol=2e-7
            )
        rows.append({"family": case.name, "passed": trials, "failed": 0})
    return rows


def validate(*, seed: int = 20261109, trials: int = 8) -> dict:
    report = shared.validate(catalog(), seed=seed, trials=trials, sample_fn=sample)
    report["metamorphic"] = metamorphic_validate(seed=seed + 1, trials=trials)
    report["libraries"] = {
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "sympy")
    }
    return report


def write_bundle(output: Path, *, seed: int = 20261109, trials: int = 8) -> dict:
    metadata = shared.write_bundle(
        output,
        seed=seed,
        trials=trials,
        cases=catalog(),
        sample_fn=sample,
        source_paths=(Path(__file__),),
        authoring="New authored library transfer problems and independent mathematical oracles; no official benchmark material read or imported.",
    )
    metadata["catalog"] = "library-transfer-v1"
    metadata["validation"]["metamorphic"] = metamorphic_validate(
        seed=seed + 1, trials=trials
    )
    metadata["validation"]["libraries"] = {
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "sympy")
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()
    if args.write and args.output is None:
        parser.error("--write requires --output")
    result = (
        write_bundle(args.output, seed=args.seed, trials=args.trials)["validation"]
        if args.write
        else validate(seed=args.seed, trials=args.trials)
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": len(result["cases"]),
                "libraries": result["libraries"],
            }
        )
    )


if __name__ == "__main__":
    main()
