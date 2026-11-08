#!/usr/bin/env python3
"""Sixteen trusted library-composition training examples, eight families.

Only source-owned correct/foil programs execute. This is not a sandbox. Existing
training cases are copied without executing or modifying their contents.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
from typing import Any, ClassVar

import numpy as np


@dataclass(frozen=True)
class Case:
    split: ClassVar[str] = "train"
    allow_underflow: ClassVar[bool] = True
    family: str
    variant: str
    purpose: str
    fold: int
    parameters: str
    requirement: str
    imports: str
    body: str
    old: str
    new: str

    @property
    def name(self) -> str:
        return f"library-{self.family}-{self.variant}"

    def code(self, *, foil: bool = False) -> str:
        body = self.body
        if foil:
            if body.count(self.old) != 1:
                raise ValueError(f"ambiguous foil mutation: {self.name}")
            body = body.replace(self.old, self.new)
        return (
            "import numpy as np\n"
            + self.imports
            + "\n\ndef solve("
            + self.parameters
            + "):\n"
            + "\n".join("    " + line for line in body.splitlines())
            + "\n"
        )

    def public(self) -> dict[str, Any]:
        prompt = (
            f"Write Python solve({self.parameters}). "
            + self.requirement
            + " Inputs satisfy the stated conditions. Return a float64 ndarray; "
            "do not mutate inputs. Include required imports and complete function, no prose."
        )
        correct, foil = self.code(), self.code(foil=True)
        if max(len((prompt + c).encode()) for c in (correct, foil)) > 3072:
            raise ValueError(f"text budget exceeded: {self.name}")
        return {
            "id": "science-" + self.name,
            "family": "library-" + self.family,
            "variant": self.variant,
            "split": "train",
            "gradient_allowed": True,
            "prompt": prompt,
            "correct": correct,
            "foil": foil,
        }


def catalog(*, revision: int = 1) -> list[Case]:
    if revision != 1:
        raise ValueError("library training catalog revision must be 1")
    rows = [
        (
            "sparse-solve",
            "fixed-boundary",
            "science_numerics",
            0,
            "rows, cols, values, n, b, fixed, prescribed",
            "COO entries define a symmetric positive-definite n by n matrix, with duplicates summed. "
            "fixed contains unique valid indices, a nonempty proper subset. Set x[fixed]=prescribed "
            "and solve A@x=b at free indices only. Return shape(n,).",
            "from scipy.sparse import coo_matrix\nfrom scipy.sparse.linalg import spsolve",
            "a = coo_matrix((values, (rows, cols)), shape=(n,n)).tocsr()\n"
            "x = np.zeros(n); x[fixed] = prescribed\n"
            "free = np.setdiff1d(np.arange(n), fixed)\n"
            "rhs = b[free] - a[free][:,fixed] @ prescribed\n"
            "x[free] = spsolve(a[free][:,free], rhs)\nreturn x",
            "b[free] - a[free][:,fixed] @ prescribed",
            "b[free]",
        ),
        (
            "sparse-solve",
            "banded-factor",
            "science_numerics",
            1,
            "lower, diagonal, upper, b",
            "lower and upper have length n-1 and describe the sub/super diagonals of a "
            "strictly diagonally-dominant tridiagonal matrix. b has shape(n,k). Factor once and solve all columns. Return shape(n,k).",
            "from scipy.sparse import diags\nfrom scipy.sparse.linalg import factorized",
            "a = diags([lower, diagonal, upper], [-1,0,1], format='csc')\n"
            "solve_column = factorized(a)\n"
            "return np.column_stack([solve_column(b[:,j]) for j in range(b.shape[1])]).astype(np.float64)",
            "[lower, diagonal, upper]",
            "[upper, diagonal, lower]",
        ),
        (
            "residual-jacobian",
            "decay-fit",
            "science_advanced_state",
            0,
            "t, y, initial",
            "Noiseless y=a*exp(-k*t)+b at at least three distinct nonnegative t, with a,k positive. "
            "initial is a length3 feasible guess with initial[0:2]>=1e-8. "
            "Use SciPy least_squares with the analytic Jacobian and positive a,k bounds. Return fitted [a,k,b].",
            "from scipy.optimize import least_squares",
            "def residual(p): return p[0]*np.exp(-p[1]*t)+p[2]-y\n"
            "def jac(p):\n    e = np.exp(-p[1]*t)\n"
            "    return np.column_stack((e, -p[0]*t*e, np.ones_like(t)))\n"
            "fit = least_squares(residual, initial, jac=jac, bounds=([1e-8,1e-8,-np.inf],np.inf), "
            "xtol=1e-12, ftol=1e-12, gtol=1e-12)\nreturn fit.x.astype(np.float64)",
            "p[0]*np.exp(-p[1]*t)+p[2]-y",
            "p[0]*np.exp(p[1]*t)+p[2]-y",
        ),
        (
            "residual-jacobian",
            "weighted-affine",
            "science_advanced_state",
            1,
            "t, y, weight",
            "Fit y approximately slope*t+intercept minimizing sum(weight*residual**2), "
            "with positive per-observation weight. Compose weighted residual and analytic Jacobian for SciPy least_squares. Return [slope,intercept].",
            "from scipy.optimize import least_squares",
            "root = np.sqrt(weight)\n"
            "def residual(p): return root*(p[0]*t+p[1]-y)\n"
            "def jac(p): return np.column_stack((root*t, root))\n"
            "fit = least_squares(residual, np.zeros(2), jac=jac, gtol=1e-12, ftol=1e-12, xtol=1e-12)\n"
            "return fit.x.astype(np.float64)",
            "np.sqrt(weight)",
            "weight",
        ),
        (
            "root-contract",
            "implicit-sensitivity",
            "science_advanced_state",
            0,
            "a, y, left, right",
            "abs(a)<0.5 and [left,right] brackets the unique root of x+a*sin(x)=y. "
            "Use a bracketed SciPy solver; return [x,dx/dy] using implicit differentiation, shape(2,).",
            "from scipy.optimize import brentq",
            "x = brentq(lambda z: z+a*np.sin(z)-y, left, right, xtol=1e-13)\n"
            "return np.array([x, 1/(1+a*np.cos(x))], dtype=np.float64)",
            "1/(1+a*np.cos(x))",
            "1+a*np.cos(x)",
        ),
        (
            "root-contract",
            "log-coordinate",
            "science_advanced_state",
            1,
            "y, low, high",
            "0<low<high bracket the solution of x+log(x)=y. Solve for u=log(x) "
            "using SciPy brentq, then return physical x, shape(1,), not the transformed coordinate.",
            "from scipy.optimize import brentq",
            "u = brentq(lambda u: np.exp(u)+u-y, np.log(low), np.log(high), xtol=1e-13)\n"
            "return np.array([np.exp(u)], dtype=np.float64)",
            "[np.exp(u)]",
            "[u]",
        ),
        (
            "quadrature-callback",
            "piecewise-moments",
            "science_advanced_simulation",
            0,
            "edges, slope, intercept",
            "On bin i, positive density is slope[i]*x+intercept[i]; edges strictly increase. "
            "Integrate each bin with SciPy quad and combine. Return [total mass, normalized first moment], shape(2,).",
            "from scipy.integrate import quad",
            "mass = 0.0; first = 0.0\n"
            "for lo,hi,a,b in zip(edges[:-1],edges[1:],slope,intercept):\n"
            "    mass += quad(lambda x: a*x+b, lo, hi)[0]\n"
            "    first += quad(lambda x: x*(a*x+b), lo, hi)[0]\n"
            "return np.array([mass, first/mass], dtype=np.float64)",
            "x*(a*x+b)",
            "a*x+b",
        ),
        (
            "quadrature-callback",
            "domain-transform",
            "science_advanced_simulation",
            1,
            "rate, low, high",
            "rate>0 and 0<=low<high. Map x=low+(high-low)*u and integrate on u in[0,1] "
            "using SciPy quad. Return [integral(exp(-rate*x)), conditional mean x] over[low,high].",
            "from scipy.integrate import quad",
            "width = high-low\n"
            "def density(u): return np.exp(-rate*(low+width*u))*width\n"
            "mass = quad(density, 0, 1)[0]\n"
            "first = quad(lambda u: (low+width*u)*density(u), 0, 1)[0]\n"
            "return np.array([mass, first/mass], dtype=np.float64)",
            "np.exp(-rate*(low+width*u))*width",
            "np.exp(-rate*(low+width*u))",
        ),
        (
            "symbolic-numeric",
            "variable-order",
            "science_contracts",
            0,
            "x, y, a",
            "For same-length vectors x,y, symbolically differentiate f=a*x**2*y+sin(y) "
            "using SymPy, then lambdify with variable order(x,y). Return rows[df/dx,df/dy], shape(len(x),2).",
            "import sympy as sp",
            "sx, sy = sp.symbols('x y')\nexpr = a*sx**2*sy+sp.sin(sy)\n"
            "gradient = sp.lambdify((sx,sy), (sp.diff(expr,sx),sp.diff(expr,sy)), 'numpy')\n"
            "dx,dy = gradient(x,y)\n"
            "return np.column_stack(np.broadcast_arrays(dx,dy)).astype(np.float64)",
            "sp.lambdify((sx,sy)",
            "sp.lambdify((sy,sx)",
        ),
        (
            "symbolic-numeric",
            "chain-coordinates",
            "science_contracts",
            1,
            "u, v",
            "Physical x=exp(u), y=u+v; f=x*y**2+sin(x). For same-length vectors u,v, "
            "use SymPy substitution and differentiation to return rows[df/du,df/dv], shape(len(u),2).",
            "import sympy as sp",
            "x,y,su,sv = sp.symbols('x y u v')\n"
            "expr = (x*y**2+sp.sin(x)).subs({x:sp.exp(su), y:su+sv})\n"
            "gradient = sp.lambdify((su,sv), (sp.diff(expr,su),sp.diff(expr,sv)), 'numpy')\n"
            "return np.column_stack(gradient(u,v)).astype(np.float64)",
            "x:sp.exp(su)",
            "x:su",
        ),
        (
            "spectral-contract",
            "linear-convolution",
            "science_numerics",
            0,
            "x, kernel",
            "Use scipy.fft real transforms and sufficient zero padding for full linear "
            "convolution of two nonempty real vectors. Return len(x)+len(kernel)-1 entries, no circular wraparound.",
            "from scipy.fft import rfft, irfft, next_fast_len",
            "size = len(x)+len(kernel)-1\nnfft = next_fast_len(size)\n"
            "spectrum = rfft(x,nfft)*rfft(kernel,nfft)\n"
            "return irfft(spectrum,nfft)[:size].astype(np.float64)",
            "rfft(kernel,nfft)",
            "np.conj(rfft(kernel,nfft))",
        ),
        (
            "spectral-contract",
            "periodic-correlation",
            "science_numerics",
            1,
            "x, y, spacing",
            "For real same-length periodic vectors, return c[k]=spacing*sum_t x[t]*y[(t-k)%n] "
            "for k=0..n-1 using scipy.fft. spacing>0; preserve this lag sign and physical scaling.",
            "from scipy.fft import fft, ifft",
            "spectrum = fft(x)*np.conj(fft(y))\n"
            "return (spacing*ifft(spectrum).real).astype(np.float64)",
            "np.conj(fft(y))",
            "fft(y)",
        ),
        (
            "optimizer-feasibility",
            "simplex-equality",
            "science_contracts",
            0,
            "b, mass",
            "Project real vector b onto x>=0 and sum(x)=mass, with mass>0. Use scipy.optimize "
            "SLSQP with analytic objective gradient and exact equality constraint. Return the projection, shape b.shape.",
            "from scipy.optimize import minimize, LinearConstraint",
            "n = len(b)\nconstraint = LinearConstraint(np.ones(n), mass, mass)\n"
            "fit = minimize(lambda x: 0.5*np.sum((x-b)**2), np.full(n,mass/n), "
            "jac=lambda x: x-b, bounds=[(0,None)]*n, constraints=constraint, method='SLSQP', "
            "options={'ftol':1e-12,'maxiter':200})\nreturn fit.x.astype(np.float64)",
            "np.ones(n), mass, mass",
            "np.ones(n), -np.inf, mass",
        ),
        (
            "optimizer-feasibility",
            "weighted-hyperplane",
            "science_contracts",
            1,
            "b, normal, target",
            "For nonzero normal, minimize 0.5*||x-b||**2 subject to normal@x=target. "
            "Use SLSQP with the analytic objective gradient and LinearConstraint, without sign bounds. Return shape b.shape.",
            "from scipy.optimize import minimize, LinearConstraint",
            "constraint = LinearConstraint(normal, target, target)\n"
            "fit = minimize(lambda x: 0.5*np.sum((x-b)**2), b.copy(), jac=lambda x: x-b, "
            "constraints=constraint, method='SLSQP', options={'ftol':1e-12,'maxiter':200})\n"
            "return fit.x.astype(np.float64)",
            "LinearConstraint(normal, target, target)",
            "LinearConstraint(np.ones_like(normal), target, target)",
        ),
        (
            "distribution-interface",
            "batched-logdensity",
            "science_advanced_probability",
            0,
            "x, location, scale",
            "For sample vector x and parameter vectors location,positive scale of length m, "
            "return normal log densities shape(len(x),m) using scipy.stats. scale denotes standard deviation, not variance.",
            "from scipy.stats import norm",
            "return norm.logpdf(x[:,None], loc=location[None,:], scale=scale[None,:]).astype(np.float64)",
            "scale=scale[None,:]",
            "scale=scale[None,:]**2",
        ),
        (
            "distribution-interface",
            "lognormal-moments",
            "science_advanced_probability",
            1,
            "probability, mean, standard_deviation",
            "For probabilities strictly between 0 and 1 and positive arithmetic mean/std, "
            "return lognormal quantiles via scipy.stats.lognorm, converting arithmetic moments to its log-space shape/scale.",
            "from scipy.stats import lognorm",
            "variance_log = np.log1p((standard_deviation/mean)**2)\n"
            "sigma = np.sqrt(variance_log)\nmu = np.log(mean)-variance_log/2\n"
            "return lognorm.ppf(probability, s=sigma, scale=np.exp(mu)).astype(np.float64)",
            "scale=np.exp(mu)",
            "scale=mean",
        ),
    ]
    return [Case(*r) for r in rows]


def sample(
    case: Case | str, rng: np.random.Generator, trial: int
) -> tuple[tuple[Any, ...], np.ndarray]:
    if isinstance(case, str):
        matches = [row for row in catalog() if row.name == case]
        if len(matches) != 1:
            raise ValueError("unknown library training case")
        case = matches[0]
    # The authored interfaces intentionally mix scalar and vector arguments.
    args: tuple[Any, ...]
    a: Any
    b: Any
    x: Any
    y: Any
    slope: Any
    mass: float
    first: float
    n = 4 + trial % 4
    family, variant = case.family, case.variant
    if family == "sparse-solve" and variant == "fixed-boundary":
        raw = rng.normal(size=(n, n))
        a = raw.T @ raw + np.eye(n)
        rows, cols = np.indices((n, n))
        values = a.ravel()
        # Duplicate COO entries are a real representation contract.
        rows = np.tile(rows.ravel(), 2)
        cols = np.tile(cols.ravel(), 2)
        values = np.concatenate((values * 0.4, values * 0.6))
        b = rng.normal(size=n)
        fixed = np.array([0, n - 1])
        prescribed = rng.uniform(0.5, 2, 2)
        free = np.arange(1, n - 1)
        ref = np.empty(n)
        ref[fixed] = prescribed
        ref[free] = np.linalg.solve(
            a[np.ix_(free, free)], b[free] - a[np.ix_(free, fixed)] @ prescribed
        )
        args = (rows, cols, values, n, b, fixed, prescribed)
    elif family == "sparse-solve":
        lower = rng.uniform(-1, -0.2, n - 1)
        upper = rng.uniform(0.3, 1.2, n - 1)
        diag = rng.uniform(3, 5, n)
        b = rng.normal(size=(n, 2 + trial % 2))
        dense = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)
        ref = np.linalg.solve(dense, b)
        args = (lower, diag, upper, b)
    elif family == "residual-jacobian" and variant == "decay-fit":
        t = np.linspace(0, 3, n + 5)
        truth = np.array(
            [rng.uniform(1, 2), rng.uniform(0.4, 0.9), rng.uniform(-0.5, 0.5)]
        )
        y = truth[0] * np.exp(-truth[1] * t) + truth[2]
        args = (t, y, truth * np.array([0.8, 1.2, 0.7]))
        ref = truth
    elif family == "residual-jacobian":
        t = np.linspace(-1, 2, n + 4)
        weight = rng.uniform(0.2, 3, len(t))
        y = 1.7 * t - 0.4 + rng.normal(0, 0.2, len(t))
        sw = sum(map(float, weight))
        mt = float(weight @ t / sw)
        my = float(weight @ y / sw)
        slope = sum(
            float(w * (x - mt) * (z - my)) for x, z, w in zip(t, y, weight)
        ) / sum(float(w * (x - mt) ** 2) for x, w in zip(t, weight))
        args = (t, y, weight)
        ref = np.array([slope, my - slope * mt])
    elif family == "root-contract" and variant == "implicit-sensitivity":
        a = rng.uniform(0.15, 0.45)
        x = rng.uniform(-1, 1)
        y = x + a * math.sin(x)
        args = (a, y, x - 1, x + 1)
        ref = np.array([x, 1 / (1 + a * math.cos(x))])
    elif family == "root-contract":
        x = rng.uniform(0.4, 3)
        args = (x + math.log(x), x * 0.3, x * 2)
        ref = np.array([x])
    elif family == "quadrature-callback" and variant == "piecewise-moments":
        edges = np.concatenate(([0.1], 0.1 + np.cumsum(rng.uniform(0.1, 0.5, n))))
        slope = rng.uniform(0.1, 0.5, n)
        intercept = rng.uniform(0.5, 2, n)
        mass = sum(
            a * (hi**2 - lo**2) / 2 + b * (hi - lo)
            for lo, hi, a, b in zip(edges[:-1], edges[1:], slope, intercept)
        )
        first = sum(
            a * (hi**3 - lo**3) / 3 + b * (hi**2 - lo**2) / 2
            for lo, hi, a, b in zip(edges[:-1], edges[1:], slope, intercept)
        )
        args = (edges, slope, intercept)
        ref = np.array([mass, first / mass])
    elif family == "quadrature-callback":
        rate = rng.uniform(0.3, 1.5)
        low = rng.uniform(0, 0.5)
        high = low + rng.uniform(1.4, 2.3)
        mass = (math.exp(-rate * low) - math.exp(-rate * high)) / rate
        first = (low / rate + 1 / rate**2) * math.exp(-rate * low) - (
            high / rate + 1 / rate**2
        ) * math.exp(-rate * high)
        args = (rate, low, high)
        ref = np.array([mass, first / mass])
    elif family == "symbolic-numeric" and variant == "variable-order":
        x = rng.uniform(0.2, 1.2, n)
        y = rng.uniform(-0.7, 0.1, n)
        a = 0.0 if trial == 0 else rng.uniform(0.5, 2)
        args = (x, y, a)
        ref = np.column_stack((2 * a * x * y, a * x * x + np.cos(y)))
    elif family == "symbolic-numeric":
        u = rng.uniform(-0.7, 0.5, n)
        v = rng.uniform(0.2, 1.1, n)
        x = np.exp(u)
        y = u + v
        args = (u, v)
        ref = np.column_stack((x * y * y + 2 * x * y + x * np.cos(x), 2 * x * y))
    elif family == "spectral-contract" and variant == "linear-convolution":
        x = rng.normal(size=n)
        kernel = rng.normal(size=n - 1)
        ref = np.zeros(2 * n - 2)
        for i, a in enumerate(x):
            for j, b in enumerate(kernel):
                ref[i + j] += a * b
        args = (x, kernel)
    elif family == "spectral-contract":
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        spacing = rng.uniform(0.2, 2)
        ref = np.array(
            [
                spacing * sum(float(x[t] * y[(t - k) % n]) for t in range(n))
                for k in range(n)
            ]
        )
        args = (x, y, spacing)
    elif family == "optimizer-feasibility" and variant == "simplex-equality":
        b = rng.uniform(-0.2, 0.2, n)
        mass = rng.uniform(2, 4)
        # All entries are active for these sampled cases; extra boundary cases alternate.
        if trial % 2:
            b[0] = rng.uniform(2, 3)
        ordered = np.sort(b)[::-1]
        partial = np.cumsum(ordered) - mass
        rho = np.flatnonzero(ordered - partial / np.arange(1, n + 1) > 0)[-1]
        theta = partial[rho] / (rho + 1)
        ref = np.maximum(b - theta, 0)
        args = (b, mass)
    elif family == "optimizer-feasibility":
        b = rng.normal(size=n)
        normal = rng.uniform(0.3, 2, n)
        target = rng.uniform(-1, 1)
        ref = b + normal * ((target - normal @ b) / (normal @ normal))
        args = (b, normal, target)
    elif family == "distribution-interface" and variant == "batched-logdensity":
        x = rng.normal(size=n)
        loc = rng.uniform(-1, 1, 3)
        scale = rng.uniform(0.3, 2, 3)
        ref = np.array(
            [
                [
                    -0.5 * ((xx - mm) / ss) ** 2
                    - math.log(ss)
                    - 0.5 * math.log(2 * math.pi)
                    for mm, ss in zip(loc, scale)
                ]
                for xx in x
            ]
        )
        args = (x, loc, scale)
    elif family == "distribution-interface":
        probability = np.linspace(0.05, 0.95, n)
        mean = rng.uniform(1, 3)
        sd = rng.uniform(0.4, 1.2)
        variance = math.log1p((sd / mean) ** 2)
        mu = math.log(mean) - variance / 2
        ref = np.array(
            [
                math.exp(mu + math.sqrt(variance) * NormalDist().inv_cdf(float(p)))
                for p in probability
            ]
        )
        args = (probability, mean, sd)
    else:
        raise ValueError(case.name)
    return args, np.asarray(ref, dtype=np.float64)


def execute(case: Case, *, foil: bool, seed: int, trials: int) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(
        compile(case.code(foil=foil), "<trusted-library-training-case>", "exec"),
        namespace,
    )
    function = namespace["solve"]
    if list(inspect.signature(function).parameters) != case.parameters.split(", "):
        raise ValueError("authored interface mismatch")
    rng = np.random.default_rng(seed)
    failures = []
    for trial in range(trials):
        args, reference = sample(case, rng, trial)
        copies = [v.copy() if isinstance(v, np.ndarray) else v for v in args]
        try:
            # SciPy may use harmless subnormal sentinels internally.
            with np.errstate(
                over="raise", divide="raise", invalid="raise", under="ignore"
            ):
                result = function(*args)
            if not isinstance(result, np.ndarray) or result.dtype != np.float64:
                raise AssertionError("output_type")
            if result.shape != reference.shape or not np.isfinite(result).all():
                raise AssertionError("output_shape_finite")
            np.testing.assert_allclose(result, reference, rtol=3e-6, atol=3e-7)
            for before, after in zip(copies, args):
                if isinstance(before, np.ndarray):
                    np.testing.assert_array_equal(before, after)
        except Exception as error:
            failures.append({"trial": trial, "type": type(error).__name__})
    return {
        "trials": trials,
        "passed": trials - len(failures),
        "failed": len(failures),
        "failures": failures,
    }


def validate(*, seed: int = 20261107, trials: int = 8) -> dict[str, Any]:
    if not 4 <= trials <= 100:
        raise ValueError("trials must be between4 and100")
    versions = {
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "sympy")
    }
    rows = []
    for case in catalog():
        public = case.public()
        case_seed = seed + int.from_bytes(
            hashlib.sha256(case.name.encode()).digest()[:4], "big"
        )
        correct = execute(case, foil=False, seed=case_seed, trials=trials)
        foil = execute(case, foil=True, seed=case_seed, trials=trials)
        if correct["failed"] or not foil["failed"]:
            raise ValueError(
                f"validation failed {case.name}: correct={correct}, foil={foil}"
            )
        rows.append(
            {
                "id": public["id"],
                "family": public["family"],
                "purpose": case.purpose,
                "fold": case.fold,
                "correct": correct,
                "foil": foil,
                "max_prompt_continuation_bytes": max(
                    len((public["prompt"] + public[k]).encode())
                    for k in ("correct", "foil")
                ),
                "correct_sha256": hashlib.sha256(
                    public["correct"].encode()
                ).hexdigest(),
                "foil_sha256": hashlib.sha256(public["foil"].encode()).hexdigest(),
            }
        )
    return {
        "status": "validated",
        "seed": seed,
        "trials_per_implementation": trials,
        "library_versions": versions,
        "cases": rows,
        "reference_method": "Analytic identities, dense small-instance solves, direct loops and standard-library inverse CDF, separate from candidate library calls.",
        "token_count_measured": False,
        "floating_policy": {
            "overflow": "raise",
            "divide": "raise",
            "invalid": "raise",
            "underflow": "ignore",
            "nonfinite_output": "reject",
        },
        "scope": "Trusted authored training only, not an official benchmark or unseen transfer score.",
    }


def training_payload() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for case in catalog():
        output.setdefault(case.purpose, [[], []])[case.fold].append(case.public())
    return output


def write_bundle(
    output: Path,
    *,
    base: Path | None = None,
    base_sha256: str | None = None,
    seed: int = 20261107,
    trials: int = 8,
) -> dict[str, Any]:
    if os.path.lexists(output):
        raise FileExistsError(output)
    if (base is None) != (base_sha256 is None):
        raise ValueError("base path and SHA must be supplied together")
    original: dict[str, Any] = {}
    base_bytes = b""
    if base is not None:
        if (
            output.resolve() == base.resolve()
            or output.resolve() in base.resolve().parents
        ):
            raise ValueError("output overlaps input")
        base_bytes = base.read_bytes()
        if hashlib.sha256(base_bytes).hexdigest() != base_sha256:
            raise ValueError("base SHA mismatch")
        original = json.loads(base_bytes)
        for folds in original.values():
            if not isinstance(folds, list) or len(folds) != 2:
                raise ValueError("base purpose/fold schema")
            for fold in folds:
                for row in fold:
                    if (
                        row.get("split", "train") != "train"
                        or row.get("gradient_allowed", True) is not True
                    ):
                        raise ValueError("base contains non-training cases")
    validation = validate(seed=seed, trials=trials)
    extension = training_payload()
    combined = json.loads(json.dumps(original))
    old_ids = {r["id"] for folds in original.values() for fold in folds for r in fold}
    for purpose, folds in extension.items():
        target = combined.setdefault(purpose, [[], []])
        for i, fold in enumerate(folds):
            for row in fold:
                if row["id"] in old_ids:
                    raise ValueError("duplicate training id")
                target[i].append(row)
    unchanged = all(
        combined[p][f][: len(fold)] == fold
        for p, folds in original.items()
        for f, fold in enumerate(folds)
    )
    if not unchanged:
        raise AssertionError("base objects changed")
    if base is not None and base.read_bytes() != base_bytes:
        raise ValueError("base changed during validation")
    metadata = {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "authoring": "Independent library/helper compositions authored without benchmark task/reference imports.",
        "new_case_count": len(catalog()),
        "new_family_count": len({c.family for c in catalog()}),
        "variants_per_family": 2,
        "old_case_count": len(old_ids),
        "combined_case_count": sum(len(f) for fs in combined.values() for f in fs),
        "purpose_count": len(combined),
        "fold_count": sum(len(fs) for fs in combined.values()),
        "base_objects_unchanged": unchanged,
        "base": {"path": str(base), "sha256": base_sha256} if base else None,
        "mapping": [
            {
                "family": c.family,
                "variant": c.variant,
                "purpose": c.purpose,
                "fold": c.fold,
            }
            for c in catalog()
        ],
        "validation": validation,
        "transfer_count": 0,
        "limits": [
            "Two variants per family are not two independent families.",
            "Numerical trials validate one program, not additional task diversity.",
            "No tokenization/model/generation/benchmark validation has occurred.",
            "This executor accepts only trusted programs in this source, not model code.",
        ],
    }
    output.mkdir(parents=True)
    for name, data in [
        ("train-extension.json", extension),
        ("train-cases.json", combined),
        ("metadata.json", metadata),
    ]:
        with (output / name).open("x") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--base-sha256")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()
    if args.write:
        if args.output is None:
            parser.error("--write requires --output")
        result = write_bundle(
            args.output,
            base=args.base,
            base_sha256=args.base_sha256,
            seed=args.seed,
            trials=args.trials,
        )
        print(
            json.dumps(
                {
                    "status": "validated",
                    "new_cases": result["new_case_count"],
                    "combined_cases": result["combined_case_count"],
                    "versions": result["validation"]["library_versions"],
                }
            )
        )
    else:
        result = validate(seed=args.seed, trials=args.trials)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "cases": len(result["cases"]),
                    "versions": result["library_versions"],
                }
            )
        )


if __name__ == "__main__":
    main()
