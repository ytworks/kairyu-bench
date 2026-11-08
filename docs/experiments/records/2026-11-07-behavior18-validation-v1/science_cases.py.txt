#!/usr/bin/env python3
"""Author-owned executable science calibration, separate from WSP evaluation.

Only the trusted programs defined in this source are executed. This is not a
sandbox or a runner for user-supplied code. No benchmark data or model is read.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class Case:
    name: str
    split: str
    purpose: str
    parameters: str
    requirement: str
    body: str
    old: str
    new: str

    def code(self, *, foil: bool = False) -> str:
        body = self.body
        if foil:
            if body.count(self.old) != 1:
                raise ValueError(f"ambiguous foil mutation: {self.name}")
            body = body.replace(self.old, self.new)
        return (
            "import numpy as np\n\ndef solve("
            + self.parameters
            + "):\n"
            + "\n".join("    " + line for line in body.splitlines())
            + "\n"
        )

    def public(self) -> dict:
        prompt = (
            "Write a NumPy-only Python function solve("
            + self.parameters
            + "). "
            + self.requirement
            + " Return a float64 ndarray with the specified shape. Do not mutate "
            "any input. Inputs satisfy the stated conditions. Include the complete "
            "function and NumPy import; no prose is needed."
        )
        correct, foil = self.code(), self.code(foil=True)
        # A conservative text-byte budget, not a tokenizer/framing measurement.
        if max(len((prompt + code).encode()) for code in (correct, foil)) > 2048:
            raise ValueError(f"text-byte budget exceeded: {self.name}")
        return {
            "id": "science-" + self.name,
            "family": self.name,
            "split": self.split,
            "gradient_allowed": self.split == "train",
            "prompt": prompt,
            "correct": correct,
            "foil": foil,
        }


def catalog() -> list[Case]:
    # The code is a general algorithm; randomized validation values never enter it.
    rows = [
        (
            "weighted-moments",
            "numerics",
            "x, w",
            "For nonempty vectors x and positive w, return [weighted mean, weighted "
            "population variance], normalized by sum(w), shape (2,).",
            "x = np.asarray(x, dtype=np.float64)\nw = np.asarray(w, dtype=np.float64)\n"
            "mean = np.dot(x, w) / w.sum()\n"
            "return np.array([mean, np.dot(w, (x-mean)**2) / w.sum()], dtype=np.float64)",
            "(x-mean)**2",
            "x**2",
        ),
        (
            "cumulative-trapezoid",
            "numerics",
            "t, x",
            "For strictly increasing sample times t and same-length values x, return "
            "the cumulative trapezoidal integral starting at zero, shape x.shape.",
            "t = np.asarray(t, dtype=np.float64)\nx = np.asarray(x, dtype=np.float64)\n"
            "areas = np.diff(t) * (x[1:] + x[:-1]) / 2\n"
            "return np.concatenate((np.zeros(1), np.cumsum(areas)))",
            " / 2",
            "",
        ),
        (
            "periodic-laplacian",
            "numerics",
            "x, dx",
            "Return the centered second spatial derivative on a periodic uniform "
            "one-dimensional grid with positive spacing dx, shape x.shape.",
            "x = np.asarray(x, dtype=np.float64)\n"
            "return (np.roll(x, 1) - 2*x + np.roll(x, -1)) / (dx*dx)",
            "(dx*dx)",
            "dx",
        ),
        (
            "row-softmax",
            "numerics",
            "x",
            "For a finite matrix x, subtract each row maximum and return normalized "
            "row exponentials, preserving matrix shape and remaining stable for large offsets.",
            "x = np.asarray(x, dtype=np.float64)\n"
            "e = np.exp(x - x.max(axis=1, keepdims=True))\n"
            "return e / e.sum(axis=1, keepdims=True)",
            "e.sum(axis=1, keepdims=True)",
            "e.sum(axis=0, keepdims=True)",
        ),
        (
            "ridge-intercept",
            "numerics",
            "x, y, penalty",
            "For design matrix x and response y, fit ridge coefficients with positive "
            "penalty and an unpenalized intercept using centered data. Return "
            "[intercept, coefficients...] of shape (x.shape[1]+1,).",
            "x = np.asarray(x, dtype=np.float64)\ny = np.asarray(y, dtype=np.float64)\n"
            "xm, ym = x.mean(axis=0), y.mean()\na = x-xm\n"
            "coef = np.linalg.solve(a.T@a + penalty*np.eye(x.shape[1]), a.T@(y-ym))\n"
            "return np.concatenate(([ym-xm@coef], coef))",
            "ym-xm@coef",
            "ym",
        ),
        (
            "markov-row-state",
            "numerics",
            "p, transition",
            "For row probability vector p and row-stochastic matrix transition, "
            "advance one Markov step then renormalize total mass. Return p.shape.",
            "p = np.asarray(p, dtype=np.float64)\n"
            "transition = np.asarray(transition, dtype=np.float64)\n"
            "out = p @ transition\nreturn out / out.sum()",
            "p @ transition",
            "transition @ p",
        ),
        (
            "orthogonal-residual",
            "numerics",
            "x, basis",
            "Columns of basis are orthonormal. Remove the projection of vector x onto "
            "their span and return the residual of shape x.shape.",
            "x = np.asarray(x, dtype=np.float64)\nbasis = np.asarray(basis, dtype=np.float64)\n"
            "return x - basis @ (basis.T @ x)",
            "x - basis",
            "x + basis",
        ),
        (
            "weighted-density",
            "numerics",
            "x, w, edges",
            "All samples x are within strictly increasing bin edges, with positive "
            "weights w. Use left-closed/right-open bins except the last includes its "
            "right edge. Return density: mass / total mass / bin width, shape "
            "(len(edges)-1,).",
            "x = np.asarray(x, dtype=np.float64)\nw = np.asarray(w, dtype=np.float64)\n"
            "edges = np.asarray(edges, dtype=np.float64)\n"
            "mass = np.histogram(x, bins=edges, weights=w)[0]\n"
            "return mass / (w.sum()*np.diff(edges))",
            "w.sum()*np.diff(edges)",
            "w.sum()",
        ),
        (
            "sample-covariance",
            "contracts",
            "x",
            "Rows are observations and columns are features; there are at least two "
            "observations. Center columns and return unbiased sample covariance, "
            "shape (features, features).",
            "x = np.asarray(x, dtype=np.float64)\na = x-x.mean(axis=0)\n"
            "return (a.T @ a) / (x.shape[0]-1)",
            "(x.shape[0]-1)",
            "x.shape[0]",
        ),
        (
            "conservative-flux",
            "contracts",
            "u, face_flux, dt, dx",
            "For cell averages u and face_flux of length len(u)+1, advance a "
            "finite-volume conservation step u - dt/dx*(right_flux-left_flux). "
            "dt and dx are positive. Return u.shape.",
            "u = np.asarray(u, dtype=np.float64)\nf = np.asarray(face_flux, dtype=np.float64)\n"
            "return u - (dt/dx) * (f[1:]-f[:-1])",
            "f[1:]-f[:-1]",
            "f[:-1]-f[1:]",
        ),
        (
            "implicit-decay",
            "contracts",
            "x, rate, dt",
            "For vector x, positive scalar rate and dt, take one implicit Euler step "
            "for dx/dt=-rate*x. Return x.shape.",
            "x = np.asarray(x, dtype=np.float64)\nreturn x / (1+dt*rate)",
            "1+dt*rate",
            "1-dt*rate",
        ),
        (
            "merge-population-moments",
            "contracts",
            "x, y",
            "Given two nonempty sample vectors, compute each batch mean and centered "
            "sum of squares, merge their statistics including the between-batch "
            "correction, and return [combined mean, population variance], shape (2,).",
            "x = np.asarray(x, dtype=np.float64)\ny = np.asarray(y, dtype=np.float64)\n"
            "n, m = x.size, y.size\nmx, my = x.mean(), y.mean()\nd = my-mx\n"
            "ss = ((x-mx)**2).sum()+((y-my)**2).sum()+d*d*n*m/(n+m)\n"
            "return np.array([(n*mx+m*my)/(n+m), ss/(n+m)], dtype=np.float64)",
            "+d*d*n*m/(n+m)",
            "",
        ),
        (
            "velocity-verlet",
            "contracts",
            "x, v, stiffness, dt",
            "For equal-length position x and velocity v with unit mass and force "
            "-stiffness*x, perform one velocity-Verlet step: half velocity, new "
            "position, new force, final half velocity. Return shape (2,len(x)): "
            "position then velocity.",
            "x = np.asarray(x, dtype=np.float64)\nv = np.asarray(v, dtype=np.float64)\n"
            "half = v - 0.5*dt*stiffness*x\npos = x+dt*half\n"
            "vel = half-0.5*dt*stiffness*pos\nreturn np.stack((pos, vel))",
            "stiffness*pos",
            "stiffness*x",
        ),
        (
            "segment-means",
            "contracts",
            "x, labels, groups",
            "labels are integer group indices in [0,groups), one per value x. "
            "Return each group's mean, with zero for empty groups, shape (groups,).",
            "x = np.asarray(x, dtype=np.float64)\nlabels = np.asarray(labels)\n"
            "count = np.bincount(labels, minlength=groups)\n"
            "total = np.bincount(labels, weights=x, minlength=groups)\n"
            "return np.divide(total, count, out=np.zeros(groups), where=count>0)",
            "total, count",
            "total, np.maximum(count-1, 1)",
        ),
        (
            "piecewise-linear",
            "contracts",
            "t, x, query",
            "For increasing knots t, values x, and in-range query points, return "
            "piecewise linear interpolation with endpoint values included, shape query.shape.",
            "t = np.asarray(t, dtype=np.float64)\nx = np.asarray(x, dtype=np.float64)\n"
            "query = np.asarray(query, dtype=np.float64)\n"
            "return np.interp(query, t, x)",
            "np.interp(query, t, x)",
            "np.interp(query, t, x).astype(np.float32)",
        ),
        (
            "batched-linear-system",
            "contracts",
            "a, b",
            "a has shape (batch,n,n) with invertible systems and b has shape "
            "(batch,n). Solve every system independently, returning b.shape.",
            "a = np.asarray(a, dtype=np.float64)\nb = np.asarray(b, dtype=np.float64)\n"
            "return np.linalg.solve(a, b[..., None])[..., 0]",
            "a, b[..., None]",
            "np.swapaxes(a, -1, -2), b[..., None]",
        ),
        (
            "kalman-joseph",
            "transfer",
            "x, covariance, observation, h, noise",
            "Perform a scalar-observation Kalman update with positive noise variance. "
            "Use innovation z-h@x and Joseph covariance update. Return updated state "
            "followed by flattened updated covariance, shape (n+n*n,).",
            "x = np.asarray(x, dtype=np.float64)\np = np.asarray(covariance, dtype=np.float64)\n"
            "h = np.asarray(h, dtype=np.float64)\nk = p@h/(h@p@h+noise)\n"
            "state = x+k*(observation-h@x)\na = np.eye(x.size)-np.outer(k, h)\n"
            "out = a@p@a.T+noise*np.outer(k, k)\n"
            "return np.concatenate((state, out.ravel()))",
            "observation-h@x",
            "observation",
        ),
        (
            "rodrigues-rotation",
            "transfer",
            "axis, angle",
            "For a nonzero 3-vector axis and scalar angle, normalize the axis and "
            "return its right-handed Rodrigues rotation matrix, shape (3,3).",
            "a = np.asarray(axis, dtype=np.float64)\na = a/np.linalg.norm(a)\nx,y,z = a\n"
            "k = np.array([[0,-z,y],[z,0,-x],[-y,x,0]], dtype=np.float64)\n"
            "return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)",
            "(1-np.cos(angle))",
            "(1+np.cos(angle))",
        ),
        (
            "log-hmm-filter",
            "transfer",
            "log_prior, log_transition, log_likelihood",
            "Transition rows are old states and columns are new states. Given log "
            "probabilities and new-state log likelihoods, propagate, condition and "
            "return normalized new-state log probabilities, shape log_prior.shape.",
            "p = np.asarray(log_prior, dtype=np.float64)\nt = np.asarray(log_transition, dtype=np.float64)\n"
            "ll = np.asarray(log_likelihood, dtype=np.float64)\n"
            "joint = p[:,None]+t\nm = joint.max(axis=0)\n"
            "post = m+np.log(np.exp(joint-m).sum(axis=0))+ll\n"
            "top = post.max()\nreturn post-top-np.log(np.exp(post-top).sum())",
            "p[:,None]+t",
            "p[:,None]+t.T",
        ),
        (
            "periodic-poisson",
            "transfer",
            "forcing, length",
            "Solve u''=forcing on a periodic uniform grid of positive length; forcing "
            "has zero mean. Use spectral differentiation and zero mean for u. "
            "Return forcing.shape.",
            "f = np.asarray(forcing, dtype=np.float64)\nk = 2*np.pi*np.fft.fftfreq(f.size, d=length/f.size)\n"
            "modes = np.fft.fft(f)\nout = np.zeros_like(modes)\n"
            "out[1:] = -modes[1:]/(k[1:]**2)\nreturn np.fft.ifft(out).real",
            "-modes[1:]",
            "modes[1:]",
        ),
        (
            "affine-conservation",
            "transfer",
            "x, c, target",
            "Project x onto c@y=target using the minimum Euclidean correction. "
            "c is nonzero. Return x.shape.",
            "x = np.asarray(x, dtype=np.float64)\nc = np.asarray(c, dtype=np.float64)\n"
            "return x + c*((target-c@x)/(c@c))",
            "(c@c)",
            "np.linalg.norm(c)",
        ),
        (
            "linear-noise-covariance",
            "transfer",
            "a, covariance, noise",
            "Propagate covariance through y=a@x plus independent process noise. "
            "a is square; covariance and noise are symmetric positive definite. "
            "Return a.shape and symmetrize roundoff.",
            "a = np.asarray(a, dtype=np.float64)\np = np.asarray(covariance, dtype=np.float64)\n"
            "q = np.asarray(noise, dtype=np.float64)\nout = a@p@a.T+q\n"
            "return (out+out.T)/2",
            "a@p@a.T+q",
            "a.T@p@a+q",
        ),
        (
            "barycentric-interpolation",
            "transfer",
            "nodes, values, weights, query",
            "nodes are distinct, weights are their barycentric polynomial weights, "
            "and query is a vector. Interpolate using weighted reciprocal distances; "
            "at an exact node return its value, shape query.shape.",
            "nodes = np.asarray(nodes, dtype=np.float64)\ny = np.asarray(values, dtype=np.float64)\n"
            "w = np.asarray(weights, dtype=np.float64)\nq = np.asarray(query, dtype=np.float64)\n"
            "out = np.empty(q.shape, dtype=np.float64)\n"
            "for i,t in enumerate(q):\n"
            "    hit = np.flatnonzero(t==nodes)\n"
            "    if hit.size: out[i] = y[hit[0]]\n"
            "    else:\n"
            "        v = w/(t-nodes)\n"
            "        out[i] = (v@y)/v.sum()\n"
            "return out",
            "w/(t-nodes)",
            "w*(t-nodes)",
        ),
        (
            "thermal-rescale",
            "transfer",
            "velocity, mass, energy",
            "Rows are particles, columns velocity components; masses and target "
            "energy are positive. Remove mass-weighted center-of-mass velocity then "
            "rescale to kinetic energy energy. Residual energy is nonzero. Return velocity.shape.",
            "v = np.asarray(velocity, dtype=np.float64)\nm = np.asarray(mass, dtype=np.float64)\n"
            "center = (m[:,None]*v).sum(axis=0)/m.sum()\nu = v-center\n"
            "kinetic = 0.5*(m[:,None]*u*u).sum()\n"
            "return u*np.sqrt(energy/kinetic)",
            "(m[:,None]*v).sum(axis=0)/m.sum()",
            "v.mean(axis=0)",
        ),
    ]
    return [
        Case(
            name,
            "transfer" if purpose == "transfer" else "train",
            "science_" + purpose,
            params,
            requirement,
            body,
            old,
            new,
        )
        for name, purpose, params, requirement, body, old, new in rows
    ]


def sample(name: str, rng: np.random.Generator, trial: int) -> tuple[tuple, np.ndarray]:
    """Independent loop/analytic references on variable-sized, unseen numeric inputs."""
    n, d = 5 + trial % 5, 2 + trial % 3
    dtype = np.float32 if trial % 2 else np.float64
    x = rng.normal(size=n).astype(dtype)
    if name == "weighted-moments":
        w = rng.uniform(0.2, 3, n).astype(dtype)
        mean = sum(float(v) * float(a) for v, a in zip(x, w)) / sum(map(float, w))
        ref = [
            mean,
            sum(float(a) * (float(v) - mean) ** 2 for v, a in zip(x, w))
            / sum(map(float, w)),
        ]
        args = (x, w)
    elif name == "cumulative-trapezoid":
        t = np.cumsum(rng.uniform(0.2, 2, n)).astype(dtype)
        ref = [0.0]
        for i in range(1, n):
            ref.append(
                ref[-1]
                + (float(t[i]) - float(t[i - 1])) * (float(x[i]) + float(x[i - 1])) / 2
            )
        args = (t, x)
    elif name == "periodic-laplacian":
        dx = 0.3 + trial * 0.07
        ref = [
            (float(x[(i - 1) % n]) - 2 * float(x[i]) + float(x[(i + 1) % n])) / dx**2
            for i in range(n)
        ]
        args = (x, dx)
    elif name == "row-softmax":
        z = (rng.normal(size=(n, d)) * 3 + 1000).astype(dtype)
        ref = []
        for row in z:
            values = [np.exp(float(v) - float(max(row))) for v in row]
            ref.append([v / sum(values) for v in values])
        args = (z,)
    elif name == "ridge-intercept":
        z = (rng.normal(size=(n, d)) + 2).astype(dtype)
        y = (rng.normal(size=n) + 3).astype(dtype)
        penalty = 0.4 + trial * 0.1
        design = np.column_stack((np.ones(n), z.astype(float)))
        regularizer = np.diag([0.0] + [penalty] * d)
        ref = np.linalg.solve(
            design.T @ design + regularizer, design.T @ y.astype(float)
        )
        args = (z, y, penalty)
    elif name == "markov-row-state":
        p = rng.uniform(0.1, 1, n)
        p /= p.sum()
        t = rng.uniform(0.1, 1, (n, n))
        t /= t.sum(axis=1, keepdims=True)
        p, t = p.astype(dtype), t.astype(dtype)
        ref = [sum(float(p[i]) * float(t[i, j]) for i in range(n)) for j in range(n)]
        ref = np.asarray(ref) / sum(ref)
        args = (p, t)
    elif name == "orthogonal-residual":
        basis = np.linalg.qr(rng.normal(size=(n, d)))[0]
        ref = x.astype(float).copy()
        for column in basis.T:
            ref -= sum(float(a) * float(b) for a, b in zip(column, x)) * column
        args = (x, basis)
    elif name == "weighted-density":
        edges = np.cumsum(rng.uniform(0.1, 2, n)).astype(dtype)
        points = rng.uniform(float(edges[0]), float(edges[-1]), 2 * n).astype(dtype)
        points[:2] = edges[[0, -1]]
        weights = rng.uniform(0.1, 3, 2 * n).astype(dtype)
        mass = np.zeros(n - 1)
        for point, weight in zip(points, weights):
            for j in range(n - 1):
                if edges[j] <= point and (point < edges[j + 1] or j == n - 2):
                    mass[j] += float(weight)
                    break
        ref = [
            mass[j] / sum(map(float, weights)) / (float(edges[j + 1]) - float(edges[j]))
            for j in range(n - 1)
        ]
        args = (points, weights, edges)
    elif name == "sample-covariance":
        z = rng.normal(size=(n, d)).astype(dtype)
        means = [sum(map(float, z[:, j])) / n for j in range(d)]
        ref = [
            [
                sum(
                    (float(row[i]) - means[i]) * (float(row[j]) - means[j]) for row in z
                )
                / (n - 1)
                for j in range(d)
            ]
            for i in range(d)
        ]
        args = (z,)
    elif name == "conservative-flux":
        flux = rng.normal(size=n + 1).astype(dtype)
        dt = 0.1
        dx = 0.7
        ref = [
            float(x[i]) - dt / dx * (float(flux[i + 1]) - float(flux[i]))
            for i in range(n)
        ]
        args = (x, flux, dt, dx)
    elif name == "implicit-decay":
        rate, dt = 1.3, 0.1 + trial * 0.01
        ref = [float(v) / (1 + rate * dt) for v in x]
        args = (x, rate, dt)
    elif name == "merge-population-moments":
        y = (rng.normal(size=n + 2) + 4).astype(dtype)
        values = list(map(float, x)) + list(map(float, y))
        mean = sum(values) / len(values)
        ref = [mean, sum((v - mean) ** 2 for v in values) / len(values)]
        args = (x, y)
    elif name == "velocity-verlet":
        v = rng.normal(size=n).astype(dtype)
        k, dt = 1.7, 0.15
        pos = []
        vel = []
        for a, b in zip(x, v):
            h = float(b) - 0.5 * dt * k * float(a)
            p = float(a) + dt * h
            pos.append(p)
            vel.append(h - 0.5 * dt * k * p)
        ref = [pos, vel]
        args = (x, v, k, dt)
    elif name == "segment-means":
        groups = 4
        labels = rng.integers(0, groups - 1, n)
        ref = []
        for j in range(groups):
            vals = [float(v) for v, label in zip(x, labels) if label == j]
            ref.append(sum(vals) / len(vals) if vals else 0.0)
        args = (x, labels, groups)
    elif name == "piecewise-linear":
        t = np.cumsum(rng.uniform(0.1, 2, n)).astype(dtype)
        query = np.concatenate((t[[0, -1]], rng.uniform(float(t[0]), float(t[-1]), d)))
        ref = []
        for point in query:
            j = min(max(np.searchsorted(t, point) - 1, 0), n - 2)
            frac = (point - float(t[j])) / (float(t[j + 1]) - float(t[j]))
            ref.append((1 - frac) * float(x[j]) + frac * float(x[j + 1]))
        args = (t, x, query)
    elif name == "batched-linear-system":
        mat = (rng.normal(size=(3, d, d)) + 0.2).astype(dtype)
        mat += np.eye(d, dtype=dtype)[None] * d
        rhs = rng.normal(size=(3, d)).astype(dtype)
        ref = np.stack(
            [
                np.linalg.solve(mat[i].astype(float), rhs[i].astype(float))
                for i in range(3)
            ]
        )
        args = (mat, rhs)
    elif name == "kalman-joseph":
        state = rng.normal(size=d).astype(dtype)
        h = rng.normal(size=d).astype(dtype)
        a = rng.normal(size=(d, d))
        cov = (a @ a.T + np.eye(d)).astype(dtype)
        z, noise = 0.7, 0.3
        hf = h.astype(float)
        pf = cov.astype(float)
        sf = state.astype(float)
        gain = pf @ hf / (float(hf @ pf @ hf) + noise)
        mean = sf + gain * (z - float(hf @ sf))
        # Algebraically equivalent covariance identity, independent of Joseph form.
        posterior = pf - np.outer(pf @ hf, pf @ hf) / (float(hf @ pf @ hf) + noise)
        ref = np.concatenate((mean, posterior.ravel()))
        args = (state, cov, z, h, noise)
    elif name == "rodrigues-rotation":
        axis = rng.normal(size=3).astype(dtype)
        angle = 0.3 + trial * 0.17
        unit = axis.astype(float) / np.linalg.norm(axis.astype(float))
        columns = []
        for e in np.eye(3):
            columns.append(
                e * np.cos(angle)
                + np.cross(unit, e) * np.sin(angle)
                + unit * np.dot(unit, e) * (1 - np.cos(angle))
            )
        ref = np.stack(columns, axis=1)
        args = (axis, angle)
    elif name == "log-hmm-filter":
        p = rng.uniform(0.1, 1, d)
        p /= p.sum()
        t = rng.uniform(0.1, 1, (d, d))
        t /= t.sum(axis=1, keepdims=True)
        likelihood = rng.uniform(0.01, 0.9, d)
        logs = tuple(np.log(v).astype(dtype) for v in (p, t, likelihood))
        prior, transition, like = [np.exp(v.astype(float)) for v in logs]
        out = np.asarray(
            [
                sum(prior[i] * transition[i, j] for i in range(d)) * like[j]
                for j in range(d)
            ]
        )
        ref = np.log(out / out.sum())
        args = logs
    elif name == "periodic-poisson":
        size = 16 + 2 * trial
        length = 2.7
        grid = np.arange(size) * length / size
        forcing = (
            np.sin(2 * np.pi * grid / length) + 0.3 * np.cos(4 * np.pi * grid / length)
        ).astype(dtype)
        # FFT float32 input quantization adds high-frequency error; reference in exact f64.
        ref = -(
            np.sin(2 * np.pi * grid / length) / (2 * np.pi / length) ** 2
            + 0.3 * np.cos(4 * np.pi * grid / length) / (4 * np.pi / length) ** 2
        )
        args = (forcing, length)
    elif name == "affine-conservation":
        c = rng.normal(size=n).astype(dtype)
        target = 1.7
        dot = sum(float(v) * float(t) for v, t in zip(c, x))
        norm = sum(float(v) ** 2 for v in c)
        ref = [float(v) + float(t) * (target - dot) / norm for v, t in zip(x, c)]
        args = (x, c, target)
    elif name == "linear-noise-covariance":
        mat = rng.normal(size=(d, d)).astype(dtype)
        b = rng.normal(size=(d, d))
        cov = (b @ b.T + np.eye(d)).astype(dtype)
        b = rng.normal(size=(d, d))
        noise = (b @ b.T * 0.1 + np.eye(d) * 0.1).astype(dtype)
        ref = [
            [
                sum(
                    float(mat[i, k]) * float(cov[k, column]) * float(mat[j, column])
                    for k in range(d)
                    for column in range(d)
                )
                + float(noise[i, j])
                for j in range(d)
            ]
            for i in range(d)
        ]
        args = (mat, cov, noise)
    elif name == "barycentric-interpolation":
        nodes = np.linspace(-1, 1, 4 + trial % 3)
        weights = np.asarray(
            [1 / np.prod(np.delete(nodes[i] - nodes, i)) for i in range(len(nodes))]
        )
        # A quadratic is exactly reproduced by all these distinct node sets.
        values = 1.3 * nodes**2 - 0.7 * nodes + 2
        query = np.concatenate((nodes[:1], rng.uniform(-1, 1, 7)))
        ref = 1.3 * query**2 - 0.7 * query + 2
        args = (nodes, values, weights, query)
    elif name == "thermal-rescale":
        v = rng.normal(size=(n, d)).astype(dtype)
        m = rng.uniform(0.2, 3, n).astype(dtype)
        energy = 2.1
        center = np.asarray(
            [
                sum(float(m[i]) * float(v[i, j]) for i in range(n)) / sum(map(float, m))
                for j in range(d)
            ]
        )
        residual = v.astype(float) - center
        kinetic = sum(
            0.5 * float(m[i]) * sum(float(t) ** 2 for t in residual[i])
            for i in range(n)
        )
        ref = residual * (energy / kinetic) ** 0.5
        args = (v, m, energy)
    else:
        raise ValueError(f"unknown authored case: {name}")
    return args, np.asarray(ref, dtype=np.float64)


def execute(
    case: Case, *, foil: bool, seed: int, trials: int, sample_fn=sample
) -> dict:
    namespace = {}
    code = case.code(foil=foil)
    exec(compile(code, "<author-owned-science-case>", "exec"), namespace)
    function = namespace["solve"]
    if list(inspect.signature(function).parameters) != case.parameters.split(", "):
        raise ValueError("interface mismatch in authored source")
    rng = np.random.default_rng(seed)
    failures = []
    for trial in range(trials):
        args, reference = sample_fn(case.name, rng, trial)
        copied = [v.copy() if isinstance(v, np.ndarray) else v for v in args]
        try:
            with np.errstate(all="raise"):
                result = function(*args)
            if not isinstance(result, np.ndarray) or result.dtype != np.float64:
                raise AssertionError("output_type_or_dtype")
            if result.shape != reference.shape:
                raise AssertionError("output_shape")
            if not np.isfinite(result).all():
                raise AssertionError("nonfinite")
            np.testing.assert_allclose(result, reference, rtol=2e-6, atol=2e-7)
            for original, now in zip(copied, args):
                if isinstance(original, np.ndarray):
                    np.testing.assert_array_equal(original, now)
        except Exception as error:
            failures.append({"trial": trial, "type": type(error).__name__})
    return {
        "trials": trials,
        "passed": trials - len(failures),
        "failed": len(failures),
        "failures": failures,
    }


def validate(
    cases: list[Case], *, seed: int = 20261106, trials: int = 8, sample_fn=sample
) -> dict:
    if not 4 <= trials <= 100:
        raise ValueError("trials must be between 4 and 100")
    ids = [c.name for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case family")
    results = []
    for case in cases:
        public = case.public()
        case_seed = seed + int.from_bytes(
            hashlib.sha256(case.name.encode()).digest()[:4], "big"
        )
        correct = execute(
            case, foil=False, seed=case_seed, trials=trials, sample_fn=sample_fn
        )
        foil = execute(
            case, foil=True, seed=case_seed, trials=trials, sample_fn=sample_fn
        )
        if correct["failed"] or not foil["failed"]:
            raise ValueError(
                f"authored validation failed: {case.name}: correct={correct}, foil={foil}"
            )
        results.append(
            {
                "id": public["id"],
                "family": case.name,
                "split": case.split,
                "correct": correct,
                "foil": foil,
                "max_prompt_continuation_bytes": max(
                    len((public["prompt"] + public[key]).encode())
                    for key in ("correct", "foil")
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
        "cases": results,
        "numpy_version": np.__version__,
        "reference_method": "independent loops or analytic identities; variable shapes/dtypes/seeds",
        "token_count_measured": False,
        "text_byte_limit": 2048,
        "scope": "Author-owned synthetic calibration, not WSP score or unseen benchmark evidence.",
    }


def training_payload(cases: list[Case]) -> dict:
    if any(c.split != "train" for c in cases):
        raise ValueError("transfer cases must never be exported as gradient input")
    output = {}
    for case in cases:
        folds = output.setdefault(case.purpose, [[], []])
        fold = sum(map(len, folds)) % 2
        folds[fold].append(case.public())
    return output


def write_bundle(
    output: Path,
    *,
    seed: int = 20261106,
    trials: int = 8,
    cases: list[Case] | None = None,
    sample_fn=sample,
    source_paths: tuple[Path, ...] = (),
    authoring: str | None = None,
) -> dict:
    if os.path.lexists(output):
        raise FileExistsError(output)
    cases = catalog() if cases is None else cases
    validation = validate(cases, seed=seed, trials=trials, sample_fn=sample_fn)
    train = [c for c in cases if c.split == "train"]
    transfer = [c for c in cases if c.split == "transfer"]
    assert set(c.name for c in train).isdisjoint(c.name for c in transfer)
    metadata = {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "train_case_count": len(train),
        "transfer_case_count": len(transfer),
        "training_families": [c.name for c in train],
        "transfer_families": [c.name for c in transfer],
        "family_disjoint": True,
        "authoring": authoring
        or "New equations/contracts authored for this tool; no benchmark corpus imported.",
        "transfer_contract": "Do not pass transfer-cases.json to a gradient collector, fit coefficients, or train weights on it. Repeated use makes it known calibration, not pristine evidence.",
        "limits": [
            "Distinct authored problem families share mathematical primitives; disjoint names are not a proof of statistical independence.",
            "Execution tests are finite coverage, not a proof for every possible input.",
            "No actual model tokenization/frame length was measured.",
            "This utility executes only its own trusted embedded code; it is not a sandbox for external model programs.",
        ],
        "validation": validation,
    }
    if source_paths:
        metadata["additional_sources"] = [
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in source_paths
        ]
    output.mkdir(parents=True)
    payloads = {
        "train-cases.json": training_payload(train),
        "transfer-cases.json": {
            "schema_version": 1,
            "split": "transfer",
            "gradient_allowed": False,
            "cases": [c.public() for c in transfer],
        },
        "metadata.json": metadata,
    }
    for name, payload in payloads.items():
        with (output / name).open("x") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--write", action="store_true", help="Validate and write a fresh split bundle"
    )
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()
    start = time.monotonic()
    if args.write:
        if args.output is None:
            parser.error("--write requires --output")
        result = write_bundle(args.output, seed=args.seed, trials=args.trials)[
            "validation"
        ]
    else:
        result = validate(catalog(), seed=args.seed, trials=args.trials)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": len(result["cases"]),
                "write": args.write,
                "seconds": time.monotonic() - start,
            }
        )
    )


if __name__ == "__main__":
    main()
