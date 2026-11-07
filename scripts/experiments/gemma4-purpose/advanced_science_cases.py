#!/usr/bin/env python3
"""Advanced authored science calibration; no benchmark/reference corpus imports.

Public dev failure concepts informed the families, so this is calibration rather
than independent final evaluation. Only trusted source-owned programs execute.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import science_cases as shared


def catalog(*, revision: int = 1) -> list[shared.Case]:
    if revision not in (1, 2):
        raise ValueError("advanced catalog revision must be 1 or 2")
    # Cases expose general algorithms, never sampled answers or test data.
    rows = [
        (
            "reset-running-stats",
            "state",
            "x, reset",
            "Scan x left to right. Before adding x[i], reset the running sum and count "
            "if reset[i] is true. Return per-prefix [mean,count], shape (len(x),2).",
            "s = 0.0; n = 0\nout = []\nfor v, flag in zip(x, reset):\n"
            "    if flag: s = 0.0; n = 0\n    s += float(v); n += 1\n"
            "    out.append([s/n, n])\nreturn np.array(out, dtype=np.float64)",
            "if flag:",
            "if False:",
        ),
        (
            "affine-state-history",
            "state",
            "a, x, forcing",
            "Starting from vector x, apply x_new=a@x+forcing[t] at each time. "
            "Return all states including the initial one, shape (len(forcing)+1,len(x)).",
            "state = np.array(x, dtype=np.float64, copy=True)\nout = [state.copy()]\n"
            "for u in forcing:\n    state = a @ state + u\n    out.append(state.copy())\n"
            "return np.array(out, dtype=np.float64)",
            "state = a @ state + u",
            "state = a @ x + u",
        ),
        (
            "capped-reservoir-chain",
            "state",
            "initial, inflow, demand, capacity",
            "At each time, add nonnegative inflow to stock, spill excess above positive "
            "capacity, then deliver min(demand,stock). Initial is in [0,capacity]. "
            "Return [remaining stock, delivered, spilled] per time, shape (len(inflow),3).",
            "stock = float(initial)\nout = []\nfor inc, req in zip(inflow, demand):\n"
            "    total = stock + inc\n    spill = max(0.0, total-capacity)\n"
            "    stock = min(total, capacity)\n    used = min(req, stock)\n"
            "    stock -= used\n    out.append([stock, used, spill])\n"
            "return np.array(out, dtype=np.float64)",
            "stock = min(total, capacity)",
            "stock = total",
        ),
        (
            "projected-gradient-chain",
            "state",
            "a, b, x, rate, steps",
            "Repeat steps times: g=a@x-b, clip g to Euclidean norm at most 1, "
            "then set x=max(0,x-rate*g) componentwise. Return the final vector, shape x.shape.",
            "y = np.array(x, dtype=np.float64, copy=True)\nfor _ in range(steps):\n"
            "    g = a @ y-b\n    g = g/max(1.0, float(np.linalg.norm(g)))\n"
            "    y = np.maximum(0.0, y-rate*g)\nreturn y",
            "max(1.0, float(np.linalg.norm(g)))",
            "max(1.0, float(np.abs(g).sum()))",
        ),
        (
            "merge-event-multiplicity",
            "counting",
            "left, right, k",
            "left and right are sorted nonempty event-time vectors. Return their first "
            "k merged events in ascending order, retaining every duplicate, shape (k,). "
            "0<k<=len(left)+len(right).",
            "i = 0; j = 0; out = []\nwhile len(out) < k:\n"
            "    if j == len(right) or (i < len(left) and left[i] <= right[j]):\n"
            "        out.append(left[i]); i += 1\n"
            "    else:\n        out.append(right[j]); j += 1\n"
            "return np.array(out, dtype=np.float64)",
            "return np.array(out, dtype=np.float64)",
            "return np.unique(out).astype(np.float64)",
        ),
        (
            "lexicographic-pair-ranking",
            "counting",
            "left, right, k",
            "List all pairs of entries, cost=left[i]+right[j]. Return the smallest k "
            "rows [cost,i,j], sorted by cost then i then j; retain equal costs. "
            "Shape (k,3), 0<k<=len(left)*len(right).",
            "i, j = np.indices((len(left), len(right)))\n"
            "cost = (np.asarray(left)[:,None]+np.asarray(right)[None,:]).ravel()\n"
            "i = i.ravel(); j = j.ravel()\norder = np.lexsort((j,i,cost))[:k]\n"
            "return np.column_stack((cost[order],i[order],j[order])).astype(np.float64)",
            "np.lexsort((j,i,cost))",
            "np.lexsort((i,j,cost))",
        ),
        (
            "budget-prefix-cardinality",
            "counting",
            "cost, budget",
            "For each nonnegative budget, buy the longest prefix of ascending nonnegative "
            "costs whose sum is <=budget. Zero-price entries count. Return counts, shape budget.shape.",
            "total = np.cumsum(np.sort(np.asarray(cost, dtype=np.float64)))\n"
            "return np.searchsorted(total, budget, side='right').astype(np.float64)",
            "side='right'",
            "side='left'",
        ),
        (
            "weighted-lattice-cardinality",
            "counting",
            "radii",
            "For each nonnegative integer r, count integer pairs (i,j) satisfying "
            "|i|+2*|j|<=r, including zero and boundary points. Return radii.shape.",
            "out = []\nfor r in radii:\n    count = 0\n"
            "    for j in range(-int(r)//2-1, int(r)//2+2):\n"
            "        width = int(r)-2*abs(j)\n"
            "        if width >= 0: count += 2*width+1\n    out.append(count)\n"
            "return np.array(out, dtype=np.float64)",
            "if width >= 0:",
            "if width > 0:",
        ),
        (
            "gaussian-pair-energy-force",
            "simulation",
            "position, scale",
            "For scalar particle coordinates and positive scale, pair energy is "
            "exp(-((x_i-x_j)/scale)^2). Sum energy over i<j and compute force=-dE/dx "
            "on every particle. Return [total energy,forces...], shape (len(position)+1,).",
            "x = np.asarray(position, dtype=np.float64)\nd = x[:,None]-x[None,:]\n"
            "e = np.exp(-(d/scale)**2)\nenergy = np.triu(e,1).sum()\n"
            "force = (2*d*e/(scale*scale)).sum(axis=1)\n"
            "return np.concatenate(([energy],force))",
            "2*d*e",
            "-2*d*e",
        ),
        (
            "periodic-kick-drift",
            "simulation",
            "x, v, stiffness, dt, steps",
            "On a periodic ring of unit masses, repeat: f=k*(roll(x,1)-2*x+roll(x,-1)); "
            "v+=dt*f; x+=dt*v. Return final x then v as two rows, shape (2,len(x)).",
            "x = np.array(x, dtype=np.float64, copy=True)\n"
            "v = np.array(v, dtype=np.float64, copy=True)\nfor _ in range(steps):\n"
            "    f = stiffness*(np.roll(x,1)-2*x+np.roll(x,-1))\n"
            "    v += dt*f\n    x += dt*v\nreturn np.stack((x,v))",
            "v += dt*f\n    x += dt*v",
            "x += dt*v\n    v += dt*f",
        ),
        (
            "reflecting-wall-flight",
            "simulation",
            "x, v, dt, length",
            "Particles drift inside [0,length] with elastic wall reflections and may "
            "cross multiple walls. Fold z=x+dt*v by period 2*length. At exact walls "
            "use the direction of the interval immediately to its right in z. "
            "Return final position and velocity, shape (2,len(x)).",
            "z = np.asarray(x,dtype=np.float64)+dt*np.asarray(v,dtype=np.float64)\n"
            "u = np.mod(z,2*length)\np = np.where(u<length,u,2*length-u)\n"
            "sign = np.where(u<length,1.0,-1.0)\nreturn np.stack((p,sign*v))",
            "2*length-u",
            "length-u",
        ),
        (
            "inelastic-two-body",
            "simulation",
            "mass, velocity, restitution",
            "Two positive masses in one dimension collide. Preserve total momentum "
            "and set relative final velocity v1'-v2'=-restitution*(v1-v2), with e in [0,1]. "
            "Return [v1',v2',lost kinetic energy], shape (3,).",
            "m = np.asarray(mass,dtype=np.float64)\nv = np.asarray(velocity,dtype=np.float64)\n"
            "center = np.dot(m,v)/m.sum()\nrel = -restitution*(v[0]-v[1])\n"
            "out = center+np.array([m[1],-m[0]])*rel/m.sum()\n"
            "lost = np.dot(m,v*v-out*out)/2\nreturn np.concatenate((out,[lost]))",
            "-restitution*",
            "restitution*",
        ),
        (
            "injected-metropolis-chain",
            "probability",
            "initial, proposals, uniforms",
            "Energy is x*x/2. At each step use the supplied proposed state y and "
            "u in (0,1). Accept iff log(u)<min(0,E(x)-E(y)). Rejected states persist. "
            "Return all states including initial and final acceptance fraction, shape (len(proposals)+2,).",
            "x = float(initial); states = [x]; accepted = 0\n"
            "for y,u in zip(proposals,uniforms):\n"
            "    if np.log(u)<min(0.0,(x*x-y*y)/2):\n"
            "        x = float(y); accepted += 1\n    states.append(x)\n"
            "return np.array(states+[accepted/len(proposals)],dtype=np.float64)",
            "(x*x-y*y)/2",
            "(y*y-x*x)/2",
        ),
        (
            "categorical-without-replacement",
            "probability",
            "weights, uniforms",
            "Positive weights define ordered categories. For each u in [0,1), choose "
            "the first cumulative remaining weight strictly greater than u*total, "
            "then zero that category's weight. Return indices, shape uniforms.shape. "
            "Number of draws is at most number of categories.",
            "w = np.array(weights,dtype=np.float64,copy=True)\nout = []\n"
            "for u in uniforms:\n    i = np.searchsorted(np.cumsum(w),u*w.sum(),side='right')\n"
            "    out.append(i); w[i] = 0.0\nreturn np.array(out,dtype=np.float64)",
            "side='right'",
            "side='left'",
        ),
        (
            "piecewise-linear-density-cdf",
            "probability",
            "knots, density, query",
            "Increasing knots carry positive density values, linear between knots. "
            "For queries within the closed knot range, integrate from the first knot "
            "and divide by total integral. Return normalized CDF, shape query.shape.",
            "t = np.asarray(knots,dtype=np.float64); y = np.asarray(density,dtype=np.float64)\n"
            "area = np.diff(t)*(y[:-1]+y[1:])/2\nprefix = np.r_[0.0,np.cumsum(area)]\n"
            "i = np.clip(np.searchsorted(t,query,side='right')-1,0,len(t)-2)\n"
            "h = query-t[i]; slope = (y[i+1]-y[i])/(t[i+1]-t[i])\n"
            "return (prefix[i]+y[i]*h+slope*h*h/2)/prefix[-1]",
            "slope*h*h/2",
            "slope*h*h",
        ),
        (
            "log-survival-event-masses",
            "probability",
            "hazards",
            "Each hazard h is in (0,1) conditional on surviving earlier steps. "
            "Return unconditional event masses at every step followed by surviving "
            "all steps, shape (len(hazards)+1,). Accumulate survival in log space.",
            "h = np.asarray(hazards,dtype=np.float64)\n"
            "logs = np.r_[0.0,np.cumsum(np.log1p(-h))]\n"
            "return np.r_[np.exp(logs[:-1])*h,np.exp(logs[-1])]",
            "logs[:-1]",
            "logs[1:]",
        ),
        (
            "block-schur-solve",
            "transfer",
            "a, b, c, d, u, v",
            "Solve [a b;c d]@[x;y]=[u;v] using a Schur complement of invertible a. "
            "The full system and Schur complement are invertible. Return concatenated x,y.",
            "aiu = np.linalg.solve(a,u); aib = np.linalg.solve(a,b)\n"
            "y = np.linalg.solve(d-c@aib,v-c@aiu)\nx = aiu-aib@y\n"
            "return np.concatenate((x,y)).astype(np.float64)",
            "v-c@aiu",
            "v+c@aiu",
        ),
        (
            "absorbing-chain-duration",
            "transfer",
            "q",
            "q is a substochastic transient-state transition matrix with spectral radius<1. "
            "Return expected steps before absorption from each state, including the next step, shape (len(q),).",
            "q = np.asarray(q,dtype=np.float64)\n"
            "return np.linalg.solve(np.eye(len(q))-q,np.ones(len(q)))",
            "np.eye(len(q))-q",
            "np.eye(len(q))+q",
        ),
        (
            "obstacle-lattice-paths",
            "transfer",
            "blocked",
            "A boolean rectangle marks blocked cells. Count right/down paths from "
            "top-left to bottom-right, using each open cell only. Return the complete "
            "path-count table, same shape as blocked; a blocked start gives all zeros.",
            "out = np.zeros(blocked.shape,dtype=np.float64)\n"
            "for i in range(blocked.shape[0]):\n    for j in range(blocked.shape[1]):\n"
            "        if blocked[i,j]: continue\n"
            "        if i == 0 and j == 0: out[i,j] = 1.0\n"
            "        else: out[i,j] = (out[i-1,j] if i else 0)+(out[i,j-1] if j else 0)\n"
            "return out",
            "if blocked[i,j]: continue",
            "if False: continue",
        ),
        (
            "cubic-hermite-value-slope",
            "transfer",
            "x, endpoints, values, slopes",
            "Evaluate the cubic Hermite interpolant and its first derivative at scalar "
            "x within two increasing endpoints, matching supplied endpoint values and "
            "slopes. Return [value,slope], shape (2,).",
            "h = endpoints[1]-endpoints[0]; t = (x-endpoints[0])/h\n"
            "y0,y1 = values; m0,m1 = slopes\n"
            "value = (2*t**3-3*t*t+1)*y0+(t**3-2*t*t+t)*h*m0+(-2*t**3+3*t*t)*y1+(t**3-t*t)*h*m1\n"
            "deriv = (6*t*t-6*t)*y0/h+(3*t*t-4*t+1)*m0+(-6*t*t+6*t)*y1/h+(3*t*t-2*t)*m1\n"
            "return np.array([value,deriv],dtype=np.float64)",
            "(3*t*t-2*t)*m1",
            "(3*t*t-2*t)*m0",
        ),
        (
            "rank-one-inverse-update",
            "transfer",
            "inverse, u, v",
            "inverse=A^-1 and 1+v@inverse@u is nonzero. Return (A+u v.T)^-1 "
            "without inverting A again, same shape as inverse.",
            "a = np.asarray(inverse,dtype=np.float64)\np = a@u; q = v@a\n"
            "return a-np.outer(p,q)/(1+v@p)",
            "1+v@p",
            "1-v@p",
        ),
        (
            "poisson-binomial-masses",
            "transfer",
            "probabilities",
            "Independent Bernoulli variables have given probabilities in [0,1]. "
            "Return probabilities of exactly 0..n successes, shape (n+1,).",
            "out = np.array([1.0])\nfor p in probabilities:\n"
            "    next_out = np.zeros(len(out)+1)\n"
            "    next_out[:-1] += (1-p)*out\n    next_out[1:] += p*out\n"
            "    out = next_out\nreturn out",
            "next_out[:-1] += (1-p)*out",
            "next_out[:-1] += p*out",
        ),
        (
            "heun-step-doubling",
            "transfer",
            "a, x, dt",
            "For dx/dt=a@x, compute one Heun step and two half-size Heun steps. "
            "Return the two-half result with Richardson correction (fine-coarse)/3, "
            "followed by that correction vector, shape (2,len(x)).",
            "def step(y,h):\n    k = a@y\n    return y+h*(k+a@(y+h*k))/2\n"
            "coarse = step(x,dt); fine = step(step(x,dt/2),dt/2)\n"
            "error = (fine-coarse)/3\nreturn np.stack((fine+error,error)).astype(np.float64)",
            "(fine-coarse)/3",
            "(fine-coarse)/2",
        ),
        (
            "whiten-recolor-rows",
            "transfer",
            "x, target_mean, target_cov",
            "Rows are observations, columns features, with positive-definite unbiased "
            "sample covariance. Whiten using its Cholesky factor, recolor to positive-"
            "definite target_cov, and shift to target_mean. Preserve row ordering and return x.shape.",
            "x = np.asarray(x,dtype=np.float64); centered = x-x.mean(axis=0)\n"
            "cov = centered.T@centered/(len(x)-1)\nl = np.linalg.cholesky(cov)\n"
            "z = np.linalg.solve(l,centered.T).T\n"
            "return z@np.linalg.cholesky(target_cov).T+target_mean",
            "np.linalg.cholesky(target_cov).T",
            "np.linalg.cholesky(target_cov)",
        ),
    ]
    cases = [
        shared.Case(
            "advanced-" + name,
            "transfer" if purpose == "transfer" else "train",
            "science_advanced_" + purpose,
            params,
            requirement,
            body,
            old,
            new,
        )
        for name, purpose, params, requirement, body, old, new in rows
    ]

    if revision == 2:
        # Prompt-only correction: retain all programs, foils, samples and oracles.
        requirements = {
            "advanced-budget-prefix-cardinality": "cost is a possibly unsorted 1D vector of nonnegative costs. First sort "
            "all costs ascending. For each nonnegative budget, buy the longest "
            "prefix of that sorted vector whose sum is <=budget. Zero-price entries "
            "count. Return counts, shape budget.shape.",
            "advanced-block-schur-solve": "a,b,c,d are 2D matrix blocks of shapes (n,n),(n,m),(m,n),(m,m); "
            "u,v are 1D vectors of shapes (n,),(m,), with n,m>=1. Solve the block "
            "system [a b;c d]@[x;y]=[u;v] using matrix multiplication and a Schur "
            "complement of invertible a. The full system and Schur complement "
            "are invertible. Return concatenate((x,y)), a 1D array of shape (n+m,).",
        }
        cases = [
            replace(case, requirement=requirements.get(case.name, case.requirement))
            for case in cases
        ]
    return cases


def sample(name: str, rng: np.random.Generator, trial: int) -> tuple[tuple, np.ndarray]:
    """Fresh numeric inputs with independent loops, identities, or alternate solvers."""
    name = name.removeprefix("advanced-")
    n, d = 4 + trial % 4, 2 + trial % 3
    dtype = np.float32 if trial % 2 else np.float64
    if name == "reset-running-stats":
        x = rng.normal(size=n).astype(dtype)
        reset = rng.random(n) < 0.4
        reset[0] = True
        reset[n // 2] = True
        ref = []
        start = 0
        for i in range(n):
            if reset[i]:
                start = i
            ref.append(
                [
                    sum(float(v) for v in x[start : i + 1]) / (i - start + 1),
                    i - start + 1,
                ]
            )
        args = (x, reset)
    elif name == "affine-state-history":
        a = (rng.normal(size=(d, d)) * 0.1).astype(dtype)
        x = rng.normal(size=d).astype(dtype)
        u = rng.normal(size=(n, d)).astype(dtype)
        ref = [x.astype(float)]
        for inc in u:
            ref.append(
                np.array(
                    [
                        sum(float(a[i, j]) * ref[-1][j] for j in range(d)) + inc[i]
                        for i in range(d)
                    ]
                )
            )
        args = (a, x, u)
    elif name == "capped-reservoir-chain":
        cap = 2.5
        initial = 1.0
        inc = rng.uniform(0, 4, n).astype(dtype)
        req = rng.uniform(0, 4, n).astype(dtype)
        inc[0] = 4
        req[0] = 0
        stock = initial
        ref = []
        for inflow, demand in zip(inc, req):
            available = min(cap, stock + float(inflow))
            delivered = min(float(demand), available)
            ref.append(
                [available - delivered, delivered, max(0, stock + float(inflow) - cap)]
            )
            stock = available - delivered
        args = (initial, inc, req, cap)
    elif name == "projected-gradient-chain":
        m = rng.normal(size=(d, d))
        a = (m.T @ m + np.eye(d)).astype(dtype)
        b = rng.normal(size=d).astype(dtype)
        x = rng.uniform(0.1, 2, d).astype(dtype)
        y = x.astype(float)
        rate = 0.2
        steps = n
        for _ in range(steps):
            g = np.array(
                [
                    sum(float(a[i, j]) * y[j] for j in range(d)) - float(b[i])
                    for i in range(d)
                ]
            )
            length = sum(v * v for v in g) ** 0.5
            y = np.array(
                [max(0, y[i] - rate * g[i] / max(1, length)) for i in range(d)]
            )
        ref = y
        args = (a, b, x, rate, steps)
    elif name == "merge-event-multiplicity":
        left = np.sort(rng.integers(0, 4, n)).astype(dtype)
        right = np.sort(rng.integers(0, 4, n + 1)).astype(dtype)
        left[:2] = 0
        right[:2] = 0
        k = 3 + trial % (2 * n - 2)
        ref = sorted(list(map(float, left)) + list(map(float, right)))[:k]
        args = (left, right, k)
    elif name == "lexicographic-pair-ranking":
        left = rng.integers(0, 3, n).astype(dtype)
        right = rng.integers(0, 3, d).astype(dtype)
        left[:] = 1 if trial % 2 == 0 else left
        right[:] = 1 if trial % 2 == 0 else right
        k = n * d - 1
        ref = sorted(
            (float(x) + float(y), i, j)
            for i, x in enumerate(left)
            for j, y in enumerate(right)
        )[:k]
        args = (left, right, k)
    elif name == "budget-prefix-cardinality":
        cost = rng.integers(0, 5, n).astype(dtype)
        cost[0] = 0
        budget = np.r_[0, np.cumsum(np.sort(cost)), sum(cost) + 1].astype(dtype)
        ref = []
        for b in budget:
            s = 0.0
            count = 0
            for v in sorted(map(float, cost)):
                if s + v > b:
                    break
                s += v
                count += 1
            ref.append(count)
        args = (cost, budget)
    elif name == "weighted-lattice-cardinality":
        radii = np.arange(n, dtype=np.int64)
        ref = [
            sum(
                abs(i) + 2 * abs(j) <= r
                for i in range(-r, r + 1)
                for j in range(-r, r + 1)
            )
            for r in radii
        ]
        args = (radii,)
    elif name == "gaussian-pair-energy-force":
        x = rng.normal(size=n).astype(dtype)
        scale = 0.7 + trial * 0.1
        energy = 0.0
        force = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                delta = float(x[i]) - float(x[j])
                e = np.exp(-((delta / scale) ** 2))
                energy += e
                f = 2 * delta * e / scale**2
                force[i] += f
                force[j] -= f
        ref = np.r_[energy, force]
        args = (x, scale)
    elif name == "periodic-kick-drift":
        x = rng.normal(size=n).astype(dtype)
        v = rng.normal(size=n).astype(dtype)
        y = x.astype(float)
        w = v.astype(float)
        stiffness = 1.4
        dt = 0.07
        steps = d
        for _ in range(steps):
            f = [
                stiffness * (y[(i - 1) % n] - 2 * y[i] + y[(i + 1) % n])
                for i in range(n)
            ]
            w = np.array([w[i] + dt * f[i] for i in range(n)])
            y = np.array([y[i] + dt * w[i] for i in range(n)])
        ref = np.stack((y, w))
        args = (x, v, stiffness, dt, steps)
    elif name == "reflecting-wall-flight":
        length = 1.7
        dt = 2.4
        x = rng.uniform(0, length, n).astype(dtype)
        v = rng.uniform(-5, 5, n).astype(dtype)
        ref = [[], []]
        for p, speed in zip(x, v):
            z = float(p) + dt * float(speed)
            interval = int(np.floor(z / length))
            phase = z - interval * length
            ref[0].append(phase if interval % 2 == 0 else length - phase)
            ref[1].append(float(speed) * (1 if interval % 2 == 0 else -1))
        args = (x, v, dt, length)
    elif name == "inelastic-two-body":
        m = rng.uniform(0.2, 3, 2).astype(dtype)
        v = rng.normal(size=2).astype(dtype)
        e = [0.0, 1.0, 0.35, 0.8][trial % 4]
        system = np.array([[float(m[0]), float(m[1])], [1, -1]])
        rhs = [
            sum(float(a) * float(b) for a, b in zip(m, v)),
            -e * (float(v[0]) - float(v[1])),
        ]
        after = np.linalg.solve(system, rhs)
        lost = sum(
            float(m[i]) * (float(v[i]) ** 2 - after[i] ** 2) / 2 for i in range(2)
        )
        ref = np.r_[after, lost]
        args = (m, v, e)
    elif name == "injected-metropolis-chain":
        initial = 0.4
        proposals = rng.normal(0, 2, n).astype(dtype)
        proposals[0] = 5
        proposals[1] = 0.0
        u = rng.uniform(0.05, 0.95, n).astype(dtype)
        y = initial
        ref = [y]
        accepted = 0
        for proposed, draw in zip(proposals, u):
            probability = min(1, np.exp((y * y - float(proposed) ** 2) / 2))
            if float(draw) < probability:
                y = float(proposed)
                accepted += 1
            ref.append(y)
        ref.append(accepted / n)
        args = (initial, proposals, u)
    elif name == "categorical-without-replacement":
        weights = rng.uniform(0.2, 3, n).astype(dtype)
        u = rng.uniform(0, 0.99, n).astype(dtype)
        u[:2] = 0
        remaining = weights.astype(float).tolist()
        ref = []
        for draw in u:
            target = float(draw) * sum(remaining)
            cumulative = 0.0
            for i, w in enumerate(remaining):
                cumulative += w
                if cumulative > target:
                    break
            ref.append(i)
            remaining[i] = 0.0
        args = (weights, u)
    elif name == "piecewise-linear-density-cdf":
        t = np.cumsum(rng.uniform(0.3, 1.5, n)).astype(dtype)
        y = rng.uniform(0.2, 3, n).astype(dtype)
        query = np.r_[t[0], t[-1], t[1], rng.uniform(t[0], t[-1], n)].astype(dtype)
        total = sum(
            (float(t[i + 1]) - float(t[i])) * (float(y[i]) + float(y[i + 1])) / 2
            for i in range(n - 1)
        )
        ref = []
        for point in query:
            area = 0.0
            for i in range(n - 1):
                width = min(
                    max(float(point) - float(t[i]), 0), float(t[i + 1]) - float(t[i])
                )
                slope = (float(y[i + 1]) - float(y[i])) / (
                    float(t[i + 1]) - float(t[i])
                )
                area += width * (float(y[i]) + float(y[i]) + slope * width) / 2
            ref.append(area / total)
        args = (t, y, query)
    elif name == "log-survival-event-masses":
        h = rng.uniform(0.001, 0.8, n).astype(dtype)
        survival = 1.0
        ref = []
        for hazard in h:
            ref.append(survival * float(hazard))
            survival *= 1 - float(hazard)
        ref.append(survival)
        args = (h,)
    elif name == "block-schur-solve":
        m = rng.normal(size=(d + 2, d + 2))
        full = m.T @ m + np.eye(d + 2)
        rhs = rng.normal(size=d + 2)
        a, b, c, dd = full[:d, :d], full[:d, d:], full[d:, :d], full[d:, d:]
        ref = np.linalg.solve(full, rhs)
        args = (a, b, c, dd, rhs[:d], rhs[d:])
    elif name == "absorbing-chain-duration":
        q = rng.uniform(0.01, 1, (d, d))
        q = q / q.sum(axis=1, keepdims=True) * 0.6
        ref = np.zeros(d)
        term = np.ones(d)
        for _ in range(100):
            ref += term
            term = q @ term
        args = (q.astype(dtype),)
        q = args[0].astype(float)
        ref = np.zeros(d)
        term = np.ones(d)
        for _ in range(100):
            ref += term
            term = q @ term
    elif name == "obstacle-lattice-paths":
        blocked = rng.random((n, d + 2)) < 0.25
        blocked[0, 0] = trial % 3 == 0
        blocked[1, 1] = True

        def paths(i, j):
            if i < 0 or j < 0 or blocked[i, j]:
                return 0
            if i == 0 and j == 0:
                return 1
            return paths(i - 1, j) + paths(i, j - 1)

        ref = [[paths(i, j) for j in range(d + 2)] for i in range(n)]
        args = (blocked,)
    elif name == "cubic-hermite-value-slope":
        endpoints = np.array([-0.7, 2.2])
        values = rng.normal(size=2)
        slopes = rng.normal(size=2)
        a, b = endpoints
        matrix = np.array(
            [
                [1, a, a * a, a**3],
                [1, b, b * b, b**3],
                [0, 1, 2 * a, 3 * a * a],
                [0, 1, 2 * b, 3 * b * b],
            ]
        )
        coefficients = np.linalg.solve(matrix, np.r_[values, slopes])
        x = [a, b, 0.3, 1.1][trial % 4]
        ref = [
            sum(coefficients[i] * x**i for i in range(4)),
            sum(i * coefficients[i] * x ** (i - 1) for i in range(1, 4)),
        ]
        args = (x, endpoints, values, slopes)
    elif name == "rank-one-inverse-update":
        m = rng.normal(size=(d, d))
        a = m.T @ m + np.eye(d)
        u = rng.normal(size=d) * 0.1
        v = rng.normal(size=d) * 0.1
        ref = np.linalg.inv(a + np.outer(u, v))
        args = (np.linalg.inv(a), u, v)
    elif name == "poisson-binomial-masses":
        p = rng.uniform(0.1, 0.9, n)
        p[0] = trial % 2
        ref = np.zeros(n + 1)
        for bits in range(1 << n):
            probability = 1.0
            count = 0
            for i in range(n):
                success = bool(bits & (1 << i))
                count += success
                probability *= p[i] if success else 1 - p[i]
            ref[count] += probability
        args = (p.astype(dtype),)
        p = args[0].astype(float)
        ref[:] = 0
        for bits in range(1 << n):
            success = [(bits >> i) & 1 for i in range(n)]
            ref[sum(success)] += np.prod(
                [p[i] if success[i] else 1 - p[i] for i in range(n)]
            )
    elif name == "heun-step-doubling":
        a = rng.normal(size=(d, d)) * 0.3
        x = rng.normal(size=d)
        dt = 0.3
        full = np.eye(d) + dt * a + (dt * dt / 2) * (a @ a)
        half = np.eye(d) + (dt / 2) * a + (dt * dt / 8) * (a @ a)
        coarse = full @ x
        fine = half @ half @ x
        error = (fine - coarse) / 3
        ref = np.stack((fine + error, error))
        args = (a, x, dt)
    elif name == "whiten-recolor-rows":
        x = rng.normal(size=(d + 5, d)).astype(dtype)
        target_mean = rng.normal(size=d)
        m = rng.normal(size=(d, d))
        cov = m.T @ m + np.eye(d)
        center = x.astype(float) - x.astype(float).mean(axis=0)
        source = sum(np.outer(row, row) for row in center) / (len(center) - 1)
        transform = (
            np.linalg.inv(np.linalg.cholesky(source)).T @ np.linalg.cholesky(cov).T
        )
        ref = center @ transform + target_mean
        args = (x, target_mean, cov)
    else:
        raise ValueError(f"unknown advanced authored case: {name}")
    return args, np.asarray(ref, dtype=np.float64)


def validate(*, seed: int = 20261108, trials: int = 8, revision: int = 1) -> dict:
    return shared.validate(
        catalog(revision=revision), seed=seed, trials=trials, sample_fn=sample
    )


def write_bundle(
    output: Path, *, seed: int = 20261108, trials: int = 8, revision: int = 1
) -> dict:
    metadata = shared.write_bundle(
        output,
        seed=seed,
        trials=trials,
        cases=catalog(revision=revision),
        sample_fn=sample,
        source_paths=(Path(__file__),),
        authoring="New authored problems informed by public dev failure concepts; no benchmark reference/test imported. Not independent final evidence.",
    )
    if revision == 2:
        metadata["catalog"] = "advanced-v2"
        metadata["catalog_revision"] = 2
        metadata["revision_scope"] = (
            "Prompt-only clarification of budget sorting and Schur matrix dimensions; implementation/sample/oracle unchanged."
        )
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--revision", type=int, choices=(1, 2), default=1)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()
    start = time.monotonic()
    if args.write:
        if args.output is None:
            parser.error("--write requires --output")
        report = write_bundle(
            args.output, seed=args.seed, trials=args.trials, revision=args.revision
        )["validation"]
    else:
        report = validate(seed=args.seed, trials=args.trials, revision=args.revision)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(report["cases"]),
                "write": args.write,
                "seconds": time.monotonic() - start,
            }
        )
    )


if __name__ == "__main__":
    main()
