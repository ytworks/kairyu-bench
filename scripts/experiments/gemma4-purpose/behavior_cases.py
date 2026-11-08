#!/usr/bin/env python3
"""Eighteen fixed behavior contracts: train12 / frozen transfer6; no model calls."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

TRAIN_SEED = 202611071
TRANSFER_SEED = 202611072
TRIALS = 16


@dataclass(frozen=True)
class Case:
    name: str
    parameters: str
    requirement: str
    body: str
    old: str
    new: str
    entrypoint: str = "solve"
    symbolic: bool = False
    exact: bool = False

    @property
    def split(self):
        return "transfer" if self.name.endswith("3") else "train"

    @property
    def seed(self):
        return TRANSFER_SEED if self.split == "transfer" else TRAIN_SEED

    @property
    def fold(self):
        return int(self.name[-1]) - 1

    def code(self, foil=False):
        name, body = self.entrypoint, self.body
        if foil and self.old == "__name__":
            name = self.new
        elif foil and self.old != "__duplicate__":
            if body.count(self.old) != 1:
                raise ValueError("foil edit must be unique")
            body = body.replace(self.old, self.new)
        text = "import numpy as np\n"
        if self.symbolic:
            text += "import sympy as sp\n"
        if foil and self.old == "__duplicate__":
            text += f"\ndef {name}({self.parameters}):\n    return np.array([-999.0])\n"
        text += f"\ndef {name}({self.parameters}):\n"
        return text + "\n".join("    " + line for line in body.splitlines()) + "\n"

    def public(self):
        extra = ""
        if self.entrypoint != "solve":
            extra = (
                " Only imports and one top-level function definition with this exact "
                "name are allowed; no other top-level statements or definitions."
            )
        return {
            "id": "behavior-" + self.name,
            "split": self.split,
            "prompt": f"Write Python {self.entrypoint}({self.parameters}). "
            + self.requirement
            + " Return a float64 ndarray with the specified shape; do not mutate inputs."
            + extra
            + " Include required imports and the complete final function, no prose.",
            "correct": self.code(),
            "foil": self.code(True),
        }


def catalog():
    return [
        Case(
            "A1",
            "A, b, s",
            "A has shape(B,n,n), each matrix is SPD; b is(B,n), s is(B,). "
            "B,n>=1. Solve A[i]x[i]=s[i]*b[i] independently, returning(B,n).",
            "A=np.asarray(A,dtype=np.float64)\nb=np.asarray(b,dtype=np.float64)\n"
            "rhs=np.asarray(s,dtype=np.float64)[:,None]*b\n"
            "return np.linalg.solve(A,rhs[...,None])[...,0]",
            "rhs[...,None])[...,0]",
            "rhs)",
        ),
        Case(
            "A2",
            "A, b, c",
            "A is(B,n,n) SPD, b,c are(B,n), B,n>=1. Solve both RHS with the same "
            "A, returning(B,n,2): last-axis0 for b,1 for c.",
            "A=np.asarray(A,dtype=np.float64)\n"
            "rhs=np.stack((b,c),axis=-1).astype(np.float64)\n"
            "return np.linalg.solve(A,rhs)",
            "axis=-1",
            "axis=1",
        ),
        Case(
            "A3",
            "A, loads",
            "A is(B,n,n) SPD and loads is(B,n,q); B,n,q>=1. Solve each column "
            "independently and return squared Euclidean norms of the solutions, shape(B,q).",
            "x=np.linalg.solve(np.asarray(A,dtype=np.float64),"
            "np.asarray(loads,dtype=np.float64))\nreturn np.sum(x*x,axis=1)",
            "axis=1",
            "axis=2",
        ),
        Case(
            "B1",
            "x, a",
            "x is a 1D real array, possibly empty, and a is real, including0. "
            "Differentiate f(t)=a*t**2+t using SymPy then lambdify. "
            "Evaluate its derivative at every x, returning x.shape.",
            "t=sp.symbols('t')\nf=a*t**2+t\n"
            "g=sp.lambdify(t,sp.diff(f,t),'numpy')(x)\n"
            "return np.broadcast_to(g,x.shape).astype(np.float64,copy=True)",
            "np.broadcast_to(g,x.shape)",
            "np.asarray(g)",
            symbolic=True,
        ),
        Case(
            "B2",
            "x, y, a",
            "x,y are same-length 1D real arrays, possibly empty; a is real including0. "
            "Use SymPy differentiation and lambdify for f=a*x*y+y**2. "
            "Return rows[df/dx,df/dy], shape(len(x),2).",
            "sx,sy=sp.symbols('x y')\nf=a*sx*sy+sy**2\n"
            "dx,dy=sp.lambdify((sx,sy),(sp.diff(f,sx),sp.diff(f,sy)),'numpy')(x,y)\n"
            "dx,dy=np.broadcast_arrays(dx,dy)\n"
            "return np.column_stack((dx,dy)).astype(np.float64)",
            "dx,dy=np.broadcast_arrays(dx,dy)",
            "pass",
            symbolic=True,
        ),
        Case(
            "B3",
            "x, a, b, c",
            "x is a 1D real array, possibly empty; a,b,c are real including0. "
            "Use SymPy then lambdify for the first and second derivatives of "
            "f(t)=a*t**3+b*t+c. Return rows[f'(x),f''(x)], shape(len(x),2).",
            "t=sp.symbols('t')\nf=a*t**3+b*t+c\n"
            "d1,d2=sp.lambdify(t,(sp.diff(f,t),sp.diff(f,t,2)),'numpy')(x)\n"
            "d1=np.broadcast_to(d1,x.shape)\nd2=np.broadcast_to(d2,x.shape)\n"
            "return np.column_stack((d1,d2)).astype(np.float64)",
            "d1=np.broadcast_to(d1,x.shape)\nd2=np.broadcast_to(d2,x.shape)",
            "pass",
            symbolic=True,
        ),
        Case(
            "C1",
            "xs",
            "xs is a 1D integer array, possibly empty. Return the sum of positive "
            "elements as shape(1,); an empty array returns[0].",
            "return np.array([sum(max(int(v),0) for v in xs)],dtype=np.float64)",
            "__name__",
            "solve",
            entrypoint="positive_mass",
            exact=True,
        ),
        Case(
            "C2",
            "xs, k",
            "xs is a 1D integer array; k is an integer>=0. Rotate left by k positions. "
            "For empty xs return shape(0,); otherwise return shape(len(xs),).",
            "if len(xs)==0:\n    return np.empty(0,dtype=np.float64)\n"
            "k=k%len(xs)\nreturn np.concatenate((xs[k:],xs[:k])).astype(np.float64)",
            "__duplicate__",
            "",
            entrypoint="rotate_left",
            exact=True,
        ),
        Case(
            "C3",
            "xs",
            "xs is a 1D integer array. Count maximal adjacent runs of equal values "
            "and return shape(1,); the empty array has0 runs.",
            "n=0 if len(xs)==0 else 1+sum(xs[i]!=xs[i-1] for i in range(1,len(xs)))\n"
            "return np.array([n],dtype=np.float64)",
            "__name__",
            "count_runs",
            entrypoint="run_count",
            exact=True,
        ),
        Case(
            "D1",
            "maps, start",
            "maps is(T,n) of integers in[0,n), n>=1, start in[0,n). "
            "At each t update state=maps[t,state]. Return every post-transition "
            "state as shape(T,); T=0 is allowed.",
            "state=int(start)\nout=[]\nfor row in maps:\n"
            "    state=int(row[state])\n    out.append(state)\n"
            "return np.asarray(out,dtype=np.float64)",
            "row[state]",
            "row[start]",
            exact=True,
        ),
        Case(
            "D2",
            "matrices, offsets, x",
            "matrices is(T,d,d), offsets is(T,d), x is(d,), d>=1. Apply "
            "x_new=matrices[t]@x+offsets[t] in increasing t order. Return the "
            "final(d,) state; T=0 returns a copy of x.",
            "state=np.asarray(x,dtype=np.float64).copy()\n"
            "for t in range(len(matrices)):\n"
            "    state=matrices[t]@state+offsets[t]\n"
            "return np.asarray(state,dtype=np.float64)",
            "range(len(matrices))",
            "reversed(range(len(matrices)))",
        ),
        Case(
            "D3",
            "keys, deltas, initial",
            "initial is(K,) account balances, keys is(N,) valid integer account "
            "indices, deltas is(N,). Apply each delta only to its key in event order. "
            "Return that account's post-event balance as(N,). Negative balances and N=0 are allowed.",
            "balances=np.asarray(initial,dtype=np.float64).copy()\nout=[]\n"
            "for key,delta in zip(keys,deltas):\n"
            "    balances[key]=balances[key]+delta\n    out.append(balances[key])\n"
            "return np.asarray(out,dtype=np.float64)",
            "balances[key]+delta",
            "initial[key]+delta",
        ),
        Case(
            "E1",
            "grid, start, moves",
            "grid is a nonempty HxW integer array, start=(row,col) is in bounds, "
            "moves is a 1D integer array with0=N,1=E,2=S,3=W. Move on the original "
            "toroidal grid and sum the start cell and every visited cell. Return(1,).",
            "r,c=map(int,start)\nh,w=grid.shape\n"
            "steps=((-1,0),(0,1),(1,0),(0,-1))\ntotal=int(grid[r,c])\n"
            "for move in moves:\n    dr,dc=steps[int(move)]\n"
            "    r,c=(r+dr)%h,(c+dc)%w\n    total+=int(grid[r,c])\n"
            "return np.array([total],dtype=np.float64)",
            "steps=((-1,0),(0,1),(1,0),(0,-1))",
            "if h<w:\n    grid=grid.T\n    r,c=c,r\n    h,w=grid.shape\n"
            "steps=((-1,0),(0,1),(1,0),(0,-1))",
            exact=True,
        ),
        Case(
            "E2",
            "points, height, width",
            "points is(N,2) integer (row,col) coordinates inside an H=height by "
            "W=width grid, H,W>=1. Return coordinates after clockwise90-degree "
            "rotation, shape(N,2), including N=0.",
            "return np.column_stack((points[:,1],height-1-points[:,0])).astype(np.float64)",
            "height-1",
            "width-1",
            exact=True,
        ),
        Case(
            "E3",
            "grid, queries",
            "grid is a nonempty HxW integer array. queries is(N,2) valid coordinates "
            "in the clockwise90-degree rotated WxH grid. Without constructing "
            "the rotated grid, look up their values in the original, returning(N,).",
            "h,w=grid.shape\n"
            "return np.asarray([grid[h-1-int(c),int(r)] for r,c in queries],dtype=np.float64)",
            "grid[h-1-int(c),int(r)]",
            "grid[w-1-int(c),int(r)]",
            exact=True,
        ),
        Case(
            "F1",
            "xs",
            "xs is a 1D integer array. Sum with alternating signs by original "
            "index, index0 positive. Zero elements still occupy a position. "
            "Return(1,), empty returns[0].",
            "return np.array([sum((-1)**i*int(v) for i,v in enumerate(xs))],dtype=np.float64)",
            "enumerate(xs)",
            "enumerate(xs[xs!=0])",
            exact=True,
        ),
        Case(
            "F2",
            "values, k",
            "values is a 1D integer array of event values; k is an integer>=1. "
            "Sum every kth event counting the original first event as number1. "
            "Zero events still count. Return(1,), empty returns[0].",
            "return np.array([sum(int(v) for v in values[k-1::k])],dtype=np.float64)",
            "values[k-1::k]",
            "values[values!=0][k-1::k]",
            exact=True,
        ),
        Case(
            "F3",
            "xs, k",
            "xs is a 1D integer array, 0<=k<=len(xs)<=10. Select exactly k "
            "elements in original order to maximize alternating-sign sum by "
            "selection position, first positive. Return(1,); k=0 returns[0].",
            "from itertools import combinations\n"
            "best=max((sum((-1)**i*int(v) for i,v in enumerate(row)) "
            "for row in combinations(xs,k)),default=-np.inf)\n"
            "return np.array([best],dtype=np.float64)",
            "combinations(xs,k)",
            "combinations(xs[xs!=0],k)",
            exact=True,
        ),
    ]


def sample(name, rng, trial):
    if not 0 <= trial < TRIALS:
        raise ValueError("fixed eight boundary and eight seeded trials required")
    # Fixed half is independent of the exported train/transfer random seeds.
    if trial < 8:
        rng = np.random.default_rng(
            81000 + ord(name[0]) * 100 + int(name[1]) * 10 + trial
        )
    n = (2, 3, 1, 4, 2, 3, 1, 4)[trial % 8]
    dtype = np.float64 if trial % 2 == 0 else np.float32
    if name[0] == "A":
        batch = (3, 3, 1, 2, 4, 1, 2, 4)[trial % 8]
        raw = rng.normal(size=(batch, n, n))
        A = (raw @ raw.transpose(0, 2, 1) + np.eye(n) * 2).astype(dtype)
        b = rng.normal(size=(batch, n)).astype(dtype)
        if name == "A1":
            s = rng.uniform(-2, 2, size=batch).astype(dtype)
            args = (A, b, s)
            ref = np.stack(
                [
                    np.linalg.solve(
                        A[i].astype(float), b[i].astype(float) * float(s[i])
                    )
                    for i in range(batch)
                ]
            )
        elif name == "A2":
            c = (rng.normal(size=(batch, n)) + 1).astype(dtype)
            args = (A, b, c)
            ref = np.stack(
                [
                    np.column_stack(
                        [
                            np.linalg.solve(A[i].astype(float), v[i].astype(float))
                            for v in (b, c)
                        ]
                    )
                    for i in range(batch)
                ]
            )
        else:
            q = (3, 1, 2, 3, 1, 4, 2, 1)[trial % 8]
            loads = rng.normal(size=(batch, n, q)).astype(dtype)
            args = (A, loads)
            ref = np.array(
                [
                    [
                        sum(
                            float(v) ** 2
                            for v in np.linalg.solve(
                                A[i].astype(float), loads[i, :, j].astype(float)
                            )
                        )
                        for j in range(q)
                    ]
                    for i in range(batch)
                ]
            )
    elif name[0] == "B":
        length = (3, 4, 0, 1, 2, 5, 3, 1)[trial % 8]
        x = rng.uniform(-2, 2, size=length).astype(dtype)
        a = 0.0 if trial % 4 in (0, 2) else float(rng.uniform(-2, 2))
        if name == "B1":
            args = (x, a)
            ref = 2 * a * x.astype(float) + 1
        elif name == "B2":
            y = rng.uniform(-2, 2, size=length).astype(dtype)
            args = (x, y, a)
            ref = np.column_stack(
                (a * y.astype(float), a * x.astype(float) + 2 * y.astype(float))
            )
        else:
            b = 0.0 if trial % 4 == 2 else float(rng.uniform(-2, 2))
            c = float(rng.uniform(-2, 2))
            args = (x, a, b, c)
            ref = np.column_stack(
                (3 * a * x.astype(float) ** 2 + b, 6 * a * x.astype(float))
            )
    elif name[0] == "C":
        xs = rng.integers(-3, 5, size=(0, 4, 1, 6, 3, 2, 5, 4)[trial % 8])
        if name == "C1":
            if trial == 0:
                xs = np.array([-2, 3, 4])
            args = (xs,)
            ref = [sum(v for v in xs.tolist() if v > 0)]
        elif name == "C2":
            k = (0, 1, 8, 2, 11, 4, 3, 6)[trial % 8]
            args = (xs, k)
            ref = [xs[(i + k) % len(xs)] for i in range(len(xs))]
        else:
            if trial == 0:
                xs = np.array([2, 2, 1, 2])
            args = (xs,)
            runs = []
            for value in xs.tolist():
                if not runs or value != runs[-1]:
                    runs.append(value)
            ref = [len(runs)]
    elif name == "D1":
        T = (2, 0, 1, 5, 3, 4, 6, 2)[trial % 8]
        maps = rng.integers(0, n, size=(T, n))
        start = 0
        if trial == 0:
            maps = np.array([[1, 0], [0, 1]])
        state = start
        out = []
        for row in maps.tolist():
            state = row[state]
            out.append(state)
        args = (maps, start)
        ref = out
    elif name == "D2":
        T = (2, 0, 1, 3, 4, 2, 5, 1)[trial % 8]
        matrices = rng.uniform(-0.5, 0.5, size=(T, n, n)).astype(dtype)
        offsets = rng.uniform(-1, 1, size=(T, n)).astype(dtype)
        x = rng.uniform(-1, 1, size=n).astype(dtype)
        if trial == 0:
            matrices = np.array([[[1, 1], [0, 1]], [[1, 0], [1, 1]]], dtype=float)
            offsets = np.zeros((2, 2))
            x = np.array([1.0, 0.0])
        state = list(map(float, x))
        for mat, off in zip(matrices.tolist(), offsets.tolist()):
            state = [
                sum(a * b for a, b in zip(row, state)) + v for row, v in zip(mat, off)
            ]
        args = (matrices, offsets, x)
        ref = state
    elif name == "D3":
        N = (3, 0, 1, 6, 4, 5, 2, 8)[trial % 8]
        initial = rng.integers(-3, 10, size=n).astype(dtype)
        keys = rng.integers(0, n, size=N)
        deltas = rng.integers(-5, 6, size=N).astype(dtype)
        if trial == 0:
            initial = np.array([10.0, 20.0])
            keys = np.array([0, 1, 0])
            deltas = np.array([1.0, 2.0, 3.0])
        balances = {i: float(v) for i, v in enumerate(initial)}
        ref = []
        for k, d in zip(keys.tolist(), deltas.tolist()):
            balances[k] += d
            ref.append(balances[k])
        args = (keys, deltas, initial)
    elif name[0] == "E":
        H = (2, 3, 1, 4, 2, 1, 3, 4)[trial % 8]
        W = (3, 2, 4, 1, 2, 1, 4, 3)[trial % 8]
        grid = np.arange(1, H * W + 1).reshape(H, W)
        if trial >= 8:
            grid = rng.integers(-5, 10, size=(H, W))
        if name == "E1":
            start = np.array([0, 0])
            moves = (
                np.array([1, 2]) if trial == 0 else rng.integers(0, 4, size=trial % 5)
            )
            r, c = start.tolist()
            total = int(grid[r, c])
            for step in moves.tolist():
                if step == 0:
                    r = (r - 1) % H
                elif step == 1:
                    c = (c + 1) % W
                elif step == 2:
                    r = (r + 1) % H
                else:
                    c = (c - 1) % W
                total += int(grid[r, c])
            args = (grid, start, moves)
            ref = [total]
        elif name == "E2":
            N = (1, 0, 3, 4, 2, 1, 6, 5)[trial % 8]
            points = np.column_stack(
                (rng.integers(0, H, size=N), rng.integers(0, W, size=N))
            )
            if trial == 0:
                points = np.array([[0, 0]])
            # Locate original labels in an independently constructed clockwise image.
            labels = np.arange(H * W).reshape(H, W)
            rotated = np.array(
                [[labels[r, c] for r in reversed(range(H))] for c in range(W)]
            )
            ref = np.asarray(
                [np.argwhere(rotated == labels[r, c])[0] for r, c in points],
                dtype=float,
            ).reshape(N, 2)
            args = (points, H, W)
        else:
            N = (1, 0, 3, 4, 2, 1, 6, 5)[trial % 8]
            queries = np.column_stack(
                (rng.integers(0, W, size=N), rng.integers(0, H, size=N))
            )
            if trial == 0:
                queries = np.array([[0, 0]])
            rotated = [[grid[r, c] for r in reversed(range(H))] for c in range(W)]
            ref = [rotated[r][c] for r, c in queries]
            args = (grid, queries)
    else:
        xs = rng.integers(-3, 6, size=(3, 0, 1, 4, 6, 2, 5, 8)[trial % 8])
        if name == "F1":
            if trial == 0:
                xs = np.array([2, 0, 5])
            args = (xs,)
            ref = [sum(map(int, xs[::2])) - sum(map(int, xs[1::2]))]
        elif name == "F2":
            k = (2, 1, 3, 2, 4, 1, 7, 3)[trial % 8]
            if trial == 0:
                xs = np.array([0, 4, 0, 8])
            args = (xs, k)
            ref = [sum(int(v) for i, v in enumerate(xs, 1) if i % k == 0)]
        else:
            k = min(len(xs), (3, 0, 1, 0, 2, 2, 5, 4)[trial % 8])
            if trial == 0:
                xs = np.array([2, 0, 5])
                k = 3
            best = [None] * (k + 1)
            best[0] = 0
            for value in xs.tolist():
                for j in range(k, 0, -1):
                    if best[j - 1] is not None:
                        candidate = (
                            best[j - 1] + (1 if (j - 1) % 2 == 0 else -1) * value
                        )
                        best[j] = (
                            candidate if best[j] is None else max(best[j], candidate)
                        )
            args = (xs, k)
            ref = [best[k]]
    return args, np.asarray(ref, dtype=np.float64)


def payload(split):
    cases = [c for c in catalog() if c.split == split]
    if split == "transfer":
        return {
            "split": "transfer",
            "gradient_allowed": False,
            "cases": [c.public() for c in cases],
        }
    if split != "train":
        raise ValueError("train/transfer required")
    # Singleton folds retain every train case as an independent objective.
    return {
        c.name[0]: [[x.public()] for x in cases if x.name[0] == c.name[0]]
        for c in cases
        if c.fold == 0
    }


def write_bundle(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    identities = {}
    for split in ("train", "transfer"):
        path = output / (split + "-cases.json")
        path.write_text(json.dumps(payload(split), ensure_ascii=False, indent=2) + "\n")
        identities[split] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cases": 12 if split == "train" else 6,
        }
    metadata = {
        "schema_version": 1,
        "cases": 18,
        "split": identities,
        "train_seed": TRAIN_SEED,
        "transfer_seed": TRANSFER_SEED,
        "trials": TRIALS,
        "token_count_measured": False,
        "max_tokens_required": 1024,
        "initial_generation_split": "train",
        "transfer_policy": "AV and frozen candidate once each after candidate spec/amplitude freeze; no optimization input",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(write_bundle(parser.parse_args().output)))
