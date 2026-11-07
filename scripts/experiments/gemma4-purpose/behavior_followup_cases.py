"""Nine new contracts after F03: train6 / held-out transfer3; no model calls."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

import behavior_cases as original_cases

from behavior_cases import (
    Case,
    TRAIN_SEED as TRAIN_SEED,
    TRANSFER_SEED as TRANSFER_SEED,
    TRIALS,
)


def catalog():
    return [
        Case("G1", "changes, start, lower, upper",
             "changes is a 1D integer array. lower<=upper are integers and start is integer. "
             "A balance starts at start and applies changes in order. Return(1,) containing1 "
             "iff the initial balance AND every intermediate balance stay in[lower,upper], else0.",
             "states=np.r_[start,start+np.cumsum(changes)]\n"
             "ok=np.all((states>=lower)&(states<=upper))\nreturn np.array([ok],dtype=float)",
             "np.all((states>=lower)&(states<=upper))",
             "(states[-1]>=lower)&(states[-1]<=upper)", exact=True),
        Case("G2", "bounds, step",
             "bounds is(T,2) integer inclusive intervals [low,high], T>=1, low<=high. "
             "step is an integer>=0. Choose one integer at every stage within its interval, "
             "with absolute change between consecutive choices<=step. Return(1,) containing1 "
             "iff a full valid sequence exists, otherwise0. An impossible prefix cannot be repaired.",
             "lo,hi=map(int,bounds[0])\nfor a,b in bounds[1:]:\n"
             "    lo=max(int(a),lo-step)\n    hi=min(int(b),hi+step)\n"
             "    if lo>hi:\n        return np.array([0.0])\n"
             "return np.array([float(lo<=hi)])",
             "    if lo>hi:\n        return np.array([0.0])", "    pass", exact=True),
        Case("G3", "maps, allowed, start",
             "maps is(T,n) integers in[0,n), allowed is(T+1,n) booleans, n>=1; "
             "start is in[0,n). Begin at start, then state=maps[t,state] each stage. "
             "Return(1,) containing1 iff allowed[0,start] and allowed[t+1,state] "
             "hold at EVERY stage, else0. T=0 is allowed.",
             "state=int(start)\nok=bool(allowed[0,state])\n"
             "for t in range(len(maps)):\n    state=int(maps[t,state])\n"
             "    ok=ok and bool(allowed[t+1,state])\nreturn np.array([ok],dtype=float)",
             "ok=ok and bool(allowed[t+1,state])",
             "ok=bool(allowed[t+1,state])", exact=True),
        Case("H1", "times, values",
             "times and values are 1D real arrays of equal length n>=2; times strictly increases. "
             "Return(n-1,) where result[i]=(values[i+1]-values[i])/(times[i+1]-times[i]). "
             "These are adjacent interval secants, not derivatives at sample points.",
             "return (np.diff(values)/np.diff(times)).astype(float)",
             "np.diff(values)/np.diff(times)", "np.gradient(values,times)[:-1]"),
        Case("H2", "values, spacing",
             "values is a 1D real array with n>=3 samples on a uniform grid; spacing is positive. "
             "Return(n-2,) containing centered two-neighbor slopes at interior sample points: "
             "result[j]=(values[j+2]-values[j])/(2*spacing). Do not use one-sided interval slopes.",
             "return ((values[2:]-values[:-2])/(2*spacing)).astype(float)",
             "(values[2:]-values[:-2])/(2*spacing)", "(values[2:]-values[1:-1])/spacing"),
        Case("H3", "times, values",
             "times and values are equal-length 1D real arrays, n>=2, times strictly increasing. "
             "For every adjacent interval compute its secant slope. Return(1,) containing "
             "the sum of ABSOLUTE interval slopes (not sample-point derivatives and not weighted by duration).",
             "slopes=np.diff(values)/np.diff(times)\nreturn np.array([np.abs(slopes).sum()],dtype=float)",
             "np.diff(values)/np.diff(times)", "np.gradient(values,times)"),
        Case("I1", "labels",
             "labels is a 1D integer array with values0..9, possibly empty. Return(1,) containing "
             "the unique most frequent label. If empty or multiple labels share the highest frequency, "
             "return the sentinel -99. Never resolve a frequency tie by numeric order.",
             "u,c=np.unique(labels,return_counts=True)\nif len(u)==0:\n"
             "    return np.array([-99.0])\nw=u[c==c.max()]\n"
             "return np.array([w[0] if len(w)==1 else -99],dtype=float)",
             "w[0] if len(w)==1 else -99", "w[0]", exact=True),
        Case("I2", "labels",
             "labels is a 1D integer array with values0..9, possibly empty. Rank DISTINCT frequency "
             "values from largest to smallest. Return(1,) containing the unique label with the SECOND "
             "largest distinct frequency. Return -88 if fewer than two distinct frequencies exist "
             "or multiple labels have that second frequency.",
             "u,c=np.unique(labels,return_counts=True)\nf=np.unique(c)\n"
             "if len(f)<2:\n    return np.array([-88.0])\nw=u[c==f[-2]]\n"
             "return np.array([w[0] if len(w)==1 else -88],dtype=float)",
             "w[0] if len(w)==1 else -88", "w[0]", exact=True),
        Case("I3", "labels, minimum",
             "labels is a 1D integer array with values0..9, possibly empty; minimum is integer>=1. "
             "Return(1,) containing the unique most frequent label ONLY when its frequency is at least "
             "minimum. Empty, tied-highest or insufficient frequency returns sentinel -77.",
             "u,c=np.unique(labels,return_counts=True)\nif len(u)==0:\n"
             "    return np.array([-77.0])\nmax_count=int(c.max())\nw=u[c==max_count]\n"
             "return np.array([w[0] if len(w)==1 and max_count>=minimum else -77],dtype=float)",
             "max_count>=minimum", "max_count>minimum", exact=True),
    ]


def sample(name, rng, trial):
    if name == "J1" or name[0] in "ABCDEF":
        return original_cases.sample("F2" if name == "J1" else name, rng, trial)
    if not 0 <= trial < TRIALS:
        raise ValueError("eight fixed plus eight seeded trials required")
    fixed = trial < 8
    n = (3, 2, 4, 1, 0, 5, 2, 3)[trial % 8]
    if name == "G1":
        fixed_changes = [[3, -3], [-3, 3], [], [0], [1, 1, -1], [-1, -1, 1], [2], [-2]]
        changes = np.array(fixed_changes[trial] if fixed else rng.integers(-3, 4, n))
        start, lower, upper = 0, -2, 2
        states, state = [start], start
        for change in changes:
            state += int(change)
            states.append(state)
        args = (changes, start, lower, upper)
        ref = [int(all(lower <= v <= upper for v in states))]
    elif name == "G2":
        fixtures = [
            ([[0, 0], [3, 3], [0, 4]], 1), ([[0, 0]], 0),
            ([[0, 1], [1, 2], [2, 3]], 1), ([[0, 0], [2, 2], [0, 4]], 0),
            ([[-2, -1], [1, 2]], 1), ([[0, 1], [0, 1]], 0),
            ([[1, 1], [0, 2], [1, 1]], 0), ([[-3, -3], [-1, -1]], 2),
        ]
        if fixed:
            b, step = fixtures[trial]
            bounds = np.asarray(b)
        else:
            bounds = np.sort(rng.integers(-4, 5, (max(1, n), 2)), axis=1)
            step = int(rng.integers(0, 3))
        reachable = set(range(int(bounds[0, 0]), int(bounds[0, 1]) + 1))
        for low, high in bounds[1:]:
            reachable = {x for x in range(int(low), int(high)+1)
                         if any(abs(x-y) <= step for y in reachable)}
        args, ref = (bounds, step), [int(bool(reachable))]
    elif name == "G3":
        if fixed:
            maps = np.array([[1, 0], [1, 0]], dtype=int) if trial != 4 else np.empty((0, 2), int)
            allowed = np.ones((len(maps)+1, 2), dtype=bool)
            if trial in (0, 2, 6):
                allowed[1, 1] = False
            if trial in (1, 3, 4):
                allowed[0, 0] = False
            if trial == 7:
                allowed[-1, 0] = False
        else:
            maps = rng.integers(0, 3, (n, 3))
            allowed = rng.random((n+1, 3)) > .3
        start, state = 0, 0
        visited = [bool(allowed[0, state])]
        for t, mapping in enumerate(maps):
            state = int(mapping[state])
            visited.append(bool(allowed[t+1, state]))
        args, ref = (maps, allowed, start), [int(all(visited))]
    elif name.startswith("H"):
        if fixed:
            ys = [[0, 1, 4], [1, -1, 2, -2], [3, 3, 3], [0, 2, 4, 6],
                  [0, 0, 4], [-3, 2, -1], [2, 1, 0, 3], [1, 0, 1]][trial]
            values = np.array(ys, dtype=float)
            times = np.cumsum(np.arange(1, len(values)+1, dtype=float))
        else:
            values = rng.normal(size=max(3, n))
            times = np.cumsum(rng.uniform(.5, 2, len(values)))
        if name == "H2":
            spacing = float(trial % 3 + 1)
            ref = [((float(values[i+1])-float(values[i])) +
                    (float(values[i])-float(values[i-1])))/(2*spacing)
                   for i in range(1, len(values)-1)]
            args = (values, spacing)
        else:
            slopes = [(float(values[i+1])-float(values[i])) /
                      (float(times[i+1])-float(times[i])) for i in range(len(values)-1)]
            args = (times, values)
            ref = slopes if name == "H1" else [sum(abs(x) for x in slopes)]
    else:
        fixtures = [[1, 1, 2, 2, 3], [], [2], [0, 0, 0, 1, 1, 2, 2],
                    [4, 4, 4, 1, 1, 2], [1, 2, 3], [0, 0, 1], [9, 9, 9, 9]]
        labels = np.array(fixtures[trial] if fixed else rng.integers(0, 5, n+3), dtype=int)
        counts = Counter(int(x) for x in labels)
        if name == "I2":
            frequencies = sorted(set(counts.values()), reverse=True)
            candidates = [x for x, count in counts.items() if len(frequencies) >= 2 and count == frequencies[1]]
            ref = [candidates[0] if len(candidates) == 1 else -88]
            args = (labels,)
        else:
            maximum = max(counts.values(), default=0)
            candidates = [x for x, count in counts.items() if count == maximum]
            if name == "I1":
                ref, args = [candidates[0] if len(candidates) == 1 else -99], (labels,)
            else:
                minimum = max(1, maximum) if fixed else int(rng.integers(1, 5))
                ref = [candidates[0] if len(candidates) == 1 and maximum >= minimum else -77]
                args = (labels, minimum)
    return args, np.asarray(ref, dtype=np.float64)


def payload(split):
    selected = [c for c in catalog() if c.split == split]
    if split == "transfer":
        return {"split": split, "gradient_allowed": False, "cases": [c.public() for c in selected]}
    if split != "train":
        raise ValueError("train/transfer required")
    return {family: [[c.public()] for c in selected if c.name[0] == family]
            for family in ("G", "H", "I")}


def offset_contrast():
    """A separate ID contrasts the observed offset error; original F2 is intact."""
    case = next(c for c in original_cases.catalog() if c.name == "F2")
    return replace(case, name="J1", new="values[0::k]")


def training_catalog():
    """All old12 and follow-up6 train cases, plus the separate J1 contrast."""
    return (
        [c for c in original_cases.catalog() if c.split == "train"]
        + [c for c in catalog() if c.split == "train"]
        + [offset_contrast()]
    )


def training_payload():
    """Preserve existing purpose/fold rows; add J as one singleton fold."""
    return {
        **original_cases.payload("train"),
        **payload("train"),
        "J": [[offset_contrast().public()]],
    }
