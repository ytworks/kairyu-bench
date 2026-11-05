"""Analyze measured finite-response directions without loading or editing a model.

Interior responses and auxiliary quantities are secant predictions, not measured
performance or guarantees. Callers supply objective constraints explicitly.
"""
from __future__ import annotations

import hashlib
import itertools
from typing import Any

import numpy as np


def _array(value: Any, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite {ndim}-dimensional array")
    return result


def solve_minimum_change(
    response: Any,
    gram: Any,
    lower: Any,
    *,
    auxiliary: Any | None = None,
    auxiliary_upper: Any | None = None,
    maximum_mass: float = 1.0,
    minimum_eigenratio: float = 1e-8,
) -> dict[str, Any]:
    """Minimize c.T @ gram @ c subject to explicit secant constraints.

    Adjacent columns are the measured +/- endpoints of one whole direction.
    Coefficients are nonnegative, sum <= maximum_mass, and each pair permits
    only one sign. All <= 256 sign branches are considered. Auxiliary rows
    are caller-defined nonnegative proxies with upper bounds; they are not
    actual interior KL/NLL measurements. SciPy is an optional analysis dependency.

    LP-reported infeasibility is distinguished from numerical/conditioning
    exclusions. A returned candidate has checked primal feasibility and KKT
    residuals for its continuous branch, not a BF16 or model-quality proof.
    """
    m = _array(response, 2, "response")
    g = _array(gram, 2, "gram")
    target = _array(lower, 1, "lower")
    columns = m.shape[1]
    if columns % 2 or not 2 <= columns <= 16:
        raise ValueError("require 1 through 8 adjacent signed endpoint pairs")
    if g.shape != (columns, columns) or target.shape != (m.shape[0],):
        raise ValueError("response, gram and lower dimensions disagree")
    if not np.isfinite(maximum_mass) or not 0 < maximum_mass <= 1:
        raise ValueError("maximum_mass must be in (0, 1]")
    if not np.isfinite(minimum_eigenratio) or not 0 < minimum_eigenratio < 1:
        raise ValueError("minimum_eigenratio must be in (0, 1)")
    scale = float(np.max(np.abs(g)))
    if scale == 0 or not np.allclose(g, g.T, rtol=1e-12, atol=scale * 1e-12):
        raise ValueError("gram must be nonzero and symmetric")
    g = (g + g.T) / 2
    if np.linalg.eigvalsh(g / scale)[0] < -1e-10:
        raise ValueError("gram must be positive semidefinite")
    a, b = -m, -target
    if (auxiliary is None) != (auxiliary_upper is None):
        raise ValueError("auxiliary and auxiliary_upper must be provided together")
    aux: np.ndarray | None = None
    if auxiliary is not None:
        aux = _array(auxiliary, 2, "auxiliary")
        upper = _array(auxiliary_upper, 1, "auxiliary_upper")
        if aux.shape[1] != columns or upper.shape != (aux.shape[0],):
            raise ValueError("auxiliary dimensions disagree")
        if np.any(aux < 0) or np.any(upper < 0):
            raise ValueError("auxiliary proxies and bounds must be nonnegative")
        a, b = np.r_[a, aux], np.r_[b, upper]
    a, b = np.r_[a, np.ones((1, columns))], np.r_[b, maximum_mass]
    row_scale = np.maximum(np.max(np.abs(a), axis=1), np.maximum(np.abs(b), 1e-12))
    a, b = a / row_scale[:, None], b / row_scale
    try:
        from scipy.optimize import linprog, minimize, nnls
    except ImportError as exc:
        raise RuntimeError("finite-response optimization requires optional scipy>=1.11") from exc

    branches: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for signs in itertools.product((0, 1), repeat=columns // 2):
        indices = np.arange(columns // 2) * 2 + signs
        q = g[np.ix_(indices, indices)]
        eigen = np.linalg.eigvalsh(q / scale)
        ratio = float(eigen[0] / eigen[-1]) if eigen[-1] > 0 else 0.0
        row: dict[str, Any] = {"indices": indices.tolist(), "eigenratio": ratio}
        if ratio < minimum_eigenratio:
            row["status"] = "excluded_conditioning"
            branches.append(row)
            continue
        # Scale the objective, never the relative costs of coordinates.
        h = q / float(eigen[-1] * scale)
        constraints = a[:, indices]
        lp = linprog(
            np.zeros(len(indices)), A_ub=constraints, b_ub=b, bounds=(0, None),
            method="highs", options={"time_limit": 5.0,
                                    "primal_feasibility_tolerance": 1e-9,
                                    "dual_feasibility_tolerance": 1e-9},
        )
        if lp.status != 0:
            row["status"] = "lp_infeasible" if lp.status == 2 else "lp_unsolved"
            row["solver_status"] = int(lp.status)
            branches.append(row)
            continue
        result = minimize(
            lambda x: 0.5 * float(x @ h @ x), lp.x, jac=lambda x: h @ x,
            constraints=[{"type": "ineq", "fun": lambda x: b - constraints @ x,
                          "jac": lambda x: -constraints}],
            bounds=[(0, None)] * len(indices), method="SLSQP",
            options={"ftol": 1e-13, "maxiter": 500},
        )
        x = np.asarray(result.x)
        cmat = np.r_[-constraints, np.eye(len(x))]
        rhs = np.r_[-b, np.zeros(len(x))]
        slack = cmat @ x - rhs
        active = np.flatnonzero(slack <= 2e-7)
        try:
            dual, _ = nnls(cmat[active].T, h @ x, maxiter=10000)
            stationarity = float(np.max(np.abs(h @ x - cmat[active].T @ dual)))
            complementarity = float(np.max(np.abs(dual * slack[active]), initial=0))
            v = cmat[active].T @ dual
            primal = 0.5 * float(x @ h @ x)
            dual_objective = float(rhs[active] @ dual - 0.5 * v @ np.linalg.solve(h, v))
            gap = primal - dual_objective
        except (RuntimeError, np.linalg.LinAlgError):
            row["status"] = "dual_unsolved"
            branches.append(row)
            continue
        valid = bool(np.isfinite(x).all() and np.min(slack) >= -2e-8
                     and stationarity <= 2e-6 and complementarity <= 2e-7
                     and -2e-7 <= gap <= 2e-7)
        row.update(
            status="verified_continuous_branch" if valid else "qp_unsolved",
            optimizer_success=bool(result.success),
            minimum_scaled_slack=float(slack.min()),
            stationarity_max=stationarity,
            complementarity_max=complementarity,
            normalized_duality_gap=gap,
        )
        if valid:
            coefficient = np.zeros(columns)
            coefficient[indices] = np.maximum(x, 0)
            prediction = m @ coefficient
            # Recheck the exact emitted coefficients in original response units.
            unscaled_slack = np.r_[prediction - target, maximum_mass - coefficient.sum()]
            if aux is not None:
                unscaled_slack = np.r_[unscaled_slack, upper - aux @ coefficient]
            if unscaled_slack.min() < -2e-8:
                row["status"] = "unscaled_validation_failed"
            else:
                row.update(coefficients=coefficient.tolist(),
                           predicted_response=prediction.tolist(),
                           predicted_auxiliary=None if aux is None else (aux @ coefficient).tolist(),
                           continuous_l2=float(np.sqrt(max(0.0, coefficient @ g @ coefficient))),
                           minimum_unscaled_slack=float(unscaled_slack.min()))
                candidates.append(row)
        branches.append(row)
    best = min(candidates, key=lambda row: row["continuous_l2"]) if candidates else None
    unresolved = sum(row["status"] not in {"verified_continuous_branch", "lp_infeasible"}
                     for row in branches)
    counts = {status: sum(row["status"] == status for row in branches)
              for status in sorted({row["status"] for row in branches})}
    return {
        "status": "surrogate_candidate" if best else "no_verified_surrogate_candidate",
        "selected": best, "branches": branches, "branch_counts": counts,
        "all_branches_resolved": unresolved == 0,
        "response_kind": "secant_prediction_requires_new_finite_measurement",
        "model_calls": 0, "checkpoint_writes": 0,
    }


def bf16_realization(
    base: Any, endpoint_deltas: Any, coefficients: Any, *, chunk_elements: int = 16384,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Round original BF16 base + combined F64 delta once (F32 then BF16 RNE).

    Return values and stored-coordinate difference metrics, without file writes.
    This measures quantization geometry only, never the resulting model response.
    """
    original64 = np.asarray(base, dtype=np.float64)
    original = original64.astype(np.float32)
    if not np.array_equal(original64, original.astype(np.float64)):
        raise ValueError("base must contain exact finite BF16 values")
    delta = np.asarray(endpoint_deltas)
    c = _array(coefficients, 1, "coefficients")
    if original.ndim != 1 or not original.size or delta.shape != (original.size, c.size):
        raise ValueError("base, endpoint_deltas and coefficients dimensions disagree")
    if type(chunk_elements) is not int or not 1 <= chunk_elements <= 1048576:
        raise ValueError("chunk_elements must be in [1, 1048576]")
    if not np.isfinite(original).all() or np.any(original.view(np.uint32) & 0xFFFF):
        raise ValueError("base must contain exact finite BF16 values")
    if c.size % 2 or np.any(c < 0) or c.sum() > 1 + 1e-10:
        raise ValueError("coefficients must be nonnegative signed pairs with sum <= 1")
    if np.any((c[0::2] > 1e-12) & (c[1::2] > 1e-12)):
        raise ValueError("opposite endpoints from one direction cannot be combined")
    active = np.flatnonzero(c)
    stored = np.empty_like(original)
    changed, square, intended_square, error_square, dot = 0, 0.0, 0.0, 0.0, 0.0
    digest = hashlib.sha256()
    for start in range(0, original.size, chunk_elements):
        end = min(start + chunk_elements, original.size)
        raw_block = delta[start:end]
        if not np.isfinite(raw_block).all():
            raise ValueError("endpoint_deltas must be finite")
        block = np.asarray(raw_block[:, active], dtype=np.float64)
        intended = block @ c[active]
        with np.errstate(over="ignore", invalid="ignore"):
            pre = (original[start:end].astype(np.float64) + intended).astype(np.float32)
            bits = pre.view(np.uint32)
            rounded_bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
            rounded = rounded_bits.view(np.float32)
        if not np.isfinite(rounded).all():
            raise ValueError("BF16 composition overflowed")
        stored[start:end] = rounded
        actual = rounded.astype(np.float64) - original[start:end].astype(np.float64)
        error = actual - intended
        changed += int(np.count_nonzero(actual))
        square += float(actual @ actual)
        intended_square += float(intended @ intended)
        error_square += float(error @ error)
        dot += float(actual @ intended)
        digest.update(rounded.astype("<f4", copy=False).tobytes())
    norm, intended_norm = float(np.sqrt(square)), float(np.sqrt(intended_square))
    return stored, {
        "changed_elements": changed, "realized_l2": norm, "continuous_l2": intended_norm,
        "rounding_relative_l2": None if intended_norm == 0 else float(np.sqrt(error_square) / intended_norm),
        "cosine_realized_to_continuous": None if norm * intended_norm == 0 else dot / (norm * intended_norm),
        "stored_f32_sha256": digest.hexdigest(),
        "rounding": "original BF16 base + F64 combined delta -> F32 -> BF16 ties-to-even once",
        "model_response_measured": False,
    }


def pareto_indices(benefits: Any, costs: Any, *, tolerance: float = 1e-12) -> list[int]:
    """Non-dominated rows: maximize every explicit benefit, minimize every cost."""
    b, c = _array(benefits, 2, "benefits"), _array(costs, 2, "costs")
    if b.shape[0] != c.shape[0] or b.shape[0] > 4096:
        raise ValueError("matching row counts <=4096 required")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    values = np.c_[b, -c]
    return [i for i, row in enumerate(values)
            if not any(np.all(other >= row - tolerance) and np.any(other > row + tolerance)
                       for j, other in enumerate(values) if i != j)]



def project_bf16_response(
    base: Any, endpoint_deltas: Any, coefficients: Any, response: Any, gram: Any,
    *, minimum_eigenratio: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project the rounded delta onto its *active signed endpoint cone*.

    Never project onto all +/- columns: their near-null cancellation could turn
    numerical residual into apparent response benefit. NNLS forbids new signs.
    A small residual is geometric evidence only, not a network linearity proof.
    """
    c = _array(coefficients, 1, "coefficients")
    m, g = _array(response, 2, "response"), _array(gram, 2, "gram")
    d = np.asarray(endpoint_deltas)
    original = np.asarray(base, dtype=np.float32)
    if m.shape[1] != c.size or g.shape != (c.size, c.size):
        raise ValueError("response and gram dimensions disagree")
    if not np.isfinite(minimum_eigenratio) or not 0 < minimum_eigenratio < 1:
        raise ValueError("minimum_eigenratio must be in (0, 1)")
    values, metrics = bf16_realization(base, d, c)
    active = np.flatnonzero(c > 0)
    report: dict[str, Any] = {
        **metrics, "response_kind": "reprojected_secant_prediction_requires_finite_measurement",
        "active_indices": active.tolist(), "projection_status": "zero_realized_delta",
        "projection_in_measured_simplex": False,
    }
    if not metrics["changed_elements"]:
        return values, report
    q = g[np.ix_(active, active)]
    scale = float(np.max(np.abs(q)))
    if not scale or not np.allclose(q, q.T, rtol=1e-12, atol=scale * 1e-12):
        raise ValueError("active gram must be nonzero and symmetric")
    q = (q + q.T) / 2
    eigen = np.linalg.eigvalsh(q / scale)
    ratio = float(eigen[0] / eigen[-1])
    report["active_gram_eigenratio"] = ratio
    if ratio < minimum_eigenratio:
        report["projection_status"] = "excluded_conditioning"
        return values, report
    cross = np.zeros(len(active))
    observed_gram = np.zeros_like(q)
    for begin in range(0, len(original), 16384):
        end = min(begin + 16384, len(original))
        block = d[begin:end, active].astype(np.float64)
        actual = values[begin:end].astype(np.float64) - original[begin:end].astype(np.float64)
        cross += block.T @ actual
        observed_gram += block.T @ block
    if not np.allclose(q, observed_gram, rtol=1e-10, atol=scale * 1e-12):
        raise ValueError("supplied active gram disagrees with endpoint coordinates")
    try:
        from scipy.optimize import nnls
    except ImportError as exc:
        raise RuntimeError("finite-response projection requires optional scipy>=1.11") from exc
    normalized = q / scale
    chol = np.linalg.cholesky(normalized)
    projected, _ = nnls(chol.T, np.linalg.solve(chol, cross / scale), maxiter=10000)
    full = np.zeros(c.size)
    full[active] = projected
    residual_square = 0.0
    for begin in range(0, len(original), 16384):
        end = min(begin + 16384, len(original))
        block = d[begin:end, active].astype(np.float64)
        actual = values[begin:end].astype(np.float64) - original[begin:end].astype(np.float64)
        residual = actual - block @ projected
        residual_square += float(residual @ residual)
    residual_l2 = float(np.sqrt(residual_square))
    report.update(
        projection_status="projected_active_cone",
        projected_coefficients=full.tolist(),
        projected_mass=float(full.sum()),
        projection_in_measured_simplex=bool(full.sum() <= 1 + 1e-9),
        projection_residual_l2=residual_l2,
        projection_relative_residual=residual_l2 / metrics["realized_l2"],
        projected_response=(m @ full).tolist(),
    )
    return values, report



def signed_simplex_grid(endpoint_count: int) -> list[np.ndarray]:
    """Fixed bounded grid: one to three directions, coefficients .25/.5/1.

    At most one sign per adjacent pair and total mass <=1. This enumerates
    proposals only; it does not execute any model or score a candidate.
    """
    if type(endpoint_count) is not int or endpoint_count % 2 or not 2 <= endpoint_count <= 16:
        raise ValueError("require 1 through 8 adjacent signed endpoint pairs")
    result = []
    for count in range(1, min(3, endpoint_count // 2) + 1):
        for active in itertools.combinations(range(endpoint_count), count):
            if len({i // 2 for i in active}) != count:
                continue
            for weights in itertools.product((0.25, 0.5, 1.0), repeat=count):
                if sum(weights) > 1:
                    continue
                coefficient = np.zeros(endpoint_count)
                coefficient[list(active)] = weights
                result.append(coefficient)
    return result
