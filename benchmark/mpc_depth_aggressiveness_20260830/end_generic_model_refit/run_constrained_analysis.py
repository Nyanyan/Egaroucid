#!/usr/bin/env python3
"""Fit endgame MPC sigma cubics that are positive on the whole engine table.

This is deliberately a separate experiment from run_analysis.py.  It changes
the six coefficients (a..f) themselves and does not multiply sigma by a global
scale.  Positivity is imposed by expressing the cubic in the Bernstein basis
over the complete u-domain and constraining every Bernstein coefficient to be
at least ``--minimum-sigma``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("end_generic_analysis", HERE / "run_analysis.py")
assert SPEC is not None and SPEC.loader is not None
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def bernstein_design(v: float, lower_u: float, upper_u: float) -> list[float]:
    t = (v - lower_u) / (upper_u - lower_u)
    one = 1.0 - t
    return [one ** 3, 3.0 * t * one * one, 3.0 * t * t * one, t ** 3]


def u_interval(ratio: float) -> tuple[float, float]:
    # v = n_discs / 64 + ratio * shallow_depth / 60.  The four corners
    # determine its full range for n_discs=0..64 and shallow_depth=0..60.
    corners = (0.0, 1.0, ratio, 1.0 + ratio)
    return min(corners), max(corners)


def nnls_four(normal: list[list[float]], target: list[float]) -> list[float]:
    """Solve the four-variable non-negative least-squares normal equation.

    Enumerating the 16 possible free-variable sets is exact for this tiny
    convex quadratic problem and avoids an external numerical dependency.
    """
    best_value = math.inf
    best = [0.0, 0.0, 0.0, 0.0]
    for mask in range(16):
        free = [index for index in range(4) if mask & (1 << index)]
        candidate = [0.0, 0.0, 0.0, 0.0]
        if free:
            matrix = [[normal[i][j] for j in free] for i in free]
            vector = [target[i] for i in free]
            trace = sum(matrix[i][i] for i in range(len(free)))
            for i in range(len(free)):
                matrix[i][i] += max(1.0e-16, trace * 1.0e-13)
            try:
                solution = BASE.solve(matrix, vector)
            except ValueError:
                continue
            if any(value < -1.0e-9 for value in solution):
                continue
            for index, value in zip(free, solution):
                candidate[index] = max(0.0, value)
        value = 0.5 * sum(
            candidate[i] * normal[i][j] * candidate[j]
            for i in range(4) for j in range(4)
        ) - sum(target[i] * candidate[i] for i in range(4))
        if value < best_value:
            best_value = value
            best = candidate
    return best


def pointwise_constrained_solution(
    normal: list[list[float]], target: list[float], constraints: list[list[float]],
    minimum_sigma: float,
) -> list[float]:
    """Solve a 4-D convex quadratic with A*x >= minimum_sigma.

    The active-set solution changes the cubic only as much as the data-fitting
    objective requires.  Unlike a penalty, the returned coefficients satisfy
    the actual table inequalities (up to floating-point tolerance).
    """
    stabilized = [row[:] for row in normal]
    trace = sum(stabilized[i][i] for i in range(4))
    for i in range(4):
        stabilized[i][i] += max(1.0e-16, trace * 1.0e-13)
    active: list[int] = []
    solution = BASE.solve(stabilized, target)
    for _ in range(100):
        multipliers: list[float] = []
        if active:
            size = 4 + len(active)
            kkt = [[0.0] * size for _ in range(size)]
            rhs = target[:] + [minimum_sigma] * len(active)
            for i in range(4):
                for j in range(4):
                    kkt[i][j] = stabilized[i][j]
            for offset, constraint_index in enumerate(active):
                design = constraints[constraint_index]
                for i in range(4):
                    kkt[i][4 + offset] = -design[i]
                    kkt[4 + offset][i] = design[i]
            try:
                kkt_solution = BASE.solve(kkt, rhs)
            except ValueError:
                # The newly added point can be linearly dependent on an
                # existing active point.  Keeping the older point is enough.
                active.pop()
                continue
            solution = kkt_solution[:4]
            multipliers = kkt_solution[4:]
            if multipliers and min(multipliers) < -1.0e-8:
                active.pop(min(range(len(active)), key=lambda i: multipliers[i]))
                continue
        values = [sum(row[i] * solution[i] for i in range(4)) for row in constraints]
        worst = min(range(len(values)), key=values.__getitem__)
        if values[worst] >= minimum_sigma - 1.0e-9:
            return solution
        if worst in active:
            raise RuntimeError("active-set solver stalled on a violated constraint")
        if len(active) >= 4:
            raise RuntimeError("active-set solver exceeded the polynomial dimension")
        active.append(worst)
    raise RuntimeError("active-set solver did not converge")


def fit_for_ratio(
    buckets: list[tuple[int, int, float, float]], ratio: float,
    minimum_sigma: float, constraint_method: str, prior_strength: float,
) -> tuple[list[float], float, tuple[float, float]]:
    lower_u, upper_u = u_interval(ratio)
    normal = [[0.0] * 4 for _ in range(4)]
    target = [0.0] * 4
    constant = 0.0
    for n_discs, shallow, sigma, weight in buckets:
        v = n_discs / 64.0 + ratio * shallow / 60.0
        design = bernstein_design(v, lower_u, upper_u)
        adjusted = sigma - minimum_sigma
        constant += weight * adjusted * adjusted
        for i in range(4):
            target[i] += weight * design[i] * adjusted
            for j in range(4):
                normal[i][j] += weight * design[i] * design[j]
    if prior_strength > 0.0:
        prior_weight = prior_strength / 121.0
        for x_index in range(11):
            for y_index in range(11):
                n_discs = 64.0 * x_index / 10.0
                shallow = 60.0 * y_index / 10.0
                v = n_discs / 64.0 + ratio * shallow / 60.0
                design = bernstein_design(v, lower_u, upper_u)
                sigma = BASE.sigma_current(n_discs, shallow)
                adjusted = sigma - minimum_sigma
                constant += prior_weight * adjusted * adjusted
                for i in range(4):
                    target[i] += prior_weight * design[i] * adjusted
                    for j in range(4):
                        normal[i][j] += prior_weight * design[i] * design[j]
    if constraint_method == "bernstein":
        heights = nnls_four(normal, target)
        controls = [minimum_sigma + value for value in heights]
    elif constraint_method in ("pointwise", "intercept_shift"):
        # In this branch the four unknowns are the Bernstein controls
        # themselves, so rebuild the target without subtracting the floor.
        target = [0.0] * 4
        constant = 0.0
        for n_discs, shallow, sigma, weight in buckets:
            v = n_discs / 64.0 + ratio * shallow / 60.0
            design = bernstein_design(v, lower_u, upper_u)
            constant += weight * sigma * sigma
            for i in range(4):
                target[i] += weight * design[i] * sigma
        if prior_strength > 0.0:
            prior_weight = prior_strength / 121.0
            for x_index in range(11):
                for y_index in range(11):
                    n_discs = 64.0 * x_index / 10.0
                    shallow = 60.0 * y_index / 10.0
                    v = n_discs / 64.0 + ratio * shallow / 60.0
                    design = bernstein_design(v, lower_u, upper_u)
                    sigma = BASE.sigma_current(n_discs, shallow)
                    constant += prior_weight * sigma * sigma
                    for i in range(4):
                        target[i] += prior_weight * design[i] * sigma
        table_constraints = []
        if constraint_method == "pointwise":
            seen: set[tuple[float, ...]] = set()
            for n_discs in range(65):
                for shallow in range(61):
                    v = n_discs / 64.0 + ratio * shallow / 60.0
                    design = bernstein_design(v, lower_u, upper_u)
                    key = tuple(round(value, 14) for value in design)
                    if key not in seen:
                        seen.add(key)
                        table_constraints.append(design)
        if constraint_method == "pointwise":
            controls = pointwise_constrained_solution(
                normal, target, table_constraints, minimum_sigma
            )
        else:
            controls = BASE.solve(normal, target)
            p3, p2, p1, p0 = bernstein_to_power(
                controls, lower_u, upper_u
            )
            candidates = [lower_u, upper_u]
            discriminant = 4.0 * p2 * p2 - 12.0 * p3 * p1
            if abs(p3) > 1.0e-14 and discriminant >= 0.0:
                root = math.sqrt(discriminant)
                candidates.extend((
                    (-2.0 * p2 - root) / (6.0 * p3),
                    (-2.0 * p2 + root) / (6.0 * p3),
                ))
            elif abs(p2) > 1.0e-14:
                candidates.append(-p1 / (2.0 * p2))
            candidates = [
                value for value in candidates
                if lower_u <= value <= upper_u
            ]
            table_minimum = min(
                p3 * value ** 3 + p2 * value ** 2 + p1 * value + p0
                for value in candidates
            )
            # Adding the same amount to all four Bernstein controls is exactly
            # an update of the polynomial intercept f, because the basis sums
            # to one.  It is not a multiplicative sigma scale.
            correction = max(0.0, minimum_sigma - table_minimum)
            controls = [value + correction for value in controls]
        heights = controls
    else:
        raise ValueError(constraint_method)
    objective = constant - 2.0 * sum(target[i] * heights[i] for i in range(4))
    objective += sum(
        heights[i] * normal[i][j] * heights[j]
        for i in range(4) for j in range(4)
    )
    return controls, objective, (lower_u, upper_u)


def bernstein_to_power(
    controls: list[float], lower_u: float, upper_u: float
) -> list[float]:
    """Return p3,p2,p1,p0 for q(v)=p3*v^3+... from Bernstein controls."""
    y0, y1, y2, y3 = controls
    t0 = y0
    t1 = 3.0 * (y1 - y0)
    t2 = 3.0 * (y0 - 2.0 * y1 + y2)
    t3 = -y0 + 3.0 * y1 - 3.0 * y2 + y3
    width = upper_u - lower_u
    p3 = t3 / width ** 3
    p2 = t2 / width ** 2 - 3.0 * lower_u * t3 / width ** 3
    p1 = (
        t1 / width - 2.0 * lower_u * t2 / width ** 2
        + 3.0 * lower_u * lower_u * t3 / width ** 3
    )
    p0 = (
        t0 - lower_u * t1 / width + lower_u * lower_u * t2 / width ** 2
        - lower_u ** 3 * t3 / width ** 3
    )
    return [p3, p2, p1, p0]


def fit_constrained_model(
    rows: list[dict[str, Any]], mode: str, minimum_sigma: float,
    constraint_method: str, prior_strength: float,
) -> dict[str, Any]:
    buckets = BASE.bucket_targets(rows, mode)
    best: tuple[float, float, list[float], tuple[float, float]] | None = None
    # Keep the same structural direction as the production model:
    # a=-1 and b=-ratio with ratio>=0.
    center = 15.0
    radius = 15.0
    for _ in range(5):
        for index in range(121):
            ratio = center - radius + 2.0 * radius * index / 120.0
            if ratio < 0.0:
                continue
            controls, objective, interval = fit_for_ratio(
                buckets, ratio, minimum_sigma, constraint_method
                , prior_strength
            )
            candidate = (objective, ratio, controls, interval)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        center = best[1]
        radius /= 10.0
    assert best is not None
    objective, ratio, controls, (lower_u, upper_u) = best
    p3, p2, p1, p0 = bernstein_to_power(controls, lower_u, upper_u)
    # The fitted polynomial is q(v), v=x+ratio*y.  Egaroucid evaluates the
    # same curve at u=-v, hence c=-p3, d=p2, e=-p1, f=p0.
    return {
        "name": (
            f"constrained_{constraint_method}_{mode}_min{minimum_sigma:g}"
            f"_prior{prior_strength:g}"
        ),
        "weighting": mode,
        "constraint_method": constraint_method,
        "prior_strength": prior_strength,
        "minimum_sigma": minimum_sigma,
        "coefficients": [-1.0, -ratio, -p3, p2, -p1, p0],
        "bernstein_controls": controls,
        "u_interval_for_v": [lower_u, upper_u],
        "fit_objective": objective,
        "bucket_count": len(buckets),
    }


def raw_sigma(model: dict[str, Any], n_discs: int, shallow: int) -> float:
    a, b, c, d, e, f = model["coefficients"]
    u = a * n_discs / 64.0 + b * shallow / 60.0
    return c * u ** 3 + d * u ** 2 + e * u + f


def domain_audit(model: dict[str, Any]) -> dict[str, Any]:
    values = [
        (raw_sigma(model, n_discs, shallow), n_discs, shallow)
        for n_discs in range(65) for shallow in range(61)
    ]
    minimum = min(values)
    maximum = max(values)
    return {
        "model": model["name"],
        "table_entries": len(values),
        "minimum_raw_sigma": minimum[0],
        "minimum_n_discs": minimum[1],
        "minimum_shallow_depth": minimum[2],
        "maximum_raw_sigma": maximum[0],
        "nonpositive_entries": sum(value <= 0.0 for value, _, _ in values),
        "below_0_5_entries": sum(value < 0.5 - 1.0e-9 for value, _, _ in values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-sigma", type=float, default=0.5)
    parser.add_argument(
        "--constraint-method",
        choices=("intercept_shift", "pointwise", "bernstein"),
        default="bernstein",
    )
    parser.add_argument("--output", type=Path, default=HERE / "results_constrained")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    historical_raw = {
        label: BASE.load_jsonl(path) for label, path in BASE.HISTORICAL_DEV.items()
    }
    historical_obs = {
        label: BASE.deduplicate_observations(rows)
        for label, rows in historical_raw.items()
    }
    june_raw = BASE.load_jsonl(BASE.HISTORICAL_HOLDOUT)
    june_obs = BASE.deduplicate_observations(june_raw)
    high_raw = BASE.load_jsonl(BASE.NEW_HIGH)
    high_obs = BASE.deduplicate_observations(high_raw)
    deep_raw = BASE.load_jsonl(BASE.NEW_DEEP)
    deep_obs = BASE.deduplicate_observations(deep_raw)
    deep30_obs = BASE.deduplicate_observations(BASE.load_jsonl(BASE.KNOWN_DEEP30))

    new_all_obs = high_obs + deep_obs
    new_train_obs = [row for row in new_all_obs if not BASE.locked_holdout(row)]
    new_holdout_obs = [row for row in new_all_obs if BASE.locked_holdout(row)]
    train_obs = [row for rows in historical_obs.values() for row in rows] + new_train_obs

    current = {"name": "current", "coefficients": list(BASE.CURRENT_COEFFICIENTS)}
    prior_grid = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)

    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        fold_train = [row for row in train_obs if BASE.cv_fold(row) != fold]
        fold_test = [row for row in train_obs if BASE.cv_fold(row) == fold]
        fold_models = [current] + [
            fit_constrained_model(
                fold_train, mode, args.minimum_sigma, args.constraint_method,
                prior_strength,
            )
            for mode in ("root_equal", "domain_equal")
            for prior_strength in prior_grid
        ]
        for model in fold_models:
            fold_rows.append({
                "fold": fold,
                "model": model["name"],
                **BASE.weighted_metrics(fold_test, model),
            })
    cv_rows: list[dict[str, Any]] = []
    model_names = ["current"] + sorted({
        str(row["model"]) for row in fold_rows if row["model"] != "current"
    })
    for model_name in model_names:
        selected = [row for row in fold_rows if row["model"] == model_name]
        weight = sum(float(row["roots"]) for row in selected)
        aggregate: dict[str, Any] = {
            "model": model_name,
            "folds": len(selected),
            "roots": int(weight),
            "observations": int(sum(float(row["observations"]) for row in selected)),
        }
        for metric in (
            "residual_rmse", "residual_mae", "sigma_rmse", "sigma_mae",
            "standardized_rms", "gaussian_nll",
        ):
            aggregate[metric] = sum(
                float(row[metric]) * float(row["roots"]) for row in selected
            ) / weight
        cv_rows.append(aggregate)

    cv_by_name = {str(row["model"]): row for row in cv_rows}
    selected_priors: dict[str, float] = {}
    for mode in ("root_equal", "domain_equal"):
        candidates = []
        for prior_strength in prior_grid:
            if prior_strength == 0.0:
                continue
            name = (
                f"constrained_{args.constraint_method}_{mode}_min"
                f"{args.minimum_sigma:g}_prior{prior_strength:g}"
            )
            candidates.append((prior_strength, cv_by_name[name]))
        selected_priors[mode] = min(
            candidates, key=lambda item: float(item[1]["gaussian_nll"])
        )[0]
    final_models = [current] + [
        fit_constrained_model(
            train_obs, mode, args.minimum_sigma, args.constraint_method,
            selected_priors[mode],
        )
        for mode in ("root_equal", "domain_equal")
    ]
    fitted_grid_models = [
        fit_constrained_model(
            train_obs, mode, args.minimum_sigma, args.constraint_method,
            prior_strength,
        )
        for mode in ("root_equal", "domain_equal")
        for prior_strength in prior_grid
    ]

    holdouts = {
        "historical_202606_deep10_18": june_obs,
        "locked_new_deep19_26": new_holdout_obs,
        "known_exact_deep30": deep30_obs,
    }
    holdout_rows = [
        {"dataset": label, "model": model["name"], **BASE.weighted_metrics(rows, model)}
        for label, rows in holdouts.items() for model in final_models
    ]

    high_holdout_roots = {BASE.root_key(row) for row in new_holdout_obs}
    high_contexts = BASE.select_contexts(
        high_raw + deep_raw, high_holdout_roots, include_level=True
    )
    june_contexts = BASE.select_contexts(june_raw, None, include_level=False)
    cut_rows: list[dict[str, Any]] = []
    for model in final_models:
        for level in range(3):
            cut_rows.append({
                "dataset": "locked_new_deep19_26",
                **BASE.simulate_cuts(high_contexts, model, level, True),
            })
        for level in range(3, 6):
            cut_rows.append({
                "dataset": "historical_202606_deep10_18_simulated_high_selectivity",
                **BASE.simulate_cuts(june_contexts, model, level, False),
            })

    coefficient_rows = []
    for model in final_models:
        coefficient_rows.append({
            "model": model["name"],
            **dict(zip(("a", "b", "c", "d", "e", "f"), model["coefficients"])),
            "global_sigma_multiplier": 1.0,
        })
    audit_rows = [domain_audit(model) for model in final_models]
    coefficient_grid_rows = [{
        "model": model["name"],
        **dict(zip(("a", "b", "c", "d", "e", "f"), model["coefficients"])),
        "global_sigma_multiplier": 1.0,
    } for model in fitted_grid_models]
    audit_grid_rows = [domain_audit(model) for model in fitted_grid_models]
    training_rows = [
        {"model": model["name"], **BASE.weighted_metrics(train_obs, model)}
        for model in final_models
    ]

    write_csv(args.output / "coefficients.csv", coefficient_rows)
    write_csv(args.output / "coefficient_grid.csv", coefficient_grid_rows)
    write_csv(args.output / "domain_audit.csv", audit_rows)
    write_csv(args.output / "domain_audit_grid.csv", audit_grid_rows)
    write_csv(args.output / "training_metrics.csv", training_rows)
    write_csv(args.output / "cv_folds.csv", fold_rows)
    write_csv(args.output / "cv_summary.csv", cv_rows)
    write_csv(args.output / "holdout_metrics.csv", holdout_rows)
    write_csv(args.output / "cut_simulation.csv", cut_rows)
    payload = {
        "method": (
            f"cubic constrained by {args.constraint_method} inequalities to "
            "minimum_sigma; no global sigma multiplier"
        ),
        "minimum_sigma": args.minimum_sigma,
        "prior_grid": list(prior_grid),
        "selected_priors_by_cv_gaussian_nll": selected_priors,
        "training_observations": len(train_obs),
        "models": final_models,
        "coefficients": coefficient_rows,
        "coefficient_grid": coefficient_grid_rows,
        "domain_audit": audit_rows,
        "domain_audit_grid": audit_grid_rows,
        "training_metrics": training_rows,
        "cv_summary": cv_rows,
        "holdout_metrics": holdout_rows,
        "cut_simulation": cut_rows,
    }
    (args.output / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "coefficients": coefficient_rows,
        "domain_audit": audit_rows,
        "cv_summary": cv_rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
