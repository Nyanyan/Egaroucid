"""Bootstrap simultaneous confidence tubes for ordered (Elo, agreement) points.

Inference is per dataset and per curve, with all proposed-model segments joined.
The tube covers the piecewise-linear interpolation of the evaluated parameters;
it makes no assumption about performance at unevaluated alpha/level settings.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


METHOD = "joint_xy_max_standardized_bootstrap_v1"
DEFAULT_SEED = 614
PROPOSED_FAMILIES = ("normal", "squared_blend", "loss_a_normal", "loss_b_loss_a")


def confidence_paths(paths):
    """Keep original parameter order, joining shared proposed endpoints once."""
    curves = {key: list(ids) for key, ids in paths.items() if key not in PROPOSED_FAMILIES}
    proposed = []
    for family in PROPOSED_FAMILIES:
        ids = paths.get(family, [])
        if not ids:
            continue
        if proposed:
            if proposed[-1] != ids[0]:
                raise ValueError("Proposed series must share adjacent endpoints")
            proposed.extend(ids[1:])
        else:
            proposed.extend(ids)
    if proposed:
        if len(set(proposed)) != len(proposed):
            raise ValueError("Proposed confidence path repeats a model")
        curves["proposed"] = proposed
    return curves


def bootstrap_agreement(hits, replicates, rng, batch_size=128):
    """Exact row bootstrap via multinomial counts of identical joint hit vectors.

    A single count vector weights all models. Grouping identical rows reduces
    memory and work without changing the distribution of a row bootstrap.
    """
    hits = np.asarray(hits, dtype=bool)
    if hits.ndim != 2 or min(hits.shape) == 0 or replicates < 2 or batch_size < 1:
        raise ValueError("Nonempty hit matrix and at least two replicates required")
    packed, counts = np.unique(np.packbits(hits, axis=1), axis=0, return_counts=True)
    patterns = np.unpackbits(packed, axis=1, count=hits.shape[1]).astype(float)
    probabilities = counts / counts.sum()
    draws = np.empty((replicates, hits.shape[1]))
    for start in range(0, replicates, batch_size):
        weights = rng.multinomial(len(hits), probabilities, size=min(batch_size, replicates - start))
        draws[start:start + len(weights)] = weights.astype(float) @ patterns / len(hits)
    return draws


def calibrate_tube(point, draws, confidence=.95):
    """Fixed bootstrap-SE sup-t calibration over every vertex and both axes.

    This is an approximate frequentist region, not an assertion that its true
    coverage equals the fraction of bootstrap draws inside it. No nested
    studentization or normal/chi-square ellipse approximation is used.
    """
    point, draws = np.asarray(point, float), np.asarray(draws, float)
    if (point.ndim != 2 or point.shape[1] != 2 or draws.ndim != 3
            or draws.shape[1:] != point.shape or len(draws) < 2
            or not 0 < confidence < 1
            or not np.isfinite(point).all() or not np.isfinite(draws).all()):
        raise ValueError("Expected finite point[K,2] and bootstrap[B,K,2] arrays")
    se = draws.std(axis=0, ddof=1)
    if (se <= 0).any():
        raise ValueError("Degenerate bootstrap coordinate: cannot estimate a simultaneous band")
    maxima = np.max(np.abs(draws - point) / se, axis=(1, 2))
    critical = float(np.quantile(maxima, confidence, method="higher"))
    low, high = point - critical * se, point + critical * se
    # Agreement is a probability. Intersecting the region with parameter space
    # preserves coverage; it does not silently change the bootstrap calibration.
    low[:, 1] = np.maximum(low[:, 1], 0)
    high[:, 1] = np.minimum(high[:, 1], 1)
    return dict(point=point.tolist(), lower=low.tolist(), upper=high.tolist(),
                standard_error=se.tolist(), critical_value=critical,
                bootstrap_max_coverage=float(np.mean(maxima <= critical)))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(data, report_path, name):
    matches = [entry for entry in data.get("sources", []) if Path(entry["path"]).name == name]
    if len(matches) != 1:
        raise ValueError(f"Simultaneous bands require one hashed source named {name} in report_data.json. "
                         "Use --ci-mode pointwise for reports without bootstrap/prediction sources.")
    entry = matches[0]
    relative = Path(entry["path"])
    candidates = [relative] if relative.is_absolute() else [base / relative for base in report_path.parents]
    found = next((path.resolve() for path in candidates if path.is_file()), None)
    if found is None:
        raise ValueError(f"Missing simultaneous-band source: {entry['path']}")
    digest = _sha256(found)
    if digest != entry.get("sha256"):
        raise ValueError(f"Simultaneous-band source SHA-256 mismatch: {found}")
    return found, dict(path=str(found), sha256=digest)


def _load_elo(path, model_ids, models):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        names = next(csv.reader(stream))[1:]
    if len(names) != len(set(names)) or set(names) != set(model_ids):
        raise ValueError("Elo bootstrap model IDs do not match the report")
    table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if table.shape[1] != len(names) + 1 or len(table) < 1000 or not np.isfinite(table).all():
        raise ValueError("Elo bootstrap requires at least 1,000 finite joint replicates")
    if len(np.unique(table[:, 0])) != len(table):
        raise ValueError("Duplicate Elo bootstrap replicate IDs")
    draws = table[:, [names.index(key) + 1 for key in model_ids]]
    bounds = np.quantile(draws, [.025, .975], axis=0).T
    expected = np.array([[model["elo_low"], model["elo_high"]] for model in models])
    if not np.allclose(bounds, expected, rtol=0, atol=1e-6):
        raise ValueError("Elo bootstrap percentiles do not match the report")
    return draws


def _load_hits(path, model_ids):
    with np.load(path, allow_pickle=False) as saved:
        names = saved["model_names"].tolist()
        moves, masks, losses = saved["predicted_policies"], saved["symmetry_target_masks"], saved["losses"]
    if len(names) != len(set(names)) or set(names) != set(model_ids):
        raise ValueError("Prediction model IDs do not match the report")
    if (moves.shape != (len(masks), len(names)) or not np.issubdtype(moves.dtype, np.integer)
            or np.any((moves < 0) | (moves >= 64)) or masks.ndim != 1
            or losses.ndim != 1 or len(losses) not in (0, len(masks))):
        raise ValueError("Invalid saved prediction dimensions or policy indices")
    hits = (np.left_shift(np.uint64(1), moves.astype(np.uint64)) & masks[:, None]) != 0
    return hits[:, [names.index(key) for key in model_ids]], losses


def build_simultaneous_bands(data, report_path, cache_path, seed=DEFAULT_SEED):
    """Validate provenance, reuse an exact-input cache, or compute all conditions."""
    if seed < 0:
        raise ValueError("Band seed must be nonnegative")
    report_path, cache_path = Path(report_path).resolve(), Path(cache_path)
    curves = confidence_paths(data["series_paths"])
    source_paths, evidence = {}, {}
    names = ["strength_elo_bootstrap_replicates.csv", "wthor_test_100000_predictions.npz"]
    if len(data["dataset_counts"]) > 1:
        names.append("mistakes_predictions.npz")
    for name in names:
        source_paths[name], evidence[name] = _source(data, report_path, name)
    signature = dict(method=METHOD, seed=seed, confidence=.95, curves=curves,
                     report_sha256=hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                                           allow_nan=False).encode("utf-8")).hexdigest(),
                     sources=evidence, numpy_version=np.__version__)
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            payload_hash = hashlib.sha256(json.dumps(cached["datasets"], sort_keys=True,
                                                    allow_nan=False).encode("utf-8")).hexdigest()
            if cached.get("signature") == signature and cached.get("datasets_sha256") == payload_hash:
                print(f"Loaded simultaneous bands: {cache_path}", flush=True)
                return cached
        except (ValueError, KeyError, TypeError):
            pass  # Truncated/obsolete cache is rebuilt from the validated sources.
    models = data["models"]
    model_ids = [model["model_id"] for model in models]
    columns = {key: i for i, key in enumerate(model_ids)}
    elo = _load_elo(source_paths[names[0]], model_ids, models)
    predictions = {name: _load_hits(source_paths[name], model_ids) for name in names[1:]}
    if len(predictions[names[1]][1]) != 0:
        raise ValueError("Overall predictions unexpectedly contain loss conditions")
    rng = np.random.default_rng(seed)
    result = dict(signature=signature, replicates=len(elo), datasets={},
                  coverage_scope="Per dataset and curve; proposed alpha segments form one curve. "
                                 "Random has a single-point joint region. Not simultaneous across curves or datasets.",
                  interpolation="Union of convex hulls of adjacent vertex rectangles; parameter order preserved.",
                  assumptions=["Fixed fitted models; independent Elo and agreement evaluation samples.",
                               "Agreement positions are resampled jointly across models, independently across rows; "
                               "no within-game or duplicate-position dependence correction.",
                               "Approximate bootstrap coverage; only evaluated vertices and their linear interpolation. "
                               "No coverage claim for actual unevaluated alpha/level performance.",
                               "No retraining, engine-setting or opening-corpus uncertainty."])
    for k, dataset in enumerate(data["dataset_counts"]):
        key = dataset["dataset"]
        name = names[1] if k == 0 else "mistakes_predictions.npz"
        hits, losses = predictions[name]
        if k:
            if len(losses) != len(hits):
                raise ValueError("Mistake predictions require per-position losses")
            hits = hits[losses >= dataset["minimum_loss"]]
        expected = np.array([model["results"][k]["top1_accuracy"] for model in models])
        if len(hits) != dataset["positions"] or not np.allclose(hits.mean(axis=0), expected, rtol=0, atol=1e-12):
            raise ValueError(f"Prediction counts/accuracy do not match report dataset {key}")
        if ((expected <= 0) | (expected >= 1)).any():
            raise ValueError(f"Boundary agreement in {key}: empirical bootstrap cannot estimate its uncertainty")
        print(f"Bootstrapping joint agreement: {key}, {len(hits):,} positions, {len(elo):,} replicates", flush=True)
        agreement = bootstrap_agreement(hits, len(elo), rng)
        point = np.column_stack(([model["elo"] for model in models], expected))
        draws = np.stack((elo, agreement), axis=2)
        result["datasets"][key] = {}
        for family, ids in curves.items():
            ix = [columns[model_id] for model_id in ids]
            result["datasets"][key][family] = dict(model_ids=ids, **calibrate_tube(point[ix], draws[:, ix]))
    result["datasets_sha256"] = hashlib.sha256(json.dumps(result["datasets"], sort_keys=True,
                                                        allow_nan=False).encode("utf-8")).hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".writing.json")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(cache_path)
    return result


def convex_hull(points):
    """Small monotone-chain hull, counterclockwise, without repeated endpoint."""
    points = sorted(set(map(tuple, np.asarray(points, float))))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    def half(ordered):
        hull = []
        for point in ordered:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], point) <= 0:
                hull.pop()
            hull.append(point)
        return hull
    return np.asarray(half(points)[:-1] + half(points[::-1])[:-1])


def tube_polygons(lower, upper):
    """Exact swept rectangles for linear interpolation, including horizontal caps."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (lower.ndim != 2 or lower.shape[1] != 2 or lower.shape != upper.shape or len(lower) == 0
            or not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper).any()):
        raise ValueError("Expected finite ordered lower/upper[K,2] bounds")
    corners = np.stack((lower, np.column_stack((upper[:, 0], lower[:, 1])), upper,
                        np.column_stack((lower[:, 0], upper[:, 1]))), axis=1)
    if len(corners) == 1:
        return [corners[0]]
    return [convex_hull(np.concatenate((a, b))) for a, b in zip(corners[:-1], corners[1:])]


def draw_tube(axis, bounds, *, color, alpha=.15, gid=None):
    lower, upper = np.asarray(bounds["lower"]).copy(), np.asarray(bounds["upper"]).copy()
    lower[:, 1] *= 100
    upper[:, 1] *= 100
    polygons = tube_polygons(lower, upper)
    # One compound path, all subpaths CCW: the nonzero winding fill forms a
    # union and composites alpha once even where adjacent segments overlap.
    paths = [MplPath(np.vstack((p, p[0])), [MplPath.MOVETO] + [MplPath.LINETO] * (len(p) - 1)
                     + [MplPath.CLOSEPOLY]) for p in polygons]
    patch = PathPatch(MplPath.make_compound_path(*paths), facecolor=color, alpha=alpha,
                      edgecolor="none", linewidth=0, zorder=1, gid=gid)
    axis.add_patch(patch)
    return patch
