"""Plot the five approved 24-model cohorts from verified saved measurements.

This module only reads experiment artifacts. It neither imports evaluation
controllers nor recalculates Elo or neural predictions. The eight-level
Egaroucid confidence tube is calibrated from saved predictions and Elo draws.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

from label_editor import (
    LABEL_PREFIX, LabelEditor, annotation_position, apply_label_positions,
    apply_text_overrides, arrange_unplaced_labels, load_label_positions,
    load_text_overrides,
)
from simultaneous_bands import METHOD, bootstrap_agreement, calibrate_tube, draw_tube


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN = SCRIPT_DIR.parents[4] / "experiment_bat/20261016_gpw_add_zebra/followup21"
DEFAULT_OUTPUT = SCRIPT_DIR / "output/gpw2026_paper_figures"
DEFAULT_FONT_SIZE = 18
AXIS_FONT_SIZE_INCREMENT = 6
DATASETS = {
    "wthor_100k": ("WTHOR 全体（共通10万局面）", 100000),
    "wthor_loss_ge_2": ("WTHOR 2石損以上", 94189),
    "quest_800-1000": ("オセロクエスト [800, 1000)", 100000),
    "quest_1400-1600": ("オセロクエスト [1400, 1600)", 100000),
    "quest_2000-2200": ("オセロクエスト [2000, 2200)", 100000),
}
PAPER_FIGURES = tuple(DATASETS)
OUTPUT_STEMS = dict(zip(PAPER_FIGURES, (
    "result-wthor-all", "result-wthor-loss2", "result-oq-800-1000",
    "result-oq-1400-1600", "result-oq-2000-2200",
)))
LEGEND_LOCATIONS = {
    "wthor_100k": "lower right",
    "wthor_loss_ge_2": "upper right",
    "quest_800-1000": "lower right",
    "quest_1400-1600": "upper right",
    "quest_2000-2200": "lower right",
}
ALPHAS = (0., .2, .4, .6, .7, .8, .9, 1., 1.2, 1.4, 1.6, 1.8, 2.)
ALPHA_NAMES = [f"alpha_{a:.1f}" for a in ALPHAS]
BETA_NAMES = ["alpha_1.0", *[f"beta_l1_{b:.1f}" for b in (.2, .4, .6, .8)], "egaroucid_l1"]
EGA_NAMES = [f"egaroucid_l{level}" for level in (1, 2, 3, 5, 9, 13, 17)]
NAMES = set(ALPHA_NAMES + BETA_NAMES + EGA_NAMES)
CURVES = {
    "proposed": dict(names=ALPHA_NAMES, label="提案手法 α", color="#D62728", linestyle="-"),
    "beta_l1": dict(names=BETA_NAMES, label="通常モデル＋L1 β", color="#D62728", linestyle="--"),
    "egaroucid": dict(names=EGA_NAMES, label="Egaroucid", color="#249972", linestyle="-"),
}
# Display names and local alpha values follow Table 5 of the GPW 2026 paper.
# The saved 13-point confidence curve is sliced for display, without recalibration.
PAPER_CURVES = {
    "level21_normal": dict(names=ALPHA_NAMES[:8], label="レベル21-通常モデル",
        color="#D62728", linestyle="-", marker="o", source="proposed", indices=list(range(8))),
    "normal_level1": dict(names=BETA_NAMES, label="通常モデル-レベル1",
        color="#D98700", linestyle="--", marker="s", source="beta_l1", indices=list(range(6))),
    "normal_loss": dict(names=ALPHA_NAMES[7:], label="通常モデル-石損モデル",
        color="#C43F72", linestyle="-.", marker="D", source="proposed", indices=list(range(7,13))),
    "egaroucid": dict(names=[*EGA_NAMES, "alpha_0.0"], label="Egaroucid",
        color="#249972", linestyle=":", marker="^", source="egaroucid_lv1_to_lv21", indices=list(range(8))),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Sources:
    """Hash and parse the same bytes, including files replaced by a publisher."""
    def __init__(self, root):
        self.root = Path(root)
        self.entries = {}

    def read(self, path, expected=None):
        path = Path(path)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        require(expected is None or digest == expected, f"SHA-256 mismatch: {path}")
        self.entries[path.relative_to(self.root).as_posix()] = dict(sha256=digest, size=len(content))
        return content

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected).decode("utf-8-sig"))

    def csv(self, path, expected=None):
        return list(csv.DictReader(io.StringIO(self.read(path, expected).decode("utf-8-sig"))))


def validate_measurements(key, names, predictions, masks, rows, curves, signature):
    """Independently recount each saved cell and check every band vertex."""
    count = DATASETS[key][1]
    require(len(names) == 24 and set(names) == NAMES, f"24-model roster mismatch: {key}")
    require(predictions.shape == (count, 24) and masks.shape == (count,), f"Cohort shape mismatch: {key}")
    require(np.all((predictions >= 0) & (predictions < 64)) and np.all(masks != 0),
            f"Missing predictions or target masks: {key}")
    hits = ((masks[:, None] >> predictions.astype(np.uint64)) & np.uint64(1)) != 0
    require(hashlib.sha256(hits.tobytes()).hexdigest() == signature, f"Bootstrap cohort mismatch: {key}")
    counts = hits.sum(axis=0)
    require(set(rows) == NAMES, f"CSV roster mismatch: {key}")
    for index, name in enumerate(names):
        row = rows[name]
        require(row["complete"] == "True" and int(row["positions"]) == int(row["completed_positions"]) == count,
                f"Incomplete result: {key}/{name}")
        require(int(row["hits"]) == int(counts[index]) and
                abs(float(row["accuracy"]) - int(counts[index]) / count) < 1e-12,
                f"NPZ/CSV agreement mismatch: {key}/{name}")
    require(set(curves) == set(CURVES), f"Confidence curves mismatch: {key}")
    for group, spec in CURVES.items():
        bounds = curves[group]
        point = np.array([[float(rows[n]["elo"]), float(rows[n]["accuracy"])] for n in spec["names"]])
        saved, low, high = [np.asarray(bounds[field], dtype=float) for field in ("point", "lower", "upper")]
        require(saved.shape == low.shape == high.shape == point.shape and
                np.isfinite([saved, low, high]).all() and np.allclose(saved, point, atol=1e-10, rtol=0),
                f"Confidence-band coordinates mismatch: {key}/{group}")
        require(np.all(low <= saved) and np.all(saved <= high) and
                np.all(low[:, 1] >= 0) and np.all(high[:, 1] <= 1), f"Invalid confidence bounds: {key}/{group}")


def load_paper_data(campaign=DEFAULT_CAMPAIGN):
    campaign = Path(campaign).resolve()
    sources = Sources(campaign)
    base, quest = campaign / "human_agreement", campaign / "quest_all_100k"
    proofs = {base: sources.json(base / "validation.json"), quest: sources.json(quest / "priority_validation.json")}
    require((proofs[base].get("status"), proofs[base].get("models"), proofs[base].get("complete_results")) ==
            ("passed", 24, 264), "Original 24-model evaluation is not verified")
    require((proofs[quest].get("status"), proofs[quest].get("models"), proofs[quest].get("datasets"),
             proofs[quest].get("complete_results"), proofs[quest].get("rows_per_band")) ==
            ("passed", 24, 3, 72, 100000), "Quest priority-three evaluation is not verified")
    identities = {folder: sources.json(folder / "identity.json", proof["identity_sha256"])
                  for folder, proof in proofs.items()}
    for folder, identity in identities.items():
        require(len(identity["model_names"]) == 24 and set(identity["model_names"]) == NAMES,
                f"Source roster mismatch: {folder}")
    require(identities[quest].get("include_literal_and_symmetry_effective_single_choice") is True and
            identities[quest].get("original_row_multiplicities_preserved") is True,
            "Quest must include single-choice rows and preserve row multiplicities")
    require(identities[quest]["source_identity_sha256"] == proofs[base]["identity_sha256"],
            "Quest and WTHOR refer to different original experiments")
    identity = identities[quest]
    ranking = sources.csv(campaign / "current_ranking.csv", identity["strength_ranking_sha256"])
    require(len(ranking) == 24 and {r["name"] for r in ranking} == NAMES, "Elo roster mismatch")
    ranking = {r["name"]: r for r in ranking}
    require(abs(np.mean([float(r["elo"]) for r in ranking.values()]) - 1500) < 1e-6, "Elo mean must be 1500")
    sources.read(campaign / "strength/strength_games.jsonl", identity["strength_raw_sha256"])
    elo_path = campaign / "strength/strength_elo_bootstrap_replicates.csv"
    elo_content = sources.read(elo_path, identity["elo_bootstrap_sha256"])
    header = next(csv.reader(io.StringIO(elo_content.decode("utf-8-sig"))))[1:]
    elo_draws = np.loadtxt(io.BytesIO(elo_content), delimiter=",", skiprows=1, ndmin=2)
    require(len(header) == 24 and set(header) == NAMES and elo_draws.shape == (20000, 25) and
            np.isfinite(elo_draws).all(), "Invalid 20,000 Elo bootstrap replicates")
    for name in header:
        interval = np.quantile(elo_draws[:, header.index(name) + 1], [.025, .975])
        require(np.allclose(interval, [float(ranking[name][k]) for k in ("elo_ci_low", "elo_ci_high")],
                            atol=1e-6, rtol=0), f"Elo interval mismatch: {name}")
    tables, bands = {}, {}
    for folder in (base, quest):
        table = sources.csv(folder / "agreement_results.csv")
        tables[folder] = {(r["dataset"], r["model"]): r for r in table}
        require(len(tables[folder]) == len(table), f"Duplicate CSV cells: {folder}")
        bands[folder] = sources.json(folder / "simultaneous_bands.json")
        b = bands[folder]
        require(b.get("method") == METHOD and b.get("replicates") == 20000 and b.get("confidence") == .95 and
                b.get("identity_sha256") == proofs[folder]["identity_sha256"] and
                b.get("elo_source_sha256") == identity["elo_bootstrap_sha256"], f"Confidence-band source mismatch: {folder}")
    result = dict(schema_version=1, campaign=str(campaign), models=list(identities[base]["model_names"]),
                  curves=deepcopy(PAPER_CURVES), source_confidence_curves=deepcopy(CURVES), datasets={}, sources=sources.entries,
                  confidence=dict(method=METHOD, replicates=20000, confidence=.95,
                                  scope="Per dataset and per curve, jointly across vertices and both axes",
                                  assumptions=bands[quest]["assumptions"]),
                  model_definitions=identity["formulas"],
                  endpoint_aliases={"beta_l1_0.0": "alpha_1.0", "beta_l1_1.0": "egaroucid_l1"})
    ega_names = PAPER_CURVES["egaroucid"]["names"]
    ega_hits = {}
    for key, (label, count) in DATASETS.items():
        folder = base if key.startswith("wthor_") else quest
        filename = key + ("_predictions.npz" if folder == base else "_priority_predictions.npz")
        entries = {e["path"]: e["sha256"] for e in proofs[folder]["prediction_files"]}
        require(filename in entries, f"Missing verified prediction file: {filename}")
        content = sources.read(folder / filename, entries[filename])
        rows = {n: tables[folder][key, n] for n in result["models"]}
        with np.load(io.BytesIO(content), allow_pickle=False) as z:
            names = list(map(str, z["model_names"]))
            require(names == identities[folder]["model_names"], f"Prediction column order mismatch: {key}")
            validate_measurements(key, names, z["predicted_policies"], z["symmetry_target_masks"], rows,
                                  bands[folder]["datasets"][key], bands[folder]["prediction_signatures"][key])
            moves = z["predicted_policies"][:, [names.index(n) for n in ega_names]].astype(np.uint64)
            ega_hits[key] = ((z["symmetry_target_masks"][:, None] >> moves) & np.uint64(1)) != 0
            if folder == quest:
                band = key.removeprefix("quest_")
                source_index = identities[quest]["bands"].index(band)
                sample_identity = identities[quest]["datasets"][source_index]
                sample_content = sources.read(quest / "samples" / (key + ".npz"), sample_identity["sample_sha256"])
                with np.load(io.BytesIO(sample_content), allow_pickle=False) as sample:
                    for field in ("records", "game_ids", "symmetry_target_masks", "source_row_indices", "original_splits"):
                        require(np.array_equal(z[field], sample[field]), f"Quest sample differs: {key}/{field}")
        for name, row in rows.items():
            require(all(abs(float(row[k]) - float(ranking[name][k])) < 1e-9
                        for k in ("elo", "elo_ci_low", "elo_ci_high")), f"CSV/Elo mismatch: {key}/{name}")
        result["datasets"][key] = dict(label=label, positions=count, rows=list(rows.values()),
            bands=deepcopy(bands[folder]["datasets"][key]),
            cohort=("Unaugmented WTHOR test; seed 613 common 100,000 rows" if key == "wthor_100k" else
                    "All unaugmented WTHOR test rows with disc loss >= 2" if folder == base else
                    "Both players in the half-open rating band; unaugmented train+val+test; 100,000 rows; single-choice rows included"))
    # Recalibrate the entire eight-vertex path; appending a separately calibrated
    # endpoint rectangle would not provide a joint 95% tube for this path.
    ega_elo = elo_draws[:, [header.index(n) + 1 for n in ega_names]]
    result["extended_egaroucid_confidence"] = dict(method=METHOD, replicates=20000,
        confidence=.95, model_names=ega_names, numpy_version=np.__version__, datasets={})
    for index, key in enumerate(DATASETS):
        seed = 614 + index
        hits = ega_hits[key]
        rows = {r["model"]: r for r in result["datasets"][key]["rows"]}
        point = np.array([[float(rows[n]["elo"]), float(rows[n]["accuracy"])] for n in ega_names])
        agreement = bootstrap_agreement(hits, 20000, np.random.default_rng(seed))
        draws = np.stack((ega_elo, agreement), axis=2)
        calibrated = calibrate_tube(point, draws)
        result["datasets"][key]["bands"]["egaroucid_lv1_to_lv21"] = calibrated
        result["extended_egaroucid_confidence"]["datasets"][key] = dict(seed=seed,
            positions=len(hits), hits_sha256=hashlib.sha256(hits.tobytes()).hexdigest(),
            critical_value=calibrated["critical_value"],
            bootstrap_max_coverage=calibrated["bootstrap_max_coverage"])
    return result


def point_label(name):
    if name == "alpha_0.0": return "α=0.0（Lv. 21）"
    if name == "egaroucid_l1": return "α=1.0\n（Lv. 1）"
    if name == "alpha_1.0": return "α=1.0（通常モデル）"
    if name == "alpha_2.0": return "α=1.0（石損モデル）"
    if name.startswith("alpha_"):
        value = float(name.removeprefix("alpha_"))
        return f"α={value if value < 1 else value-1:.1f}"
    if name.startswith("beta_l1_"): return "α=" + name.removeprefix("beta_l1_")
    return "Lv. " + name.removeprefix("egaroucid_l")


def paper_curve_bounds(dataset, spec):
    """Keep the saved critical value while selecting the displayed vertices."""
    bounds = deepcopy(dataset["bands"][spec["source"]])
    for field in ("point", "lower", "upper", "standard_error"):
        require(max(spec["indices"]) < len(bounds[field]),
                "Confidence tube is missing a displayed endpoint")
        bounds[field] = [bounds[field][i] for i in spec["indices"]]
        require(len(bounds[field]) == len(spec["names"]),
                "Every displayed curve vertex requires confidence bounds")
    return bounds


def paper_y_limits(dataset):
    """Round each cohort's confidence bounds outward to multiples of 10%."""
    values = np.concatenate([np.asarray(paper_curve_bounds(dataset, spec)[side])[:, 1] * 100
                             for spec in PAPER_CURVES.values() for side in ("lower", "upper")])
    return (max(0, int(np.floor((values.min() - 2) / 10) * 10)),
            min(100, int(np.ceil((values.max() + 2) / 10) * 10)))


def mark_best_agreement(dataset, annotations):
    """Set initial default percentages; never call after applying user edits."""
    if dataset["rows"][0]["dataset"] != "wthor_100k":
        return ensure_best_agreement(dataset, annotations)
    best_hits = max(int(row["hits"]) for row in dataset["rows"])
    best = {row["model"]: row for row in dataset["rows"] if int(row["hits"]) == best_hits}
    result = {}
    for uid, annotation in annotations.items():
        name = uid.rsplit("/", 1)[-1]
        if name not in best:
            continue
        row = best[name]
        percent = 100 * int(row["hits"]) / int(row["positions"])
        # Refresh an existing suffix rather than duplicating a saved annotation.
        text = re.sub(r"[,，、\n]\s*\d+(?:\.\d+)?\s*[%％]\s*$", "", annotation.get_text())
        separator = "\n" if text == "石損モデル" else ", "
        annotation.set_text(f"{text}{separator}{percent:.2f}%")
        result[name] = dict(hits=best_hits, positions=int(row["positions"]),
                            accuracy_percent=percent, text=annotation.get_text())
    ensure_best_agreement(dataset, annotations)
    return result


def describe_best_agreement(dataset, annotations):
    """Record measured maxima and current label text without mutating artists."""
    best_hits = max(int(row["hits"]) for row in dataset["rows"])
    best = {row["model"]:row for row in dataset["rows"] if int(row["hits"]) == best_hits}
    return {name:dict(hits=best_hits, positions=int(best[name]["positions"]),
                     accuracy_percent=100 * best_hits / int(best[name]["positions"]),
                     text=annotation.get_text())
            for uid, annotation in annotations.items()
            if (name := uid.rsplit("/", 1)[-1]) in best}


def ensure_best_agreement(dataset, annotations):
    """Keep endpoint alpha labels and the four requested overall-WTHOR values."""
    best = describe_best_agreement(dataset, annotations)
    show_percent = dataset["rows"][0]["dataset"] == "wthor_100k"
    rows = {row["model"]: row for row in dataset["rows"]}
    requested = {"alpha_0.0", "alpha_1.0", "alpha_0.7", "alpha_0.8"}
    old_endpoints = {"alpha_0.0": "α=0.0", "alpha_1.0": "通常モデル",
                     "alpha_2.0": "石損モデル", "egaroucid_l1": "α=1.0"}
    for uid, annotation in annotations.items():
        name = uid.rsplit("/", 1)[-1]
        text = annotation.get_text()
        # Migrate only former default names; preserve manual names and positions.
        if name in old_endpoints and text.startswith(old_endpoints[name]):
            tail = text[len(old_endpoints[name]):]
            if not tail or re.fullmatch(r"[,，、\n]\s*\d+(?:\.\d+)?\s*[%％]", tail):
                text = point_label(name) + tail
                annotation.set_text(text)
        if not show_percent:
            if name in best:
                annotation.set_text(re.sub(r"(?:[,，、\n]\s*|\s+)?[(（]?\d+(?:\.\d+)?\s*[%％][)）]?\s*$", "", text))
            continue
        if name not in requested:
            continue
        if re.search(r"\d+(?:\.\d+)?\s*[%％]", text):
            continue
        separator = "\n" if text == "石損モデル" or re.fullmatch(r"α=\d+(?:\.\d+)?", text) else ", "
        percent = 100 * int(rows[name]["hits"]) / int(rows[name]["positions"])
        annotation.set_text(f"{text}{separator}{percent:.2f}%")
    return describe_best_agreement(dataset, annotations)


def create_paper_figure(data, key, ci_alpha=.15, font_size=DEFAULT_FONT_SIZE):
    from plot_rating_vs_human_match import configure_plotting
    configure_plotting()
    axis_font_size = font_size + AXIS_FONT_SIZE_INCREMENT
    plt.rcParams.update({"font.size": font_size, "axes.labelsize": axis_font_size,
                         "xtick.labelsize": axis_font_size, "ytick.labelsize": axis_font_size,
                         "axes.titlesize": font_size + 2, "pdf.fonttype": 42})
    dataset = data["datasets"][key]
    rows = {r["model"]: r for r in dataset["rows"]}
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=.10, right=.98, bottom=.10, top=.97)
    ax.set_gid("dataset:" + key)
    for group, spec in PAPER_CURVES.items():
        draw_tube(ax, paper_curve_bounds(dataset, spec), color=spec["color"], alpha=ci_alpha,
                  gid="simultaneous-band:" + group)
        ordered = [rows[n] for n in spec["names"]]
        ax.plot([float(r["elo"]) for r in ordered], [100 * float(r["accuracy"]) for r in ordered],
                color=spec["color"], linestyle=spec["linestyle"], lw=2, label=spec["label"], zorder=3,
                gid="curve:" + group)
    annotations = {}
    for name in data["models"]:
        row = rows[name]
        group = ("normal_level1" if name == "egaroucid_l1" or name.startswith("beta_")
                 else "egaroucid" if name.startswith("egaroucid_")
                 else "normal_loss" if float(name.removeprefix("alpha_")) > 1 else "level21_normal")
        spec = PAPER_CURVES[group]
        color, marker = spec["color"], spec["marker"]
        xy = float(row["elo"]), 100 * float(row["accuracy"])
        ax.plot(*xy, marker=marker, ms=7, linestyle="none", color=color,
                markerfacecolor="white" if marker == "s" else color, zorder=4, gid="model-point:" + name)
        uid = key + "/" + name
        annotation = ax.annotate(point_label(name), xy, xytext=(9, 8), textcoords="offset points",
            color=color, fontsize=font_size - 1, annotation_clip=False, multialignment="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=.7), zorder=5)
        annotation.set_gid(LABEL_PREFIX + uid)
        annotations[uid] = annotation
    mark_best_agreement(dataset, annotations)
    # Include horizontal and vertical band bounds before arranging labels.
    all_bounds = np.concatenate([np.asarray(paper_curve_bounds(d, spec)[side])
        for d in data["datasets"].values() for spec in PAPER_CURVES.values() for side in ("lower", "upper")])
    x0, x1 = all_bounds[:, 0].min(), all_bounds[:, 0].max()
    ax.set_xlim(x0 - (x1 - x0) * .08, x1 + (x1 - x0) * .08)
    y_limits = paper_y_limits(dataset)
    ax.set_ylim(*y_limits)
    ax.set_yticks(np.arange(y_limits[0], y_limits[1] + 1, 10))
    ax.set_yticks(np.arange(y_limits[0] + 5, y_limits[1], 10), minor=True)
    ax.set_xlabel("Eloレーティング")
    ax.set_ylabel("着手一致率（%）")
    # Overlay the confidence fill and line in each series' legend sample.
    legend_handles = [
        (Patch(facecolor=spec["color"], edgecolor="none", alpha=ci_alpha),
         Line2D([], [], color=spec["color"], linestyle=spec["linestyle"], lw=2))
        for spec in PAPER_CURVES.values()
    ]
    ax.legend(legend_handles, [spec["label"] for spec in PAPER_CURVES.values()],
              handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
              loc=LEGEND_LOCATIONS[key], ncol=1, frameon=False,
              borderaxespad=.8, fontsize=axis_font_size)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.grid(color="#E5EAF0", lw=.7)
    ax.grid(which="minor", axis="y", color="#E5EAF0", lw=.7)
    ax.set_axisbelow(True)
    return fig, annotations


def render_paper(campaign=DEFAULT_CAMPAIGN, output_dir=DEFAULT_OUTPUT, figures=PAPER_FIGURES,
                 formats=("png", "svg", "pdf"), ci_alpha=.15, dpi=300,
                 label_positions_path=None, adjust_labels=False, reset_labels=False,
                 font_size=DEFAULT_FONT_SIZE):
    from plot_rating_vs_human_match import enable_interactive_backend, save_figure
    require(bool(figures) and len(set(figures)) == len(figures) and all(k in DATASETS for k in figures), "Unknown or duplicate paper figure")
    require(bool(formats) and all(f in ("png", "svg", "pdf") for f in formats), "Formats must be png, svg, pdf")
    require(0 <= ci_alpha <= 1 and dpi >= 1 and font_size >= 8, "Invalid opacity, resolution, or font size")
    start = time.monotonic()
    output = Path(output_dir).resolve()
    source = Path(campaign).resolve()
    require(output != source and source not in output.parents, "Paper outputs must be outside the experiment directory")
    output.mkdir(parents=True, exist_ok=True)

    def progress(message):
        print(message, flush=True)
        with (output / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    progress("VERIFY: five fixed cohorts, 24 models, saved NPZ/CSV/Elo and joint XY bands; ETA provisional")
    data = load_paper_data(campaign)
    positions_path = Path(label_positions_path or output / "label_positions.json").resolve()
    saved = {} if reset_labels else load_label_positions(positions_path)
    texts = {} if reset_labels else load_text_overrides(positions_path)
    if adjust_labels: enable_interactive_backend()
    (output / "plot_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(data["datasets"][PAPER_FIGURES[0]]["rows"][0])
    with (output / "agreement_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for dataset in data["datasets"].values(): writer.writerows(dataset["rows"])
    generated = []
    manifest = dict(schema_version=1, campaign=str(source), models=24, figures=list(figures),
        datasets={k:dict(label=v["label"], positions=v["positions"], cohort=v["cohort"]) for k,v in data["datasets"].items()},
        sources=data["sources"], curves=deepcopy(PAPER_CURVES), confidence=data["confidence"],
        graph_title=False, figure_size_inches=[12, 6.5], legend_locations=LEGEND_LOCATIONS,
        y_limits_percent={k:paper_y_limits(v) for k,v in data["datasets"].items()},
        axis_font_size=font_size + AXIS_FONT_SIZE_INCREMENT,
        legend_font_size=font_size + AXIS_FONT_SIZE_INCREMENT,
        default_model_labels={n:point_label(n) for n in data["models"]},
        output_stems=OUTPUT_STEMS, automatic_copy=False, output_directory=str(output),
        display_alpha="Local 0..1 within each of the three proposed curves (paper Table 5)",
        confidence_display="The saved proposed tube retains its calibration; the full eight-level Egaroucid tube is recalibrated jointly",
        extended_egaroucid_confidence=data["extended_egaroucid_confidence"],
        endpoint_aliases=data["endpoint_aliases"], ci_alpha=ci_alpha, dpi=dpi, font_size=font_size,
        scatter_errorbars=False, label_positions_path=str(positions_path), label_positions={},
        highest_agreement={}, text_overrides={}, outputs=generated)
    manifest["highest_agreement_display_datasets"] = ["wthor_100k"]
    manifest["agreement_value_display_models"] = {"wthor_100k": ["alpha_0.0", "alpha_1.0", "alpha_0.7", "alpha_0.8"]}
    for index, key in enumerate(figures):
        fig, annotations = create_paper_figure(data, key, ci_alpha, font_size)
        apply_label_positions(fig, saved.get(key, {}))
        text_targets = apply_text_overrides(fig, texts.get(key, {}))
        ensure_best_agreement(data["datasets"][key], annotations)
        arrange_unplaced_labels(fig, annotations, set(saved.get(key, {})))

        def save():
            manifest["highest_agreement"][key] = describe_best_agreement(data["datasets"][key], annotations)
            stem = OUTPUT_STEMS[key]
            for name in save_figure(fig, output, stem, formats, dpi):
                if name not in generated: generated.append(name)
            manifest["label_positions"][key] = {uid:asdict(annotation_position(a)) for uid,a in annotations.items()}
            manifest["text_overrides"][key] = {uid:t.artist.get_text() for uid,t in text_targets.items()}
            (output / "plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            if adjust_labels:
                fig._label_editor = LabelEditor(fig, annotations, key, positions_path, save, text_targets,
                    before_save=lambda: ensure_best_agreement(data["datasets"][key], annotations))
                fig.canvas.manager.set_window_title(key + " — ドラッグ／ダブルクリック編集／Sで保存／Rで復元")
                progress(f"EDIT {index + 1}/{len(figures)} {key}: S saves; close continues")
                with plt.rc_context({"keymap.save": [], "keymap.home": [k for k in plt.rcParams["keymap.home"] if k.lower() != "r"]}):
                    plt.show(block=True)
            else:
                save()
                elapsed = time.monotonic() - start
                eta = elapsed / (index + 1) * (len(figures) - index - 1)
                progress(f"PLOT {index + 1}/{len(figures)} {key}; elapsed {elapsed:.1f}s; ETA {eta:.1f}s (provisional)")
        finally:
            plt.close(fig)
    progress(f"COMPLETE: {len(generated)} files; elapsed {time.monotonic() - start:.1f}s; {output}")
    return generated
