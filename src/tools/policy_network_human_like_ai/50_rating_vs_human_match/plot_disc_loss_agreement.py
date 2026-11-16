"""Render paper-style agreement plots with lower disc loss on the right."""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from label_editor import (LABEL_PREFIX, LabelEditor, annotation_position,
                          apply_label_positions, apply_text_overrides,
                          load_label_positions, load_text_overrides, save_label_positions)
from simultaneous_bands import draw_tube


SCRIPT_DIR = Path(__file__).resolve().parent
ALPHAS = (0., .2, .4, .6, .7, .8, .9, 1., 1.2, 1.4, 1.6, 1.8, 2.)
ALPHA_NAMES = [f"alpha_{value:.1f}" for value in ALPHAS]
CURVES = {
    "level21_normal": dict(names=ALPHA_NAMES[:8], label="レベル21-通常モデル",
                           color="#D62728", linestyle="-", marker="o"),
    "normal_level1": dict(names=["alpha_1.0", *[f"beta_l1_{v:.1f}" for v in (.2, .4, .6, .8)], "egaroucid_l1"],
                          label="通常モデル-レベル1", color="#D98700", linestyle="--", marker="s"),
    "normal_loss": dict(names=ALPHA_NAMES[7:], label="通常モデル-石損モデル",
                        color="#C43F72", linestyle="-.", marker="D"),
    "egaroucid": dict(names=[*[f"egaroucid_l{v}" for v in (1, 2, 3, 5, 9, 13, 17)], "alpha_0.0"],
                      label="Egaroucid", color="#249972", linestyle=":", marker="^"),
    "extra_loss": dict(names=["alpha_3.0", "alpha_4.0"], label="石損4・6以上モデル",
                       color="#C43F72", linestyle="None", marker="D"),
    "zebra": dict(names=["zebra_d1_e2", "zebra_d3_e6", "zebra_d5_e10", "zebra_d9_e18"],
                  label="Zebra（中盤/完全読み）", color="#7952A3", linestyle="--", marker="v"),
    "random": dict(names=["random"], label="ランダム", color="#626A73", linestyle="None", marker="X"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point_label(name, accuracy):
    if name == "alpha_0.0":
        return f"α=0.0\n（Lv. 21）\n{100*accuracy:.2f}%"
    if name == "alpha_1.0":
        return f"α=1.0\n（通常モデル）\n{100*accuracy:.2f}%"
    if name == "alpha_2.0":
        return "α=1.0（石損モデル）"
    if name in ("alpha_3.0", "alpha_4.0"):
        return f"石損{4 if name == 'alpha_3.0' else 6}以上モデル"
    if name == "egaroucid_l1":
        return "α=1.0\n（Lv. 1）"
    if name.startswith("alpha_"):
        value = float(name[6:])
        text = f"α={value if value < 1 else value-1:.1f}"
        return text + (f", {100*accuracy:.2f}%" if name in ("alpha_0.7", "alpha_0.8") else "")
    if name.startswith("beta_l1_"):
        return "α=" + name[8:]
    if name.startswith("egaroucid_l"):
        return "Lv. " + name.split("_l")[1]
    if name.startswith("zebra_"):
        return "/".join(v[1:] for v in name.split("_")[1:])
    if name == "random":
        return "ランダム"
    raise ValueError("Unknown model ID: " + name)


def model_group(name):
    if name in ("alpha_3.0", "alpha_4.0"):
        return "extra_loss"
    if name == "egaroucid_l1" or name.startswith("beta_l1_"):
        return "normal_level1"
    if name.startswith("egaroucid_l"):
        return "egaroucid"
    if name.startswith("zebra_"):
        return "zebra"
    if name == "random":
        return "random"
    return "normal_loss" if float(name[6:]) > 1 else "level21_normal"


def selected_names(data, with_zebra, x_max=None, exclude_extra_loss=False):
    names = [row["model"] for row in data["rows"]]
    require(len(names) == len(set(names)), "Duplicate model IDs")
    losses = {row["model"]: row["mean_disc_loss"] for row in data["rows"]}
    return [name for name in names if (with_zebra or not name.startswith("zebra_"))
            and (not exclude_extra_loss or name not in CURVES["extra_loss"]["names"])
            and (x_max is None or 0 <= losses[name] <= x_max)]


def mean_draws(mean, covariance, rng, count):
    covariance = np.asarray(covariance, dtype=float)
    require(covariance.shape == (len(mean), len(mean)) and np.isfinite(covariance).all(),
            "Invalid mean covariance dimensions")
    require(np.allclose(covariance, covariance.T, atol=1e-12), "Covariance must be symmetric")
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    require(values.min() >= -1e-10, "Covariance is not positive semidefinite")
    draws = mean + rng.standard_normal((count, len(mean))) @ (vectors * np.sqrt(np.maximum(values, 0))).T
    zero = np.diag(covariance) == 0
    draws[:, zero] = mean[zero]
    return draws


def confidence_tubes(data, replicates=20000, seed=20261116):
    rows = {row["model"]: row for row in data["rows"]}
    names = data["covariance_model_order"]
    require(set(names) == set(rows) and len(names) == len(rows), "Covariance model order differs")
    point = np.array([[rows[n]["mean_disc_loss"], rows[n]["wthor_accuracy"]] for n in names])
    covariance_x, covariance_y = np.asarray(data["covariance_x"]), np.asarray(data["covariance_y"])
    rng = np.random.default_rng(seed)
    draws = np.stack((mean_draws(point[:, 0], covariance_x, rng, replicates),
                      mean_draws(point[:, 1], covariance_y, rng, replicates)), axis=2)
    se = np.column_stack((np.sqrt(np.maximum(np.diag(covariance_x), 0)),
                          np.sqrt(np.maximum(np.diag(covariance_y), 0))))
    result = {}
    for key, spec in CURVES.items():
        models = [name for name in spec["names"] if name in names]
        if not models:
            continue
        cols = [names.index(name) for name in models]
        p, s = point[cols], se[cols]
        error = np.divide(draws[:, cols] - p, s, out=np.zeros_like(draws[:, cols]), where=s > 0)
        critical = float(np.quantile(np.max(np.abs(error), axis=(1, 2)), .95, method="higher"))
        low, high = p - critical*s, p + critical*s
        low[:, 0] = np.maximum(low[:, 0], 0)
        low[:, 1] = np.maximum(low[:, 1], 0)
        high[:, 1] = np.minimum(high[:, 1], 1)
        result[key] = dict(model_ids=models, point=p.tolist(), lower=low.tolist(), upper=high.tolist(),
                           standard_error=s.tolist(), critical_value=critical)
    return result


def draw_bounds(axis, spec, bounds):
    if spec["linestyle"] != "None":
        draw_tube(axis, bounds, color=spec["color"], alpha=.15)
    else:
        # Unconnected models get individual regions rather than an invented path.
        for i in range(len(bounds["point"])):
            draw_tube(axis, {field: [bounds[field][i]] for field in ("point", "lower", "upper")},
                      color=spec["color"], alpha=.15)


def make_figure(data, tubes, with_zebra, x_max=None, exclude_extra_loss=False):
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator
    from plot_rating_vs_human_match import configure_plotting
    configure_plotting()
    plt.rcParams.update({"font.size": 18, "axes.labelsize": 24, "xtick.labelsize": 21,
                         "ytick.labelsize": 21, "pdf.fonttype": 42})
    require(x_max is None or np.isfinite(x_max) and x_max > 0, "X maximum must be positive")
    names = selected_names(data, with_zebra, x_max, exclude_extra_loss)
    rows = {row["model"]: row for row in data["rows"]}
    fig = plt.figure(figsize=(20, 9.8))
    main = fig.add_axes([.055, .29, .62, .66])
    detail = fig.add_axes([.74, .29, .245, .66])
    main.set_xlim((10.85, -.17) if x_max is None else (x_max, 0))
    main.set_ylim(16, 74)
    detail.set_xlim(.96, -.075)
    detail.set_ylim(59.5, 70.2)
    axes = {"main": main, "detail": detail}
    annotations = {}
    active = {key: spec for key, spec in CURVES.items()
              if key in tubes and any(n in names for n in spec["names"])}
    for axis in axes.values():
        for key, spec in active.items():
            draw_bounds(axis, spec, tubes[key])
            ordered = [rows[n] for n in spec["names"] if n in names]
            axis.plot([r["mean_disc_loss"] for r in ordered], [100*r["wthor_accuracy"] for r in ordered],
                      color=spec["color"], linestyle=spec["linestyle"], lw=2, zorder=3)
        for name in names:
            row, spec = rows[name], CURVES[model_group(name)]
            x, y = row["mean_disc_loss"], 100*row["wthor_accuracy"]
            axis.plot(x, y, marker=spec["marker"], ms=7, linestyle="none", color=spec["color"],
                      markerfacecolor="white" if spec["marker"] == "s" else spec["color"], zorder=4)
        axis.set_xlabel("平均石損（石／着手）", labelpad=12)
        axis.set_ylabel("着手一致率（%）", labelpad=12)
        axis.grid(color="#E5EAF0", lw=.7)
        axis.set_axisbelow(True)
    main.yaxis.set_major_locator(MultipleLocator(10))
    main.yaxis.set_minor_locator(MultipleLocator(5))
    main.grid(which="minor", axis="y", color="#E5EAF0", lw=.7)
    main.xaxis.set_major_locator(MultipleLocator(2 if x_max is None else 1))
    detail.xaxis.set_major_locator(MultipleLocator(.2))
    detail.yaxis.set_major_locator(MultipleLocator(2))
    for name in names:
        row, spec = rows[name], CURVES[model_group(name)]
        region = "detail" if row["mean_disc_loss"] <= .85 else "main"
        axis = axes[region]
        uid = region + "/" + name
        annotation = axis.annotate(point_label(name, row["wthor_accuracy"]),
            (row["mean_disc_loss"], 100*row["wthor_accuracy"]), xytext=(9, 8), textcoords="offset points",
            color=spec["color"], fontsize=17, multialignment="center", annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=spec["color"], lw=.7), zorder=5)
        annotation.set_gid(LABEL_PREFIX + uid)
        annotations[uid] = annotation
    handles = []
    labels = []
    for key, spec in active.items():
        handles.append((Patch(facecolor=spec["color"], edgecolor="none", alpha=.15),
                        Line2D([], [], color=spec["color"], linestyle=spec["linestyle"], lw=2,
                               marker=spec["marker"] if spec["linestyle"] == "None" else None)))
        labels.append(spec["label"])
    fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
               loc="upper center", bbox_to_anchor=(.52, .18), ncol=4,
               frameon=False, fontsize=17, columnspacing=2)
    reference_handles = []
    for ref in data["references"]:
        in_range = x_max is None or 0 <= ref["value"] <= x_max
        if in_range:
            main.axvline(ref["value"], color=ref["color"], ls=ref["linestyle"], lw=1.5, alpha=.8, zorder=1)
        suffix = "" if in_range else "（範囲外）"
        reference_handles.append(Line2D([], [], color=ref["color"], ls=ref["linestyle"], lw=1.5,
            label=f'{ref["label"]}：{ref["value"]:.4f}{suffix}'))
    fig.legend(reference_handles, [line.get_label() for line in reference_handles], loc="lower center",
               bbox_to_anchor=(.52, .01), ncol=2, frameon=False, fontsize=15, columnspacing=2.5)
    return fig, axes, annotations


def render(data_path, output_dir, variants=(True, False), edit=None, x_max=None, arrange=False,
           exclude_extra_loss=False):
    import matplotlib
    if edit is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from disc_loss_label_layout import arrange_labels, inspect_layout, route_leaders
    data_path, output_dir = Path(data_path), Path(output_dir)
    data = json.loads(data_path.read_text(encoding="utf-8-sig"))
    tubes = confidence_tubes(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    positions_path = output_dir / "label_positions.json"
    positions = load_label_positions(positions_path)
    texts = load_text_overrides(positions_path)
    manifests = {}
    for with_zebra in variants:
        variant = "with-zebra" if with_zebra else "without-zebra"
        suffix = "" if x_max is None else f"-x{x_max:g}"
        base_key = "result-wthor-all-disc-loss-" + variant + suffix
        key = base_key + ("-no-extra-loss" if exclude_extra_loss else "")
        fig, axes, annotations = make_figure(data, tubes, with_zebra, x_max, exclude_extra_loss)
        seed_positions = {}
        if exclude_extra_loss:
            seed_positions.update(positions.get(base_key, {}))
        if x_max is not None:
            full_key = "result-wthor-all-disc-loss-" + variant
            if exclude_extra_loss and full_key + "-no-extra-loss" in positions:
                full_key += "-no-extra-loss"
            seed_positions.update({uid: value for uid, value in positions.get(full_key, {}).items()
                                   if uid.startswith("detail/")})
        seed_positions.update(positions.get(key, {}))
        apply_label_positions(fig, seed_positions)
        text_targets = apply_text_overrides(fig, texts.get(key, {}))
        if arrange:
            for _ in range(4):
                report = inspect_layout(fig, annotations)
                if not any(report.values()):
                    break
                affected = set(report["line"] + report["point"] + report["legend"]
                               + report["outside"] + report["leader_outside"])
                affected.update(uid for pair in report["text"] for uid in pair)
                affected.update(pair[0] for pair in report["leader"])
                arrange_labels(fig, annotations, set(annotations)-affected)
            save_label_positions(positions_path, key,
                                 {uid: annotation_position(a) for uid, a in annotations.items()})

        def save():
            route_leaders(fig, annotations)
            for extension in ("png", "svg", "pdf"):
                fig.savefig(output_dir / f"{key}.{extension}", dpi=220, facecolor="white")
            layout = inspect_layout(fig, annotations)
            visible = selected_names(data, with_zebra, x_max, exclude_extra_loss)
            eligible = selected_names(data, with_zebra, exclude_extra_loss=exclude_extra_loss)
            excluded = sorted(set(eligible) - set(visible))
            manifest = dict(variant=variant, model_count=len(visible), model_ids=visible,
                reversed_x_axes=all(ax.xaxis_inverted() for ax in axes.values()),
                x_limits={name: list(axis.get_xlim()) for name, axis in axes.items()},
                out_of_range_models=excluded,
                excluded_models=sorted(set(selected_names(data, with_zebra)) - set(eligible)),
                out_of_range_references=[r["key"] for r in data["references"]
                                         if x_max is not None and not 0 <= r["value"] <= x_max],
                reference_lines=data["references"], labels={uid: asdict(annotation_position(a)) for uid, a in annotations.items()},
                label_overlaps=layout["text"], layout_collisions=layout,
                status="passed" if not any(layout.values()) else "needs_layout_review",
                input_sha256=sha256(data_path),
                confidence=dict(method="joint_xy_multivariate_normal_mean_max_standardized_v1",
                    confidence=.95, replicates=20000, seed=20261116,
                    scope="Per displayed family; evaluated vertices and their linear interpolation only.",
                    assumptions=data.get("assumptions", [])),
                paper_style={key: {k: v for k, v in spec.items() if k != "names"} for key, spec in CURVES.items()},
                output_sha256={ext: sha256(output_dir / f"{key}.{ext}") for ext in ("png", "svg", "pdf")})
            (output_dir / f"{key}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifests[variant] = manifest

        if edit == variant:
            editor = LabelEditor(fig, annotations, key, positions_path, save, text_targets)
            fig._label_editor = editor
            with plt.rc_context({"keymap.save": []}):
                plt.show(block=True)
        else:
            save()
        plt.close(fig)
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output/disc_loss_agreement")
    parser.add_argument("--zebra", choices=("both", "with", "without"), default="both")
    parser.add_argument("--edit", choices=("with-zebra", "without-zebra"))
    parser.add_argument("--x-max", type=float, help="Set the main X axis from this value to exactly zero")
    parser.add_argument("--exclude-extra-loss", action="store_true",
                        help="Omit models additionally trained on moves losing at least 4 or 6 discs")
    parser.add_argument("--arrange-labels", action="store_true", help="Reposition and save labels clear of plotted objects")
    args = parser.parse_args()
    variants = (True, False) if args.zebra == "both" else (args.zebra == "with",)
    if args.edit:
        variants = (args.edit == "with-zebra",)
    manifests = render(args.data, args.output_dir, variants, args.edit, args.x_max, args.arrange_labels,
                       args.exclude_extra_loss)
    for key, manifest in manifests.items():
        print(f"{key}: {manifest['model_count']} models; X={manifest['x_limits']['main']}; layout={manifest['status']}")


if __name__ == "__main__":
    main()
