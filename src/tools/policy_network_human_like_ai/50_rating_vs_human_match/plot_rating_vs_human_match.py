#!/usr/bin/env python3
"""Render rating/agreement figures from saved report_data.json, without inference."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

from label_editor import (
    LABEL_PREFIX, LabelEditor, annotation_position, apply_label_positions,
    apply_text_overrides, load_label_positions, load_text_overrides, update_leader_line,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parents[4] / "experiment_bat/20261007_2/report_data.json"
DEFAULT_CI_ALPHA = 0.10
STYLES = {
    "Egaroucid": {"label": "Egaroucid", "color": "#249972", "marker": "o"},
    "Edax": {"label": "Edax", "color": "#2468B4", "marker": "s"},
    "normal": {"label": "通常NN＋Egaroucid L21", "color": "#D62728", "marker": "^"},
    "squared_blend": {"label": "通常NN＋石損NN", "color": "#D55E00", "marker": "D"},
    "random": {"label": "ランダム（参考）", "color": "#747C85", "marker": "X"},
}
FIGURES = (
    "model_overview", "elo_vs_agreement", "mistake_model_comparison",
    "threshold_trends", "agreement_heatmap", "mistake_elo_vs_agreement",
)
EDITABLE_FIGURES = ("elo_vs_agreement", "mistake_elo_vs_agreement", "model_overview",
                    "mistake_model_comparison", "threshold_trends", "agreement_heatmap")
LINE_STYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)))


def validate_report(document):
    """Join measurements by dataset ID, so input row order cannot mix cohorts."""
    data = deepcopy(document)
    models = data["models"]
    datasets = data["dataset_counts"]
    if not models or not datasets:
        raise ValueError("models and dataset_counts must not be empty")
    dataset_ids = [row["dataset"] for row in datasets]
    model_ids = [model["model_id"] for model in models]
    if len(set(dataset_ids)) != len(dataset_ids) or len(set(model_ids)) != len(model_ids):
        raise ValueError("Duplicate model or dataset ID")
    if any(row["positions"] <= 0 for row in datasets):
        raise ValueError("Dataset positions must be positive")
    if datasets[0].get("minimum_loss") is not None:
        raise ValueError("The first dataset must be the overall agreement sample")
    if any(row.get("minimum_loss") is None for row in datasets[1:]):
        raise ValueError("Subsequent datasets must specify minimum_loss")
    for model in models:
        if model["group"] not in STYLES:
            raise ValueError(f"Unknown model group: {model['group']}")
        check_interval(model["elo"], model["elo_low"], model["elo_high"], model["model_id"])
        rows = {row["dataset"]: row for row in model["results"]}
        if len(rows) != len(model["results"]) or set(rows) != set(dataset_ids):
            raise ValueError(f"Dataset mismatch: {model['model_id']}")
        model["results"] = [rows[key] for key in dataset_ids]
        model["color"] = STYLES[model["group"]]["color"]
        for row, dataset in zip(model["results"], datasets):
            if row["positions"] != dataset["positions"]:
                raise ValueError(f"Position count mismatch: {model['model_id']}/{row['dataset']}")
            if row.get("model_id", model["model_id"]) != model["model_id"]:
                raise ValueError(f"Model ID mismatch: {model['model_id']}")
            check_interval(row["top1_accuracy"], row["top1_ci95_lower"], row["top1_ci95_upper"],
                           f"{model['model_id']}/{row['dataset']}", probability=True)
    covered = set()
    if not data["series_paths"]:
        raise ValueError("series_paths must not be empty")
    for family, ids in data["series_paths"].items():
        if family not in STYLES or not ids or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid series: {family}")
        if not set(ids).issubset(model_ids):
            raise ValueError(f"Unknown model in series: {family}")
        covered.update(ids)
    if covered != set(model_ids):
        raise ValueError("Every model must belong to a series path")
    data["groups"] = {key: deepcopy(STYLES[key]) for key in data["series_paths"]}
    return data


def check_interval(point, low, high, label, probability=False):
    if not all(math.isfinite(value) for value in (point, low, high)):
        raise ValueError(f"Non-finite estimate or confidence interval: {label}")
    if not low <= point <= high:
        raise ValueError(f"Estimate outside 95% confidence interval: {label}")
    if probability and not 0 <= low <= high <= 1:
        raise ValueError(f"Agreement must be a probability, not a percentage: {label}")


def read_report(path):
    return validate_report(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def configure_plotting():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in ("Meiryo", "Yu Gothic", "Noto Sans CJK JP", "IPAexGothic")
                   if name in installed), "sans-serif")
    plt.rcParams.update({
        "font.family": family, "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.unicode_minus": False, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white", "axes.titlepad": 12,
    })


def draw_ci_line(axis, x, y, low, high, *, color, ci_alpha=DEFAULT_CI_ALPHA, **kwargs):
    """Connect pointwise y intervals in the supplied level/alpha order.

    The shading is a visual interpolation, not a simultaneous confidence band
    or a fitted regression. Elo uncertainty is drawn separately as x error bars.
    """
    if not 0 <= ci_alpha <= 1:
        raise ValueError("CI alpha must be between 0 and 1")
    band = axis.fill_between(x, low, high, color=color, alpha=ci_alpha,
                             linewidth=0, zorder=1)
    line, = axis.plot(x, y, color=color, zorder=3, **kwargs)
    return line, band


class ReportPlots:
    def __init__(self, data, ci_alpha=DEFAULT_CI_ALPHA):
        if not 0 <= ci_alpha <= 1:
            raise ValueError("CI alpha must be between 0 and 1")
        self.data = validate_report(data)
        self.models = self.data["models"]
        self.index = {model["model_id"]: model for model in self.models}
        self.datasets = self.data["dataset_counts"]
        self.paths = self.data["series_paths"]
        self.ci_alpha = ci_alpha
        self.ci_note = (f"帯：一致率の95% Wilson区間（不透明度{ci_alpha:.0%}）。"
                        "横線：Eloの95%区間。帯は各点の区間を接続したものです。")

    def legend(self):
        return [Line2D([], [], color=STYLES[key]["color"], marker=STYLES[key]["marker"],
                       label=STYLES[key]["label"], lw=1.5, ms=6) for key in self.paths]

    @staticmethod
    def grid(axis):
        axis.grid(color="#E5EAF0", lw=.7)
        axis.set_axisbelow(True)

    def row_labels(self, axis):
        axis.set_yticks(range(len(self.models)))
        axis.set_yticklabels([model["model"] for model in self.models])
        axis.set_ylim(len(self.models) - .4, -.7)
        for tick, model in zip(axis.get_yticklabels(), self.models):
            tick.set_color(model["color"])

    def row_guides(self, axis, heatmap=False):
        for i, model in enumerate(self.models):
            if i and model["group"] != self.models[i - 1]["group"]:
                axis.axhline(i - .5, color="white" if heatmap else "#DCE2EA", lw=2 if heatmap else .7)
            if model["group"] == "squared_blend" and not heatmap:
                axis.axhspan(i - .5, i + .5, color="#FFF3E7", zorder=-2)

    def elo_limits(self):
        low = min(model["elo_low"] for model in self.models)
        high = max(model["elo_high"] for model in self.models)
        return min(0, math.floor((low - 60) / 100) * 100), math.ceil((high + 180) / 100) * 100

    def agreement_limit(self, indices, padding=3):
        high = max(model["results"][k]["top1_ci95_upper"] * 100
                   for model in self.models for k in indices)
        return min(100, math.ceil(high + padding))

    def interval_point(self, axis, x, y, low, high, style, yerr=None, size=5.5):
        axis.errorbar(x, y, xerr=[[x - low], [high - x]], yerr=yerr,
                      fmt=style["marker"], color=style["color"],
                      ecolor=style["color"], alpha=1.0,
                      ms=size, lw=1, capsize=2.5, zorder=4)

    def model_overview(self):
        figure, axes = plt.subplots(1, 2, figsize=(17, max(8, len(self.models) * .51)), sharey=True)
        for i, model in enumerate(self.models):
            result = model["results"][0]
            values = [(model["elo"], model["elo_low"], model["elo_high"]),
                      tuple(result[key] * 100 for key in ("top1_accuracy", "top1_ci95_lower", "top1_ci95_upper"))]
            for j, (point, low, high) in enumerate(values):
                self.interval_point(axes[j], point, i, low, high, STYLES[model["group"]])
                axes[j].text(high + (30 if j == 0 else .7), i,
                             f"{point:.1f}" if j == 0 else f"{point:.3f}%",
                             va="center", color=model["color"])
        self.row_labels(axes[0])
        for axis in axes:
            self.row_guides(axis)
            axis.grid(axis="x", color="#E5EAF0")
            axis.set_axisbelow(True)
        axes[0].set(title=f"相対Elo（{len(self.models)}モデル）", xlabel="Elo", xlim=self.elo_limits())
        axes[1].set(title=f"人間との着手一致率（{self.datasets[0]['positions']:,}局面）",
                    xlabel="Top-1一致率（%）", xlim=(0, self.agreement_limit([0], 8)))
        figure.suptitle("対戦の強さと、人間の着手の再現", fontsize=18, y=.99)
        figure.legend(handles=self.legend(), loc="upper center", bbox_to_anchor=(.59, .955), ncol=3, frameon=False)
        figure.subplots_adjust(left=.235, right=.985, top=.86, bottom=.09, wspace=.15)
        figure.text(.235, .025, "横線：入力データに保存された95%信頼区間。EloはXOT局面単位のブートストラップ、\n"
                    "一致率はWilson区間（棋譜内相関の補正なし）。", fontsize=9, color="#596575")
        return figure

    def mistake_model_comparison(self):
        count = len(self.datasets) - 1
        figure, axes = self.panels(count, 3, (20, max(9, math.ceil(count / 3) * len(self.models) * .39)), sharey=True)
        for k, axis in enumerate(axes[:count], 1):
            for i, model in enumerate(self.models):
                row = model["results"][k]
                point, low, high = (row[key] * 100 for key in ("top1_accuracy", "top1_ci95_lower", "top1_ci95_upper"))
                self.interval_point(axis, point, i, low, high, STYLES[model["group"]])
                axis.text(high + .7, i, f"{point:.2f}", va="center", fontsize=9, color=model["color"])
            axis.set(xlim=(0, self.agreement_limit(range(1, len(self.datasets)), 6)), xlabel="着手一致率（%）")
            axis.set_title(self.dataset_title(k), fontsize=12)
            self.row_guides(axis)
            axis.grid(axis="x", color="#E5EAF0")
            axis.set_axisbelow(True)
            if (k - 1) % 3 == 0:
                self.row_labels(axis)
                axis.tick_params(labelleft=True)
        figure.suptitle(f"石損した人間の着手との一致率 — 全{len(self.models)}モデル", fontsize=18, y=.994)
        figure.legend(handles=self.legend(), loc="upper center", bbox_to_anchor=(.58, .969), ncol=3, frameon=False)
        figure.subplots_adjust(left=.205, right=.985, top=.90, bottom=.066, wspace=.18, hspace=.22)
        figure.text(.205, .02, "各石損条件に該当する局面を使用。条件間の局面は重複しています。\n"
                    "横線は95% Wilson区間（棋譜内相関の補正なし）。", fontsize=9, color="#596575")
        return figure

    @staticmethod
    def panels(count, columns, size, sharey=False):
        if count < 1:
            raise ValueError("This figure requires at least one loss-conditioned dataset or series")
        figure, axes = plt.subplots(math.ceil(count / columns), columns, figsize=size, sharey=sharey, squeeze=False)
        axes = list(axes.flat)
        for axis in axes[count:]:
            axis.set_visible(False)
        return figure, axes

    def dataset_title(self, index):
        dataset = self.datasets[index]
        return f"{dataset['label']}（n={dataset['positions']:,}）"

    def threshold_trends(self):
        if len(self.datasets) < 2:
            raise ValueError("threshold_trends requires loss-conditioned datasets")
        paths = [(family, ids) for family, ids in self.paths.items() if family != "random"]
        figure, axes = self.panels(len(paths), 2, (15.5, 5.3 * math.ceil(len(paths) / 2)), sharey=True)
        x = np.arange(len(self.datasets) - 1)
        for axis, (family, ids) in zip(axes, paths):
            style = STYLES[family]
            for i, model_id in enumerate(ids):
                model = self.index[model_id]
                rows = model["results"][1:]
                label = "B0.0" if family == "squared_blend" and model_id == "alpha_1.0" else model["short"]
                draw_ci_line(axis, x, [row["top1_accuracy"] * 100 for row in rows],
                             [row["top1_ci95_lower"] * 100 for row in rows],
                             [row["top1_ci95_upper"] * 100 for row in rows], color=style["color"],
                             ci_alpha=self.ci_alpha, marker=style["marker"],
                             linestyle=LINE_STYLES[i % len(LINE_STYLES)], ms=5, lw=1.5, label=label)
            axis.set_title(style["label"], fontsize=13)
            axis.set_xticks(x)
            axis.set_xticklabels([dataset["label"] for dataset in self.datasets[1:]])
            axis.set_ylim(0, self.agreement_limit(range(1, len(self.datasets))))
            self.grid(axis)
            axis.legend(loc="upper right", ncol=3, fontsize=9)
        for axis in axes[::2]:
            axis.set_ylabel("着手一致率（%）")
        figure.suptitle("石損の閾値による一致率の変化", fontsize=18, y=.992)
        figure.subplots_adjust(left=.065, right=.99, top=.92, bottom=.11, hspace=.28, wspace=.14)
        figure.text(.065, .022, "Nは通常NN＋L21の通常NN側のα、Bは石損NN側のα（B0.0は通常NN、B1.0は石損NN単体）。\n"
                    f"横軸は累積条件を等間隔で表示。帯は95% Wilson区間（不透明度{self.ci_alpha:.0%}、同時信頼帯ではありません）。",
                    fontsize=9, color="#596575")
        return figure

    def agreement_heatmap(self):
        matrix = np.asarray([[row["top1_accuracy"] * 100 for row in model["results"]] for model in self.models])
        figure, axis = plt.subplots(figsize=(max(12, len(self.datasets) * 2.28), max(8, len(self.models) * .56)))
        maximum = max(70, self.agreement_limit(range(len(self.datasets)), 0))
        plot = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=maximum, aspect="auto")
        axis.set_xticks(range(len(self.datasets)))
        axis.set_xticklabels([dataset["label"] + f"\nn={dataset['positions']:,}" for dataset in self.datasets], fontsize=10)
        axis.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=10)
        self.row_labels(axis)
        for i in range(len(self.models)):
            for j in range(len(self.datasets)):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=10,
                          color="white" if matrix[i, j] >= maximum * .61 else "#172C42")
        self.row_guides(axis, heatmap=True)
        axis.axvline(.5, color="white", lw=3)
        for spine in axis.spines.values():
            spine.set_visible(False)
        figure.colorbar(plot, ax=axis, fraction=.034, pad=.035).set_label("着手一致率（%）")
        figure.suptitle("全体と石損条件別の着手一致率", fontsize=18, x=.60, y=.986)
        figure.subplots_adjust(left=.245, right=.96, top=.86, bottom=.06)
        figure.text(.245, .016, "各セルは点推定値。95%信頼区間は比較図・折れ線図で確認できます。", fontsize=9, color="#596575")
        return figure

    def scatter_panel(self, axis, k):
        axis.set_gid(f"dataset:{self.datasets[k]['dataset']}")
        for family, ids in self.paths.items():
            if len(ids) < 2:
                continue
            models = [self.index[key] for key in ids]
            draw_ci_line(axis, [model["elo"] for model in models],
                         [model["results"][k]["top1_accuracy"] * 100 for model in models],
                         [model["results"][k]["top1_ci95_lower"] * 100 for model in models],
                         [model["results"][k]["top1_ci95_upper"] * 100 for model in models],
                         color=STYLES[family]["color"], ci_alpha=self.ci_alpha, lw=1.4)
        for model in self.models:
            row = model["results"][k]
            y = row["top1_accuracy"] * 100
            self.interval_point(axis, model["elo"], y, model["elo_low"], model["elo_high"], STYLES[model["group"]],
                                yerr=[[y - row["top1_ci95_lower"] * 100], [row["top1_ci95_upper"] * 100 - y]])
        blend_ids = [key for key in self.paths.get("squared_blend", []) if key != "alpha_1.0"]
        for model in self.models:
            model_id = model["model_id"]
            offset = (6, 9)
            if model_id in blend_ids[:-1]:
                i = blend_ids.index(model_id)
                choices = ([(17, -3), (14, -12), (-15, -16), (-15, -16)] if k == 0 else
                           [(10, -2), (10, 6), (-12, 10), (-22, 10)] if k == 1 else
                           [(10, -2), (10, 2), (-15, -16), (-15, -16)])
                offset = choices[i % len(choices)]
            elif model_id == "alpha_1.0" and k:
                offset = (10, -16)
            elif model_id == "alpha_0.0":
                offset = (-24, -16) if k < 3 else (8, 8)
            elif model_id == "edax_l21":
                offset = (-18, 9) if k < 3 else (-24, 22)
            annotation = axis.annotate(
                model["short"], (model["elo"], model["results"][k]["top1_accuracy"] * 100),
                xytext=offset, textcoords="offset points", fontsize=10, color=model["color"], weight="bold",
                arrowprops={"arrowstyle": "-", "color": model["color"], "lw": .8, "shrinkA": 3, "shrinkB": 5},
                zorder=5)
            annotation.set_gid(f"{LABEL_PREFIX}{self.datasets[k]['dataset']}/{model_id}")
            update_leader_line(annotation)
        upper = self.agreement_limit([0] if k == 0 else range(1, len(self.datasets)), 7 if k == 0 else 3)
        axis.set(title=self.dataset_title(k), xlim=self.elo_limits(), ylim=(0, upper),
                 xlabel=f"相対Elo（{len(self.models)}モデル）", ylabel="着手一致率（%）")
        self.grid(axis)

    def elo_vs_agreement(self):
        indices = [0]
        if len(self.datasets) > 1:
            loss_two = next((k for k, dataset in enumerate(self.datasets)
                             if dataset.get("minimum_loss") == 2), None)
            if loss_two is None:
                raise ValueError("elo_vs_agreement requires a dataset with minimum_loss=2")
            indices.append(loss_two)
        count = len(indices)
        figure, axes = self.panels(count, count, (16 if count == 2 else 9, 7.5))
        for k, axis in zip(indices, axes):
            self.scatter_panel(axis, k)
        figure.suptitle("対戦レートと人間の着手一致率", fontsize=18, y=.99)
        figure.legend(handles=self.legend(), loc="upper center", bbox_to_anchor=(.52, .945), ncol=3, frameon=False)
        figure.subplots_adjust(left=.065, right=.985, top=.79, bottom=.17, wspace=.19)
        figure.text(.065, .025, "Bは石損NN側のα（B1.0は石損NN単体）、N1.0は通常NN単体、E21はEgaroucid L21、D21はEdax L21、Rはランダム。\n"
                    + self.ci_note + "\n各区間は点ごとの区間で、同時信頼帯やモデル間の差の検定ではありません。左右は縦軸の範囲が異なります。",
                    fontsize=9, color="#596575")
        return figure

    def mistake_elo_vs_agreement(self):
        count = len(self.datasets) - 1
        figure, axes = self.panels(count, 2, (16, 5.33 * math.ceil(count / 2)))
        for k, axis in enumerate(axes[:count], 1):
            self.scatter_panel(axis, k)
        figure.suptitle("対戦レートと、石損した着手との一致率", fontsize=18, y=.992)
        figure.legend(handles=self.legend(), loc="upper center", bbox_to_anchor=(.53, .969), ncol=3, frameon=False)
        figure.subplots_adjust(left=.07, right=.985, top=.90, bottom=.08, hspace=.35, wspace=.20)
        figure.text(.07, .015, self.ci_note + "\n線はレベル・αの順に接続。区間は点ごとの95%区間で、同時信頼帯ではありません。\n"
                    "石損条件は累積しており、同じ局面が複数条件に含まれます。一致率の区間は棋譜内相関の補正なし。",
                    fontsize=9, color="#596575")
        return figure


def save_figure(figure, output_dir, name, formats, dpi):
    generated = []
    for extension in formats:
        destination = output_dir / f"{name}.{extension}"
        temporary = destination.with_name(f".{name}.rendering.{extension}")
        figure.savefig(temporary, format=extension, dpi=dpi, bbox_inches="tight", pad_inches=.18)
        temporary.replace(destination)
        generated.append(destination.name)
    return generated


def enable_interactive_backend():
    failures = []
    for backend in ("QtAgg", "TkAgg"):
        try:
            plt.switch_backend(backend)
            # Backend imports can succeed even when the GUI runtime is broken.
            with plt.ioff():
                probe = plt.figure(figsize=(1, 1))
                plt.close(probe)
            return
        except Exception as error:
            failures.append(f"{backend}: {error}")
    raise ValueError("ラベル編集にはQtまたはTkを使えるPythonが必要です。POLICY_PYTHONを確認してください。\n"
                     + "\n".join(failures))


def render_report(input_path, output_dir, figures=FIGURES, formats=("png", "svg"),
                  ci_alpha=DEFAULT_CI_ALPHA, dpi=200, label_positions_path=None,
                  adjust_labels=False, reset_labels=False):
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    positions_path = Path(label_positions_path or output_dir / "label_positions.json").resolve()
    if any(name not in FIGURES for name in figures):
        raise ValueError("Unknown figure name")
    if adjust_labels:
        # Batch rendering uses Agg; only the editor needs a desktop window.
        enable_interactive_backend()
    # Hash and parse the same bytes, even if an aggregator replaces the input.
    source = input_path.read_bytes()
    plots = ReportPlots(json.loads(source.decode("utf-8-sig")), ci_alpha)
    saved_positions = {} if reset_labels else load_label_positions(positions_path)
    saved_texts = {} if reset_labels else load_text_overrides(positions_path)
    configure_plotting()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    labels_used = {}
    texts_used = {}
    manifest = {
        "input": str(input_path), "input_sha256": hashlib.sha256(source).hexdigest(),
        "models": len(plots.models), "datasets": plots.datasets, "series_paths": plots.paths,
        "styles": plots.data["groups"], "ci_transparency": 1 - ci_alpha, "ci_alpha": ci_alpha,
        "ci_errorbar_alpha": 1.0,
        "confidence_intervals": "Saved pointwise 95% intervals; y = Wilson, x = XOT-cluster bootstrap. "
                                "Connecting y bounds is visual interpolation, not a simultaneous confidence band.",
        "label_positions_path": str(positions_path), "label_positions": labels_used,
        "text_overrides": texts_used, "outputs": generated,
    }
    for name in figures:
        if name not in FIGURES:
            raise ValueError(f"Unknown figure: {name}")
        figure = getattr(plots, name)()
        annotations = apply_label_positions(figure, saved_positions.get(name, {}))
        text_targets = apply_text_overrides(figure, saved_texts.get(name, {}))

        def save_current_figure():
            for filename in save_figure(figure, output_dir, name, formats, dpi):
                if filename not in generated:
                    generated.append(filename)
            labels_used[name] = {key: asdict(annotation_position(label)) for key, label in annotations.items()}
            texts_used[name] = {key: target.artist.get_text() for key, target in text_targets.items()}
            (output_dir / "plot_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        try:
            if adjust_labels:
                editor = LabelEditor(figure, annotations, name, positions_path, save_current_figure, text_targets)
                figure._label_editor = editor  # Keep callbacks alive for the entire window lifetime.
                figure.canvas.manager.set_window_title(
                    f"{name} — 点ラベルをドラッグ / 全文字をダブルクリックで変更 / Sで保存 / Rで元に戻す")
                print(f"Editing {name}: drag to move; double-click to rename; S saves JSON and images; R resets. "
                      "Close the window to continue.", flush=True)
                # Matplotlib's default S shortcut opens a file dialog; use our save instead.
                with plt.rc_context({"keymap.save": [], "keymap.home": [
                    key for key in plt.rcParams["keymap.home"] if key.lower() != "r"
                ]}):
                    plt.show(block=True)
            else:
                save_current_figure()
                print(f"Created {name}", flush=True)
        finally:
            plt.close(figure)
    return generated


def make_argparser():
    parser = argparse.ArgumentParser(description="保存済みreport_data.jsonからレートと着手一致率の図を描画します。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="集計JSON（既定：experiment_bat/20261007_2/report_data.json）")
    parser.add_argument("--output-dir", type=Path, help="出力先（既定：このフォルダ/output/入力JSONの親フォルダ名）")
    parser.add_argument("--figures", nargs="+", choices=FIGURES,
                        help="描画・編集する図（既定：全6種類）")
    parser.add_argument("--formats", nargs="+", choices=("png", "svg"), default=["png", "svg"])
    ci_options = parser.add_mutually_exclusive_group()
    ci_options.add_argument("--ci-alpha", type=float, default=DEFAULT_CI_ALPHA,
                            help="95%%信頼区間の帯の不透明度。0=透明、1=不透明、既定0.10。十字は常に不透明")
    ci_options.add_argument("--ci-transparency", type=float,
                            help="従来オプション：帯の透過率（1−alpha）。0.90で--ci-alpha 0.10と同じ")
    parser.add_argument("--adjust-labels", action="store_true", help="タイトル・軸名・凡例など全文字と点ラベルの位置を編集しSキーで保存")
    parser.add_argument("--label-positions", type=Path, help="位置JSON（既定：出力先/label_positions.json）")
    parser.add_argument("--reset-labels", action="store_true", help="保存済み位置を読み込まず、既定の位置から開始")
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main(argv=None):
    parser = make_argparser()
    args = parser.parse_args(argv)
    ci_alpha = args.ci_alpha if args.ci_transparency is None else 1 - args.ci_transparency
    if not 0 <= ci_alpha <= 1 or args.dpi < 1:
        parser.error("--ci-alpha / --ci-transparency must be in [0, 1]; --dpi must be positive")
    output = args.output_dir or SCRIPT_DIR / "output" / args.input.resolve().parent.name
    figures = args.figures or (EDITABLE_FIGURES if args.adjust_labels else FIGURES)
    try:
        generated = render_report(args.input, output, figures, args.formats, ci_alpha, args.dpi,
                                  args.label_positions, args.adjust_labels, args.reset_labels)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Saved {len(generated)} figures: {output.resolve()}")


if __name__ == "__main__":
    main()
