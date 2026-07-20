#!/usr/bin/env python3
"""推定Eloと人間との1位着手一致率の関係を作図する。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "rating_vs_human_top1_data.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "rating_vs_human_top1"
BASE_FONT_SIZE = 20
TITLE_FONT_SIZE = 32
AXIS_LABEL_FONT_SIZE = 24
LABEL_FONT_SIZE = 20
MARKER_DIAMETER_POINTS = 16.0


@dataclass(frozen=True)
class Result:
    series: str
    label: str
    rating: float
    top1: float


SERIES_STYLES = {
    "console": {
        "legend": "Egaroucid for Console",
        "marker": "o",
        "color": "#005AFF",
    },
    "blend": {
        "legend": "ブレンド方策",
        "marker": "s",
        "color": "#FF4B00",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="推定Eloと人間との1位着手一致率を作図します。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"入力CSV (既定値: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"拡張子を除いたPNG出力パス (既定値: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def select_japanese_font() -> str:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Yu Gothic", "Meiryo", "Noto Sans CJK JP", "IPAexGothic"):
        if candidate in installed:
            return candidate
    return "sans-serif"


def read_results(path: Path) -> list[Result]:
    results: list[Result] = []
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            result = Result(
                series=row["系列"],
                label=row["ラベル"],
                rating=float(row["推定Elo"]),
                top1=float(row["1位着手一致率"]),
            )
            if result.series not in SERIES_STYLES:
                raise ValueError(f"未知の系列です: {result.series}")
            results.append(result)
    if not results:
        raise ValueError(f"入力データが空です: {path}")
    return results


def label_candidates() -> list[tuple[float, float]]:
    candidates: list[tuple[float, float]] = []
    for vertical in (22.0, 32.0, 44.0, 58.0, 76.0, 98.0):
        for horizontal in (12.0, 32.0, 54.0, 82.0, 110.0):
            candidates.extend(
                [
                    (horizontal, vertical),
                    (-horizontal, vertical),
                    (horizontal, -vertical),
                    (-horizontal, -vertical),
                ]
            )
    return candidates


def padded_bbox(bbox: Bbox, x_padding: float, y_padding: float) -> Bbox:
    return Bbox.from_extents(
        bbox.x0 - x_padding,
        bbox.y0 - y_padding,
        bbox.x1 + x_padding,
        bbox.y1 + y_padding,
    )


def build_point_obstacles(axis: plt.Axes, results: list[Result]) -> list[Bbox]:
    transform = axis.transData
    marker_radius = (
        MARKER_DIAMETER_POINTS / 2.0 + 2.0
    ) * axis.figure.dpi / 72.0
    point_boxes: list[Bbox] = []
    for result in results:
        point = transform.transform((result.rating, result.top1))
        point_boxes.append(
            Bbox.from_bounds(
                point[0] - marker_radius,
                point[1] - marker_radius,
                marker_radius * 2.0,
                marker_radius * 2.0,
            )
        )
    return point_boxes


def place_labels_without_overlap(
    figure: plt.Figure,
    axis: plt.Axes,
    results: list[Result],
    reserved_boxes: list[Bbox],
) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = padded_bbox(axis.get_window_extent(renderer), -4.0, -4.0)
    blocked_boxes = [
        padded_bbox(box, 3.0, 3.0)
        for box in build_point_obstacles(axis, results)
    ]
    blocked_boxes.extend(
        padded_bbox(box, 5.0, 5.0)
        for box in reserved_boxes
    )

    display_points = {
        result.label: axis.transData.transform((result.rating, result.top1))
        for result in results
    }

    def local_density(result: Result) -> float:
        point = display_points[result.label]
        return sum(
            np.exp(-np.linalg.norm(point - other_point) / 110.0)
            for label, other_point in display_points.items()
            if label != result.label
        )

    for result in sorted(results, key=local_density, reverse=True):
        style = SERIES_STYLES[result.series]
        display_label = result.label.replace("alpha", "α")
        placed = False
        for dx, dy in label_candidates():
            annotation = axis.annotate(
                display_label,
                (result.rating, result.top1),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx > 0 else "right",
                va="bottom" if dy > 0 else "top",
                fontsize=LABEL_FONT_SIZE,
                color=style["color"],
                weight="bold",
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": "#FAFBFC",
                    "edgecolor": "none",
                    "alpha": 0.97,
                },
                zorder=5,
            )
            annotation.update_positions(renderer)
            label_box = padded_bbox(
                annotation.get_window_extent(renderer), 3.0, 2.0
            )
            inside_axes = (
                axes_box.x0 <= label_box.x0
                and label_box.x1 <= axes_box.x1
                and axes_box.y0 <= label_box.y0
                and label_box.y1 <= axes_box.y1
            )
            overlaps_box = any(
                label_box.overlaps(blocked_box)
                for blocked_box in blocked_boxes
            )
            if inside_axes and not overlaps_box:
                blocked_boxes.append(label_box)
                placed = True
                break
            annotation.remove()
        if not placed:
            raise RuntimeError(
                f"{result.label}のラベルを他の要素と重ならない位置へ配置できません"
            )


def plot_results(results: list[Result], output_stem: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": select_japanese_font(),
            "font.size": BASE_FONT_SIZE,
            "font.weight": "bold",
            "axes.unicode_minus": False,
            "axes.edgecolor": "#4C566A",
            "axes.linewidth": 1.0,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.color": "#344054",
            "ytick.color": "#344054",
            "text.color": "#20242A",
        }
    )

    figure, axis = plt.subplots(figsize=(16.0, 10.0), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#FAFBFC")

    for series, style in SERIES_STYLES.items():
        group = [result for result in results if result.series == series]
        axis.scatter(
            [result.rating for result in group],
            [result.top1 for result in group],
            s=MARKER_DIAMETER_POINTS**2,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=1.4,
            alpha=0.95,
            label=style["legend"],
            zorder=3,
        )

    axis.set_title(
        "推定Eloレーティングと人間との1位着手一致率",
        fontsize=TITLE_FONT_SIZE,
        pad=18,
        weight="bold",
    )
    axis.set_xlabel(
        "推定Eloレーティング",
        fontsize=AXIS_LABEL_FONT_SIZE,
        labelpad=12,
        weight="bold",
    )
    axis.set_ylabel(
        "人間との1位着手一致率 (%)",
        fontsize=AXIS_LABEL_FONT_SIZE,
        labelpad=12,
        weight="bold",
    )

    rating_lower = min(result.rating for result in results)
    rating_upper = max(result.rating for result in results)
    rating_padding = max(250.0, (rating_upper - rating_lower) * 0.1)
    x_min = np.floor((rating_lower - rating_padding) / 250.0) * 250.0
    x_max = np.ceil((rating_upper + rating_padding) / 250.0) * 250.0
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(49.5, 66.5)
    tick_min = np.ceil(x_min / 500.0) * 500.0
    tick_max = np.floor(x_max / 500.0) * 500.0
    axis.set_xticks(np.arange(tick_min, tick_max + 1.0, 500.0))
    axis.set_yticks(np.arange(50, 67, 2))
    axis.tick_params(axis="both", labelsize=BASE_FONT_SIZE)
    for tick_label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick_label.set_fontweight("bold")
    axis.grid(which="major", color="#D7DCE3", linewidth=0.8, alpha=0.68)
    axis.set_axisbelow(True)

    legend = axis.legend(
        loc="lower right",
        prop={"size": BASE_FONT_SIZE, "weight": "bold"},
        markerscale=1.15,
        frameon=True,
        facecolor="white",
        edgecolor="#C8CDD5",
        framealpha=0.97,
        borderpad=0.7,
    )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    place_labels_without_overlap(
        figure,
        axis,
        results,
        [legend.get_window_extent(renderer)],
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    figure.savefig(png_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"wrote {png_path}")


def main() -> None:
    args = parse_args()
    plot_results(read_results(args.input), args.output)


if __name__ == "__main__":
    main()
