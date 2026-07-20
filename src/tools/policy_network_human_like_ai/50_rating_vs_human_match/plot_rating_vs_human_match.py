#!/usr/bin/env python3
"""推定Eloと人間との1位着手一致率の関係を作図する。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "rating_vs_human_top1_data.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "rating_vs_human_top1"
SOURCE_RATING_CONFIDENCE = 0.95
DEFAULT_RATING_CONFIDENCE = 0.68
MARKER_SIZE_POINTS = 7.5
ERROR_BAR_CAP_POINTS = 3.0


@dataclass(frozen=True)
class Result:
    series: str
    label: str
    rating: float
    rating_ci95_half_width: float
    top1: float
    top1_ci_lower: float
    top1_ci_upper: float


SERIES_STYLES = {
    "console": {
        "legend": "Egaroucid for Console",
        "marker": "o",
        "color": "#2878B5",
    },
    "blend": {
        "legend": "ブレンド方策",
        "marker": "s",
        "color": "#D64B32",
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="推定Eloと人間との1位着手一致率を信頼区間付きで作図します。"
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
    parser.add_argument(
        "--rating-confidence",
        type=float,
        default=DEFAULT_RATING_CONFIDENCE,
        help=(
            "横方向に表示する推定Elo信頼区間の信頼水準 "
            f"(0より大きく1未満、既定値: {DEFAULT_RATING_CONFIDENCE})"
        ),
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
                rating_ci95_half_width=float(row["推定Eloの95%信頼区間半幅"]),
                top1=float(row["1位着手一致率"]),
                top1_ci_lower=float(row["1位着手一致率の95%信頼区間下限"]),
                top1_ci_upper=float(row["1位着手一致率の95%信頼区間上限"]),
            )
            if result.series not in SERIES_STYLES:
                raise ValueError(f"未知の系列です: {result.series}")
            if not result.top1_ci_lower <= result.top1 <= result.top1_ci_upper:
                raise ValueError(f"{result.label}の一致率が信頼区間外です")
            results.append(result)
    if not results:
        raise ValueError(f"入力データが空です: {path}")
    return results


def confidence_percent(confidence: float) -> str:
    return f"{confidence * 100:g}%"


def rating_ci_scale(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("推定Elo信頼区間の信頼水準は0より大きく1未満にしてください")
    source_z = NormalDist().inv_cdf(0.5 + SOURCE_RATING_CONFIDENCE / 2.0)
    target_z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return target_z / source_z


def label_candidates() -> list[tuple[float, float]]:
    candidates: list[tuple[float, float]] = []
    for vertical in (14.0, 20.0, 28.0, 38.0, 50.0, 65.0):
        for horizontal in (8.0, 18.0, 32.0, 48.0, 65.0):
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


def segment_intersects_bbox(
    segment: tuple[float, float, float, float],
    bbox: Bbox,
    padding: float = 2.0,
) -> bool:
    x0, y0, x1, y1 = segment
    box = padded_bbox(bbox, padding, padding)
    if abs(y0 - y1) < 1e-9:
        return box.y0 <= y0 <= box.y1 and max(min(x0, x1), box.x0) <= min(
            max(x0, x1), box.x1
        )
    if abs(x0 - x1) < 1e-9:
        return box.x0 <= x0 <= box.x1 and max(min(y0, y1), box.y0) <= min(
            max(y0, y1), box.y1
        )
    raise ValueError("ラベル配置で扱える線分は水平または垂直だけです")


def build_plot_obstacles(
    axis: plt.Axes,
    results: list[Result],
    rating_ci_half_widths: dict[str, float],
) -> tuple[list[Bbox], list[tuple[float, float, float, float]]]:
    transform = axis.transData
    point_boxes: list[Bbox] = []
    segments: list[tuple[float, float, float, float]] = []
    marker_radius = (MARKER_SIZE_POINTS / 2.0 + 2.0) * axis.figure.dpi / 72.0
    cap_radius = ERROR_BAR_CAP_POINTS * axis.figure.dpi / 72.0

    for result in results:
        left = transform.transform(
            (result.rating - rating_ci_half_widths[result.label], result.top1)
        )
        right = transform.transform(
            (result.rating + rating_ci_half_widths[result.label], result.top1)
        )
        lower = transform.transform((result.rating, result.top1_ci_lower))
        upper = transform.transform((result.rating, result.top1_ci_upper))
        point = transform.transform((result.rating, result.top1))

        segments.extend(
            [
                (left[0], left[1], right[0], right[1]),
                (lower[0], lower[1], upper[0], upper[1]),
                (left[0], left[1] - cap_radius, left[0], left[1] + cap_radius),
                (
                    right[0],
                    right[1] - cap_radius,
                    right[0],
                    right[1] + cap_radius,
                ),
                (
                    lower[0] - cap_radius,
                    lower[1],
                    lower[0] + cap_radius,
                    lower[1],
                ),
                (
                    upper[0] - cap_radius,
                    upper[1],
                    upper[0] + cap_radius,
                    upper[1],
                ),
            ]
        )
        point_boxes.append(
            Bbox.from_bounds(
                point[0] - marker_radius,
                point[1] - marker_radius,
                marker_radius * 2.0,
                marker_radius * 2.0,
            )
        )
    return point_boxes, segments


def place_labels_without_overlap(
    figure: plt.Figure,
    axis: plt.Axes,
    results: list[Result],
    rating_ci_half_widths: dict[str, float],
    reserved_boxes: list[Bbox],
) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = padded_bbox(axis.get_window_extent(renderer), -3.0, -3.0)
    point_boxes, segments = build_plot_obstacles(
        axis, results, rating_ci_half_widths
    )
    blocked_boxes = [padded_bbox(box, 2.0, 2.0) for box in point_boxes]
    blocked_boxes.extend(padded_bbox(box, 4.0, 4.0) for box in reserved_boxes)

    display_points = {
        result.label: axis.transData.transform((result.rating, result.top1))
        for result in results
    }

    def local_density(result: Result) -> float:
        point = display_points[result.label]
        return sum(
            np.exp(
                -np.linalg.norm(point - other_point) / 90.0
            )
            for label, other_point in display_points.items()
            if label != result.label
        )

    for result in sorted(results, key=local_density, reverse=True):
        style = SERIES_STYLES[result.series]
        display_label = result.label.replace("alpha", "α")
        placed = False
        for dx, dy in label_candidates():
            horizontal_alignment = "left" if dx > 0 else "right"
            vertical_alignment = "bottom" if dy > 0 else "top"
            annotation = axis.annotate(
                display_label,
                (result.rating, result.top1),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=horizontal_alignment,
                va=vertical_alignment,
                fontsize=9.5,
                color=style["color"],
                weight="semibold",
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": "#FAFBFC",
                    "edgecolor": "none",
                    "alpha": 0.96,
                },
                zorder=5,
            )
            annotation.update_positions(renderer)
            label_box = padded_bbox(
                annotation.get_window_extent(renderer), 2.0, 1.5
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
            crosses_line = any(
                segment_intersects_bbox(segment, label_box)
                for segment in segments
            )
            if inside_axes and not overlaps_box and not crosses_line:
                blocked_boxes.append(label_box)
                placed = True
                break
            annotation.remove()
        if not placed:
            raise RuntimeError(
                f"{result.label}のラベルを他の要素と重ならない位置へ配置できません"
            )


def plot_results(
    results: list[Result],
    output_stem: Path,
    rating_confidence: float,
) -> None:
    ci_scale = rating_ci_scale(rating_confidence)
    rating_ci_half_widths = {
        result.label: result.rating_ci95_half_width * ci_scale
        for result in results
    }
    plt.rcParams.update(
        {
            "font.family": select_japanese_font(),
            "font.size": 11,
            "axes.unicode_minus": False,
            "axes.edgecolor": "#4C566A",
            "axes.linewidth": 0.8,
            "xtick.color": "#344054",
            "ytick.color": "#344054",
            "text.color": "#20242A",
        }
    )

    figure, axis = plt.subplots(figsize=(12.8, 8.0), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#FAFBFC")

    for series, style in SERIES_STYLES.items():
        group = [result for result in results if result.series == series]
        x = np.array([result.rating for result in group])
        y = np.array([result.top1 for result in group])
        xerr = np.array(
            [rating_ci_half_widths[result.label] for result in group]
        )
        yerr = np.array(
            [
                [result.top1 - result.top1_ci_lower for result in group],
                [result.top1_ci_upper - result.top1 for result in group],
            ]
        )

        axis.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=style["marker"],
            markersize=MARKER_SIZE_POINTS,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=1.0,
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.15,
            capsize=ERROR_BAR_CAP_POINTS,
            capthick=1.15,
            alpha=0.88,
            label=style["legend"],
            zorder=3,
        )

    axis.set_title(
        "推定Eloレーティングと人間との1位着手一致率",
        fontsize=17,
        pad=16,
        weight="semibold",
    )
    axis.set_xlabel("推定Eloレーティング", fontsize=12, labelpad=10)
    axis.set_ylabel("人間との1位着手一致率 (%)", fontsize=12, labelpad=10)
    rating_lower = min(
        result.rating - rating_ci_half_widths[result.label]
        for result in results
    )
    rating_upper = max(
        result.rating + rating_ci_half_widths[result.label]
        for result in results
    )
    rating_padding = max(150.0, (rating_upper - rating_lower) * 0.05)
    x_min = rating_lower - rating_padding
    x_max = rating_upper + rating_padding
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(49.5, 66.5)
    tick_min = np.ceil(x_min / 500.0) * 500.0
    tick_max = np.floor(x_max / 500.0) * 500.0
    axis.set_xticks(np.arange(tick_min, tick_max + 1.0, 500.0))
    axis.set_yticks(np.arange(50, 67, 2))
    axis.grid(which="major", color="#D7DCE3", linewidth=0.7, alpha=0.72)
    axis.set_axisbelow(True)

    legend = axis.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#C8CDD5",
        framealpha=0.96,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    information = axis.text(
        0.01,
        0.985,
        f"横線: 推定Eloの{confidence_percent(rating_confidence)}信頼区間　"
        "縦線: 1位着手一致率の95%信頼区間\n"
        "強さ測定: 各組50対戦セット　一致率測定: 各モデル100,000局面",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        color="#596273",
        bbox={
            "boxstyle": "square,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#D7DCE3",
            "alpha": 0.92,
        },
    )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    place_labels_without_overlap(
        figure,
        axis,
        results,
        rating_ci_half_widths,
        [
            legend.get_window_extent(renderer),
            information.get_window_extent(renderer),
        ],
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    figure.savefig(png_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"wrote {png_path}")


def main() -> None:
    args = parse_args()
    plot_results(
        read_results(args.input),
        args.output,
        args.rating_confidence,
    )


if __name__ == "__main__":
    main()
