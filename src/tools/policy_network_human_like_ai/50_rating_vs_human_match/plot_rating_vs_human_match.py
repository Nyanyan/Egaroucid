#!/usr/bin/env python3
"""推定Eloと人間との1位着手一致率の関係を作図する。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "rating_vs_human_top1_data.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "rating_vs_human_top1"
DEFAULT_LABEL_POSITIONS = SCRIPT_DIR / "label_positions.json"
BASE_FONT_SIZE = 30
TITLE_FONT_SIZE = 48
AXIS_LABEL_FONT_SIZE = 38
LABEL_FONT_SIZE = 30
MARKER_DIAMETER_POINTS = 16.0
LEADER_LINE_THRESHOLD_POINTS = 18.0
ERROR_BAR_LINE_WIDTH = 2.2
ERROR_BAR_CAP_SIZE = 7.0
ERROR_BAR_ALPHA = 0.65


@dataclass(frozen=True)
class Result:
    series: str
    label: str
    rating: float
    rating_ci_half_width: float
    top1: float
    top1_ci_lower: float
    top1_ci_upper: float


@dataclass(frozen=True)
class LabelPosition:
    dx: float
    dy: float
    horizontal_alignment: str
    vertical_alignment: str


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
    parser.add_argument(
        "--label-positions",
        type=Path,
        default=DEFAULT_LABEL_POSITIONS,
        help=f"ラベル位置JSON (既定値: {DEFAULT_LABEL_POSITIONS})",
    )
    parser.add_argument(
        "--adjust-labels",
        action="store_true",
        help="図を開き、ラベルをドラッグして位置を調整します。",
    )
    parser.add_argument(
        "--reset-labels",
        action="store_true",
        help="保存済み位置を無視し、自動配置を調整開始位置にします。",
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
                rating_ci_half_width=float(
                    row["推定Eloの95%信頼区間半幅"]
                ),
                top1=float(row["1位着手一致率"]),
                top1_ci_lower=float(
                    row["1位着手一致率の95%信頼区間下限"]
                ),
                top1_ci_upper=float(
                    row["1位着手一致率の95%信頼区間上限"]
                ),
            )
            if result.series not in SERIES_STYLES:
                raise ValueError(f"未知の系列です: {result.series}")
            if result.rating_ci_half_width < 0.0:
                raise ValueError(
                    f"{result.label}のElo信頼区間半幅が負です"
                )
            if not result.top1_ci_lower <= result.top1 <= result.top1_ci_upper:
                raise ValueError(
                    f"{result.label}の着手一致率が信頼区間外です"
                )
            results.append(result)
    if not results:
        raise ValueError(f"入力データが空です: {path}")
    return results


def load_label_positions(
    path: Path,
    valid_labels: set[str],
) -> dict[str, LabelPosition]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as json_file:
        document = json.load(json_file)
    if document.get("version") != 1:
        raise ValueError(f"未対応のラベル位置ファイルです: {path}")

    positions: dict[str, LabelPosition] = {}
    allowed_horizontal = {"left", "center", "right"}
    allowed_vertical = {"top", "center", "bottom"}
    for label, raw_position in document.get("positions", {}).items():
        if label not in valid_labels:
            raise ValueError(f"入力データに存在しないラベルです: {label}")
        position = LabelPosition(
            dx=float(raw_position["dx"]),
            dy=float(raw_position["dy"]),
            horizontal_alignment=str(raw_position["horizontal_alignment"]),
            vertical_alignment=str(raw_position["vertical_alignment"]),
        )
        if not math.isfinite(position.dx) or not math.isfinite(position.dy):
            raise ValueError(f"{label}のラベル位置が有限値ではありません")
        if position.horizontal_alignment not in allowed_horizontal:
            raise ValueError(f"{label}の水平方向の文字揃えが不正です")
        if position.vertical_alignment not in allowed_vertical:
            raise ValueError(f"{label}の垂直方向の文字揃えが不正です")
        positions[label] = position
    return positions


def save_label_positions(
    path: Path,
    positions: dict[str, LabelPosition],
) -> None:
    document = {
        "version": 1,
        "positions": {
            label: {
                "dx": round(position.dx, 3),
                "dy": round(position.dy, 3),
                "horizontal_alignment": position.horizontal_alignment,
                "vertical_alignment": position.vertical_alignment,
            }
            for label, position in sorted(positions.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as json_file:
        json.dump(document, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")
    temporary_path.replace(path)


def label_candidates() -> list[tuple[float, float]]:
    candidates = [
        (15.0, 0.0),
        (-15.0, 0.0),
        (0.0, 15.0),
        (0.0, -15.0),
        (12.0, 12.0),
        (-12.0, 12.0),
        (12.0, -12.0),
        (-12.0, -12.0),
    ]
    for distance in (20.0, 28.0, 38.0, 52.0, 70.0):
        candidates.extend(
            [
                (distance, 0.0),
                (-distance, 0.0),
                (0.0, distance),
                (0.0, -distance),
                (distance, distance),
                (-distance, distance),
                (distance, -distance),
                (-distance, -distance),
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


def position_for_offset(dx: float, dy: float) -> LabelPosition:
    return LabelPosition(
        dx=dx,
        dy=dy,
        horizontal_alignment="left" if dx > 0 else "right" if dx < 0 else "center",
        vertical_alignment="bottom" if dy > 0 else "top" if dy < 0 else "center",
    )


def add_label_annotation(
    axis: plt.Axes,
    result: Result,
    position: LabelPosition,
) -> Annotation:
    style = SERIES_STYLES[result.series]
    annotation = axis.annotate(
        result.label.replace("alpha", "α"),
        (result.rating, result.top1),
        xytext=(position.dx, position.dy),
        textcoords="offset points",
        ha=position.horizontal_alignment,
        va=position.vertical_alignment,
        fontsize=LABEL_FONT_SIZE,
        color=style["color"],
        weight="bold",
        bbox={
            "boxstyle": "square,pad=0.12",
            "facecolor": "#FFFFFF",
            "edgecolor": "none",
            "alpha": 1.0,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": style["color"],
            "linewidth": 2.0,
            "alpha": 0.8,
            "shrinkA": 4.0,
            "shrinkB": MARKER_DIAMETER_POINTS / 2.0,
            "zorder": 2.5,
        },
        zorder=5,
    )
    update_leader_line(annotation)
    return annotation


def update_leader_line(annotation: Annotation) -> None:
    if annotation.arrow_patch is None:
        return
    dx, dy = annotation.xyann
    distance = math.hypot(float(dx), float(dy))
    annotation.arrow_patch.set_visible(distance > LEADER_LINE_THRESHOLD_POINTS)


def label_bbox(annotation: Annotation, renderer: object) -> Bbox:
    return Text.get_window_extent(annotation, renderer)


def annotation_position(annotation: Annotation) -> LabelPosition:
    dx, dy = annotation.xyann
    return LabelPosition(
        dx=float(dx),
        dy=float(dy),
        horizontal_alignment=annotation.get_ha(),
        vertical_alignment=annotation.get_va(),
    )


def place_labels_without_overlap(
    figure: plt.Figure,
    axis: plt.Axes,
    results: list[Result],
    reserved_boxes: list[Bbox],
    preferred_positions: dict[str, LabelPosition] | None = None,
) -> dict[str, Annotation]:
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
    annotations: dict[str, Annotation] = {}
    preferred_positions = preferred_positions or {}

    for result in results:
        position = preferred_positions.get(result.label)
        if position is None:
            continue
        annotation = add_label_annotation(axis, result, position)
        annotation.update_positions(renderer)
        blocked_boxes.append(
            padded_bbox(label_bbox(annotation, renderer), 3.0, 2.0)
        )
        annotations[result.label] = annotation

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

    remaining_results = [
        result for result in results if result.label not in annotations
    ]
    for result in sorted(remaining_results, key=local_density, reverse=True):
        placed = False
        for dx, dy in label_candidates():
            annotation = add_label_annotation(
                axis,
                result,
                position_for_offset(dx, dy),
            )
            annotation.update_positions(renderer)
            label_box = padded_bbox(
                label_bbox(annotation, renderer), 3.0, 2.0
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
                annotations[result.label] = annotation
                placed = True
                break
            annotation.remove()
        if not placed:
            raise RuntimeError(
                f"{result.label}のラベルを他の要素と重ならない位置へ配置できません"
            )
    return annotations


def create_plot(
    results: list[Result],
    label_positions: dict[str, LabelPosition] | None = None,
) -> tuple[plt.Figure, dict[str, Annotation]]:
    plt.rcParams.update(
        {
            "font.family": select_japanese_font(),
            "font.size": BASE_FONT_SIZE,
            "font.weight": "bold",
            "axes.unicode_minus": False,
            "axes.edgecolor": "#4C566A",
            "axes.facecolor": "#FFFFFF",
            "axes.linewidth": 1.0,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "xtick.color": "#344054",
            "ytick.color": "#344054",
            "text.color": "#20242A",
        }
    )

    figure, axis = plt.subplots(figsize=(16.0, 10.0), constrained_layout=True)
    figure.patch.set_facecolor("#FFFFFF")
    axis.set_facecolor("#FFFFFF")

    for series, style in SERIES_STYLES.items():
        group = [result for result in results if result.series == series]
        ratings = np.array([result.rating for result in group], dtype=float)
        top1_values = np.array([result.top1 for result in group], dtype=float)
        axis.errorbar(
            ratings,
            top1_values,
            xerr=np.array(
                [result.rating_ci_half_width for result in group],
                dtype=float,
            ),
            yerr=np.array(
                [
                    [
                        result.top1 - result.top1_ci_lower
                        for result in group
                    ],
                    [
                        result.top1_ci_upper - result.top1
                        for result in group
                    ],
                ],
                dtype=float,
            ),
            fmt="none",
            ecolor=to_rgba(style["color"], ERROR_BAR_ALPHA),
            elinewidth=ERROR_BAR_LINE_WIDTH,
            capsize=ERROR_BAR_CAP_SIZE,
            capthick=ERROR_BAR_LINE_WIDTH,
            zorder=2.5,
        )
        axis.scatter(
            ratings,
            top1_values,
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

    rating_lower = min(
        result.rating - result.rating_ci_half_width
        for result in results
    )
    rating_upper = max(
        result.rating + result.rating_ci_half_width
        for result in results
    )
    rating_padding = max(100.0, (rating_upper - rating_lower) * 0.03)
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
    annotations = place_labels_without_overlap(
        figure,
        axis,
        results,
        [legend.get_window_extent(renderer)],
        label_positions,
    )
    return figure, annotations


def save_plot(figure: plt.Figure, output_stem: Path) -> Path:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    figure.savefig(
        png_path,
        dpi=180,
        facecolor="#FFFFFF",
        transparent=False,
    )
    return png_path


def plot_results(
    results: list[Result],
    output_stem: Path,
    label_positions: dict[str, LabelPosition] | None = None,
) -> None:
    figure, _ = create_plot(results, label_positions)
    png_path = save_plot(figure, output_stem)
    plt.close(figure)
    print(f"wrote {png_path}")


class LabelEditor:
    def __init__(
        self,
        figure: plt.Figure,
        annotations: dict[str, Annotation],
        output_stem: Path,
        positions_path: Path,
    ) -> None:
        self.figure = figure
        self.annotations = annotations
        self.output_stem = output_stem
        self.positions_path = positions_path
        self.initial_positions = {
            label: annotation_position(annotation)
            for label, annotation in annotations.items()
        }
        self.drag_state: tuple[Annotation, float, float, float, float] | None = None
        self.connections = [
            figure.canvas.mpl_connect("button_press_event", self.on_press),
            figure.canvas.mpl_connect("motion_notify_event", self.on_motion),
            figure.canvas.mpl_connect("button_release_event", self.on_release),
            figure.canvas.mpl_connect("key_press_event", self.on_key_press),
        ]

    def on_press(self, event: object) -> None:
        if getattr(event, "button", None) != 1:
            return
        event_x = getattr(event, "x", None)
        event_y = getattr(event, "y", None)
        if event_x is None or event_y is None:
            return
        for annotation in reversed(list(self.annotations.values())):
            contains, _ = annotation.contains(event)
            if not contains:
                continue
            dx, dy = annotation.xyann
            self.drag_state = (
                annotation,
                float(event_x),
                float(event_y),
                float(dx),
                float(dy),
            )
            return

    def on_motion(self, event: object) -> None:
        if self.drag_state is None:
            return
        event_x = getattr(event, "x", None)
        event_y = getattr(event, "y", None)
        if event_x is None or event_y is None:
            return
        annotation, start_x, start_y, start_dx, start_dy = self.drag_state
        pixels_to_points = 72.0 / self.figure.dpi
        annotation.xyann = (
            start_dx + (float(event_x) - start_x) * pixels_to_points,
            start_dy + (float(event_y) - start_y) * pixels_to_points,
        )
        update_leader_line(annotation)
        self.figure.canvas.draw_idle()

    def on_release(self, _event: object) -> None:
        self.drag_state = None

    def current_positions(self) -> dict[str, LabelPosition]:
        return {
            label: annotation_position(annotation)
            for label, annotation in self.annotations.items()
        }

    def save(self) -> None:
        save_label_positions(self.positions_path, self.current_positions())
        png_path = save_plot(self.figure, self.output_stem)
        print(f"saved label positions to {self.positions_path}", file=sys.stderr)
        print(f"saved plot to {png_path}", file=sys.stderr)

    def reset(self) -> None:
        for label, position in self.initial_positions.items():
            annotation = self.annotations[label]
            annotation.xyann = (position.dx, position.dy)
            annotation.set_ha(position.horizontal_alignment)
            annotation.set_va(position.vertical_alignment)
            update_leader_line(annotation)
        self.figure.canvas.draw_idle()
        print("restored label positions used at startup", file=sys.stderr)

    def on_key_press(self, event: object) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key in {"s", "ctrl+s"}:
            self.save()
        elif key == "r":
            self.reset()


def adjust_labels(
    results: list[Result],
    output_stem: Path,
    positions_path: Path,
    label_positions: dict[str, LabelPosition] | None = None,
) -> None:
    figure, annotations = create_plot(results, label_positions)
    editor = LabelEditor(figure, annotations, output_stem, positions_path)
    figure._label_editor = editor  # type: ignore[attr-defined]
    manager = figure.canvas.manager
    if manager is not None:
        manager.set_window_title(
            "ラベル位置調整: ドラッグで移動、Sで保存、Rで起動時位置へ戻す"
        )
    print(
        "Drag labels with the left mouse button. "
        "Press S to save JSON and PNG, R to restore startup positions.",
        file=sys.stderr,
    )
    plt.show()


def main() -> None:
    args = parse_args()
    results = read_results(args.input)
    valid_labels = {result.label for result in results}
    label_positions = (
        {}
        if args.reset_labels
        else load_label_positions(args.label_positions, valid_labels)
    )
    if args.adjust_labels:
        adjust_labels(
            results,
            args.output,
            args.label_positions,
            label_positions,
        )
    else:
        plot_results(results, args.output, label_positions)


if __name__ == "__main__":
    main()
