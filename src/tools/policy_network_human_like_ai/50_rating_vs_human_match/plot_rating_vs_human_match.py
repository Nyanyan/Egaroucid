#!/usr/bin/env python3
"""推定Eloと人間との1位着手一致率の関係を作図する。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "rating_vs_human_top1_data.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "rating_vs_human_top1"


@dataclass(frozen=True)
class Result:
    series: str
    label: str
    rating: float
    rating_ci_half_width: float
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

# ラベル同士が重ならないように、点からの表示位置をポイント単位で指定する。
LABEL_OFFSETS = {
    "level 1": (9, -2),
    "level 3": (8, 5),
    "level 5": (8, -10),
    "level 7": (-34, 8),
    "level 9": (-35, -13),
    "level 11": (-48, 8),
    "level 13": (-49, -14),
    "level 15": (8, 8),
    "level 17": (8, -14),
    "level 19": (8, -3),
    "alpha=0.0": (8, 8),
    "alpha=0.2": (8, -13),
    "alpha=0.4": (-53, 7),
    "alpha=0.6": (8, 7),
    "alpha=0.8": (8, 7),
    "alpha=1.0": (8, 7),
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
                rating_ci_half_width=float(row["推定Eloの95%信頼区間半幅"]),
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


def plot_results(results: list[Result], output_stem: Path) -> None:
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
        xerr = np.array([result.rating_ci_half_width for result in group])
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
            markersize=7.5,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=1.0,
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.15,
            capsize=3.0,
            capthick=1.15,
            alpha=0.88,
            label=style["legend"],
            zorder=3,
        )

        for result in group:
            dx, dy = LABEL_OFFSETS[result.label]
            display_label = result.label.replace("alpha", "α")
            axis.annotate(
                display_label,
                (result.rating, result.top1),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9.5,
                color=style["color"],
                weight="semibold",
                zorder=4,
            )

    axis.set_title(
        "推定Eloレーティングと人間との1位着手一致率",
        fontsize=17,
        pad=16,
        weight="semibold",
    )
    axis.set_xlabel("推定Eloレーティング", fontsize=12, labelpad=10)
    axis.set_ylabel("人間との1位着手一致率 (%)", fontsize=12, labelpad=10)
    axis.set_xlim(-950, 3550)
    axis.set_ylim(49.5, 66.5)
    axis.set_xticks(np.arange(-500, 3501, 500))
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

    axis.text(
        0.01,
        0.985,
        "横線: 推定Eloの95%信頼区間　縦線: 1位着手一致率の95%信頼区間\n"
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
