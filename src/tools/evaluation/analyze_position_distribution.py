"""Compare measured board-feature distributions with independent GGS leaves.

The classifier is deliberately diagnostic: it only uses the basic board
statistics requested by the investigation and must not be interpreted as a
proof that either population is globally in or out of the training support.
Duplicate-frequency and D4-unique populations are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260828
DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),             (0, 1),
    (1, -1),  (1, 0),   (1, 1),
)
CORNERS = (0, 7, 56, 63)
X_SQUARES = (9, 14, 49, 54)
C_SQUARES = (1, 6, 8, 15, 48, 55, 57, 62)
FEATURES = (
    "legal_count",
    "opponent_legal_count",
    "player_discs",
    "opponent_discs",
    "disc_difference",
    "legal_difference",
    "player_corner_count",
    "opponent_corner_count",
    "player_x_count",
    "opponent_x_count",
    "player_c_count",
    "opponent_c_count",
)


def board_cells(text: str) -> str:
    cells = text.strip().split()[0]
    if len(cells) != 64 or any(cell not in "-XO" for cell in cells):
        raise ValueError("invalid normalized board: {!r}".format(text))
    return cells


def legal_count(cells: str, side: str) -> int:
    other = "O" if side == "X" else "X"
    result = 0
    for square, cell in enumerate(cells):
        if cell != "-":
            continue
        row, col = divmod(square, 8)
        legal = False
        for dr, dc in DIRECTIONS:
            rr, cc = row + dr, col + dc
            seen_other = False
            while 0 <= rr < 8 and 0 <= cc < 8 and cells[rr * 8 + cc] == other:
                seen_other = True
                rr += dr
                cc += dc
            if seen_other and 0 <= rr < 8 and 0 <= cc < 8 and cells[rr * 8 + cc] == side:
                legal = True
                break
        result += legal
    return result


def transform(cells: str, symmetry: int) -> str:
    output = ["-"] * 64
    for row in range(8):
        for col in range(8):
            r, c = row, col
            if symmetry >= 4:
                c = 7 - c
            for _ in range(symmetry % 4):
                r, c = c, 7 - r
            output[r * 8 + c] = cells[row * 8 + col]
    return "".join(output)


def canonical_d4(cells: str) -> str:
    return min(transform(cells, symmetry) for symmetry in range(8))


def features(text: str) -> Dict[str, object]:
    cells = board_cells(text)
    player_legal = legal_count(cells, "X")
    opponent_legal = legal_count(cells, "O")
    player_discs = cells.count("X")
    opponent_discs = cells.count("O")
    result: Dict[str, object] = {
        "canonical_board": canonical_d4(cells),
        "legal_count": player_legal,
        "opponent_legal_count": opponent_legal,
        "player_discs": player_discs,
        "opponent_discs": opponent_discs,
        "disc_difference": player_discs - opponent_discs,
        "legal_difference": player_legal - opponent_legal,
        "player_corner_count": sum(cells[square] == "X" for square in CORNERS),
        "opponent_corner_count": sum(cells[square] == "O" for square in CORNERS),
        "player_x_count": sum(cells[square] == "X" for square in X_SQUARES),
        "opponent_x_count": sum(cells[square] == "O" for square in X_SQUARES),
        "player_c_count": sum(cells[square] == "X" for square in C_SQUARES),
        "opponent_c_count": sum(cells[square] == "O" for square in C_SQUARES),
    }
    return result


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def add_training(rows: List[Dict[str, object]], path: Path, phase: int) -> None:
    if not path.is_file():
        return
    for item in read_tsv(path):
        if item.get("sample_kind") != "frequency":
            continue
        result: Dict[str, object] = {
            "group": "training_phase{}_frequency".format(phase),
            "phase": phase,
            "sample_id": item["sample_id"],
            "cluster": "{}|{}".format(item.get("source_file", ""), item.get("game_index_in_file", "")),
            "source_category": item.get("source_category", ""),
            "random_moves": item.get("random_moves", ""),
            "moves_after_random": item.get("moves_after_random", ""),
            "teacher": item.get("teacher", ""),
            "board": item["board"],
        }
        result.update(features(item["board"]))
        rows.append(result)


def add_ggs(rows: List[Dict[str, object]], path: Path, cohort: str) -> None:
    if not path.is_file():
        return
    source_rows = read_tsv(path)
    specifications = ((0, 35), (1, 36), (5, 40), (9, 44))
    for item in source_rows:
        for depth, phase in specifications:
            board_column = "root_normalized_board" if depth == 0 else "depth_{}_pv_leaf_board".format(depth)
            board = item.get(board_column, "")
            if not board:
                continue
            result: Dict[str, object] = {
                "group": (
                    "{}_root_phase35".format(cohort)
                    if depth == 0
                    else "{}_leaf_depth{}_phase{}".format(cohort, depth, phase)
                ),
                "phase": phase,
                "sample_id": item["sample_id"],
                "cluster": item["sample_id"],
                "source_category": "independent_ggs_log",
                "random_moves": "",
                "moves_after_random": "",
                "teacher": "",
                "board": board,
            }
            result.update(features(board))
            rows.append(result)


def timing_bucket(row: Mapping[str, object]) -> str:
    source = str(row.get("source_category", ""))
    if source == "ggs_random_setup_starting_board":
        return "ggs_random_setup"
    if source == "ggs_random_setup_2_starting_board":
        return "ggs_random_setup_2"
    text = str(row.get("moves_after_random", "")).strip()
    if text:
        value = int(float(text))
        if value == 0:
            return "after_random_0"
        if value <= 4:
            return "after_random_1_4"
        if value <= 8:
            return "after_random_5_8"
        if value <= 16:
            return "after_random_9_16"
        return "after_random_17_plus"
    return "not_reconstructable_or_not_random"


def origin_bucket(row: Mapping[str, object]) -> str:
    source = str(row.get("source_category", ""))
    if source == "egaroucid_vs_edax":
        return "egaroucid_vs_edax"
    if source in ("egaroucid_selfplay", "book_start_selfplay"):
        return "egaroucid_selfplay"
    if source == "ggs_random_setup_starting_board":
        return "ggs_random_setup"
    if source == "ggs_random_setup_2_starting_board":
        return "ggs_random_setup_2"
    if str(row.get("random_moves", "")).strip():
        return "normal_initial_board_random_legal_moves"
    if source == "independent_ggs_log":
        return "independent_ggs_log"
    return "other"


def attach_exact_values(frame: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    exact_by_key: Dict[Tuple[int, str], float] = {}
    paths = {
        30: report_dir / "egaroucid_phase30_root.tsv",
        35: report_dir / "egaroucid_phase35_root.tsv",
        40: report_dir / "egaroucid_phase40_root.tsv",
        44: report_dir / "egaroucid_phase44_root.tsv",
    }
    for phase, path in paths.items():
        if not path.is_file():
            continue
        for item in read_tsv(path):
            if item.get("exact_complete") != "1" or not item.get("exact_value"):
                continue
            exact_by_key[(phase, item["sample_id"])] = float(item["exact_value"])
    for ggs_path in (
        report_dir / "egaroucid_ggs96_exact_root.tsv",
        report_dir / "egaroucid_ggs_latest96_root.tsv",
    ):
        if ggs_path.is_file():
            for item in read_tsv(ggs_path):
                if item.get("exact_complete") == "1" and item.get("exact_value"):
                    exact_by_key[(35, item["sample_id"])] = float(item["exact_value"])
    frame = frame.copy()
    frame["exact_value"] = [
        exact_by_key.get((int(phase), str(sample_id)), math.nan)
        if group in ("training_phase30_frequency", "training_phase35_frequency",
                     "training_phase40_frequency", "training_phase44_frequency")
        or str(group).endswith("_root_phase35")
        else math.nan
        for phase, sample_id, group in zip(frame["phase"], frame["sample_id"], frame["group"])
    ]
    frame["teacher"] = pd.to_numeric(frame["teacher"], errors="coerce")
    return frame


def percentile_interval(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def classifier_rows(frame: pd.DataFrame, bootstrap: int) -> List[Dict[str, object]]:
    pairs = tuple(
        (cohort, phase, "{}_{}".format(cohort, suffix))
        for cohort in ("ggs_latest96", "ggs_seeded96")
        for phase, suffix in (
            (35, "root_phase35"),
            (36, "leaf_depth1_phase36"),
            (40, "leaf_depth5_phase40"),
            (44, "leaf_depth9_phase44"),
        )
    )
    output: List[Dict[str, object]] = []
    rng = np.random.default_rng(SEED)
    for cohort, phase, ggs_group in pairs:
        training_group = "training_phase{}_frequency".format(phase)
        for population in ("frequency_with_duplicates", "unique_d4"):
            selected = frame[frame["group"].isin((training_group, ggs_group))].copy()
            if population == "unique_d4":
                selected = selected.drop_duplicates(["group", "canonical_board"])
            if selected.empty or selected["group"].nunique() != 2:
                continue
            y = (selected["group"] == ggs_group).astype(int).to_numpy()
            x = selected[list(FEATURES)].astype(float).to_numpy()
            groups = selected["canonical_board"].astype(str).to_numpy()
            folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED),
            )
            probabilities = cross_val_predict(
                model, x, y, groups=groups, cv=folds, method="predict_proba"
            )[:, 1]
            auc = float(roc_auc_score(y, probabilities))
            balanced = float(balanced_accuracy_score(y, probabilities >= 0.5))

            unique_clusters = {
                label: np.unique(groups[y == label]) for label in (0, 1)
            }
            auc_boot: List[float] = []
            balanced_boot: List[float] = []
            for _ in range(bootstrap):
                indices: List[int] = []
                for label in (0, 1):
                    candidates = unique_clusters[label]
                    drawn = rng.choice(candidates, size=len(candidates), replace=True)
                    for cluster in drawn:
                        member = np.flatnonzero((groups == cluster) & (y == label))
                        indices.extend(member.tolist())
                yy = y[indices]
                pp = probabilities[indices]
                auc_boot.append(float(roc_auc_score(yy, pp)))
                balanced_boot.append(float(balanced_accuracy_score(yy, pp >= 0.5)))
            auc_lo, auc_hi = percentile_interval(auc_boot)
            bal_lo, bal_hi = percentile_interval(balanced_boot)
            output.append({
                "cohort": cohort,
                "phase": phase,
                "training_group": training_group,
                "ggs_group": ggs_group,
                "population": population,
                "training_rows": int((y == 0).sum()),
                "ggs_rows": int((y == 1).sum()),
                "classifier": "standardized_logistic_regression",
                "cv": "5-fold StratifiedGroupKFold grouped by D4 board",
                "roc_auc": auc,
                "roc_auc_ci95_low": auc_lo,
                "roc_auc_ci95_high": auc_hi,
                "balanced_accuracy": balanced,
                "balanced_accuracy_ci95_low": bal_lo,
                "balanced_accuracy_ci95_high": bal_hi,
                "seed": SEED,
                "bootstrap_replicates": bootstrap,
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()

    rows: List[Dict[str, object]] = []
    add_training(rows, report_dir / "exact_input_30_empties.tsv", 30)
    add_training(rows, report_dir / "exact_input_25_empties.tsv", 35)
    add_training(rows, report_dir / "phase36_sample" / "sample_positions.tsv", 36)
    add_training(rows, report_dir / "exact_input_20_empties.tsv", 40)
    add_training(rows, report_dir / "exact_input_16_empties.tsv", 44)
    add_ggs(rows, report_dir / "egaroucid_ggs96_root.tsv", "ggs_seeded96")
    add_ggs(rows, report_dir / "egaroucid_ggs_latest96_root.tsv", "ggs_latest96")
    if not rows:
        raise SystemExit("no input rows found")
    frame = attach_exact_values(pd.DataFrame(rows), report_dir)
    frame["random_timing_bucket"] = [timing_bucket(row) for row in rows]
    frame["origin_bucket"] = [origin_bucket(row) for row in rows]
    frame.to_csv(report_dir / "position_distribution_features.csv", index=False)

    summaries: List[Dict[str, object]] = []
    for group, group_frame in frame.groupby("group", sort=True):
        for population, selected in (
            ("frequency_with_duplicates", group_frame),
            ("unique_d4", group_frame.drop_duplicates("canonical_board")),
        ):
            for feature in FEATURES:
                values = selected[feature].astype(float)
                summaries.append({
                    "group": group,
                    "phase": int(selected["phase"].iloc[0]),
                    "population": population,
                    "n": len(selected),
                    "feature": feature,
                    "mean": values.mean(),
                    "stddev": values.std(ddof=0),
                    "p10": values.quantile(0.10),
                    "median": values.median(),
                    "p90": values.quantile(0.90),
                })
    pd.DataFrame(summaries).to_csv(
        report_dir / "position_distribution_summary.csv", index=False
    )

    category_rows: List[Dict[str, object]] = []
    training = frame[frame["group"].str.startswith("training_phase")]
    for keys, selected in training.groupby(
        ["phase", "source_category", "origin_bucket", "random_timing_bucket"],
        dropna=False,
        sort=True,
    ):
        phase, source, origin, timing = keys
        exact = selected.dropna(subset=["teacher", "exact_value"])
        row: Dict[str, object] = {
            "phase": int(phase),
            "source_category": source,
            "origin_bucket": origin,
            "random_timing_bucket": timing,
            "frequency_sample_rows": len(selected),
            "unique_d4_positions": selected["canonical_board"].nunique(),
            "teacher_exact_rows": len(exact),
            "teacher_minus_exact_bias": (
                float((exact["teacher"] - exact["exact_value"]).mean())
                if len(exact) else math.nan
            ),
            "teacher_exact_mae": (
                float((exact["teacher"] - exact["exact_value"]).abs().mean())
                if len(exact) else math.nan
            ),
        }
        if len(exact) >= 3 and exact["exact_value"].var(ddof=0) > 0:
            slope, intercept = np.polyfit(exact["exact_value"], exact["teacher"], 1)
            row["teacher_exact_slope"] = slope
            row["teacher_exact_intercept"] = intercept
        else:
            row["teacher_exact_slope"] = math.nan
            row["teacher_exact_intercept"] = math.nan
        for feature in FEATURES:
            row[feature + "_mean"] = float(selected[feature].mean())
            row[feature + "_stddev"] = float(selected[feature].std(ddof=0))
        category_rows.append(row)
    pd.DataFrame(category_rows).to_csv(
        report_dir / "position_category_summary.csv", index=False
    )
    diagnostics = classifier_rows(frame, args.bootstrap)
    pd.DataFrame(diagnostics).to_csv(
        report_dir / "position_distribution_classifier.csv", index=False
    )
    with (report_dir / "position_distribution_classifier.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(
            {
                "status": "diagnostic_only_not_support_proof",
                "features": list(FEATURES),
                "seed": SEED,
                "results": diagnostics,
            },
            destination,
            ensure_ascii=False,
            indent=2,
        )
        destination.write("\n")


if __name__ == "__main__":
    main()
