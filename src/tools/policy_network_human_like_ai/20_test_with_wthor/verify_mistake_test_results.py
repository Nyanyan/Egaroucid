#!/usr/bin/env python3
"""Independently recount saved predictions and check engine/model agreement."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_mistake_test_models import MODEL_NAMES, THRESHOLDS, atomic_json, verify_subsets
from evaluate_wthor_blend_from_cache import ReadOnlyHintCache
from evaluate_wthor_blend_human_match import equivalent_targets
from wthor_human_match_evaluation import BoardState
from wthor_human_match_experiment import file_sha256


def run(args):
    output = args.output_dir
    summary = json.loads((output / "mistake_test_summary.json").read_text(encoding="utf-8"))
    counts, records, losses, indices, _, manifests = verify_subsets(args.datasets_dir)
    if (summary["identity"]["dataset_manifests"] != manifests
            or summary["identity"]["dataset_counts_sha256"] != file_sha256(args.datasets_dir / "dataset_counts.json")
            or summary["dataset_counts"] != counts["datasets"]):
        raise ValueError("result dataset identity differs")
    predictions_path = output / "predictions.npz"
    if summary["predictions_sha256"] != file_sha256(predictions_path):
        raise ValueError("prediction checksum differs")
    with np.load(predictions_path, allow_pickle=False) as saved:
        if tuple(saved["model_names"]) != MODEL_NAMES:
            raise ValueError("model roster differs")
        for actual, expected in ((saved["source_indices"], indices), (saved["losses"], losses),
                                 (saved["human_policies"], records["policy"])):
            if not np.array_equal(actual, expected):
                raise ValueError("prediction source rows differ")
        predicted = saved["predicted_policies"].copy()
        saved_targets = saved["symmetry_target_masks"].copy()
    if predicted.shape != (len(records), 27) or np.any(predicted < 0) or np.any(predicted > 63):
        raise ValueError("missing/invalid predicted moves")
    targets_by_label, legal_by_key, key_rows = {}, {}, {}
    expected_targets = np.empty(len(records), dtype=np.uint64)
    legal_masks = np.empty(len(records), dtype=np.uint64)
    for i, row in enumerate(records):
        player, opponent, side, label = (int(row[f]) for f in ("player", "opponent", "color", "policy"))
        key = (player, opponent, side) if side == 0 else (opponent, player, side)
        if key not in legal_by_key:
            legal_by_key[key] = sum(1 << move for move in BoardState(*key).legal_policies(side))
        target_key = (player, opponent, label)
        if target_key not in targets_by_label:
            targets_by_label[target_key] = sum(1 << move for move in equivalent_targets(player, opponent, label))
        expected_targets[i] = targets_by_label[target_key]
        legal_masks[i] = legal_by_key[key]
        key_rows.setdefault(key, []).append(i)
    if not np.array_equal(saved_targets, expected_targets):
        raise ValueError("saved symmetry targets differ from independently computed targets")
    move_masks = np.left_shift(np.uint64(1), predicted.astype(np.uint64))
    if np.any((move_masks & legal_masks[:, None]) == 0):
        raise ValueError("illegal predicted move")
    # Validate the engine endpoints against the persisted search cache. In
    # particular, alpha=0 sorts scores before applying native order to ties.
    for engine in ("egaroucid", "edax"):
        for level in (1, 5, 9, 13, 17, 21):
            hint_count = 64 if engine == "egaroucid" and level == 21 else 1
            directory = output / "edax" if engine == "edax" else output
            path = directory / f"{'edax_' if engine == 'edax' else ''}hint_score_cache_level{level}_hint{hint_count}.sqlite3"
            cache = ReadOnlyHintCache(path)
            try:
                hints = cache.load(level, hint_count, list(key_rows))
            finally:
                cache.close()
            if set(hints) != set(key_rows):
                raise ValueError("incomplete engine cache")
            name = "alpha_0.0" if engine == "egaroucid" and level == 21 else f"{engine}_l{level}"
            col = MODEL_NAMES.index(name)
            for key, hint in hints.items():
                expected = max(hint.move_order, key=lambda move: hint.scores[move]) if hint_count == 64 else hint.move_order[0]
                if np.any(predicted[key_rows[key], col] != expected):
                    raise ValueError(f"prediction differs from cache: {name}, {key}")
            print(f"Verified {name}: {len(hints):,} states", flush=True)
    matched = (move_masks & expected_targets[:, None]) != 0
    exact = predicted == records["policy"][:, None]
    rows = {(r["model"], r["minimum_loss"]): r for r in summary["results"]}
    if len(rows) != 108:
        raise ValueError("expected 108 result rows")
    for col, name in enumerate(MODEL_NAMES):
        for threshold in THRESHOLDS:
            mask = losses >= threshold
            n, hits, strict = int(mask.sum()), int(matched[mask, col].sum()), int(exact[mask, col].sum())
            row = rows[name, threshold]
            if (row["positions"], row["requested_positions"], row["top1_hits"], row["exact_top1_hits"], row["status"]) != (n, n, hits, strict, "complete"):
                raise ValueError(f"result recount differs: {name}, {threshold}")
            if row["top1_accuracy"] != (hits / n if n else None):
                raise ValueError("accuracy differs from hits/positions")
    report = dict(status="passed", model_count=27, threshold_count=4, result_rows=108,
                  source_positions=counts["source_positions"], union_positions=len(records),
                  checks=["source dataset checksums and exact nested membership", "source labels and indices",
                          "independent symmetry and legal move computation", "all 12 engine endpoints match caches",
                          "all 108 hit counts and accuracies independently recounted"],
                  summary_sha256=file_sha256(output / "mistake_test_summary.json"),
                  predictions_sha256=summary["predictions_sha256"])
    atomic_json(output / "verification.json", report)
    lines = ["# ミスした着手に対する27モデルの一致率", "",
             "WTHOR_1977_2024の水増し前test全体から、Egaroucidレベル10の解析による石損で抽出しました。"
             "各条件は「以上」の累積条件で、同じ局面が複数の条件に含まれます。", "",
             "| 石損 | 局面数 | 棋譜数 |", "| --- | ---: | ---: |"]
    for row in counts["datasets"]:
        lines.append(f"| {row['minimum_loss']}石以上 | {row['positions']:,} | {row['games']:,} |")
    lines.extend(["", "一致率はTop-1で、盤面を保つ対称変換で等価な着手も一致とします。"
                  "通常NNのα混合が `alpha`、石損重み付きNNのα混合が `loss_weighted_alpha`、"
                  "両NNのβ混合が `model_blend_beta` です。α=0はEgaroucidレベル21、α=1はNN単体です。", "",
                  "| モデル | 1石以上 | 2石以上 | 4石以上 | 6石以上 |", "| --- | ---: | ---: | ---: | ---: |"])
    for name in MODEL_NAMES:
        values = [rows[name, t]["top1_accuracy_percent"] for t in THRESHOLDS]
        lines.append("| " + name + " | " + " | ".join(f"{v:.4f}%" if v is not None else "対象なし" for v in values) + " |")
    lines.extend(["", "全108条件の一致数を保存済みの予測着手から独立に検算しました。"
                  "予測手の合法性、対称性、12種類のエンジン設定と探索キャッシュの一致も確認しています。", "",
                  "- [一致数・評価件数付きの一覧](mistake_test_accuracy_matrix.csv)",
                  "- [完全一致率と95%区間を含む詳細](mistake_test_summary.csv)",
                  "- [実験条件と全結果](mistake_test_summary.json)",
                  "- [検算結果](verification.json)", ""])
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
