"""Write a compact bilingual progress report for root-move precomputation.

The generator keeps every completed position in a durable JSON-lines companion
file until the selected compaction interval. This tool reads both that file
and the last compacted state, without modifying either, so its counts remain
accurate between compactions.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

import generate_ggs_root_teacher
from r14_random_setup_probability import fraction_json, r14_random_setup_probability


STATE_SUFFIX = ".state.json"
PROBABILITY_SOURCE_JAPANESE_DESCRIPTIONS = {
    "enumerated_population": (
        "`random_setup(14)` で作り得る局面集合の列挙と、回転・反射を一つの代表局面へまとめる処理",
        "このファイル単独では、各局面が選ばれる確率を決めない",
    ),
    "sampling_steps": (
        "セルの選択、O 石数の選択、色の割当てを行うローカルの抽選手順",
        "alternate 方式用のツールであり、`random_setup(14)` を直接実行してはいない",
    ),
    "repository_description": (
        "このリポジトリが主方式・alternate 方式との対応関係として記録している説明",
        "現在 GGS で動いているソースや版を証明するものではない",
    ),
}


def output_path_from_state(state_path: Path) -> Path:
    if not state_path.name.endswith(STATE_SUFFIX):
        raise ValueError(f"state path must end with {STATE_SUFFIX}: {state_path}")
    return state_path.with_name(state_path.name[: -len(STATE_SUFFIX)])


def _read_state(state_path: Path) -> dict[str, Any]:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read state {state_path}: {error}") from error
    if state.get("schema") != generate_ggs_root_teacher.TEACHER_SCHEMA:
        raise ValueError(f"{state_path}: unsupported state schema")
    for field in ("roots", "results", "rejections"):
        expected_type = list if field == "roots" else dict
        if not isinstance(state.get(field), expected_type):
            raise ValueError(f"{state_path}: invalid {field}")
    return state


def _pending_record_count(output_path: Path) -> int:
    path = generate_ggs_root_teacher._pending_updates_path(output_path)
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read pending updates {path}: {error}") from error


def _fraction_from_json(value: dict[str, int]) -> Fraction:
    """Restore an exact rational value recorded in a JSON-friendly form."""
    return Fraction(value["numerator"], value["denominator"])


def _format_fraction_with_percent(value: Fraction) -> str:
    """Show the exact value first; the decimal percentage is display-only."""
    return f"{value.numerator}/{value.denominator} (約 {float(value * 100):.6f}%)"


def _probability_sums(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return exact sums for a probability-ordered ``random_setup(14)`` calculation.

    The sums apply only to the boards listed in this teacher calculation.  They
    are not a claim about all ``s8r14`` starts, because the other start-board
    construction is outside this input directory.
    """
    if state.get("root_order") != generate_ggs_root_teacher.ROOT_ORDER_GGS_R14_PROBABILITY:
        return None

    roots = state["roots"]
    root_set = set(roots)
    accepted_set = set(state["results"])
    rejected_set = set(state["rejections"])
    if len(root_set) != len(roots):
        raise ValueError("probability-ordered state has duplicate roots")
    if accepted_set & rejected_set:
        raise ValueError("probability-ordered state has a root in both results and rejections")
    processed_set = accepted_set | rejected_set
    if not processed_set <= root_set:
        raise ValueError("probability-ordered state has a processed root outside its roots")

    def total(boards: set[str]) -> Fraction:
        return sum((r14_random_setup_probability(board) for board in boards), Fraction(0))

    requested = total(root_set)
    accepted = total(accepted_set)
    rejected = total(rejected_set)
    remaining = total(root_set - processed_set)
    if accepted + rejected + remaining != requested:
        raise ValueError("probability-ordered state has inconsistent probability sums")
    priority_manifest = state.get("priority_manifest")
    if not isinstance(priority_manifest, dict):
        raise ValueError("probability-ordered state has no frozen priority-file provenance")
    sha256 = priority_manifest.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("probability-ordered state has an invalid priority-file SHA-256")
    metadata_sha256 = priority_manifest.get("metadata_sha256")
    if not isinstance(metadata_sha256, str) or len(metadata_sha256) != 64:
        raise ValueError("probability-ordered state has an invalid priority metadata SHA-256")
    frozen_tie_seed = priority_manifest.get("tie_seed")
    if frozen_tie_seed is not None and (
        isinstance(frozen_tie_seed, bool) or not isinstance(frozen_tie_seed, int)
    ):
        raise ValueError("probability-ordered state has an invalid frozen priority-file tie seed")
    if state.get("priority_manifest_tie_seed") != frozen_tie_seed:
        raise ValueError("probability-ordered state does not record its frozen priority-file tie seed")
    input_audit = priority_manifest.get("input_audit")
    if not isinstance(input_audit, dict):
        raise ValueError("probability-ordered state has no frozen priority-file input audit")
    sources = input_audit.get("local_probability_model_sources")
    if not isinstance(sources, dict) or not isinstance(sources.get("files"), list):
        raise ValueError("probability-ordered state has no local probability-model sources")
    if state.get("r14_local_probability_model_sources") != sources:
        raise ValueError(
            "probability-ordered state does not match its frozen probability-model sources"
        )
    source_files: list[dict[str, str]] = []
    for index, source in enumerate(sources["files"]):
        if not isinstance(source, dict):
            raise ValueError(f"probability-ordered state has invalid source file {index}")
        identifier = source.get("id")
        path = source.get("relative_path")
        digest = source.get("sha256")
        role = source.get("role")
        limitation = source.get("limitation")
        if not all(
            isinstance(value, str) and value
            for value in (identifier, path, digest, role, limitation)
        ):
            raise ValueError(f"probability-ordered state has invalid source file {index}")
        if identifier not in PROBABILITY_SOURCE_JAPANESE_DESCRIPTIONS:
            raise ValueError(f"probability-ordered state has an unknown source file {identifier}")
        source_files.append(
            {
                "id": identifier,
                "relative_path": path,
                "sha256": digest,
                "role": role,
                "limitation": limitation,
            }
        )
    return {
        "requested": fraction_json(requested),
        "accepted": fraction_json(accepted),
        "rejected": fraction_json(rejected),
        "processed": fraction_json(accepted + rejected),
        "remaining": fraction_json(remaining),
        "priority_manifest_sha256": sha256,
        "priority_manifest_metadata_sha256": metadata_sha256,
        "priority_manifest_tie_seed": frozen_tie_seed,
        "local_probability_model_source_files": source_files,
    }


def progress_counts(state_path: Path) -> dict[str, Any]:
    """Return counts from the compacted state plus durable pending updates."""
    state = _read_state(state_path)
    output_path = output_path_from_state(state_path)
    compacted_accepted = len(state["results"])
    compacted_rejected = len(state["rejections"])
    effective = copy.deepcopy(state)
    generate_ggs_root_teacher._apply_pending_updates(output_path, effective)
    accepted = len(effective["results"])
    rejected = len(effective["rejections"])
    requested = len(effective["roots"])
    processed = accepted + rejected
    probability_sums = _probability_sums(effective)
    return {
        "requested": requested,
        "accepted": accepted,
        "rejected": rejected,
        "processed": processed,
        "remaining": requested - processed,
        "compacted_accepted": compacted_accepted,
        "compacted_rejected": compacted_rejected,
        "pending_records": _pending_record_count(output_path),
        "state_schema": str(state["schema"]),
        "probability_sums": probability_sums,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_progress_report(state_path: Path, report_path: Path) -> dict[str, Any]:
    counts = progress_counts(state_path)
    output_path = output_path_from_state(state_path)
    pending_path = generate_ggs_root_teacher._pending_updates_path(output_path)
    probability_sums = counts["probability_sums"]
    if probability_sums is None:
        probability_japanese = ""
        probability_english = ""
    else:
        exact = {
            key: _fraction_from_json(probability_sums[key])
            for key in ("requested", "accepted", "rejected", "processed", "remaining")
        }
        japanese_source_lines = "\n".join(
            (
                f"  - `{source['relative_path']}`: "
                f"{PROBABILITY_SOURCE_JAPANESE_DESCRIPTIONS[source['id']][0]}。"
                f"制限: {PROBABILITY_SOURCE_JAPANESE_DESCRIPTIONS[source['id']][1]}。"
                f" SHA-256: `{source['sha256']}`"
            )
            for source in probability_sums["local_probability_model_source_files"]
        )
        english_source_lines = "\n".join(
            (
                f"  - `{source['relative_path']}`: {source['role']}. "
                f"Limitation: {source['limitation']}. SHA-256: `{source['sha256']}`"
            )
            for source in probability_sums["local_probability_model_source_files"]
        )
        probability_japanese = f"""
- この計算の対象局面が表す出現確率の合計: {_format_fraction_with_percent(exact['requested'])}
- 採用済み局面が表す出現確率の合計: {_format_fraction_with_percent(exact['accepted'])}
- 不採用局面が表す出現確率の合計: {_format_fraction_with_percent(exact['rejected'])}
- 処理済み局面が表す出現確率の合計: {_format_fraction_with_percent(exact['processed'])}
- 未処理局面が表す出現確率の合計: {_format_fraction_with_percent(exact['remaining'])}
- この順番を固定した入力ファイルの SHA-256: `{probability_sums['priority_manifest_sha256']}`
- 入力ファイルのメタデータの SHA-256: `{probability_sums['priority_manifest_metadata_sha256']}`
- 同じ確率の局面を並べるため、固定順位ファイルに記録された seed: `{probability_sums['priority_manifest_tie_seed']}`
- 確率式の根拠ファイル:
{japanese_source_lines}

ここでいう「出現確率の合計」は、`records321_14_random_setup` にある `random_setup(14)` で作られる開始局面だけについて、各局面の回転・反射の個数と O 石数から求めた確率を足した値である。回転・反射は盤面の幾何学的な 8 通りであり、X と O の交換や手番の変更は含めない。`random_setup_2` で作られる開始局面はこの値に含めない。このため、ここに示す百分率を `s8r14` 全体のカバー率と解釈してはならない。式は上記3ファイルから作ったローカルの確率モデルを根拠にしており、現行 GGS サーバーのソースそのものと主張するものではない。整数や丸め誤差で順位を決めないため、判定には先頭の分数を用い、百分率は読みやすさのための表示だけである。

「確率順序記録ファイル」の定義:

- 出典: 上記3ファイル、開始局面一覧、および開始局面一覧の監査結果。
- 目的: 途中までの計算で、`random_setup(14)` の中で出現しやすい局面から処理する。
- 具体対象: 各代表局面、その有理数の確率、順位、同率時の seed、入力ファイルと根拠ファイルの SHA-256。
- 役割: 計算順・再開・進捗の確率合計を再現可能にする。
- 前後関係: 開始局面一覧を監査した後、教師計算の前に作成し、教師計算はその順を変更せずに読む。
- 候補語: 「確率順序記録ファイル」「出現確率順の局面一覧」。
- 初出定義: この報告書のこの節。
"""
        probability_english = f"""
- Sum of occurrence probabilities represented by all positions in this calculation: {_format_fraction_with_percent(exact['requested'])}
- Sum represented by accepted positions: {_format_fraction_with_percent(exact['accepted'])}
- Sum represented by rejected positions: {_format_fraction_with_percent(exact['rejected'])}
- Sum represented by processed positions: {_format_fraction_with_percent(exact['processed'])}
- Sum represented by remaining positions: {_format_fraction_with_percent(exact['remaining'])}
- SHA-256 of the fixed input file that established this order: `{probability_sums['priority_manifest_sha256']}`
- SHA-256 of the metadata sidecar for that input file: `{probability_sums['priority_manifest_metadata_sha256']}`
- Seed recorded in the frozen priority file for ordering equal-probability positions: `{probability_sums['priority_manifest_tie_seed']}`
- Local files used as evidence for the formula:
{english_source_lines}

These sums cover only positions made by `random_setup(14)` in `records321_14_random_setup`. They add the probability of each representative using its number of distinct rotations/reflections and its O-disc count. The rotations/reflections are the eight geometric board transformations; they do not exchange X and O or change side to move. They exclude positions made by `random_setup_2`, so they are not coverage percentages for all `s8r14` starts. The formula is a local probability model derived from the three files above, not a claim about the current GGS server source. The fractions are the exact values used for accounting; displayed percentages are only for readability.

Definition of “probability-order record file”:

- Source: the three files above, the starting-position list, and the audit of that list.
- Purpose: process more frequently occurring positions within `random_setup(14)` first when the calculation is incomplete.
- Concrete target: each representative position, its exact rational probability, rank, equal-probability seed, and SHA-256 values for input and source files.
- Role: make calculation order, resumption, and probability-sum progress reproducible.
- Preceding and following steps: create it after auditing the starting-position list and before teacher calculation; teacher calculation reads it without changing its order.
- Candidate terms: “probability-order record file” and “position list ordered by occurrence probability”.
- Initial definition: this section of this report.
"""
    text = f"""# 開始局面の最初の手の事前計算：進捗報告

## 日本語

- 対象局面数: {counts['requested']}
- 採用局面数（品質検査を通過し、最初の手を記録できた局面）: {counts['accepted']}
- 不採用局面数（品質検査を通過しなかった局面）: {counts['rejected']}
- 処理済み局面数: {counts['processed']}/{counts['requested']}
- 未処理局面数: {counts['remaining']}
- 状態ファイルへ統合済みの採用局面数: {counts['compacted_accepted']}
- 状態ファイルへ統合済みの不採用局面数: {counts['compacted_rejected']}
- 状態ファイルへの統合待ちとして安全に追記済みの記録数: {counts['pending_records']}
- 状態ファイル形式: `{counts['state_schema']}`
{probability_japanese}

状態ファイルへ統合済みの件数には、直近の統合以降に完了した局面は含まれない。一方、追記ファイル `{pending_path.name}` の各記録は局面の完了時点で安全に保存される。`--resume` を指定して再開すると、この追記ファイルの内容を最初に状態ファイルへ反映する。したがって、「採用局面数」「不採用局面数」「処理済み局面数」は、統合待ちの安全な追記記録も含む実際に再開可能な件数である。

この報告書は進捗ファイルを読むだけであり、探索結果、状態ファイル、大会用の開始局面の手の表を変更しない。

## English

- Requested positions: {counts['requested']}
- Accepted positions, including durable append records: {counts['accepted']}
- Rejected positions, including durable append records: {counts['rejected']}
- Processed positions, including durable append records: {counts['processed']}/{counts['requested']}
- Remaining positions: {counts['remaining']}
- Accepted positions already compacted into the state file: {counts['compacted_accepted']}
- Rejected positions already compacted into the state file: {counts['compacted_rejected']}
- Records in the durable append file: {counts['pending_records']}
- State-file format: `{counts['state_schema']}`
{probability_english}

The compacted counts do not include positions completed after the latest compaction. Records in `{pending_path.name}` are saved when each position completes and are applied to the state file first by `--resume`. The counts described as including durable append records are the actual resumable processed counts.

This report only reads files; it does not change search results, the state file, or the tournament table of moves for starting positions.
"""
    _atomic_write_text(report_path, text)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = write_progress_report(args.state, args.output)
    print(
        f"processed {counts['processed']}/{counts['requested']} "
        f"accepted={counts['accepted']} rejected={counts['rejected']} "
        f"pending={counts['pending_records']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
