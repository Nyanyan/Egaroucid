#!/usr/bin/env python3
"""Show policy, Egaroucid, blended, and WTHOR values for one board.

The defaults reproduce the position after::

    f5d6c3d3c4f4f6f3e6e7

WTHOR frequencies are counted by exact side-relative bitboards and side to
move, rather than by transcript prefix.  This includes games that reach the
same position through a different move order and matches the grouping used by
the existing WTHOR experiments in this directory tree.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BLEND_DIR = SCRIPT_DIR.parent / "30_blend_with_egaroucid"
sys.path.insert(0, str(BLEND_DIR))

from blend_policy_with_egaroucid import (  # noqa: E402
    POLICY_SIZE,
    BlendedPolicy,
    BoardState,
    board_to_features,
    default_egaroucid_exe,
    default_weights_file,
    geometric_blend_distribution,
    normalize_policy_on_legal,
    parse_hint_move_order,
    parse_side,
    policy_to_coord,
    side_to_egaroucid_char,
)


DEFAULT_TRANSCRIPT = "f5d6c3d3c4f4f6f3e6e7"
DEFAULT_POSITION = (
    "------------------XO-O----XXOO-----XOX-----OOX------O----------- X"
)
DEFAULT_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
BOARD_SAMPLE_SIZE = 19
BOARD_DTYPE = np.dtype(
    [
        ("player", "<u8"),
        ("opponent", "<u8"),
        ("color", "i1"),
        ("policy", "i1"),
        ("score", "i1"),
    ],
    align=False,
)
assert BOARD_DTYPE.itemsize == BOARD_SAMPLE_SIZE


def default_wthor_board_data_dir() -> Path:
    data_root = os.environ.get("EGAROUCID_DATA")
    if data_root:
        return Path(data_root) / "train_data" / "board_data" / "records1"
    return REPO_ROOT / "train_data" / "board_data" / "records1"


def parse_position(text: str) -> BoardState:
    parts = text.strip().rsplit(maxsplit=1)
    if len(parts) != 2 or len(parts[0]) != 64:
        raise ValueError(
            "position must be a 64-character X/O/- board followed by X or O"
        )
    board, side_text = parts
    return BoardState.from_board_string(board, parse_side(side_text))


def verify_transcript_position(transcript: str, position: str) -> BoardState:
    transcript_state = BoardState.from_transcript(transcript)
    position_state = parse_position(position)
    transcript_key = (
        transcript_state.black,
        transcript_state.white,
        transcript_state.side,
    )
    position_key = (
        position_state.black,
        position_state.white,
        position_state.side,
    )
    if transcript_key != position_key:
        raise ValueError(
            "the transcript does not produce the supplied board and side: "
            f"transcript={transcript_state.to_egaroucid_board()}, "
            f"position={position_state.to_egaroucid_board()}"
        )
    return position_state


def parse_alphas(text: str) -> List[float]:
    alphas = [float(token.strip()) for token in text.split(",") if token.strip()]
    if not alphas:
        raise argparse.ArgumentTypeError("at least one alpha is required")
    if any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise argparse.ArgumentTypeError("every alpha must be in [0, 1]")
    if len(set(alphas)) != len(alphas):
        raise argparse.ArgumentTypeError("alphas must not contain duplicates")
    return alphas


def alpha_label(alpha: float) -> str:
    label = f"{alpha:.10g}"
    if "." not in label and "e" not in label.lower():
        label += ".0"
    return label


def discover_wthor_files(board_data_dir: Path) -> List[Path]:
    def sort_key(path: Path) -> Tuple[int, str]:
        return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)

    files = sorted(board_data_dir.glob("*.dat"), key=sort_key)
    if not files:
        raise FileNotFoundError(
            f"no WTHOR board-data .dat files found in {board_data_dir}; "
            "set EGAROUCID_DATA or pass --board-data-dir"
        )
    return files


def count_wthor_moves(
    state: BoardState,
    legal_policies: Sequence[int],
    board_data_dir: Path,
    batch_size: int,
) -> Tuple[np.ndarray, int, int, List[Path]]:
    """Count exact-position WTHOR samples and return counts by policy index."""
    if batch_size <= 0:
        raise ValueError("WTHOR batch size must be positive")

    dat_files = discover_wthor_files(board_data_dir)
    player, opponent = state.player_opponent_bits(state.side)
    player_u64 = np.uint64(player)
    opponent_u64 = np.uint64(opponent)
    legal_mask = np.zeros(POLICY_SIZE, dtype=bool)
    legal_mask[np.asarray(legal_policies, dtype=np.int64)] = True
    counts = np.zeros(POLICY_SIZE, dtype=np.int64)
    scanned_samples = 0
    matching_samples = 0
    invalid_matching_samples = 0

    for path in dat_files:
        file_size = path.stat().st_size
        if file_size % BOARD_SAMPLE_SIZE:
            raise ValueError(
                f"WTHOR board-data file has trailing bytes: {path} "
                f"({file_size} bytes)"
            )
        file_samples = file_size // BOARD_SAMPLE_SIZE
        records = np.memmap(
            path,
            dtype=BOARD_DTYPE,
            mode="r",
            shape=(file_samples,),
        )
        try:
            for start in range(0, file_samples, batch_size):
                batch = records[start : min(file_samples, start + batch_size)]
                matches = (
                    (batch["player"] == player_u64)
                    & (batch["opponent"] == opponent_u64)
                    & (batch["color"] == state.side)
                )
                matched_policies = np.asarray(
                    batch["policy"][matches],
                    dtype=np.int64,
                )
                matching_samples += len(matched_policies)
                if len(matched_policies):
                    in_range = (0 <= matched_policies) & (
                        matched_policies < POLICY_SIZE
                    )
                    valid = np.zeros(len(matched_policies), dtype=bool)
                    valid[in_range] = legal_mask[matched_policies[in_range]]
                    invalid_matching_samples += int(np.count_nonzero(~valid))
                    counts += np.bincount(
                        matched_policies[valid],
                        minlength=POLICY_SIZE,
                    ).astype(np.int64, copy=False)
                scanned_samples += len(batch)
        finally:
            del records

    if invalid_matching_samples:
        raise ValueError(
            "matching WTHOR samples contain invalid or illegal move labels: "
            f"{invalid_matching_samples}"
        )
    if int(np.sum(counts)) != matching_samples:
        raise RuntimeError("WTHOR matching-sample accounting mismatch")
    return counts, matching_samples, scanned_samples, dat_files


def ordered_legal_moves(
    raw_hint: str,
    legal_policies: Sequence[int],
) -> List[int]:
    legal_set = set(legal_policies)
    result: List[int] = []
    seen = set()
    for policy in parse_hint_move_order(raw_hint):
        if policy in legal_set and policy not in seen:
            result.append(policy)
            seen.add(policy)
    for policy in legal_policies:
        if policy not in seen:
            result.append(policy)
            seen.add(policy)
    return result


def make_result(args: argparse.Namespace) -> dict:
    state = verify_transcript_position(args.transcript, args.position)
    legal_policies = state.legal_policies(state.side)
    if not legal_policies:
        raise ValueError("the supplied side has no legal move")
    if not args.weights.is_file():
        raise FileNotFoundError(f"policy-network weights not found: {args.weights}")
    if not args.egaroucid_exe.is_file():
        raise FileNotFoundError(
            f"Egaroucid for Console executable not found: {args.egaroucid_exe}"
        )

    wthor_counts, wthor_matches, wthor_scanned, dat_files = count_wthor_moves(
        state,
        legal_policies,
        args.board_data_dir,
        args.wthor_batch_size,
    )

    blender = BlendedPolicy(
        weights=args.weights,
        egaroucid_exe=args.egaroucid_exe,
        egaroucid_level=args.egaroucid_level,
        egaroucid_threads=args.egaroucid_threads,
        egaroucid_timeout_sec=args.egaroucid_timeout_sec,
        score_temperature=args.score_temperature,
        legal_mask_policy=True,
    )
    try:
        if blender.policy_network is None:
            raise RuntimeError("the local policy network was not initialized")
        policy_raw = blender.policy_network.predict(
            board_to_features(state, state.side)
        )[0]
        policy_legal = normalize_policy_on_legal(policy_raw, legal_policies)
        scores, raw_hint = blender.cached_hint_scores(state, state.side)
        egaroucid_policy = blender.egaroucid_distribution(
            scores,
            legal_policies,
        )
    finally:
        blender.close()

    blends: Dict[str, np.ndarray] = {}
    for alpha in args.alphas:
        label = alpha_label(alpha)
        blends[label] = geometric_blend_distribution(
            policy_legal,
            egaroucid_policy,
            legal_policies,
            alpha,
        )

    move_order = ordered_legal_moves(raw_hint, legal_policies)
    rows = []
    for policy in move_order:
        count = int(wthor_counts[policy])
        rows.append(
            {
                "move": policy_to_coord(policy),
                "policy_index": int(policy),
                "policy_network_raw": float(policy_raw[policy]),
                "policy_network": float(policy_legal[policy]),
                "egaroucid_score": float(scores[policy]),
                "egaroucid_policy": float(egaroucid_policy[policy]),
                "blend": {
                    label: float(distribution[policy])
                    for label, distribution in blends.items()
                },
                "wthor_count": count,
                "wthor_probability": (
                    count / wthor_matches if wthor_matches else 0.0
                ),
            }
        )

    result = {
        "transcript": args.transcript,
        "position": state.to_egaroucid_board(state.side),
        "side": side_to_egaroucid_char(state.side),
        "legal_moves": [policy_to_coord(policy) for policy in legal_policies],
        "policy_network": {
            "weights": str(args.weights.resolve()),
            "raw_64way_legal_mass": float(
                np.sum(policy_raw[np.asarray(legal_policies, dtype=np.int64)])
            ),
            "reported_policy": "raw 64-way softmax and legal-masked renormalized softmax",
        },
        "egaroucid": {
            "executable": str(args.egaroucid_exe.resolve()),
            "level": args.egaroucid_level,
            "threads": args.egaroucid_threads,
            "score_temperature": args.score_temperature,
            "policy_formula": "softmax(score / score_temperature) over legal moves",
        },
        "blend": {
            "alphas": [float(alpha) for alpha in args.alphas],
            "formula": "normalize(policy_network^alpha * egaroucid_policy^(1-alpha))",
        },
        "wthor": {
            "board_data_dir": str(args.board_data_dir.resolve()),
            "files": [str(path.resolve()) for path in dat_files],
            "scanned_position_samples": wthor_scanned,
            "matching_position_samples": wthor_matches,
            "match_rule": "exact player/opponent bitboards and side; no symmetry merge",
        },
        "rows": rows,
    }
    if args.show_raw_hint:
        result["egaroucid"]["raw_hint"] = raw_hint
    return result


def format_probability(value: float) -> str:
    return f"{value:.10g}"


def print_markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---:" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def print_human_result(result: dict) -> None:
    print(f"transcript: {result['transcript']}")
    print(f"position:   {result['position']}")
    print("Policy Network: raw=64-way softmax, policy=legal-masked and renormalized")
    print(
        "Egaroucid Policy: softmax(score / "
        f"{result['egaroucid']['score_temperature']:g}) over legal moves"
    )
    print(
        "WTHOR: "
        f"{result['wthor']['matching_position_samples']:,} exact matches / "
        f"{result['wthor']['scanned_position_samples']:,} position samples"
    )
    print()
    print("Policy Network / Egaroucid / WTHOR")
    print_markdown_table(
        (
            "move",
            "network raw",
            "network policy",
            "eval",
            "eval policy",
            "WTHOR count",
            "WTHOR probability",
        ),
        (
            (
                row["move"],
                format_probability(row["policy_network_raw"]),
                format_probability(row["policy_network"]),
                f"{row['egaroucid_score']:g}",
                format_probability(row["egaroucid_policy"]),
                f"{row['wthor_count']:,}",
                format_probability(row["wthor_probability"]),
            )
            for row in result["rows"]
        ),
    )

    print()
    print("Blended Policy")
    alpha_labels = [alpha_label(alpha) for alpha in result["blend"]["alphas"]]
    print_markdown_table(
        ("move", *(f"alpha={label}" for label in alpha_labels)),
        (
            (
                row["move"],
                *(
                    format_probability(row["blend"][label])
                    for label in alpha_labels
                ),
            )
            for row in result["rows"]
        ),
    )

    if "raw_hint" in result["egaroucid"]:
        print()
        print("Raw Egaroucid hint output")
        print(result["egaroucid"]["raw_hint"].rstrip())


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print Policy Network, Egaroucid, blended-policy, and exact-position "
            "WTHOR move probabilities for one Othello position."
        )
    )
    parser.add_argument("--transcript", default=DEFAULT_TRANSCRIPT)
    parser.add_argument(
        "--position",
        default=DEFAULT_POSITION,
        help=(
            "64-character X/O/- board plus side; values beginning with '-' "
            "must use --position=<value>"
        ),
    )
    parser.add_argument("--weights", type=Path, default=default_weights_file())
    parser.add_argument(
        "--egaroucid-exe",
        type=Path,
        default=default_egaroucid_exe(),
    )
    parser.add_argument("--egaroucid-level", type=int, default=21)
    parser.add_argument("--egaroucid-threads", type=int, default=1)
    parser.add_argument("--egaroucid-timeout-sec", type=float, default=60.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument(
        "--alphas",
        type=parse_alphas,
        default=list(DEFAULT_ALPHAS),
        help="comma-separated blend alphas (default: 0.0,0.2,...,1.0)",
    )
    parser.add_argument(
        "--board-data-dir",
        type=Path,
        default=default_wthor_board_data_dir(),
    )
    parser.add_argument("--wthor-batch-size", type=int, default=1_000_000)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-raw-hint", action="store_true")
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    if args.score_temperature <= 0.0:
        raise ValueError("score temperature must be positive")
    result = make_result(args)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_human_result(result)


if __name__ == "__main__":
    main()
