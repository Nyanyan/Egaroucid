#!/usr/bin/env python3
"""
Minimal GTP engine that plays the argmax move of the blended distribution.

This is intended for the strength tests in ../40_test_strength.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from blend_policy_with_egaroucid import (
    BLACK,
    WHITE,
    BlendedPolicy,
    BoardState,
    coord_to_policy,
    default_egaroucid_exe,
    default_weights_file,
    parse_side,
    policy_to_coord,
    side_to_gtp_color,
)


def parse_color(text: str) -> int:
    return parse_side(text)


class BlendGtpEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state = BoardState.initial()
        self.last_move_stone_loss = None
        self.blender = BlendedPolicy(
            weights=args.weights,
            egaroucid_exe=args.egaroucid_exe,
            egaroucid_level=args.egaroucid_level,
            egaroucid_threads=args.egaroucid_threads,
            egaroucid_timeout_sec=args.egaroucid_timeout_sec,
            cache_egaroucid=args.cache_egaroucid,
            hint_cache_db=args.hint_cache_db,
            policy_server_host=args.policy_server_host,
            policy_server_port=args.policy_server_port,
            policy_server_timeout_sec=args.policy_server_timeout_sec,
            score_temperature=args.score_temperature,
            legal_mask_policy=not args.no_legal_mask_policy,
            minimum_available_memory_mib=(
                args.minimum_available_memory_mib
            ),
            estimated_egaroucid_memory_mib=(
                args.estimated_egaroucid_memory_mib
            ),
        )

    def clear_board(self) -> str:
        self.state = BoardState.initial()
        self.last_move_stone_loss = None
        return ""

    def play(self, color: str, move: str) -> str:
        self.last_move_stone_loss = None
        side = parse_color(color)
        move = move.strip().lower()
        if move == "pass":
            self.state.side = side ^ 1
            return ""
        self.state.apply_move(side, coord_to_policy(move))
        return ""

    def genmove(self, color: str) -> str:
        self.last_move_stone_loss = None
        side = parse_color(color)
        legal = self.state.legal_policies(side)
        if not legal:
            self.state.side = side ^ 1
            return "pass"
        blended, _, _, hint_scores, _ = self.blender.blended_distribution(self.state, side, self.args.blend_param)
        best_policy = max(legal, key=lambda policy: float(blended[policy]))
        if self.args.measure_move_stone_loss:
            if not hint_scores:
                hint_scores, _ = self.blender.cached_hint_scores(self.state, side)
            scored_legal = [policy for policy in legal if policy in hint_scores]
            if best_policy in hint_scores and scored_legal:
                self.last_move_stone_loss = max(hint_scores[policy] for policy in scored_legal) - hint_scores[best_policy]
        self.state.apply_move(side, best_policy)
        return policy_to_coord(best_policy)

    def setboard(self, args: list[str]) -> str:
        if len(args) < 2:
            raise ValueError("setboard requires '<64 squares> <side>'")
        board = args[0]
        side = parse_color(args[1])
        self.state = BoardState.from_board_string(board, side)
        self.last_move_stone_loss = None
        return ""

    def get_last_move_stone_loss(self) -> str:
        if self.last_move_stone_loss is None:
            return "unavailable"
        return f"{self.last_move_stone_loss:.10g}"

    def dispatch(self, command: str, args: list[str]) -> tuple[bool, str]:
        command = command.lower()
        if command == "protocol_version":
            return True, "2"
        if command == "name":
            return True, "EgaroucidPolicyBlend"
        if command == "version":
            return True, "0.1"
        if command == "list_commands":
            return True, "\n".join(
                [
                    "protocol_version",
                    "name",
                    "version",
                    "list_commands",
                    "boardsize",
                    "clear_board",
                    "komi",
                    "time_settings",
                    "time_left",
                    "play",
                    "genmove",
                    "last_move_stone_loss",
                    "setboard",
                    "quit",
                ]
            )
        if command == "boardsize":
            if args and args[0] != "8":
                raise ValueError("only boardsize 8 is supported")
            return True, ""
        if command in ("komi", "time_settings", "time_left"):
            return True, ""
        if command == "clear_board":
            return True, self.clear_board()
        if command == "play":
            if len(args) != 2:
                raise ValueError("play requires '<color> <move>'")
            return True, self.play(args[0], args[1])
        if command == "genmove":
            if len(args) != 1:
                raise ValueError("genmove requires '<color>'")
            return True, self.genmove(args[0])
        if command == "last_move_stone_loss":
            return True, self.get_last_move_stone_loss()
        if command == "setboard":
            return True, self.setboard(args)
        if command == "quit":
            return False, ""
        raise ValueError(f"unknown command: {command}")

    @staticmethod
    def write_response(ok: bool, payload: str = "") -> None:
        prefix = "=" if ok else "?"
        if payload:
            sys.stdout.write(prefix + " " + payload + "\n\n")
        else:
            sys.stdout.write(prefix + "\n\n")
        sys.stdout.flush()

    def loop(self) -> None:
        try:
            for raw_line in sys.stdin:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                command = parts[0]
                args = parts[1:]
                try:
                    keep_running, payload = self.dispatch(command, args)
                    self.write_response(True, payload)
                    if not keep_running:
                        break
                except Exception as exc:
                    if self.args.debug:
                        traceback.print_exc(file=sys.stderr)
                    self.write_response(False, str(exc))
        finally:
            self.blender.close()


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GTP wrapper for blended policy/Egaroucid move selection.")
    parser.add_argument("--weights", type=Path, default=default_weights_file())
    parser.add_argument("--blend-param", "--alpha", dest="blend_param", type=float, required=True)
    parser.add_argument("--egaroucid-exe", type=Path, default=default_egaroucid_exe())
    parser.add_argument("--egaroucid-level", type=int, default=21)
    parser.add_argument("--egaroucid-threads", type=int, default=1)
    parser.add_argument("--egaroucid-timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--minimum-available-memory-mib",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--estimated-egaroucid-memory-mib",
        type=float,
        default=0.0,
    )
    parser.add_argument("--cache-egaroucid", action="store_true")
    parser.add_argument("--hint-cache-db", type=Path, default=None)
    parser.add_argument("--policy-server-host", default=None)
    parser.add_argument("--policy-server-port", type=int, default=None)
    parser.add_argument("--policy-server-timeout-sec", type=float, default=30.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument(
        "--measure-move-stone-loss",
        action="store_true",
        help="Measure the selected move against the best Egaroucid move in estimated disc-difference units.",
    )
    parser.add_argument("--no-legal-mask-policy", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-l", "--level", default=None, help="Accepted for compatibility; ignored.")
    parser.add_argument("-t", "--threads", default=None, help="Accepted for compatibility; ignored.")
    parser.add_argument("-quiet", action="store_true", help="Accepted for compatibility; ignored.")
    parser.add_argument("-nobook", action="store_true", help="Accepted for compatibility; ignored.")
    parser.add_argument("-gtp", action="store_true", help="Accepted for compatibility; ignored.")
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    BlendGtpEngine(args).loop()


if __name__ == "__main__":
    main()
