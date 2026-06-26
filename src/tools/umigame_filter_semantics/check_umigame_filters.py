#!/usr/bin/env python3
"""
Check the lightweight Umigame score-range and integration-error contracts.

This intentionally does not parse real .egbk3 book files. It verifies the
source-level wiring for the GUI settings and runs an independent small-tree
model for recursive black-perspective score-range filtering with cumulative
local-loss pruning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BLACK = 0
WHITE = 1
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Result:
    b: int
    w: int


@dataclass(frozen=True)
class Edge:
    black_score: int
    child: str


@dataclass(frozen=True)
class Node:
    player: int
    edges: tuple[Edge, ...] = ()
    terminal: Result | None = None


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require_contains(path: Path, needle: str) -> None:
    text = read(path)
    if needle not in text:
        raise AssertionError(f"{path.relative_to(REPO_ROOT)} does not contain expected text: {needle}")


def require_absent(path: Path, needle: str) -> None:
    text = read(path)
    if needle in text:
        raise AssertionError(f"{path.relative_to(REPO_ROOT)} still contains removed text: {needle}")


def require_order(path: Path, needles: tuple[str, ...]) -> None:
    text = read(path)
    pos = -1
    for needle in needles:
        next_pos = text.find(needle, pos + 1)
        if next_pos == -1:
            raise AssertionError(f"{path.relative_to(REPO_ROOT)} is missing expected ordered text: {needle}")
        pos = next_pos


def check_source_contracts() -> None:
    gui_common = REPO_ROOT / "src/gui/function/const/gui_common.hpp"
    menu_definition = REPO_ROOT / "src/gui/function/menu_definition.hpp"
    menu = REPO_ROOT / "src/gui/function/menu.hpp"
    umigame = REPO_ROOT / "src/engine/umigame.hpp"
    main_scene = REPO_ROOT / "src/gui/main_scene.hpp"
    regression_runner = REPO_ROOT / "src/tools/umigame_filter_semantics/run_umigame_regression.py"
    regression_verifier = REPO_ROOT / "src/tools/umigame_filter_semantics/umigame_regression.cpp"

    require_contains(gui_common, "int umigame_value_score_min;")
    require_contains(gui_common, "int umigame_value_score_max;")
    require_contains(gui_common, "int umigame_value_integration_error;")
    require_contains(menu_definition, "init_2bars(language.get(\"operation\", \"generate_random_board\", \"score_range\")")
    require_contains(menu_definition, "&menu_elements->umigame_value_score_min")
    require_contains(menu_definition, "&menu_elements->umigame_value_score_max")
    require_contains(menu_definition, "language.get(\"display\", \"cell\", \"integration_error\")")
    require_absent(menu_definition, "umigame_value_apply_setting")

    require_contains(menu, "String range_str = format_bar_value(*bar_elem1) + U\"~\" + format_bar_value(*bar_elem2);")
    require_absent(menu, "init_2bars_player_loss")

    require_contains(umigame, "Umigame_condition(int score_min, int score_max)")
    require_contains(umigame, "Umigame_condition(int score_min, int score_max, int integration_error)")
    require_contains(umigame, "struct Umigame_cache_key")
    require_contains(umigame, "int remaining_error;")
    require_contains(umigame, "std::vector<Book_value> moves = get_registered_moves_with_value(b);")
    require_contains(umigame, "inline bool get_registered_move_value(Board *b, int policy, int *value)")
    require_contains(umigame, "int local_loss = best_value - move.value;")
    require_contains(umigame, "condition.accepts_black_score(umigame_book_value_to_black_score(move.value, player))")
    require_contains(umigame, "condition.uses_original_best_moves()")
    require_absent(umigame, "max_move_loss")
    require_absent(umigame, "get_all_moves_within_child_loss")

    require_contains(main_scene, "UMIGAME_VALUE_SETTING_DEBOUNCE_MSEC")
    require_contains(main_scene, "void update_umigame_value_settings()")
    require_contains(main_scene, "void add_umigame_job_if_displayed(int cell, Board board, int player, int request_id, const Umigame_condition& condition)")
    require_order(
        main_scene,
        (
            "if (!book.contain(board))",
            "condition.accepts_black_score(umigame_book_value_to_black_score(elem.value, player))",
        ),
    )
    require_absent(main_scene, "umigame_value_apply_setting")

    require_contains(regression_runner, 'DEFAULT_BOOK = REPO_ROOT / "bin" / "document" / "book.egbk3"')
    require_contains(regression_verifier, 'constexpr const char* DEFAULT_BOOK_FILE = "document/book.egbk3";')


def mover_value(edge: Edge, player: int) -> int:
    return edge.black_score if player == BLACK else -edge.black_score


def brute_force(tree: dict[str, Node], node_id: str, score_min: int, score_max: int, remaining_error: int) -> Result:
    node = tree[node_id]
    if node.terminal is not None:
        return node.terminal
    if not node.edges:
        return Result(1, 1)

    best = max(mover_value(edge, node.player) for edge in node.edges)
    kept = [
        (edge, remaining_error - (best - mover_value(edge, node.player)))
        for edge in node.edges
        if score_min <= edge.black_score <= score_max and best - mover_value(edge, node.player) <= remaining_error
    ]
    if not kept:
        return Result(1, 1)

    child_results = [
        brute_force(tree, edge.child, score_min, score_max, child_remaining_error)
        for edge, child_remaining_error in kept
    ]
    if node.player == BLACK:
        return Result(min(result.b for result in child_results), sum(result.w for result in child_results))
    return Result(sum(result.b for result in child_results), min(result.w for result in child_results))


def is_displayed(candidate_in_book: bool, score_black: int, score_min: int, score_max: int) -> bool:
    return candidate_in_book and score_min <= score_black <= score_max


def check_independent_model() -> None:
    tree = {
        "root": Node(BLACK, (Edge(8, "white"), Edge(4, "top_loss"))),
        "white": Node(WHITE, (Edge(-3, "d"), Edge(3, "e"), Edge(9, "f"))),
        "top_loss": Node(WHITE, terminal=Result(2, 30)),
        "d": Node(BLACK, terminal=Result(5, 7)),
        "e": Node(BLACK, terminal=Result(11, 13)),
        "f": Node(BLACK, terminal=Result(17, 19)),
    }

    # Full range with Integration Error = 0 keeps only local best moves, so it
    # preserves original Umigame branch selection.
    assert brute_force(tree, "root", -64, 64, 0) == Result(5, 7)

    # Range filtering is recursive. Here the best white reply has black score
    # -3, so range 0..10 prunes it. With Integration Error = 0 the non-best
    # in-range white replies are still pruned by local loss.
    assert brute_force(tree, "root", 0, 10, 0) == Result(1, 1)

    # Widening the integration budget admits the in-range white reply with
    # local loss 6.
    assert brute_force(tree, "root", 0, 10, 6) == Result(2, 43)

    assert is_displayed(candidate_in_book=True, score_black=3, score_min=0, score_max=6)
    assert not is_displayed(candidate_in_book=True, score_black=8, score_min=0, score_max=6)
    assert not is_displayed(candidate_in_book=False, score_black=3, score_min=0, score_max=6)


def main() -> None:
    check_source_contracts()
    check_independent_model()
    print("Umigame score-range semantics checks passed.")


if __name__ == "__main__":
    main()
