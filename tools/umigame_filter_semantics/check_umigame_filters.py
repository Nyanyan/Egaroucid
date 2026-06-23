#!/usr/bin/env python3
"""
Check Umigame filter semantics that are easy to regress from the GUI.

This intentionally does not parse real .egbk3 book files. It verifies the
source-level contracts for UI field mapping and runs an independent small-tree
model for Umigame filtering:

- Errors per Move: local mover-perspective loss from the best child.
- Max Allowed Eval: display-only filter for the current candidate move.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BLACK = 0
WHITE = 1


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Result:
    b: int
    w: int


@dataclass(frozen=True)
class Edge:
    score: int
    child: str


@dataclass(frozen=True)
class Node:
    player: int
    edges: tuple[Edge, ...] = ()
    terminal: Result | None = None


def require_contains(path: Path, needle: str) -> None:
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        raise AssertionError(f"{path.relative_to(REPO_ROOT)} does not contain expected text: {needle}")


def require_order(path: Path, needles: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    pos = -1
    for needle in needles:
        next_pos = text.find(needle, pos + 1)
        if next_pos == -1:
            raise AssertionError(f"{path.relative_to(REPO_ROOT)} is missing expected ordered text: {needle}")
        pos = next_pos


def check_source_contracts() -> None:
    menu_definition = REPO_ROOT / "src/gui/function/menu_definition.hpp"
    menu = REPO_ROOT / "src/gui/function/menu.hpp"
    umigame = REPO_ROOT / "src/engine/umigame.hpp"
    book = REPO_ROOT / "src/engine/book.hpp"
    main_scene = REPO_ROOT / "src/gui/main_scene.hpp"

    require_contains(
        menu_definition,
        "init_2bars_player_loss(language.get(\"display\", \"cell\", \"max_child_loss\"), "
        "&menu_elements->umigame_value_black_max_loss, &menu_elements->umigame_value_white_max_loss",
    )
    require_contains(menu, 'Format(U"B", format_player_loss_bar_value(*bar_elem1), U" W", format_player_loss_bar_value(*bar_elem2))')
    require_contains(menu, 'return value >= max_elem ? U"Inf" : Format(value);')
    require_order(
        menu,
        (
            "handle_circle1.draw(Palette::Black);",
            "handle_circle2.draw(Palette::White);",
        ),
    )

    require_contains(umigame, "std::vector<Book_value> moves = book.get_all_moves_within_child_loss(b, condition.max_move_loss);")
    require_contains(umigame, "policies.emplace_back(move.policy);")
    require_contains(umigame, "explicit Umigame_condition(int max_move_loss)")
    if "is_within_umigame_black_score_interval" in umigame.read_text(encoding="utf-8"):
        raise AssertionError("Umigame recursion must not filter by absolute eval interval")
    if "black_max_loss" in umigame.read_text(encoding="utf-8") or "white_max_loss" in umigame.read_text(encoding="utf-8"):
        raise AssertionError("Umigame search condition must not carry display-side eval limits")

    require_contains(main_scene, "bool is_umigame_value_eval_allowed(int score_black) const")
    require_contains(main_scene, "score_black > umigame_value_applied_black_max_loss")
    require_contains(main_scene, "score_black < -umigame_value_applied_white_max_loss")
    require_contains(main_scene, "void add_umigame_job_if_displayed(int cell, Board board, int player, int request_id, const Umigame_condition& condition)")
    require_contains(main_scene, "Umigame_condition condition(umigame_value_applied_max_move_loss);")

    require_order(
        book,
        (
            "max_value = std::max(max_value, value);",
            "if (value_policy.first >= max_value - max_child_loss)",
        ),
    )


def brute_force(
    tree: dict[str, Node],
    node_id: str,
    max_move_loss: int,
) -> Result:
    node = tree[node_id]
    if node.terminal is not None:
        return node.terminal
    if not node.edges:
        return Result(1, 1)

    best = max(edge.score for edge in node.edges)
    kept = [edge for edge in node.edges if edge.score >= best - max_move_loss]
    if not kept:
        return Result(1, 1)

    child_results = [
        brute_force(tree, edge.child, max_move_loss)
        for edge in kept
    ]
    if node.player == BLACK:
        return Result(min(result.b for result in child_results), sum(result.w for result in child_results))
    return Result(sum(result.b for result in child_results), min(result.w for result in child_results))


def is_display_allowed(score_black: int, black_limit: int | None, white_limit: int | None) -> bool:
    if black_limit is not None and score_black > black_limit:
        return False
    if white_limit is not None and score_black < -white_limit:
        return False
    return True


def check_independent_model() -> None:
    tree = {
        "root": Node(BLACK, (Edge(6, "a"), Edge(4, "b"), Edge(1, "c"))),
        "white": Node(WHITE, (Edge(7, "d"), Edge(5, "e"), Edge(0, "f"))),
        "a": Node(WHITE, terminal=Result(3, 7)),
        "b": Node(WHITE, terminal=Result(5, 2)),
        "c": Node(WHITE, terminal=Result(9, 9)),
        "d": Node(BLACK, terminal=Result(11, 13)),
        "e": Node(BLACK, terminal=Result(17, 19)),
        "f": Node(BLACK, terminal=Result(23, 29)),
    }

    # Errors per Move = 0 keeps only the local best child.
    assert brute_force(tree, "root", 0) == Result(3, 7)

    # Errors per Move = 2 keeps scores +6 and +4, but not +1.
    assert brute_force(tree, "root", 2) == Result(3, 9)

    # Max Allowed Eval is display-only. A current move with Black-perspective
    # score +6 is hidden by B4, but this filter is not applied inside recursion.
    assert not is_display_allowed(score_black=6, black_limit=4, white_limit=8)
    assert is_display_allowed(score_black=4, black_limit=4, white_limit=8)
    assert is_display_allowed(score_black=-8, black_limit=4, white_limit=8)
    assert not is_display_allowed(score_black=-9, black_limit=4, white_limit=8)

    # Inf is represented here by None: that side does not filter display.
    assert is_display_allowed(score_black=99, black_limit=None, white_limit=8)
    assert is_display_allowed(score_black=-99, black_limit=4, white_limit=None)


def main() -> None:
    check_source_contracts()
    check_independent_model()
    print("Umigame UI mapping and filter semantics checks passed.")


if __name__ == "__main__":
    main()
