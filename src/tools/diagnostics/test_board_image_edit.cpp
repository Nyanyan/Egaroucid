#include <cassert>
#include <iostream>
#include "../../gui/function/board_image_edit.hpp"

int main() {
    for (int mark = 0; mark <= BOARD_IMAGE_MARK_MAX; ++mark) {
        const auto [tool, style] = board_image_tool_from_mark(mark);
        assert(board_image_mark_from_tool(tool, style) == mark);
    }
    assert(board_image_mark_from_tool(0, 0) == 0);
    assert(board_image_mark_from_tool(1, 0) == 1);
    assert(board_image_mark_from_tool(0, 1) == 2);
    assert(board_image_mark_from_tool(1, 1) == 3);
    assert(board_image_mark_from_tool(BOARD_IMAGE_TOOL_VALUE, 0) == 4);
    assert(board_image_mark_from_tool(BOARD_IMAGE_TOOL_ERASE, 0) == 5);

    Board_image_edit edit;
    assert(edit.marks[0] == BOARD_IMAGE_NOMARK && !edit.values[0]);
    edit.marks[0] = BOARD_IMAGE_FLIP;
    edit.marks[63] = BOARD_IMAGE_VALUE;
    edit.values[63] = -6;
    Board_image_arrow_input input;
    assert(!input.click(-1, &edit) && input.start == -1);
    assert(!input.click(0, &edit) && input.start == 0);
    assert(!input.click(64, &edit) && input.start == 0);
    assert(input.click(63, &edit) && input.start == -1);
    assert(!input.click(0, &edit));
    assert(!input.click(63, &edit) && edit.arrows.size() == 1); // duplicate
    assert(!input.click(7, &edit));
    assert(!input.click(7, &edit) && input.start == -1); // zero length
    assert(!input.click(63, &edit));
    assert(input.click(0, &edit) && edit.arrows.size() == 2); // reverse is distinct
    assert(!input.click(8, &edit));
    input.cancel();
    assert(input.start == -1);

    const Board_image_position a{1, 2, 0}, b{4, 8, 1}, opposite_turn{1, 2, 1};
    Board_image_edit_cache cache;
    cache.save(a, edit);
    assert(cache.load(b).marks[0] == BOARD_IMAGE_NOMARK);
    assert(cache.load(opposite_turn).arrows.empty());
    auto restored = cache.load(a);
    assert(restored.marks[0] == BOARD_IMAGE_FLIP);
    assert(restored.values[63] == -6 && restored.arrows == edit.arrows);
    restored.clear();
    assert(cache.load(a).arrows.size() == 2); // independent snapshot
    cache.save(b, restored);
    cache.retain({a, b});
    assert(cache.positions.size() == 2); // navigation preserves both
    cache.retain({b});
    cache.retain({a, b}); // deleting and recreating A must not restore old marks
    assert(cache.load(a).marks[0] == BOARD_IMAGE_NOMARK);
    cache.save(a, edit);
    cache.retain({a, a}); // board exists in both branches, then only one
    cache.retain({a});
    assert(cache.load(a).arrows.size() == 2);
    cache.save(a, restored);
    assert(cache.load(a).arrows.empty() && !cache.load(a).values[63]);
    assert(cache.load(a).marks[63] == BOARD_IMAGE_NOMARK);
    cache.positions.clear(); // new game / replaced progression
    assert(cache.load(a).arrows.empty());
    cache.save(a, edit);
    cache.retain({});
    assert(cache.positions.empty());
    std::cout << "PASS: mark compatibility, arrow endpoints, clear, snapshots, and progression lifetime\n";
}
