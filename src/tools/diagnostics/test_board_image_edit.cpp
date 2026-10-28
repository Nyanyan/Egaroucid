#include <cassert>
#include <iostream>
#include "../../gui/function/board_image_edit.hpp"

int main() {
    const int supported[] = {0, 1, 2, 3, 4, 5, 11};
    for (int mark : supported) {
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
    edit.marks[0] = BOARD_IMAGE_BRECT;
    edit.marks[63] = BOARD_IMAGE_VALUE;
    edit.values[63] = -6;
    const Board_image_position a{1, 2, 0}, b{4, 8, 1}, opposite_turn{1, 2, 1};
    Board_image_edit_cache cache;
    cache.save(a, edit);
    assert(cache.load(b).marks[0] == BOARD_IMAGE_NOMARK);
    assert(cache.load(opposite_turn).marks[0] == BOARD_IMAGE_NOMARK);
    auto restored = cache.load(a);
    assert(restored.marks[0] == BOARD_IMAGE_BRECT && restored.values[63] == -6);
    restored.clear();
    assert(cache.load(a).marks[0] == BOARD_IMAGE_BRECT);
    cache.save(b, restored);
    cache.retain({a, b});
    assert(cache.positions.size() == 2);
    cache.retain({b});
    cache.retain({a, b});
    assert(cache.load(a).marks[0] == BOARD_IMAGE_NOMARK);
    cache.save(a, edit);
    cache.retain({a, a});
    cache.retain({a});
    assert(cache.load(a).values[63] == -6);
    cache.save(a, restored);
    assert(cache.load(a).marks[63] == BOARD_IMAGE_NOMARK && !cache.load(a).values[63]);
    cache.positions.clear();
    assert(cache.load(a).marks[0] == BOARD_IMAGE_NOMARK);
    cache.save(a, edit);
    cache.retain({});
    assert(cache.positions.empty());
    std::cout << "PASS: supported marks, compatibility, snapshots, clear, and progression lifetime\n";
}
