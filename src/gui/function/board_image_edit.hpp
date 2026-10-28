/*
    Egaroucid Project
    @file board_image_edit.hpp
        Session-only board image annotations
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

// Keep the original six IDs compatible with existing setting.json files.
constexpr int BOARD_IMAGE_BRECT = 0;
constexpr int BOARD_IMAGE_BSTAR = 1;
constexpr int BOARD_IMAGE_WRECT = 2;
constexpr int BOARD_IMAGE_WSTAR = 3;
constexpr int BOARD_IMAGE_VALUE = 4;
constexpr int BOARD_IMAGE_NOMARK = 5;
constexpr int BOARD_IMAGE_ORECT = 6;
constexpr int BOARD_IMAGE_OSTAR = 7;
constexpr int BOARD_IMAGE_BCIRCLE = 8;
constexpr int BOARD_IMAGE_WCIRCLE = 9;
constexpr int BOARD_IMAGE_OCIRCLE = 10;
constexpr int BOARD_IMAGE_FLIP = 11;
constexpr int BOARD_IMAGE_MARK_MAX = BOARD_IMAGE_FLIP;

constexpr int BOARD_IMAGE_TOOL_RECT = 0;
constexpr int BOARD_IMAGE_TOOL_STAR = 1;
constexpr int BOARD_IMAGE_TOOL_CIRCLE = 2;
constexpr int BOARD_IMAGE_TOOL_FLIP = 3;
constexpr int BOARD_IMAGE_TOOL_VALUE = 4;
constexpr int BOARD_IMAGE_TOOL_ERASE = 6;
constexpr int BOARD_IMAGE_STYLE_BLACK = 0;
constexpr int BOARD_IMAGE_STYLE_WHITE = 1;
constexpr int BOARD_IMAGE_STYLE_OUTLINE = 2;

inline int board_image_mark_from_tool(int tool, int style) {
    constexpr int marks[3][3] = {
        {BOARD_IMAGE_BRECT, BOARD_IMAGE_WRECT, BOARD_IMAGE_ORECT},
        {BOARD_IMAGE_BSTAR, BOARD_IMAGE_WSTAR, BOARD_IMAGE_OSTAR},
        {BOARD_IMAGE_BCIRCLE, BOARD_IMAGE_WCIRCLE, BOARD_IMAGE_OCIRCLE}
    };
    if (BOARD_IMAGE_TOOL_RECT <= tool && tool <= BOARD_IMAGE_TOOL_CIRCLE) {
        return marks[tool][std::clamp(style, BOARD_IMAGE_STYLE_BLACK, BOARD_IMAGE_STYLE_OUTLINE)];
    }
    switch (tool) {
    case BOARD_IMAGE_TOOL_FLIP: return BOARD_IMAGE_FLIP;
    case BOARD_IMAGE_TOOL_VALUE: return BOARD_IMAGE_VALUE;
    default: return BOARD_IMAGE_NOMARK;
    }
}

inline std::pair<int, int> board_image_tool_from_mark(int mark) {
    constexpr int tools[] = {BOARD_IMAGE_TOOL_RECT, BOARD_IMAGE_TOOL_STAR, BOARD_IMAGE_TOOL_CIRCLE, BOARD_IMAGE_TOOL_FLIP, BOARD_IMAGE_TOOL_VALUE, BOARD_IMAGE_TOOL_ERASE};
    for (int tool : tools) {
        for (int style = BOARD_IMAGE_STYLE_BLACK; style <= BOARD_IMAGE_STYLE_OUTLINE; ++style) {
            if (board_image_mark_from_tool(tool, style) == mark) {
                return {tool, style};
            }
        }
    }
    return {BOARD_IMAGE_TOOL_RECT, BOARD_IMAGE_STYLE_BLACK};
}


struct Board_image_edit {
    std::array<int, 64> marks;
    std::array<std::optional<int>, 64> values;

    Board_image_edit() { clear(); }

    void clear() {
        marks.fill(BOARD_IMAGE_NOMARK);
        values.fill(std::nullopt);
    }
};


using Board_image_position = std::tuple<uint64_t, uint64_t, int>;

struct Board_image_edit_cache {
    std::map<Board_image_position, Board_image_edit> positions;

    Board_image_edit load(const Board_image_position& position) const {
        const auto it = positions.find(position);
        return it == positions.end() ? Board_image_edit{} : it->second;
    }

    void save(const Board_image_position& position, const Board_image_edit& edit) {
        positions[position] = edit;
    }

    void retain(const std::vector<Board_image_position>& live_positions) {
        for (auto it = positions.begin(); it != positions.end();) {
            if (std::find(live_positions.begin(), live_positions.end(), it->first) == live_positions.end()) {
                it = positions.erase(it);
            } else {
                ++it;
            }
        }
    }
};
