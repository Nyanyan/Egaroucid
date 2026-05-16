/*
    Egaroucid Project

    @file shortcut_setting_common.hpp
        Shared UI helpers for shortcut settings scenes
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "const/gui_common.hpp"
#include "shortcut_key.hpp"

constexpr int SHORTCUT_SETTINGS_LIST_SX = 30;
constexpr int SHORTCUT_SETTINGS_LIST_WIDTH = 740;
constexpr int SHORTCUT_SETTINGS_LIST_SY = 78;
constexpr int SHORTCUT_SETTINGS_LIST_ROW_HEIGHT = 30;

inline Rect draw_shortcut_settings_row_background(const Colors& colors, int row_idx, int sy) {
    Rect rect;
    rect.x = SHORTCUT_SETTINGS_LIST_SX;
    rect.y = sy;
    rect.w = SHORTCUT_SETTINGS_LIST_WIDTH;
    rect.h = SHORTCUT_SETTINGS_LIST_ROW_HEIGHT;
    if (row_idx % 2) {
        rect.draw(colors.dark_green).drawFrame(1.0, colors.white);
    } else {
        rect.draw(colors.green).drawFrame(1.0, colors.white);
    }
    return rect;
}

inline void draw_shortcut_settings_scroll_head(const Fonts& fonts, const Colors& colors, int strt_idx, int sy) {
    if (strt_idx > 0) {
        fonts.font(U"︙").draw(15, Arg::bottomCenter(X_CENTER, sy - 6), colors.white);
    }
}

inline void draw_shortcut_settings_scroll_tail(const Fonts& fonts, const Colors& colors, int next_idx, int total, int sy) {
    if (next_idx < total) {
        fonts.font(U"︙").draw(15, Arg::topCenter(X_CENTER, sy + 6), colors.white);
    }
}

inline String get_shortcut_function_description(const String& function_name) {
    return shortcut_keys.get_shortcut_key_description(function_name);
}

