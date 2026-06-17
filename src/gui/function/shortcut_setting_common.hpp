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
#include "input.hpp"
#include "scroll.hpp"
#include "shortcut_key.hpp"

constexpr int SHORTCUT_SETTINGS_LIST_SX = 30;
constexpr int SHORTCUT_SETTINGS_LIST_WIDTH = 740;
constexpr int SHORTCUT_SETTINGS_LIST_SY = 78;
constexpr int SHORTCUT_SETTINGS_LIST_ROW_HEIGHT = 30;
constexpr int SHORTCUT_SETTINGS_SEARCH_RESULT_LIST_SY = SHORTCUT_SETTINGS_LIST_SY + SHORTCUT_SETTINGS_LIST_ROW_HEIGHT;
constexpr int SHORTCUT_SETTINGS_SEARCH_LABEL_RIGHT_X = 115;
constexpr int SHORTCUT_SETTINGS_SEARCH_BOX_SX = 125;
constexpr int SHORTCUT_SETTINGS_SEARCH_BOX_SY = 43;
constexpr int SHORTCUT_SETTINGS_SEARCH_BOX_WIDTH = 520;
constexpr int SHORTCUT_SETTINGS_SEARCH_BOX_HEIGHT = 30;
constexpr size_t SHORTCUT_SETTINGS_SEARCH_MAX_CHARS = 80;

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

inline String get_shortcut_settings_fallback_label(const std::string& key, const String& fallback) {
    String label = language.get("common", key);
    if (label == U"?") {
        return fallback;
    }
    return label;
}

inline bool sanitize_shortcut_settings_search_area(TextAreaEditState& search_area) {
    String sanitized = search_area.text.replaced(U"\r", U"").replaced(U"\n", U"");
    if (sanitized == search_area.text) {
        return false;
    }
    size_t old_cursor = search_area.cursorPos;
    search_area.text = sanitized;
    search_area.cursorPos = std::min<size_t>(old_cursor, sanitized.size());
    search_area.scrollY = 0.0;
    search_area.rebuildGlyphs();
    return true;
}

inline bool draw_shortcut_settings_search_box(
    const Fonts& fonts,
    const Colors& colors,
    TextAreaEditState& search_area,
    bool enabled = true
) {
    fonts.font(get_shortcut_settings_fallback_label("search", U"Search")).draw(
        15,
        Arg::rightCenter(SHORTCUT_SETTINGS_SEARCH_LABEL_RIGHT_X, SHORTCUT_SETTINGS_SEARCH_BOX_SY + SHORTCUT_SETTINGS_SEARCH_BOX_HEIGHT / 2),
        colors.white
    );
    bool changed = text_area_with_ime_candidate_window(
        search_area,
        Vec2{ SHORTCUT_SETTINGS_SEARCH_BOX_SX, SHORTCUT_SETTINGS_SEARCH_BOX_SY },
        SizeF{ SHORTCUT_SETTINGS_SEARCH_BOX_WIDTH, SHORTCUT_SETTINGS_SEARCH_BOX_HEIGHT },
        SHORTCUT_SETTINGS_SEARCH_MAX_CHARS,
        enabled
    );
    changed |= sanitize_shortcut_settings_search_area(search_area);
    return changed;
}

inline String get_shortcut_settings_normalized_search_text(const TextAreaEditState& search_area) {
    return search_area.text.trimmed().lowercased();
}

inline bool shortcut_function_matches_search(int function_idx, const String& normalized_search_text) {
    if (normalized_search_text.isEmpty()) {
        return true;
    }
    const String function_name = shortcut_keys.shortcut_keys[function_idx].name.lowercased();
    const String function_description = get_shortcut_function_description(shortcut_keys.shortcut_keys[function_idx].name).lowercased();
    return function_name.includes(normalized_search_text) || function_description.includes(normalized_search_text);
}

inline std::vector<int> get_filtered_shortcut_function_indices(const TextAreaEditState& search_area) {
    std::vector<int> indices;
    const String normalized_search_text = get_shortcut_settings_normalized_search_text(search_area);
    for (int i = 0; i < static_cast<int>(shortcut_keys.shortcut_keys.size()); ++i) {
        if (shortcut_function_matches_search(i, normalized_search_text)) {
            indices.emplace_back(i);
        }
    }
    return indices;
}

inline int find_shortcut_function_filter_position(const std::vector<int>& filtered_function_indices, int function_idx) {
    for (int i = 0; i < static_cast<int>(filtered_function_indices.size()); ++i) {
        if (filtered_function_indices[i] == function_idx) {
            return i;
        }
    }
    return -1;
}

inline void sync_shortcut_settings_scroll_manager(
    Scroll_manager& scroll_manager,
    int filtered_count,
    int n_functions_on_window,
    bool reset_to_top
) {
    scroll_manager.set_n_elem(filtered_count);
    if (reset_to_top) {
        scroll_manager.set_strt_idx(0.0);
        return;
    }
    const int max_strt_idx = std::max(filtered_count - n_functions_on_window, 0);
    scroll_manager.set_strt_idx(std::min<double>(scroll_manager.get_strt_idx_double(), max_strt_idx));
}

inline void draw_shortcut_settings_no_match_message(const Fonts& fonts, const Colors& colors, int sy) {
    fonts.font(get_shortcut_settings_fallback_label("no_match", U"No matches")).draw(
        15,
        Arg::topCenter(X_CENTER, sy + 40),
        colors.white
    );
}
