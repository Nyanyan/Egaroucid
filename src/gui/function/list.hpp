/*
    Egaroucid Project

    @file list.hpp
        Shared helpers for GUI list operations
    @date 2025
    @author
    @license GPL-3.0-or-later
*/

#pragma once
#include <Siv3D.hpp>
#include <algorithm>
#include <vector>
#include <functional>

bool move_folder(const String& source_path, const String& target_parent_path, const String& folder_name);

namespace gui_list {

struct VerticalListGeometry {
    int list_left = 0;
    int list_top = 0;
    int list_width = 0;
    int row_height = 0;
    int visible_row_count = 0;
};

inline Rect make_row_rect(const VerticalListGeometry& geom, int row_offset) {
    return Rect(geom.list_left,
                geom.list_top + row_offset * geom.row_height,
                geom.list_width,
                geom.row_height);
}

inline int compute_drop_index_for_items(
    const Vec2& cursor_pos,
    const VerticalListGeometry& geom,
    int first_visible_row,
    int first_item_row,
    int item_count
) {
    if (item_count <= 0 || geom.row_height <= 0 || geom.visible_row_count <= 0) {
        return -1;
    }

    int row_index = first_item_row;
    for (int item = 0; item < item_count; ++item, ++row_index) {
        if (row_index < first_visible_row || row_index >= first_visible_row + geom.visible_row_count) {
            continue;
        }
        int visible_row = row_index - first_visible_row;
        Rect rect = make_row_rect(geom, visible_row);
        if (rect.contains(cursor_pos)) {
            double mid_y = rect.y + rect.h / 2.0;
            if (cursor_pos.y < mid_y) {
                return item;
            }
            return std::min(item + 1, item_count);
        }
    }

    Rect list_bounds(geom.list_left,
                     geom.list_top,
                     geom.list_width,
                     geom.row_height * geom.visible_row_count);
    if (list_bounds.contains(cursor_pos)) {
        return item_count;
    }
    return -1;
}

namespace detail {

template <class VectorT>
inline void reorder_single_vector(VectorT& vec, int from_idx, int target_idx) {
    if (vec.empty() || from_idx < 0 || from_idx >= static_cast<int>(vec.size())) {
        return;
    }
    auto entry = vec[from_idx];
    vec.erase(vec.begin() + from_idx);
    int clamped_target = std::clamp(target_idx, 0, static_cast<int>(vec.size()));
    vec.insert(vec.begin() + clamped_target, entry);
}

inline void reorder_aux_vectors(int from_idx, int target_idx) {
    (void)from_idx;
    (void)target_idx;
}

template <class VectorT, class... Rest>
inline void reorder_aux_vectors(int from_idx, int target_idx, VectorT& vec, Rest&... rest) {
    reorder_single_vector(vec, from_idx, target_idx);
    reorder_aux_vectors(from_idx, target_idx, rest...);
}

}  // namespace detail

template <class T, class... AuxVectors>
inline bool reorder_parallel(std::vector<T>& primary, int from_idx, int insert_idx, AuxVectors&... aux_vectors) {
    int size = static_cast<int>(primary.size());
    if (size == 0 || from_idx < 0 || from_idx >= size) {
        return false;
    }
    int clamped_target = std::clamp(insert_idx, 0, size);
    if (clamped_target == from_idx || clamped_target == from_idx + 1) {
        return false;
    }

    auto entry = primary[from_idx];
    primary.erase(primary.begin() + from_idx);

    int target_after_removal = clamped_target;
    if (target_after_removal > from_idx) {
        target_after_removal -= 1;
    }
    target_after_removal = std::clamp(target_after_removal, 0, static_cast<int>(primary.size()));
    primary.insert(primary.begin() + target_after_removal, entry);

    detail::reorder_aux_vectors(from_idx, target_after_removal, aux_vectors...);
    return true;
}

inline bool is_forbidden_folder_char(char32 ch) {
    static constexpr char32 forbidden_chars[] = U"\\/:*?\"<>|";
    if (ch < 0x20) {
        return true;
    }
    for (char32 f : forbidden_chars) {
        if (f == U'\0') {
            break;
        }
        if (ch == f) {
            return true;
        }
    }
    return false;
}

inline String sanitize_folder_text(const String& text) {
    String filtered;
    filtered.reserve(text.size());
    for (char32 ch : text) {
        if (!is_forbidden_folder_char(ch)) {
            filtered.push_back(ch);
        }
    }
    return filtered;
}

inline bool is_valid_folder_name(const String& name) {
    String trimmed = name.trimmed();
    if (trimmed.isEmpty()) {
        return false;
    }
    if (trimmed == U"." || trimmed == U"..") {
        return false;
    }
    for (char32 ch : trimmed) {
        if (is_forbidden_folder_char(ch)) {
            return false;
        }
    }
    return true;
}

inline void sanitize_text_area(TextAreaEditState& area) {
    String sanitized = sanitize_folder_text(area.text);
    if (sanitized != area.text) {
        size_t old_cursor = area.cursorPos;
        area.text = sanitized;
        area.cursorPos = std::min<size_t>(old_cursor, sanitized.size());
        area.rebuildGlyphs();
    }
}

inline String normalize_directory_base(const String& base_dir) {
    if (base_dir.isEmpty()) {
        return base_dir;
    }
    if (base_dir.ends_with(U"/") || base_dir.ends_with(U"\\")) {
        return base_dir;
    }
    return base_dir + U"/";
}

inline bool create_folder_with_initializer(
    const String& base_dir,
    const String& folder_name,
    const std::function<void(const String&)>& after_create = nullptr
) {
    String trimmed = folder_name.trimmed();
    if (!is_valid_folder_name(trimmed)) {
        return false;
    }
    String normalized_base = normalize_directory_base(base_dir);
    String target_dir = normalized_base + trimmed + U"/";
    if (FileSystem::Exists(target_dir)) {
        return false;
    }
    if (!FileSystem::CreateDirectories(target_dir)) {
        return false;
    }
    if (after_create) {
        after_create(target_dir);
    }
    return true;
}

inline bool rename_folder_in_directory(
    const String& base_dir,
    const String& current_name,
    const String& new_name
) {
    String trimmed_new = new_name.trimmed();
    if (!is_valid_folder_name(trimmed_new)) {
        return false;
    }
    String trimmed_current = current_name.trimmed();
    if (trimmed_current == trimmed_new) {
        return true;
    }
    String normalized_base = normalize_directory_base(base_dir);
    String source_path = normalized_base + trimmed_current;
    if (!FileSystem::Exists(source_path)) {
        return false;
    }
    return move_folder(source_path, normalized_base, trimmed_new);
}

}  // namespace gui_list
