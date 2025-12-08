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

}  // namespace gui_list
