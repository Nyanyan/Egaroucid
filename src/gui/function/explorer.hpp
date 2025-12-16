/*
    Egaroucid Project

    @file explorer.hpp
        Shared helpers for explorer-style folder navigation and GUI list operations
    @date 2025
    @author GitHub Copilot
    @license GPL-3.0-or-later
*/

#pragma once
#include <Siv3D.hpp>
#include <algorithm>
#include <functional>
#include <vector>
#include "util.hpp"

bool move_folder(const String& source_path, const String& target_parent_path, const String& folder_name);

namespace explorer {

inline void trim_trailing_slash(std::string& path) {
    while (!path.empty() && path.back() == '/') {
        path.pop_back();
    }
}

struct PathState {
    std::string subfolder;

    bool has_parent() const {
        return !subfolder.empty();
    }

    void clear() {
        subfolder.clear();
    }

    bool navigate_to_parent() {
        if (subfolder.empty()) {
            return false;
        }
        trim_trailing_slash(subfolder);
        size_t pos = subfolder.find_last_of('/');
        if (pos == std::string::npos) {
            subfolder.clear();
        } else {
            subfolder = subfolder.substr(0, pos);
        }
        return true;
    }

    void navigate_to_child(const String& folder_name) {
        if (folder_name.isEmpty()) {
            return;
        }
        String trimmed = folder_name.trimmed();
        if (trimmed.isEmpty()) {
            return;
        }
        if (!subfolder.empty()) {
            subfolder += "/";
        }
        subfolder += trimmed.narrow();
    }
};

inline std::string build_root_dir_narrow(const std::string& document_dir, const std::string& relative_root) {
    std::string root = document_dir;
    if (root.empty() || (root.back() != '/' && root.back() != '\\')) {
        root += "/";
    }
    root += relative_root;
    if (!root.empty() && root.back() == '/') {
        root.pop_back();
    }
    return root;
}

inline String build_root_dir(const std::string& document_dir, const std::string& relative_root) {
    return Unicode::Widen(build_root_dir_narrow(document_dir, relative_root));
}

inline String build_current_dir(const String& root_dir, const PathState& state) {
    String dir = root_dir;
    if (!dir.ends_with(U"/")) {
        dir += U"/";
    }
    if (!state.subfolder.empty()) {
        dir += Unicode::Widen(state.subfolder);
        if (!dir.ends_with(U"/")) {
            dir += U"/";
        }
    }
    return dir;
}

inline String build_current_dir(const std::string& root_dir, const PathState& state) {
    return build_current_dir(Unicode::Widen(root_dir), state);
}

inline std::vector<String> enumerate_subfolders(const std::string& root_dir, const PathState& state) {
    return enumerate_subdirectories_generic(root_dir, state.subfolder);
}

inline String compose_path_label(const String& root_label, const PathState& state) {
    return build_path_label(root_label, Unicode::Widen(state.subfolder));
}

inline std::string make_child_subfolder(const PathState& state, const std::string& child) {
    if (child.empty()) {
        return state.subfolder;
    }
    if (state.subfolder.empty()) {
        return child;
    }
    return state.subfolder + "/" + child;
}

}  // namespace explorer

namespace gui_list {

// Drag and drop color constants
namespace DragColors {
    constexpr ColorF DraggedItemBackground = ColorF(1.0, 1.0, 0.0, 0.25);  // Yellow with 25% alpha for dragged item background
    constexpr ColorF DropTargetFrame = ColorF(1.0, 1.0, 0.0);  // Yellow for drop target frame
    constexpr ColorF DraggedItemAlpha = ColorF(1.0, 1.0, 1.0, 0.5);  // Semi-transparent white for dragged item
    constexpr int DropTargetFrameThickness = 3;
}

struct VerticalListGeometry {
    int list_left = 0;
    int list_top = 0;
    int list_width = 0;
    int row_height = 0;
    int visible_row_count = 0;
};

// Auto-scroll when dragging near list edges
inline bool update_drag_auto_scroll(
    const Vec2& cursor_pos,
    const VerticalListGeometry& geom,
    double& scroll_position,
    int total_items,
    double scroll_speed = 0.15
) {
    constexpr double SCROLL_ZONE_HEIGHT = 30.0;
    int list_bottom = geom.list_top + geom.row_height * geom.visible_row_count;
    
    bool scrolled = false;
    
    // Scroll up when cursor is near top
    if (cursor_pos.y >= geom.list_top && cursor_pos.y < geom.list_top + SCROLL_ZONE_HEIGHT) {
        if (scroll_position > 0) {
            scroll_position = std::max(0.0, scroll_position - scroll_speed);
            scrolled = true;
        }
    }
    // Scroll down when cursor is near bottom
    else if (cursor_pos.y > list_bottom - SCROLL_ZONE_HEIGHT && cursor_pos.y <= list_bottom) {
        double max_scroll = std::max(0.0, static_cast<double>(total_items - geom.visible_row_count));
        if (scroll_position < max_scroll) {
            scroll_position = std::min(max_scroll, scroll_position + scroll_speed);
            scrolled = true;
        }
    }
    
    return scrolled;
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

struct InlineEditLayout {
    double primary_x = 0.0;
    double primary_width = 0.0;
    double secondary_x = 0.0;
    double secondary_width = 0.0;
    double text_y = 0.0;
    double field_height = 0.0;
    double back_x = 0.0;
    double ok_x = 0.0;
    double buttons_y = 0.0;
};

struct InlineEditLayoutParams {
    double row_y = 0.0;
    double row_height = 0.0;
    double list_left = 0.0;
    double list_width = 0.0;
    double left_margin = 0.0;
    double control_margin = 10.0;
    double field_height = 30.0;
    double secondary_width = 70.0;
    double back_button_width = 80.0;
    double back_button_height = 30.0;
    double ok_button_width = 70.0;
};

inline InlineEditLayout compute_inline_edit_layout(const InlineEditLayoutParams& params) {
    InlineEditLayout layout;
    layout.field_height = params.field_height;
    double row_center = params.row_y + params.row_height / 2.0;
    layout.text_y = row_center - layout.field_height / 2.0 - 2.0;
    layout.primary_x = params.list_left + params.left_margin;

    layout.ok_x = params.list_left + params.list_width - params.control_margin - params.ok_button_width;
    layout.back_x = layout.ok_x - params.control_margin - params.back_button_width;

    layout.secondary_width = params.secondary_width;
    layout.secondary_x = layout.back_x - params.control_margin - layout.secondary_width;
    layout.primary_width = layout.secondary_x - layout.primary_x - params.control_margin;
    if (layout.primary_width < 120.0) {
        layout.primary_width = 120.0;
    }
    layout.buttons_y = row_center - params.back_button_height / 2.0;
    return layout;
}

}  // namespace gui_list