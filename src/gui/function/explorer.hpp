/*
    Egaroucid Project

    @file explorer.hpp
        Shared helpers for explorer-style folder navigation
    @date 2025
    @author
    @license GPL-3.0-or-later
*/

#pragma once
#include <Siv3D.hpp>
#include "util.hpp"

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
