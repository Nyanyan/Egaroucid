/*
    Egaroucid Project

    @file util.hpp
        GUI Utility
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <string>
#include <vector>
#include "const/gui_common.hpp"
#include "button.hpp"

std::string get_extension(std::string file) {
    std::string res;
    bool dot_found = false;
    for (int i = (int)file.size() - 1; i >= 0; --i) {
        if (file[i] == '.') {
            dot_found = true;
            break;
        }
        res.insert(0, {file[i]});
    }
    if (dot_found) {
        return res;
    }
    return "";
}

std::vector<int> get_put_order(Graph_resources graph_resources, History_elem current_history_elem) {
    std::vector<int> put_order;
    int inspect_switch_n_discs = INF;
    if (graph_resources.branch == GRAPH_MODE_INSPECT) {
        if (graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
            inspect_switch_n_discs = graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs();
        } else {
            std::cerr << "get_put_order: no node found in inspect mode" << std::endl;
        }
    }
    // std::cerr << inspect_switch_n_discs << std::endl;
    for (History_elem& history_elem : graph_resources.nodes[GRAPH_MODE_NORMAL]) {
        if (history_elem.board.n_discs() + 1 >= inspect_switch_n_discs || history_elem.board.n_discs() >= current_history_elem.board.n_discs()) {
            break;
        }
        if (history_elem.next_policy != -1) {
            put_order.emplace_back(history_elem.next_policy);
        }
    }
    if (inspect_switch_n_discs != INF) {
        if (graph_resources.nodes[GRAPH_MODE_INSPECT][0].policy != -1) {
            put_order.emplace_back(graph_resources.nodes[GRAPH_MODE_INSPECT][0].policy);
        }
        for (History_elem& history_elem : graph_resources.nodes[GRAPH_MODE_INSPECT]) {
            if (history_elem.board.n_discs() >= current_history_elem.board.n_discs()) {
                break;
            }
            if (history_elem.next_policy != -1) {
                put_order.emplace_back(history_elem.next_policy);
            }
        }
    }
    return put_order;
}

std::vector<int> get_put_player(Board initial_board, int initial_player, std::vector<int> put_order) {
    Flip flip;
    Board board = initial_board.copy();
    int player = initial_player;
    std::vector<int> put_player;
    for (int cell: put_order) {
        put_player.emplace_back(player);
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
        player ^= 1;
        if (board.get_legal() == 0) {
            board.pass();
            player ^= 1;
        }
    }
    return put_player;
}

std::string get_transcript(Graph_resources graph_resources, History_elem current_history_elem) {
    std::vector<int> put_order = get_put_order(graph_resources, current_history_elem);
    std::string transcript;
    for (int cell: put_order) {
        transcript += idx_to_coord(cell);
    }
    return transcript;
}

std::vector<std::pair<Board, int>> get_board_player_list(Graph_resources graph_resources, History_elem current_history_elem) {
    std::vector<std::pair<Board, int>> res;
    int inspect_switch_n_discs = INF;
    if (graph_resources.branch == GRAPH_MODE_INSPECT) {
        if (graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
            inspect_switch_n_discs = graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs();
        } else {
            std::cerr << "no node found in inspect mode" << std::endl;
        }
    }
    std::cerr << inspect_switch_n_discs << std::endl;
    Flip flip;
    for (History_elem& history_elem : graph_resources.nodes[GRAPH_MODE_NORMAL]) {
        if (history_elem.board.n_discs() + 1 >= inspect_switch_n_discs || history_elem.board.n_discs() >= current_history_elem.board.n_discs()) {
            break;
        }
        if (history_elem.next_policy != -1) {
            Board board = history_elem.board;
            int player = history_elem.player;
            calc_flip(&flip, &board, history_elem.next_policy);
            board.move_board(&flip);
            player ^= 1;
            if (board.get_legal() == 0) {
                board.pass();
                player ^= 1;
            }
            res.emplace_back(std::make_pair(board, player));
        }
    }
    if (inspect_switch_n_discs != INF) {
        if (graph_resources.nodes[GRAPH_MODE_INSPECT][0].policy != -1) {
            res.emplace_back(std::make_pair(graph_resources.nodes[GRAPH_MODE_INSPECT][0].board, graph_resources.nodes[GRAPH_MODE_INSPECT][0].player));
        }
        for (History_elem& history_elem : graph_resources.nodes[GRAPH_MODE_INSPECT]) {
            if (history_elem.board.n_discs() >= current_history_elem.board.n_discs()) {
                break;
            }
            if (history_elem.next_policy != -1) {
                Board board = history_elem.board;
                int player = history_elem.player;
                calc_flip(&flip, &board, history_elem.next_policy);
                board.move_board(&flip);
                player ^= 1;
                if (board.get_legal() == 0) {
                    board.pass();
                    player ^= 1;
                }
                res.emplace_back(std::make_pair(board, player));
            }
        }
    }
    return res;
}

// Utility function to enumerate directories in a given path
inline std::vector<String> enumerate_direct_subdirectories(const std::string& document_dir, const std::string& subfolder) {
    std::vector<String> result;
    
    String base = Unicode::Widen(document_dir) + U"games/" + Unicode::Widen(subfolder);
    if (base.size() && base.back() != U'/') base += U"/";
    
    // Convert to absolute path for comparison
    String abs_base = FileSystem::FullPath(base);
    if (abs_base.size() && abs_base.back() != U'/' && abs_base.back() != U'\\') abs_base += U"/";
    
    Array<FilePath> list = FileSystem::DirectoryContents(base);
    Array<String> real_folders;
    for (const auto& path : list) {
        if (FileSystem::IsDirectory(path) && FileSystem::Exists(path)) {
            String abs_path = FileSystem::FullPath(path);
            if (abs_path.size() && abs_path.back() != U'/' && abs_path.back() != U'\\') abs_path += U"/";
            
            String name = path;
            while (name.size() && (name.back() == U'/' || name.back() == U'\\')) name.pop_back();
            size_t pos = name.lastIndexOf(U'/');
            if (pos == String::npos) pos = name.lastIndexOf(U'\\');
            if (pos != String::npos) name = name.substr(pos + 1);
            
            // Check if this is a direct child directory
            String expected_abs_path = abs_base + name + U"/";
            
            if (name.size() && name != U"." && name != U".." && abs_path == expected_abs_path) {
                real_folders.emplace_back(name);
            }
        }
    }
    std::sort(real_folders.begin(), real_folders.end());
    for (auto& n : real_folders) result.emplace_back(n);
    
    return result;
}

// Generic directory enumeration function (not limited to games/)
inline std::vector<String> enumerate_subdirectories_generic(const std::string& base_dir, const std::string& subfolder) {
    std::vector<String> result;
    
    String base = Unicode::Widen(base_dir);
    if (base.size() && base.back() != U'/') base += U"/";
    if (!subfolder.empty()) {
        base += Unicode::Widen(subfolder);
        if (base.size() && base.back() != U'/') base += U"/";
    }
    
    // Convert to absolute path for comparison
    String abs_base = FileSystem::FullPath(base);
    if (abs_base.size() && abs_base.back() != U'/' && abs_base.back() != U'\\') abs_base += U"/";
    
    Array<FilePath> list = FileSystem::DirectoryContents(base);
    Array<String> real_folders;
    for (const auto& path : list) {
        if (FileSystem::IsDirectory(path) && FileSystem::Exists(path)) {
            String abs_path = FileSystem::FullPath(path);
            if (abs_path.size() && abs_path.back() != U'/' && abs_path.back() != U'\\') abs_path += U"/";
            
            String name = path;
            while (name.size() && (name.back() == U'/' || name.back() == U'\\')) name.pop_back();
            size_t pos = name.lastIndexOf(U'/');
            if (pos == String::npos) pos = name.lastIndexOf(U'\\');
            if (pos != String::npos) name = name.substr(pos + 1);
            
            // Check if this is a direct child directory
            String expected_abs_path = abs_base + name + U"/";
            
            if (name.size() && name != U"." && name != U".." && abs_path == expected_abs_path) {
                real_folders.emplace_back(name);
            }
        }
    }
    std::sort(real_folders.begin(), real_folders.end());
    for (auto& n : real_folders) result.emplace_back(n);
    
    return result;
}

// Create a new folder in the specified directory
bool create_folder_in_directory(const String& base_dir, const String& folder_name) {
    if (folder_name.empty() || folder_name.contains(U"/") || folder_name.contains(U"\\")) {
        std::cerr << "Invalid folder name" << std::endl;
        return false;
    }
    
    String folder_path = base_dir;
    if (!folder_path.ends_with(U"/") && !folder_path.ends_with(U"\\")) {
        folder_path += U"/";
    }
    folder_path += folder_name;
    
    if (FileSystem::Exists(folder_path)) {
        std::cerr << "Folder already exists: " << folder_path.narrow() << std::endl;
        return false;
    }
    
    bool created = FileSystem::CreateDirectories(folder_path);
    if (created) {
        std::cerr << "Created folder: " << folder_path.narrow() << std::endl;
    } else {
        std::cerr << "Failed to create folder: " << folder_path.narrow() << std::endl;
    }
    return created;
}

// Move a folder from source to target directory
bool move_folder(const String& source_path, const String& target_parent_path, const String& folder_name) {
    String full_source = source_path;
    String full_target = target_parent_path;
    
    if (!full_target.ends_with(U"/") && !full_target.ends_with(U"\\")) {
        full_target += U"/";
    }
    full_target += folder_name;
    
    // Check if source and target are the same
    if (FileSystem::FullPath(full_source) == FileSystem::FullPath(full_target)) {
        std::cerr << "Source and target are the same" << std::endl;
        return false;
    }
    
    // Check for circular reference
    String source_abs = FileSystem::FullPath(full_source);
    String target_abs = FileSystem::FullPath(full_target);
    auto ensure_trailing_separator = [](String path) {
        if (!path.ends_with(U"/") && !path.ends_with(U"\\")) {
            path += U"/";
        }
        return path;
    };
    String source_abs_with_sep = ensure_trailing_separator(source_abs);
    if (target_abs.starts_with(source_abs_with_sep)) {
        std::cerr << "Cannot move folder into its own subdirectory" << std::endl;
        return false;
    }
    
    // Ensure target parent exists
    if (!FileSystem::Exists(target_parent_path)) {
        FileSystem::CreateDirectories(target_parent_path);
    }
    
    // Check if target already exists
    if (FileSystem::Exists(full_target)) {
        std::cerr << "Target folder already exists" << std::endl;
        return false;
    }
    
    // Use system command for folder move
    std::string cmd = "move \"" + full_source.narrow() + "\" \"" + full_target.narrow() + "\"";
    int result = system(cmd.c_str());
    
    if (result == 0) {
        std::cerr << "Successfully moved folder" << std::endl;
        return true;
    } else {
        std::cerr << "Failed to move folder (error code: " << result << ")" << std::endl;
        return false;
    }
}

inline String build_path_label(const String& root, const String& suffix) {
    if (suffix.isEmpty()) {
        return root;
    }
    return root + suffix;
}

inline bool draw_up_navigation_button(Button& button, bool can_go_up) {
    if (!can_go_up) {
        return false;
    }
    button.enable();
    button.draw();
    return button.clicked();
}
