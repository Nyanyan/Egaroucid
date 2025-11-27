/*
    Egaroucid Project

    @file draw.hpp
        Drawing board / information
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"

void draw_board(Fonts fonts, Colors colors, History_elem history_elem, bool monochrome) {
    String coord_x = U"abcdefgh";
    Color dark_gray_color = colors.dark_gray;
    if (monochrome) {
        dark_gray_color = colors.black;
    }
    for (int i = 0; i < HW; ++i) {
        fonts.font_bold(i + 1).draw(15, Arg::center(BOARD_SX - BOARD_COORD_SIZE, BOARD_SY + BOARD_CELL_SIZE * i + BOARD_CELL_SIZE / 2), dark_gray_color);
        fonts.font_bold(coord_x[i]).draw(15, Arg::center(BOARD_SX + BOARD_CELL_SIZE * i + BOARD_CELL_SIZE / 2, BOARD_SY - BOARD_COORD_SIZE - 2), dark_gray_color);
    }
    for (int i = 0; i < HW_M1; ++i) {
        Line(BOARD_SX + BOARD_CELL_SIZE * (i + 1), BOARD_SY, BOARD_SX + BOARD_CELL_SIZE * (i + 1), BOARD_SY + BOARD_CELL_SIZE * HW).draw(BOARD_CELL_FRAME_WIDTH, dark_gray_color);
        Line(BOARD_SX, BOARD_SY + BOARD_CELL_SIZE * (i + 1), BOARD_SX + BOARD_CELL_SIZE * HW, BOARD_SY + BOARD_CELL_SIZE * (i + 1)).draw(BOARD_CELL_FRAME_WIDTH, dark_gray_color);
    }
    Circle(BOARD_SX + 2 * BOARD_CELL_SIZE, BOARD_SY + 2 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(dark_gray_color);
    Circle(BOARD_SX + 2 * BOARD_CELL_SIZE, BOARD_SY + 6 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(dark_gray_color);
    Circle(BOARD_SX + 6 * BOARD_CELL_SIZE, BOARD_SY + 2 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(dark_gray_color);
    Circle(BOARD_SX + 6 * BOARD_CELL_SIZE, BOARD_SY + 6 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(dark_gray_color);
    s3d::RoundRect(BOARD_SX, BOARD_SY, BOARD_CELL_SIZE * HW, BOARD_CELL_SIZE * HW, BOARD_ROUND_DIAMETER).drawFrame(0, BOARD_ROUND_FRAME_WIDTH, colors.white);
    Flip flip;
    int board_arr[HW2];
    history_elem.board.translate_to_arr(board_arr, history_elem.player);
    for (int cell = 0; cell < HW2; ++cell) {
        int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
        int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
        if (board_arr[cell] == BLACK) {
            Circle(x, y, DISC_SIZE).draw(colors.black);
        } else if (board_arr[cell] == WHITE) {
            if (monochrome) {
                Circle(x, y, DISC_SIZE).draw(colors.white).drawFrame(BOARD_DISC_FRAME_WIDTH, 0, colors.black);
            } else {
                Circle(x, y, DISC_SIZE).draw(colors.white);
            }
        }
    }
}

void draw_board(Fonts fonts, Colors colors, History_elem history_elem) {
    draw_board(fonts, colors, history_elem, false);
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

void draw_info(Colors colors, History_elem history_elem, Fonts fonts, Menu_elements menu_elements, bool pausing_in_pass, std::string principal_variation, bool forced_opening_found) {
    s3d::RoundRect round_rect{ INFO_SX, INFO_SY, INFO_WIDTH, INFO_HEIGHT, INFO_RECT_RADIUS };
    round_rect.drawFrame(INFO_RECT_THICKNESS, colors.white);
    // 1st line
    int dy = 6;
    String moves_line;
    if (history_elem.board.get_legal()) {
        moves_line = language.get("info", "ply") + Format(history_elem.board.n_discs() - 3) + language.get("info", "moves");
        bool black_to_move = history_elem.player == BLACK;
        if (black_to_move ^ pausing_in_pass) {
            moves_line += U" " + language.get("info", "black");
        } else {
            moves_line += U" " + language.get("info", "white");
        }
        bool ai_to_move = (menu_elements.ai_put_black && history_elem.player == BLACK) || (menu_elements.ai_put_white && history_elem.player == WHITE);
        if (ai_to_move ^ pausing_in_pass) {
            moves_line += U" (" + language.get("info", "ai") + U")";
        } else {
            moves_line += U" (" + language.get("info", "human") + U")";
        }
    } else {
        moves_line = language.get("info", "game_end");
    }
    fonts.font(moves_line).draw(15, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + dy));
    dy += 23;
    // 2nd line
    String opening_info = language.get("info", "opening_name") + U": ";
    if (menu_elements.show_opening_name) {
        opening_info += Unicode::FromUTF8(history_elem.opening_name);
    }
    fonts.font(opening_info).draw(12, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + dy));
    if (menu_elements.show_ai_focus) {
        dy += 24;
    } else {
        dy += 27;
    }
    // 3rd line
    int black_discs, white_discs;
    if (history_elem.player == BLACK) {
        black_discs = history_elem.board.count_player();
        white_discs = history_elem.board.count_opponent();
    } else {
        black_discs = history_elem.board.count_opponent();
        white_discs = history_elem.board.count_player();
    }
    Line(INFO_SX + INFO_WIDTH / 2, INFO_SY + dy, INFO_SX + INFO_WIDTH / 2, INFO_SY + dy + INFO_DISC_RADIUS * 2).draw(2, colors.dark_gray);
    if (menu_elements.show_ai_focus) {
        Rect(INFO_SX + 10, INFO_SY + dy - 2, AI_FOCUS_INFO_COLOR_RECT_WIDTH, INFO_DISC_RADIUS * 2 + 4).draw(colors.black_advantage);
        Rect(INFO_SX + INFO_WIDTH - 10 - AI_FOCUS_INFO_COLOR_RECT_WIDTH, INFO_SY + dy - 2, AI_FOCUS_INFO_COLOR_RECT_WIDTH, INFO_DISC_RADIUS * 2 + 4).draw(colors.white_advantage);
        fonts.font(black_discs).draw(20, Arg::center(INFO_SX + 138, INFO_SY + dy + INFO_DISC_RADIUS));
        Circle(INFO_SX + 100, INFO_SY + dy + INFO_DISC_RADIUS, INFO_DISC_RADIUS).draw(colors.black);
        Circle(INFO_SX + INFO_WIDTH - 100, INFO_SY + dy + INFO_DISC_RADIUS, INFO_DISC_RADIUS).draw(colors.white);
        fonts.font(language.get("info", "black_advantage")).draw(12, Arg::center(INFO_SX + 10 + (AI_FOCUS_INFO_COLOR_RECT_WIDTH - INFO_DISC_RADIUS * 2 - 6) / 2, INFO_SY + dy + INFO_DISC_RADIUS), colors.black);
        fonts.font(language.get("info", "white_advantage")).draw(12, Arg::center(INFO_SX + INFO_WIDTH - 10 - (AI_FOCUS_INFO_COLOR_RECT_WIDTH - INFO_DISC_RADIUS * 2 - 6) / 2, INFO_SY + dy + INFO_DISC_RADIUS), colors.black);
        fonts.font(white_discs).draw(20, Arg::center(INFO_SX + INFO_WIDTH - 138, INFO_SY + dy + INFO_DISC_RADIUS));
        dy += 28;
    } else {
        Circle(INFO_SX + 70, INFO_SY + dy + INFO_DISC_RADIUS, INFO_DISC_RADIUS).draw(colors.black);
        Circle(INFO_SX + INFO_WIDTH - 70, INFO_SY + dy + INFO_DISC_RADIUS, INFO_DISC_RADIUS).draw(colors.white);
        fonts.font(black_discs).draw(20, Arg::center(INFO_SX + 110, INFO_SY + dy + INFO_DISC_RADIUS));
        fonts.font(white_discs).draw(20, Arg::center(INFO_SX + INFO_WIDTH - 110, INFO_SY + dy + INFO_DISC_RADIUS));
        dy += 30;
    }
    // 4th line
    if (menu_elements.show_ai_focus) {
        // AI's focus Mode
        const double linewidth = 3;
        const double width = AI_FOCUS_INFO_COLOR_RECT_WIDTH - linewidth;
        const double height = 22;
        const double lleft = INFO_SX + 10 + linewidth / 2;
        const double rright = INFO_SX + INFO_WIDTH - 10 - AI_FOCUS_INFO_COLOR_RECT_WIDTH + linewidth / 2 + width;
        const double up = INFO_SY + dy + linewidth / 2 + 2;
        Line{ lleft, up, lleft + width / 2, up }.draw(LineStyle::SquareDot, linewidth, colors.blue);
        Line{ lleft, up + height, lleft + width / 2, up + height }.draw(LineStyle::SquareDot, linewidth, colors.blue);
        Line{ lleft, up, lleft, up + height }.draw(LineStyle::SquareDot, linewidth, colors.blue);
        Line{ lleft + width, up, lleft + width, up + height }.draw(linewidth, colors.blue);
        Line{ lleft + width / 2, up, lleft + width, up }.draw(linewidth, colors.blue);
        Line{ lleft + width / 2, up + height, lleft + width, up + height }.draw(linewidth, colors.blue);
        fonts.font(language.get("info", "good_point")).draw(12, Arg::center(lleft + width / 2, up + height / 2));
        Line{ rright, up, rright - width / 2, up }.draw(LineStyle::SquareDot, linewidth, colors.red);
        Line{ rright, up + height, rright - width / 2, up + height }.draw(LineStyle::SquareDot, linewidth, colors.red);
        Line{ rright, up, rright, up + height }.draw(LineStyle::SquareDot, linewidth, colors.red);
        Line{ rright - width, up, rright - width, up + height }.draw(linewidth, colors.red);
        Line{ rright - width / 2, up, rright - width, up }.draw(linewidth, colors.red);
        Line{ rright - width / 2, up + height, rright - width, up + height }.draw(linewidth, colors.red);
        fonts.font(language.get("info", "bad_point")).draw(12, Arg::center(rright - width / 2, up + height / 2));
        String level_info = language.get("common", "level") + U" " + Format(menu_elements.level);
        bool is_forced = menu_elements.force_specified_openings && forced_opening_found;
        if (is_forced) {
            fonts.font(level_info).draw(11, Arg::center(INFO_SX + INFO_WIDTH / 2, up + height / 4));
            fonts.font(language.get("info", "forced")).draw(11, Arg::center(INFO_SX + INFO_WIDTH / 2, up + height * 3 / 4));
        } else {
            fonts.font(level_info).draw(12, Arg::center(INFO_SX + INFO_WIDTH / 2, up + height / 2));
        }
        dy += 23;
    } else {
        // Normal Mode
        String level_info = language.get("common", "level") + U" " + Format(menu_elements.level) + U" (";
        if (menu_elements.level <= LIGHT_LEVEL) {
            level_info += language.get("info", "light");
        } else if (menu_elements.level <= STANDARD_MAX_LEVEL) {
            level_info += language.get("info", "standard");
        } else if (menu_elements.level <= PRAGMATIC_MAX_LEVEL) {
            level_info += language.get("info", "pragmatic");
        } else if (menu_elements.level <= ACCURATE_MAX_LEVEL) {
            level_info += language.get("info", "accurate");
        } else {
            level_info += language.get("info", "danger");
        }
        level_info += U")";
        bool is_forced = menu_elements.force_specified_openings && forced_opening_found;
        if (is_forced) {
            level_info += U" " + language.get("info", "forced");
        }
        fonts.font(level_info).draw(12, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + dy));
        dy += 18;
    }
    // 5th line
    String pv_info = language.get("info", "principal_variation") + U": ";
    String pv_info2 = U"";
    bool use_second_line = false;
    if (menu_elements.show_principal_variation) {
        if (principal_variation.size() > 15 * 2) { // 15 moves and more?
            int center = principal_variation.size() / 2 / 2 * 2;
            if (center > 18 * 2) {
                center = 18 * 2;
            }
            std::string pv1 = principal_variation.substr(0, center);
            std::string pv2 = principal_variation.substr(center);
            pv_info += Unicode::Widen(pv1);
            pv_info2 = Unicode::Widen(pv2);
            use_second_line = true;
        } else {
            pv_info += Unicode::Widen(principal_variation);
        }
    }
    if (use_second_line) {
        fonts.font(pv_info).draw(11, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + dy));
        fonts.font(pv_info2).draw(11, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + dy + 12));
    } else {
        fonts.font(pv_info).draw(13, Arg::center(INFO_SX + INFO_WIDTH / 2, ((INFO_SY + dy) + (INFO_SY + INFO_HEIGHT - INFO_RECT_THICKNESS / 2)) / 2));
    }
}



struct ExplorerDrawResult {
    bool folderClicked = false;
    bool folderDoubleClicked = false;
    String clickedFolder;
    bool importClicked = false;
    bool gameDoubleClicked = false;
    int importIndex = -1;
    bool deleteClicked = false;
    int deleteIndex = -1;
    bool upButtonClicked = false;
    bool parentFolderDoubleClicked = false;  // New: parent folder navigation
    
    // Drag and drop functionality
    bool drag_started = false;
    bool drop_completed = false;
    int dragged_game_index = -1;
    String dragged_folder_name;
    String drop_target_folder;
    bool is_dragging_game = false;
    bool is_dragging_folder = false;
    bool drop_on_parent = false;  // New: drop on parent folder
};

// Drag state management structure
struct ExplorerDragState {
    bool is_dragging = false;
    bool is_dragging_game = false;
    bool is_dragging_folder = false;
    int dragged_game_index = -1;
    String dragged_folder_name;
    Vec2 drag_start_pos;
    Vec2 drag_offset;
    Vec2 current_mouse_pos;
    bool mouse_was_down_last_frame = false;
    bool mouse_just_pressed = false;  // Add this for consistent state
    bool mouse_just_released = false; // Add this for consistent state
    
    // Constants
    static constexpr double DRAG_THRESHOLD = 2.0;  // Reduced from 5.0 to make drag easier
    
    void reset_drag_preparation() {
        dragged_game_index = -1;
        dragged_folder_name.clear();
        drag_start_pos = Vec2(0, 0);
        drag_offset = Vec2(0, 0);
    }
    
    void reset_drag_state() {
        is_dragging = false;
        is_dragging_game = false;
        is_dragging_folder = false;
        reset_drag_preparation();
    }
    
    void update_mouse_state() {
        current_mouse_pos = Cursor::Pos();
        bool mouse_is_down = MouseL.pressed();
        mouse_just_pressed = mouse_is_down && !mouse_was_down_last_frame;
        mouse_just_released = !mouse_is_down && mouse_was_down_last_frame;
        mouse_was_down_last_frame = mouse_is_down;
    }
};

// Click state management structure
struct ExplorerClickState {
    uint64_t last_click_time = 0;
    String last_clicked_folder;
    int last_clicked_game_index = -1;
    static constexpr uint64_t DOUBLE_CLICK_TIME_MS = 400;
};

// Helper function to handle drag drop detection
template <class FontsT, class ColorsT, class ResourcesT, class LanguageT>
inline ExplorerDrawResult handle_drag_drop(
    ExplorerDragState& drag_state,
    const std::vector<String>& folders_display,
    Scroll_manager& scroll_manager,
    int item_height,
    int n_games_on_window,
    bool has_parent
) {
    ExplorerDrawResult res;
    
    // Handle drag end (mouse release)
    if (drag_state.is_dragging && !MouseL.pressed()) {
        drag_state.is_dragging = false;
        
        // Check if we're dropping on parent folder first
        bool dropped_on_parent = false;
        if (has_parent) {
            int parent_row = 0;
            int parent_sy = IMPORT_GAME_SY + 8 + (parent_row - scroll_manager.get_strt_idx_int()) * item_height;
            
            if (parent_row >= scroll_manager.get_strt_idx_int() && 
                parent_row < scroll_manager.get_strt_idx_int() + n_games_on_window) {
                Rect parent_rect(IMPORT_GAME_SX, parent_sy, IMPORT_GAME_WIDTH, item_height);
                if (parent_rect.contains(drag_state.current_mouse_pos)) {
                    dropped_on_parent = true;
                }
            }
        }
        
        // Check if we're dropping on a folder
        bool dropped_on_folder = false;
        String target_folder;
        
        if (!dropped_on_parent) {
            int parent_offset = has_parent ? 1 : 0;
            for (int folder_idx = 0; folder_idx < (int)folders_display.size(); ++folder_idx) {
                int folder_row = parent_offset + folder_idx;
                int folder_sy = IMPORT_GAME_SY + 8 + (folder_row - scroll_manager.get_strt_idx_int()) * item_height;
                
                if (folder_row >= scroll_manager.get_strt_idx_int() && 
                    folder_row < scroll_manager.get_strt_idx_int() + n_games_on_window) {
                    Rect folder_rect(IMPORT_GAME_SX, folder_sy, IMPORT_GAME_WIDTH, item_height);
                    if (folder_rect.contains(drag_state.current_mouse_pos)) {
                        String candidate_target = folders_display[folder_idx];
                        // Don't allow dropping a folder on itself
                        if (drag_state.is_dragging_folder && drag_state.dragged_folder_name == candidate_target) {
                            continue;
                        }
                        dropped_on_folder = true;
                        target_folder = candidate_target;
                        break;
                    }
                }
            }
        }
        
        if (dropped_on_parent) {
            if (drag_state.is_dragging_game) {
                std::cerr << "Moving game " << drag_state.dragged_game_index << " to parent folder" << std::endl;
            } else if (drag_state.is_dragging_folder) {
                std::cerr << "Moving folder '" << drag_state.dragged_folder_name.narrow() << "' to parent folder" << std::endl;
            }
            res.drop_completed = true;
            res.drop_on_parent = true;
            res.is_dragging_game = drag_state.is_dragging_game;
            res.is_dragging_folder = drag_state.is_dragging_folder;
            res.dragged_game_index = drag_state.dragged_game_index;
            res.dragged_folder_name = drag_state.dragged_folder_name;
        } else if (dropped_on_folder) {
            if (drag_state.is_dragging_game) {
                std::cerr << "Moving game " << drag_state.dragged_game_index << " to folder '" << target_folder.narrow() << "'" << std::endl;
            } else if (drag_state.is_dragging_folder) {
                std::cerr << "Moving folder '" << drag_state.dragged_folder_name.narrow() << "' to folder '" << target_folder.narrow() << "'" << std::endl;
            }
            res.drop_completed = true;
            res.drop_target_folder = target_folder;
            res.is_dragging_game = drag_state.is_dragging_game;
            res.is_dragging_folder = drag_state.is_dragging_folder;
            res.dragged_game_index = drag_state.dragged_game_index;
            res.dragged_folder_name = drag_state.dragged_folder_name;
        }
        
        // Reset drag state
        drag_state.reset_drag_state();
    }
    
    return res;
}

// Helper function to update mouse state and handle drag start
inline void update_mouse_state_and_drag_start(ExplorerDragState& drag_state, ExplorerDrawResult& res) {
    drag_state.update_mouse_state();
    
    // Reset drag state when mouse is released (but only if not dragging)
    if (drag_state.mouse_just_released && !drag_state.is_dragging) {
        drag_state.reset_drag_preparation();
        return;  // Early return to avoid further processing
    }
    
    // Check for drag start (if mouse is pressed and we have valid drag targets)
    if (MouseL.pressed() && !drag_state.is_dragging) {
        if ((drag_state.dragged_game_index >= 0 || !drag_state.dragged_folder_name.empty()) && 
            drag_state.drag_start_pos.x != 0 && drag_state.drag_start_pos.y != 0) {
            double distance = drag_state.drag_start_pos.distanceFrom(drag_state.current_mouse_pos);
            if (distance > ExplorerDragState::DRAG_THRESHOLD) {
                // Start dragging
                if (drag_state.dragged_game_index >= 0) {
                    drag_state.is_dragging = true;
                    drag_state.is_dragging_game = true;
                    res.drag_started = true;
                    res.is_dragging_game = true;
                    res.dragged_game_index = drag_state.dragged_game_index;
                } else if (!drag_state.dragged_folder_name.empty()) {
                    drag_state.is_dragging = true;
                    drag_state.is_dragging_folder = true;
                    res.drag_started = true;
                    res.is_dragging_folder = true;
                    res.dragged_folder_name = drag_state.dragged_folder_name;
                }
            }
        }
    }
}

template <class FontsT, class ColorsT, class ResourcesT, class LanguageT>
inline ExplorerDrawResult DrawExplorerList(
    const std::vector<String>& folders_display,
    const std::vector<Game_abstract>& games,
    std::vector<ImageButton>& delete_buttons,
    Scroll_manager& scroll_manager,
    Button& up_button,
    int item_height,
    int n_games_on_window,
    bool has_parent,
    FontsT& fonts,
    ColorsT& colors,
    ResourcesT& resources,
    LanguageT& language,
    const std::string& document_dir,
    const std::string& current_subfolder
) {
    ExplorerDrawResult res;
    
    // Static state management
    static ExplorerDragState drag_state;
    static ExplorerClickState click_state;
    
    uint64_t current_time = Time::GetMillisec();
    
    // Update mouse state and handle drag operations
    update_mouse_state_and_drag_start(drag_state, res);
    
    // Handle drag drop detection
    ExplorerDrawResult drag_result = handle_drag_drop<FontsT, ColorsT, ResourcesT, LanguageT>(
        drag_state, folders_display, scroll_manager, item_height, n_games_on_window, has_parent);
    if (drag_result.drop_completed) {
        return drag_result;
    }
    
    // Check if there are any items to display
    int parent_offset = has_parent ? 1 : 0;
    int total_items = parent_offset + (int)folders_display.size() + (int)games.size();
    
    // Only show "no game available" if we're at root and have no items
    if (total_items == 0) {
        fonts.font(language.get("in_out", "no_game_available")).draw(20, Arg::center(X_CENTER, Y_CENTER), colors.white);
        return res;
    }
    
    // If we only have parent folder (empty subfolder), show a message but still show parent folder
    bool empty_subfolder = has_parent && (folders_display.size() == 0 && games.size() == 0);
    if (empty_subfolder) {
        fonts.font(language.get("in_out", "empty_folder")).draw(16, Arg::center(X_CENTER, Y_CENTER + 50), colors.white);
    }
    
    // Draw list items
    ExplorerDrawResult list_result = draw_explorer_list_items<FontsT, ColorsT, ResourcesT, LanguageT>(
        folders_display, games, delete_buttons, scroll_manager, 
        item_height, n_games_on_window, has_parent, fonts, colors, resources, language,
        drag_state, click_state, current_time
    );
    
    if (list_result.folderClicked || list_result.folderDoubleClicked || 
        list_result.gameDoubleClicked || list_result.deleteClicked || 
        list_result.parentFolderDoubleClicked) {
        return list_result;
    }
    
    // Draw dragged items on top (最前面に描画)
    draw_dragged_items<FontsT, ColorsT, ResourcesT>(drag_state, games, folders_display, item_height, fonts, colors, resources);
    
    return res;
}

// Helper function to draw list items
template <class FontsT, class ColorsT, class ResourcesT, class LanguageT>
inline ExplorerDrawResult draw_explorer_list_items(
    const std::vector<String>& folders_display,
    const std::vector<Game_abstract>& games,
    std::vector<ImageButton>& delete_buttons,
    Scroll_manager& scroll_manager,
    int item_height,
    int n_games_on_window,
    bool has_parent,
    FontsT& fonts,
    ColorsT& colors,
    ResourcesT& resources,
    LanguageT& language,
    ExplorerDragState& drag_state,
    ExplorerClickState& click_state,
    uint64_t current_time
) {
    ExplorerDrawResult res;
    
    int sy = IMPORT_GAME_SY;
    int strt_idx_int = scroll_manager.get_strt_idx_int();
    if (strt_idx_int > 0) {
        fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, sy }, colors.white);
    }
    sy += 8;
    
    int parent_offset = has_parent ? 1 : 0;
    int total_rows = parent_offset + (int)folders_display.size() + (int)games.size();
    
    for (int row = strt_idx_int; row < std::min(total_rows, strt_idx_int + n_games_on_window); ++row) {
        if (row < 0 || row >= total_rows) {
            continue;
        }
        
        Rect rect(IMPORT_GAME_SX, sy, IMPORT_GAME_WIDTH, item_height);
        if (row % 2) {
            rect.draw(colors.dark_green).drawFrame(1.0, colors.white);
        } else {
            rect.draw(colors.green).drawFrame(1.0, colors.white);
        }

        // Handle parent folder
        if (has_parent && (row - strt_idx_int) == 0 && row == 0) {
            ExplorerDrawResult parent_result = draw_parent_folder<FontsT, ColorsT>(rect, drag_state, fonts, colors, sy, item_height, click_state, current_time);
            if (parent_result.parentFolderDoubleClicked) {
                return parent_result;
            }
        } 
        // Handle folders
        else if (row - parent_offset < (int)folders_display.size()) {
            int folder_idx = row - parent_offset;
            if (folder_idx >= 0 && folder_idx < (int)folders_display.size()) {
                ExplorerDrawResult folder_result = draw_folder_item<FontsT, ColorsT, ResourcesT>(
                    folders_display[folder_idx], rect, row, drag_state, fonts, colors, resources,
                    sy, item_height, click_state, current_time
                );
                if (folder_result.folderClicked || folder_result.folderDoubleClicked) {
                    return folder_result;
                }
            }
        } 
        // Handle games
        else {
            int i = row - parent_offset - (int)folders_display.size();
            if (i >= 0 && i < (int)games.size()) {
                ExplorerDrawResult game_result = draw_game_item<FontsT, ColorsT>(
                    games[i], i, rect, row, delete_buttons, drag_state, fonts, colors,
                    sy, item_height, click_state, current_time
                );
                if (game_result.gameDoubleClicked || game_result.deleteClicked) {
                    return game_result;
                }
            }
        }
        sy += item_height;
    }
    
    if (strt_idx_int + n_games_on_window < total_rows) {
        fonts.font(U"︙").draw(15, Arg::topCenter(X_CENTER, IMPORT_GAME_SY + item_height * n_games_on_window + 10), colors.white);
    }
    scroll_manager.draw();
    scroll_manager.update();
    
    return res;
}

// Helper function to draw parent folder
template <class FontsT, class ColorsT>
inline ExplorerDrawResult draw_parent_folder(
    const Rect& rect,
    ExplorerDragState& drag_state,
    FontsT& fonts,
    ColorsT& colors,
    int sy,
    int item_height,
    ExplorerClickState& click_state,
    uint64_t current_time
) {
    ExplorerDrawResult res;
    
    // Handle drop on parent folder (visual feedback)
    if (rect.contains(drag_state.current_mouse_pos) && (drag_state.is_dragging_game || drag_state.is_dragging_folder)) {
        rect.draw(colors.yellow.withAlpha(64));
    }
    
    // Draw parent folder (..) 
    fonts.font(U"↑..").draw(15, Arg::leftCenter(IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, sy + item_height / 2), colors.white);
    
    // Handle parent folder double-click
    if (rect.leftClicked() && !drag_state.is_dragging) {
        static String last_clicked_parent;
        static uint64 last_parent_click_time = 0;
        
        // Check for double-click first (regardless of drag preparation)
        if (last_clicked_parent == U"parent" && current_time - last_parent_click_time < ExplorerClickState::DOUBLE_CLICK_TIME_MS) {
            // Double-click detected - clear any drag preparation and execute double-click
            std::cerr << "Navigating to parent folder" << std::endl;
            drag_state.reset_drag_preparation();
            res.parentFolderDoubleClicked = true;
            last_clicked_parent.clear();
            last_parent_click_time = 0;
        } else {
            // Single click - always update click state for potential double-click detection
            last_clicked_parent = U"parent";
            last_parent_click_time = current_time;
        }
    }
    
    return res;
}

// Helper function to draw folder item
template <class FontsT, class ColorsT, class ResourcesT>
inline ExplorerDrawResult draw_folder_item(
    const String& fname,
    const Rect& rect,
    int row,
    ExplorerDragState& drag_state,
    FontsT& fonts,
    ColorsT& colors,
    ResourcesT& resources,
    int sy,
    int item_height,
    ExplorerClickState& click_state,
    uint64_t current_time
) {
    ExplorerDrawResult res;
    
    double folder_icon_scale = (double)(rect.h - 2 * 10) / (double)resources.folder.height();
    
    // Drag and drop for folders
    bool is_being_dragged = (drag_state.is_dragging_folder && drag_state.dragged_folder_name == fname);
    Color folder_bg_color = is_being_dragged ? colors.yellow.withAlpha(64) : // ドラッグ中は元の位置を半透明に
                           (row % 2 ? colors.dark_green : colors.green);
    
    // Draw folder icon and name at original position (半透明でドラッグ中のアイテムを表示)
    Color icon_alpha = is_being_dragged ? colors.white.withAlpha(128) : colors.white;
    resources.folder.scaled(folder_icon_scale).draw(Arg::leftCenter(IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, sy + item_height / 2), icon_alpha);
    fonts.font(fname).draw(15, Arg::leftCenter(IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10 + 30, sy + item_height / 2), icon_alpha);
    
    // Handle folder drag preparation - only on initial press
    if (drag_state.mouse_just_pressed && rect.contains(drag_state.current_mouse_pos) && 
        !drag_state.is_dragging && drag_state.dragged_game_index == -1 && drag_state.dragged_folder_name.empty()) {
        drag_state.dragged_folder_name = fname;
        drag_state.drag_start_pos = drag_state.current_mouse_pos;
        drag_state.drag_offset = drag_state.current_mouse_pos - Vec2(rect.x, rect.y);
    }
    
    // Handle folder click and double-click
    if (rect.leftClicked() && !drag_state.is_dragging) {
        // Check for double-click first (regardless of drag preparation)
        if (click_state.last_clicked_folder == fname && current_time - click_state.last_click_time < ExplorerClickState::DOUBLE_CLICK_TIME_MS) {
            // Double-click detected - clear drag preparation and execute double-click
            std::cerr << "Navigating to folder: " << fname.narrow() << std::endl;
            drag_state.reset_drag_preparation();
            res.folderDoubleClicked = true;
            res.clickedFolder = fname;
            click_state.last_clicked_folder.clear();
            click_state.last_click_time = 0;
            return res;
        } else {
            // Single click - only process if no drag preparation exists
            if (drag_state.dragged_game_index == -1 && drag_state.dragged_folder_name.empty()) {
                res.folderClicked = true;
                res.clickedFolder = fname;
                click_state.last_clicked_folder = fname;
                click_state.last_click_time = current_time;
                return res;
            } else {
                // Update click state for potential double-click detection, but don't process single click
                click_state.last_clicked_folder = fname;
                click_state.last_click_time = current_time;
            }
        }
    }
    
    return res;
}
// Helper function to draw game item
template <class FontsT, class ColorsT>
inline ExplorerDrawResult draw_game_item(
    const Game_abstract& game,
    int game_index,
    const Rect& rect,
    int row,
    std::vector<ImageButton>& delete_buttons,
    ExplorerDragState& drag_state,
    FontsT& fonts,
    ColorsT& colors,
    int sy,
    int item_height,
    ExplorerClickState& click_state,
    uint64_t current_time
) {
    ExplorerDrawResult res;
    
    // Drag and drop for games
    bool is_being_dragged = (drag_state.is_dragging_game && drag_state.dragged_game_index == game_index);
    Color game_bg_color = is_being_dragged ? colors.yellow.withAlpha(64) : // ドラッグ中は元の位置を半透明に
                         (row % 2 ? colors.dark_green : colors.green);
    
    // Always draw background, but make it more transparent when being dragged
    rect.draw(game_bg_color);
    
    // テキスト色をドラッグ状態に応じて調整
    Color text_color = is_being_dragged ? colors.white.withAlpha(128) : colors.white;
    
    int winner = -1;
    if (game.black_score != GAME_DISCS_UNDEFINED && game.white_score != GAME_DISCS_UNDEFINED) {
        if (game.black_score > game.white_score) {
            winner = IMPORT_GAME_WINNER_BLACK;
        } else if (game.black_score < game.white_score) {
            winner = IMPORT_GAME_WINNER_WHITE;
        } else {
            winner = IMPORT_GAME_WINNER_DRAW;
        }
    }
    
    // Handle game drag preparation - only on initial press
    if (drag_state.mouse_just_pressed && rect.contains(drag_state.current_mouse_pos) && 
        !drag_state.is_dragging && drag_state.dragged_game_index == -1 && drag_state.dragged_folder_name.empty()) {
        drag_state.dragged_game_index = game_index;
        drag_state.drag_start_pos = drag_state.current_mouse_pos;
        drag_state.drag_offset = drag_state.current_mouse_pos - Vec2(rect.x, rect.y);
    }
    
    // Show delete button only if delete_buttons vector has sufficient size
    if (game_index < (int)delete_buttons.size()) {
        delete_buttons[game_index].move(IMPORT_GAME_SX + 1, sy + 1);
        delete_buttons[game_index].draw();
        if (delete_buttons[game_index].clicked()) {
            res.deleteClicked = true;
            res.deleteIndex = game_index;
            return res;
        }
    }
    
    String date = game.date.substr(0, 10).replace(U"_", U"/");
    fonts.font(date).draw(15, IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, sy + 2, text_color);
    
    // Draw player rectangles and scores
    Rect black_player_rect;
    black_player_rect.w = IMPORT_GAME_PLAYER_WIDTH;
    black_player_rect.h = IMPORT_GAME_PLAYER_HEIGHT;
    black_player_rect.y = sy + 1;
    black_player_rect.x = IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + IMPORT_GAME_DATE_WIDTH;
    
    if (winner == IMPORT_GAME_WINNER_BLACK) {
        black_player_rect.draw(colors.darkred);
    } else if (winner == IMPORT_GAME_WINNER_WHITE) {
        black_player_rect.draw(colors.darkblue);
    } else if (winner == IMPORT_GAME_WINNER_DRAW) {
        black_player_rect.draw(colors.chocolate);
    }
    
    int upper_center_y = black_player_rect.y + black_player_rect.h / 2;
    
    // Draw black player name with size adjustment
    for (int font_size = 15; font_size >= 12; --font_size) {
        if (fonts.font(game.black_player).region(font_size, Vec2{0, 0}).w <= IMPORT_GAME_PLAYER_WIDTH - 4) {
            fonts.font(game.black_player).draw(font_size, Arg::rightCenter(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH - 2, upper_center_y), text_color);
            break;
        } else if (font_size == 12) {
            String player = game.black_player;
            while (fonts.font(player).region(font_size, Vec2{0, 0}).w > IMPORT_GAME_PLAYER_WIDTH - 4) {
                for (int i2 = 0; i2 < 4; ++i2) {
                    player.pop_back();
                }
                player += U"...";
            }
            fonts.font(player).draw(font_size, Arg::rightCenter(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH - 2, upper_center_y), text_color);
        }
    }
    
    // Draw scores
    String black_score = U"??";
    String white_score = U"??";
    if (game.black_score != GAME_DISCS_UNDEFINED && game.white_score != GAME_DISCS_UNDEFINED) {
        black_score = ToString(game.black_score);
        white_score = ToString(game.white_score);
    }
    double hyphen_w = fonts.font(U"-").region(15, Vec2{0, 0}).w;
    fonts.font(black_score).draw(15, Arg::rightCenter(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH + IMPORT_GAME_SCORE_WIDTH / 2 - hyphen_w / 2 - 1, upper_center_y), text_color);
    fonts.font(U"-").draw(15, Arg::center(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH + IMPORT_GAME_SCORE_WIDTH / 2, upper_center_y), text_color);
    fonts.font(white_score).draw(15, Arg::leftCenter(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH + IMPORT_GAME_SCORE_WIDTH / 2 + hyphen_w / 2 + 1, upper_center_y), text_color);
    
    // Draw white player rectangle
    Rect white_player_rect;
    white_player_rect.w = IMPORT_GAME_PLAYER_WIDTH;
    white_player_rect.h = IMPORT_GAME_PLAYER_HEIGHT;
    white_player_rect.y = sy + 1;
    white_player_rect.x = black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH + IMPORT_GAME_SCORE_WIDTH;
    
    if (winner == IMPORT_GAME_WINNER_BLACK) {
        white_player_rect.draw(colors.darkblue);
    } else if (winner == IMPORT_GAME_WINNER_WHITE) {
        white_player_rect.draw(colors.darkred);
    } else if (winner == IMPORT_GAME_WINNER_DRAW) {
        white_player_rect.draw(colors.chocolate);
    }
    
    // Draw white player name with size adjustment
    for (int font_size = 15; font_size >= 12; --font_size) {
        if (fonts.font(game.white_player).region(font_size, Vec2{0, 0}).w <= IMPORT_GAME_PLAYER_WIDTH - 4) {
            fonts.font(game.white_player).draw(font_size, Arg::leftCenter(white_player_rect.x + 2, upper_center_y), text_color);
            break;
        } else if (font_size == 12) {
            String player = game.white_player;
            while (fonts.font(player).region(font_size, Vec2{0, 0}).w > IMPORT_GAME_PLAYER_WIDTH - 4) {
                for (int i2 = 0; i2 < 4; ++i2) {
                    player.pop_back();
                }
                player += U"...";
            }
            fonts.font(player).draw(font_size, Arg::leftCenter(white_player_rect.x + 2, upper_center_y), text_color);
        }
    }
    
    fonts.font(game.memo).draw(12, IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, black_player_rect.y + black_player_rect.h, colors.white);
    
    // Handle game double-click for importing (avoiding delete button area)
    Rect game_click_area = rect;
    if (game_index < (int)delete_buttons.size()) {
        // Exclude delete button area (top-left corner)
        game_click_area.x += 20;
        game_click_area.w -= 20;
    }
    
    if (game_click_area.leftClicked() && !drag_state.is_dragging) {
        // Check for double-click first (regardless of drag preparation)
        if (click_state.last_clicked_game_index == game_index && current_time - click_state.last_click_time < ExplorerClickState::DOUBLE_CLICK_TIME_MS) {
            // Double-click detected - clear drag preparation and execute double-click
            std::cerr << "Importing game: " << game_index << std::endl;
            drag_state.reset_drag_preparation();
            res.gameDoubleClicked = true;
            res.importIndex = game_index;
            click_state.last_clicked_game_index = -1;
            click_state.last_click_time = 0;
            return res;
        } else {
            // Single click - only process if no drag preparation exists
            if (drag_state.dragged_game_index == -1 && drag_state.dragged_folder_name.empty()) {
                click_state.last_clicked_game_index = game_index;
                click_state.last_click_time = current_time;
                // Clear folder click state when clicking on game
                click_state.last_clicked_folder.clear();
            } else {
                // Update click state for potential double-click detection, but don't process single click
                click_state.last_clicked_game_index = game_index;
                click_state.last_click_time = current_time;
                // Clear folder click state when clicking on game
                click_state.last_clicked_folder.clear();
            }
        }
    }
    
    return res;
}

// Helper function to draw dragged items
template <class FontsT, class ColorsT, class ResourcesT>
inline void draw_dragged_items(
    const ExplorerDragState& drag_state,
    const std::vector<Game_abstract>& games,
    const std::vector<String>& folders_display,
    int item_height,
    FontsT& fonts,
    ColorsT& colors,
    ResourcesT& resources
) {
    // Draw dragged game at mouse position if dragging
    if (drag_state.is_dragging_game && drag_state.dragged_game_index >= 0 && drag_state.dragged_game_index < (int)games.size()) {
        // Use drag_offset to maintain relative position from where user clicked
        Vec2 drag_pos = drag_state.current_mouse_pos - drag_state.drag_offset;
        Rect drag_rect(drag_pos.x, drag_pos.y, IMPORT_GAME_WIDTH, item_height);
        
        // Enhanced visual styling for dragged game - more prominent
        drag_rect.draw(colors.yellow.withAlpha(220)).drawFrame(3.0, colors.white);
        
        // Draw simplified game info with better visibility
        const auto& game = games[drag_state.dragged_game_index];
        String date = game.date.substr(0, 10).replace(U"_", U"/");
        fonts.font(date).draw(12, drag_rect.x + 10, drag_rect.y + 2, colors.black);
        fonts.font(game.black_player + U" vs " + game.white_player).draw(10, drag_rect.x + 10, drag_rect.y + 15, colors.black);
        
        // Add a shadow effect for more depth
        Rect shadow_rect(drag_pos.x + 2, drag_pos.y + 2, IMPORT_GAME_WIDTH, item_height);
        shadow_rect.draw(colors.black.withAlpha(64));
    }
    
    // Draw dragged folder at mouse position if dragging
    if (drag_state.is_dragging_folder && !drag_state.dragged_folder_name.empty()) {
        Vec2 drag_pos = drag_state.current_mouse_pos - drag_state.drag_offset;
        Rect drag_rect(drag_pos.x, drag_pos.y, IMPORT_GAME_WIDTH, item_height);
        
        // Add shadow effect first
        Rect shadow_rect(drag_pos.x + 2, drag_pos.y + 2, IMPORT_GAME_WIDTH, item_height);
        shadow_rect.draw(colors.black.withAlpha(64));
        
        // Enhanced visual styling for dragged folder - more prominent
        Color folder_bg_color = colors.blue.withAlpha(220);
        drag_rect.draw(folder_bg_color).drawFrame(3.0, colors.white);
        
        // Draw folder icon and name
        double folder_icon_scale = (double)(drag_rect.h - 2 * 10) / (double)resources.folder.height();
        resources.folder.scaled(folder_icon_scale).draw(Arg::leftCenter(drag_rect.x + IMPORT_GAME_LEFT_MARGIN + 10, drag_rect.y + item_height / 2));
        fonts.font(drag_state.dragged_folder_name).draw(15, Arg::leftCenter(drag_rect.x + IMPORT_GAME_LEFT_MARGIN + 10 + 30, drag_rect.y + item_height / 2), colors.white);
    }
}

// Backward compatibility overload for existing code
template <class FontsT, class ColorsT, class ResourcesT, class LanguageT>
inline ExplorerDrawResult DrawExplorerList(
    const std::vector<String>& folders_display,
    const std::vector<Game_abstract>& games,
    std::vector<Button>& import_buttons,
    std::vector<ImageButton>& delete_buttons,
    Scroll_manager& scroll_manager,
    Button& up_button,
    bool showImportButtons,
    int item_height,
    int n_games_on_window,
    bool has_parent,
    FontsT& fonts,
    ColorsT& colors,
    ResourcesT& resources,
    LanguageT& language
) {
    // Call the main function with empty document_dir and current_subfolder
    return DrawExplorerList(
        folders_display, games, delete_buttons, scroll_manager,
        up_button, item_height,
        n_games_on_window, has_parent, fonts, colors, resources, language,
        "", ""
    );
}