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
    RoundRect(BOARD_SX, BOARD_SY, BOARD_CELL_SIZE * HW, BOARD_CELL_SIZE * HW, BOARD_ROUND_DIAMETER).drawFrame(0, BOARD_ROUND_FRAME_WIDTH, colors.white);
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

void draw_info(Colors colors, History_elem history_elem, Fonts fonts, Menu_elements menu_elements, bool pausing_in_pass, std::string principal_variation) {
    RoundRect round_rect{ INFO_SX, INFO_SY, INFO_WIDTH, INFO_HEIGHT, INFO_RECT_RADIUS };
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
        const double linewidth = 3;
        const double width = AI_FOCUS_INFO_COLOR_RECT_WIDTH - linewidth;
        const double height = 20;
        const double lleft = INFO_SX + 10 + linewidth / 2;
        const double rright = INFO_SX + INFO_WIDTH - 10 - AI_FOCUS_INFO_COLOR_RECT_WIDTH + linewidth / 2 + width;
        const double up = INFO_SY + dy - 2 + linewidth / 2;
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
        fonts.font(level_info).draw(12, Arg::center(INFO_SX + INFO_WIDTH / 2, up + height / 2));
        dy += 23;
    } else {
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
    bool openExplorerClicked = false;
    
    // Drag and drop functionality
    bool dragStarted = false;
    bool dropCompleted = false;
    int draggedGameIndex = -1;
    String draggedFolderName;
    String dropTargetFolder;
    bool isDraggingGame = false;
    bool isDraggingFolder = false;
    bool dropOnParent = false;  // New: drop on parent folder
};

template <class FontsT, class ColorsT, class ResourcesT, class LanguageT>
inline ExplorerDrawResult DrawExplorerList(
    const std::vector<String>& folders_display,
    const std::vector<Game_abstract>& games,
    std::vector<ImageButton>& delete_buttons,
    Scroll_manager& scroll_manager,
    Button& up_button,
    Button& open_explorer_button,
    int itemHeight,
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
    
    // Static variables for drag and drop state management
    static bool is_dragging = false;
    static bool is_dragging_game = false;
    static bool is_dragging_folder = false;
    static int dragged_game_index = -1;
    static String dragged_folder_name;
    static Vec2 drag_start_pos;
    static Vec2 drag_offset;  // Offset from click position to maintain relative positioning
    static Vec2 current_mouse_pos;
    
    // Static variables for double-click detection
    static uint64_t last_click_time = 0;
    static String last_clicked_folder;
    static int last_clicked_game_index = -1;
    static constexpr uint64_t DOUBLE_CLICK_TIME_MS = 400;
    static constexpr double DRAG_THRESHOLD = 0.5; // Minimum distance to start drag
    
    current_mouse_pos = Cursor::Pos();
    uint64_t current_time = Time::GetMillisec();
    
    // Handle drag end (mouse release)
    if (is_dragging && !MouseL.pressed()) {
        is_dragging = false;
        
        // Check if we're dropping on parent folder first
        bool dropped_on_parent = false;
        if (has_parent) {
            int parent_row = 0;
            int parent_sy = IMPORT_GAME_SY + 8 + (parent_row - scroll_manager.get_strt_idx_int()) * itemHeight;
            
            if (parent_row >= scroll_manager.get_strt_idx_int() && 
                parent_row < scroll_manager.get_strt_idx_int() + n_games_on_window) {
                Rect parent_rect(IMPORT_GAME_SX, parent_sy, IMPORT_GAME_WIDTH, itemHeight);
                if (parent_rect.contains(current_mouse_pos)) {
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
                int folder_row = parent_offset + folder_idx;  // Add parent offset
                int folder_sy = IMPORT_GAME_SY + 8 + (folder_row - scroll_manager.get_strt_idx_int()) * itemHeight;
                
                if (folder_row >= scroll_manager.get_strt_idx_int() && 
                    folder_row < scroll_manager.get_strt_idx_int() + n_games_on_window) {
                    Rect folder_rect(IMPORT_GAME_SX, folder_sy, IMPORT_GAME_WIDTH, itemHeight);
                    if (folder_rect.contains(current_mouse_pos)) {
                        dropped_on_folder = true;
                        target_folder = folders_display[folder_idx];
                        break;
                    }
                }
            }
        }
        
        if (dropped_on_parent) {
            res.dropCompleted = true;
            res.dropOnParent = true;
            res.isDraggingGame = is_dragging_game;
            res.isDraggingFolder = is_dragging_folder;
            res.draggedGameIndex = dragged_game_index;
            res.draggedFolderName = dragged_folder_name;
        } else if (dropped_on_folder) {
            res.dropCompleted = true;
            res.dropTargetFolder = target_folder;
            res.isDraggingGame = is_dragging_game;
            res.isDraggingFolder = is_dragging_folder;
            res.draggedGameIndex = dragged_game_index;
            res.draggedFolderName = dragged_folder_name;
        }
        
        // Reset drag state
        is_dragging_game = false;
        is_dragging_folder = false;
        dragged_game_index = -1;
        dragged_folder_name.clear();
        drag_start_pos = Vec2(0, 0);
        drag_offset = Vec2(0, 0);
    }
    
    // Clean up drag preparation if mouse is released without drag
    if (!MouseL.pressed() && !is_dragging) {
        if (dragged_game_index >= 0 || !dragged_folder_name.empty()) {
            dragged_game_index = -1;
            dragged_folder_name.clear();
            drag_start_pos = Vec2(0, 0);
            drag_offset = Vec2(0, 0);
        }
    }
    
    // Check if mouse has moved enough to start dragging
    if (MouseL.pressed() && !is_dragging) {
        if (drag_start_pos.x != 0 && drag_start_pos.y != 0) {
            double distance = drag_start_pos.distanceFrom(current_mouse_pos);
            if (distance > DRAG_THRESHOLD) {
                // Start dragging if we have a pending drag item
                if (dragged_game_index >= 0) {
                    is_dragging = true;
                    is_dragging_game = true;
                    res.dragStarted = true;
                    res.isDraggingGame = true;
                    res.draggedGameIndex = dragged_game_index;
                } else if (!dragged_folder_name.empty()) {
                    is_dragging = true;
                    is_dragging_folder = true;
                    res.dragStarted = true;
                    res.isDraggingFolder = true;
                    res.draggedFolderName = dragged_folder_name;
                }
            }
        }
    }
    
    // Parent folder is handled as the first item in the list when has_parent is true
    
    // "Open in Explorer" button in the top-right area
    open_explorer_button.draw();
    if (open_explorer_button.clicked()) {
        res.openExplorerClicked = true;
        return res;
    }
    
    // Check if there are any items to display
    int parent_offset = has_parent ? 1 : 0;  // Add parent folder as first item if has_parent
    int total_items = parent_offset + (int)folders_display.size() + (int)games.size();
    
    // Only show "no game available" if we're at root and have no items
    if (total_items == 0) {
        fonts.font(language.get("in_out", "no_game_available")).draw(20, Arg::center(X_CENTER, Y_CENTER), colors.white);
        return res;
    }
    
    // If we only have parent folder (empty subfolder), show a message but still show parent folder
    bool empty_subfolder = has_parent && (folders_display.size() == 0 && games.size() == 0);
    if (empty_subfolder) {
        fonts.font(U"このフォルダは空です").draw(16, Arg::center(X_CENTER, Y_CENTER + 50), colors.white);
    }
    
    int sy = IMPORT_GAME_SY;
    int strt_idx_int = scroll_manager.get_strt_idx_int();
    if (strt_idx_int > 0) {
        fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, sy }, colors.white);
    }
    sy += 8;
    int total_rows = parent_offset + (int)folders_display.size() + (int)games.size();
    for (int row = strt_idx_int; row < std::min(total_rows, strt_idx_int + n_games_on_window); ++row) {
        // Additional safety check: ensure row is within valid range
        if (row < 0 || row >= total_rows) {
            continue;
        }
        
        Rect rect;
        rect.y = sy;
        rect.x = IMPORT_GAME_SX;
        rect.w = IMPORT_GAME_WIDTH;
        rect.h = itemHeight;
        if (row % 2) {
            rect.draw(colors.dark_green).drawFrame(1.0, colors.white);
        } else {
            rect.draw(colors.green).drawFrame(1.0, colors.white);
        }

        // Handle parent folder as first item
        if (has_parent && row == 0) {
            // Handle drop on parent folder (visual feedback)
            if (rect.contains(current_mouse_pos) && (is_dragging_game || is_dragging_folder)) {
                rect.draw(colors.yellow.withAlpha(64));
            }
            
            // Draw parent folder (..) with special icon or styling
            double folder_icon_scale = (double)(rect.h - 2 * 10) / (double)resources.folder.height();
            // resources.folder.scaled(folder_icon_scale).draw(Arg::leftCenter(IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, sy + itemHeight / 2));
            // fonts.font(U"↑..").draw(15, Arg::leftCenter(IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10 + 30, sy + itemHeight / 2), colors.white);
            fonts.font(U"↑..").draw(15, Arg::leftCenter(IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, sy + itemHeight / 2), colors.white);
            
            // Handle parent folder double-click
            if (rect.leftClicked() && !is_dragging) {
                static String last_clicked_parent;
                static uint64 last_parent_click_time = 0;
                uint64 current_time = Time::GetMillisec();
                
                if (last_clicked_parent == U"parent" && current_time - last_parent_click_time < DOUBLE_CLICK_TIME_MS) {
                    // Double-click detected
                    res.parentFolderDoubleClicked = true;
                    last_clicked_parent.clear();
                    last_parent_click_time = 0;
                    return res;
                } else {
                    // Single click
                    last_clicked_parent = U"parent";
                    last_parent_click_time = current_time;
                }
            }
        } else if (row - parent_offset < (int)folders_display.size()) {
            // Additional bounds check for folders_display access
            int folder_idx = row - parent_offset;
            if (folder_idx >= 0 && folder_idx < (int)folders_display.size()) {
                String fname = folders_display[folder_idx];
                double folder_icon_scale = (double)(rect.h - 2 * 10) / (double)resources.folder.height();
                
                // Drag and drop for folders
                bool is_being_dragged = (is_dragging_folder && dragged_folder_name == fname);
                Color folder_bg_color = is_being_dragged ? colors.yellow.withAlpha(128) : 
                                       (row % 2 ? colors.dark_green : colors.green);
                
                if (is_being_dragged) {
                    // Draw dragged folder at mouse position with transparency - fix positioning
                    Vec2 drag_pos = current_mouse_pos - drag_offset;
                    Rect drag_rect(drag_pos.x, drag_pos.y, IMPORT_GAME_WIDTH, itemHeight);
                    drag_rect.draw(folder_bg_color).drawFrame(2.0, colors.white);
                    resources.folder.scaled(folder_icon_scale).draw(Arg::leftCenter(drag_rect.x + IMPORT_GAME_LEFT_MARGIN + 10, drag_rect.y + itemHeight / 2));
                    fonts.font(fname).draw(15, Arg::leftCenter(drag_rect.x + IMPORT_GAME_LEFT_MARGIN + 10 + 30, drag_rect.y + itemHeight / 2), colors.white);
                }
                
                resources.folder.scaled(folder_icon_scale).draw(Arg::leftCenter(IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, sy + itemHeight / 2));
                fonts.font(fname).draw(15, Arg::leftCenter(IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10 + 30, sy + itemHeight / 2), colors.white);
                
                // Handle folder drag preparation and click
                if (rect.leftPressed() && !is_dragging && dragged_game_index == -1 && dragged_folder_name.empty()) {
                    // Prepare for potential drag - record relative position from click
                    dragged_folder_name = fname;
                    drag_start_pos = current_mouse_pos;
                    drag_offset = current_mouse_pos - Vec2(rect.x, rect.y);  // Store relative position within rect
                }
                
                // Handle folder click and double-click (only if not dragging)
                if (rect.leftClicked() && !is_dragging) {
                    if (last_clicked_folder == fname && current_time - last_click_time < DOUBLE_CLICK_TIME_MS) {
                        // Double-click detected - prevent console window by not using System::LaunchFile
                        res.folderDoubleClicked = true;
                        res.clickedFolder = fname;
                        last_clicked_folder.clear();
                        last_click_time = 0;
                        // Clear drag state
                        dragged_folder_name.clear();
                        drag_start_pos = Vec2(0, 0);
                        drag_offset = Vec2(0, 0);
                        return res;
                    } else {
                        // Single click
                        res.folderClicked = true;
                        res.clickedFolder = fname;
                        last_clicked_folder = fname;
                        last_click_time = current_time;
                        // Clear drag state if it was just a click
                        if (drag_start_pos.distanceFrom(current_mouse_pos) <= DRAG_THRESHOLD) {
                            dragged_folder_name.clear();
                            drag_start_pos = Vec2(0, 0);
                            drag_offset = Vec2(0, 0);
                        }
                        return res;
                    }
                }
            }
        } else {
            // Always show games, but conditionally show import buttons
            int i = row - parent_offset - (int)folders_display.size();
            
            // Check bounds for games vector access
            if (i >= 0 && i < (int)games.size()) {
                // Drag and drop for games
                bool is_being_dragged = (is_dragging_game && dragged_game_index == i);
                Color game_bg_color = is_being_dragged ? colors.yellow.withAlpha(128) : 
                                     (row % 2 ? colors.dark_green : colors.green);
                
                // Override rect color for dragged items
                if (is_being_dragged) {
                    rect.draw(game_bg_color).drawFrame(2.0, colors.white);
                }
                
                int winner = -1;
                if (games[i].black_score != GAME_DISCS_UNDEFINED && games[i].white_score != GAME_DISCS_UNDEFINED) {
                    if (games[i].black_score > games[i].white_score) {
                        winner = IMPORT_GAME_WINNER_BLACK;
                    } else if (games[i].black_score < games[i].white_score) {
                        winner = IMPORT_GAME_WINNER_WHITE;
                    } else {
                        winner = IMPORT_GAME_WINNER_DRAW;
                    }
                }
                
                // Handle game drag preparation
                if (rect.leftPressed() && !is_dragging && dragged_game_index == -1 && dragged_folder_name.empty()) {
                    // Prepare for potential drag - record relative position from click
                    dragged_game_index = i;
                    drag_start_pos = current_mouse_pos;
                    drag_offset = current_mouse_pos - Vec2(rect.x, rect.y);  // Store relative position within rect
                }
                
                // Show delete button only if delete_buttons vector has sufficient size
                if (i < (int)delete_buttons.size()) {
                    delete_buttons[i].move(IMPORT_GAME_SX + 1, sy + 1);
                    delete_buttons[i].draw();
                    if (delete_buttons[i].clicked()) {
                        res.deleteClicked = true;
                        res.deleteIndex = i;
                        return res;
                    }
                }
                String date = games[i].date.substr(0, 10).replace(U"_", U"/");
                fonts.font(date).draw(15, IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, sy + 2, colors.white);
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
                for (int font_size = 15; font_size >= 12; --font_size) {
                    if (fonts.font(games[i].black_player).region(font_size, Vec2{0, 0}).w <= IMPORT_GAME_PLAYER_WIDTH - 4) {
                        fonts.font(games[i].black_player).draw(font_size, Arg::rightCenter(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH - 2, upper_center_y), colors.white);
                        break;
                    } else if (font_size == 12) {
                        String player = games[i].black_player;
                        while (fonts.font(player).region(font_size, Vec2{0, 0}).w > IMPORT_GAME_PLAYER_WIDTH - 4) {
                            for (int i2 = 0; i2 < 4; ++i2) {
                                player.pop_back();
                            }
                            player += U"...";
                        }
                        fonts.font(player).draw(font_size, Arg::rightCenter(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH - 2, upper_center_y), colors.white);
                    }
                }
                String black_score = U"??";
                String white_score = U"??";
                if (games[i].black_score != GAME_DISCS_UNDEFINED && games[i].white_score != GAME_DISCS_UNDEFINED) {
                    black_score = ToString(games[i].black_score);
                    white_score = ToString(games[i].white_score);
                }
                double hyphen_w = fonts.font(U"-").region(15, Vec2{0, 0}).w;
                fonts.font(black_score).draw(15, Arg::rightCenter(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH + IMPORT_GAME_SCORE_WIDTH / 2 - hyphen_w / 2 - 1, upper_center_y), colors.white);
                fonts.font(U"-").draw(15, Arg::center(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH + IMPORT_GAME_SCORE_WIDTH / 2, upper_center_y), colors.white);
                fonts.font(white_score).draw(15, Arg::leftCenter(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH + IMPORT_GAME_SCORE_WIDTH / 2 + hyphen_w / 2 + 1, upper_center_y), colors.white);
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
                for (int font_size = 15; font_size >= 12; --font_size) {
                    if (fonts.font(games[i].white_player).region(font_size, Vec2{0, 0}).w <= IMPORT_GAME_PLAYER_WIDTH - 4) {
                        fonts.font(games[i].white_player).draw(font_size, Arg::leftCenter(white_player_rect.x + 2, upper_center_y), colors.white);
                        break;
                    } else if (font_size == 12) {
                        String player = games[i].white_player;
                        while (fonts.font(player).region(font_size, Vec2{0, 0}).w > IMPORT_GAME_PLAYER_WIDTH - 4) {
                            for (int i2 = 0; i2 < 4; ++i2) {
                                player.pop_back();
                            }
                            player += U"...";
                        }
                        fonts.font(player).draw(font_size, Arg::leftCenter(white_player_rect.x + 2, upper_center_y), colors.white);
                    }
                }
                fonts.font(games[i].memo).draw(12, IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, black_player_rect.y + black_player_rect.h, colors.white);
                
                // Handle game double-click for importing (avoiding delete button area)
                Rect game_click_area = rect;
                if (i < (int)delete_buttons.size()) {
                    // Exclude delete button area (top-left corner)
                    game_click_area.x += 20;
                    game_click_area.w -= 20;
                }
                
                if (game_click_area.leftClicked() && !is_dragging) {
                    if (last_clicked_game_index == i && current_time - last_click_time < DOUBLE_CLICK_TIME_MS) {
                        // Double-click detected
                        res.gameDoubleClicked = true;
                        res.importIndex = i;
                        last_clicked_game_index = -1;
                        last_click_time = 0;
                        // Clear drag state
                        dragged_game_index = -1;
                        drag_start_pos = Vec2(0, 0);
                        drag_offset = Vec2(0, 0);
                        return res;
                    } else {
                        // Single click
                        last_clicked_game_index = i;
                        last_click_time = current_time;
                        // Clear folder click state when clicking on game
                        last_clicked_folder.clear();
                        // Clear drag state if it was just a click
                        if (drag_start_pos.distanceFrom(current_mouse_pos) <= DRAG_THRESHOLD) {
                            dragged_game_index = -1;
                            drag_start_pos = Vec2(0, 0);
                            drag_offset = Vec2(0, 0);
                        }
                    }
                }
            } // End of bounds check
        }
        sy += itemHeight;
    }
    int total_rows2 = parent_offset + (int)folders_display.size() + (int)games.size();
    if (strt_idx_int + n_games_on_window < total_rows2) {
        fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, 415}, colors.white);
    }
    scroll_manager.draw();
    scroll_manager.update();
    
    // Draw dragged game at mouse position if dragging
    if (is_dragging_game && dragged_game_index >= 0 && dragged_game_index < (int)games.size()) {
        // Use drag_offset to maintain relative position from where user clicked
        Vec2 drag_pos = current_mouse_pos - drag_offset;
        Rect drag_rect(drag_pos.x, drag_pos.y, IMPORT_GAME_WIDTH, itemHeight);
        drag_rect.draw(colors.yellow.withAlpha(200)).drawFrame(2.0, colors.white);
        
        // Draw simplified game info
        const auto& game = games[dragged_game_index];
        String date = game.date.substr(0, 10).replace(U"_", U"/");
        fonts.font(date).draw(12, drag_rect.x + 10, drag_rect.y + 2, colors.black);
        fonts.font(game.black_player + U" vs " + game.white_player).draw(10, drag_rect.x + 10, drag_rect.y + 15, colors.black);
    }
    
    return res;
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
    Button& open_explorer_button,
    bool showImportButtons,
    int itemHeight,
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
        up_button, open_explorer_button, itemHeight,
        n_games_on_window, has_parent, fonts, colors, resources, language,
        "", ""
    );
}