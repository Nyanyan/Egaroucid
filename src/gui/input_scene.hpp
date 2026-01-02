/*
    Egaroucid Project

    @file input.hpp
        Input scenes
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include <algorithm>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"
#include "draw.hpp"

// Forward declaration
inline bool load_game_from_json(
    Common_resources& data,
    Opening& opening,
    const String& json_path,
    const String& game_date,
    const std::string& subfolder
);

class Import_text : public App::Scene {
private:
    Button back_button;
    Button import_button;
    bool done;
    std::string text;
    std::vector<History_elem> n_history;
    std::vector<History_elem> history_until_now;
    bool imported_from_position;
    TextAreaEditState text_area;
    int text_input_format;
    Game_import_t imported_game;

public:
    Import_text(const InitData& init) : IScene{ init } {
        back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        import_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "import"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);done = false;
        import_button.disable();
        imported_from_position = false;
        text_input_format = TEXT_INPUT_FORMAT_NONE;
        text.clear();
        for (History_elem history_elem : getData().graph_resources.nodes[getData().graph_resources.branch]) {
            if (getData().history_elem.board.n_discs() >= history_elem.board.n_discs()) {
                history_until_now.emplace_back(history_elem);
            }
        }
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        const int icon_width = SCENE_ICON_WIDTH;
        getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
        getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
        int sy = 20 + icon_width + 30;
        getData().fonts.font(language.get("in_out", "input_text")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
        text_area.active = true;
        bool text_changed = SimpleGUI::TextArea(text_area, Vec2{X_CENTER - 300, sy + 40}, SizeF{600, 150}, INPUT_STR_MAX_SIZE);
        getData().fonts.font(language.get("in_out", "you_can_paste_with_ctrl_v") + U" / " + language.get("in_out", "you_can_drag_and_drop_game_file")).draw(13, Arg::topCenter(X_CENTER, sy + 195), getData().colors.white);
        bool return_pressed = false;
        if (text_area.text.size()) {
            if (text_area.text[text_area.text.size() - 1] == '\n') {
                return_pressed = true;
            }
        }
        if (DragDrop::HasNewFilePaths()) {
            std::string path = DragDrop::GetDroppedFilePaths()[0].path.narrow();
            std::ifstream ifs(path);
            if (ifs.fail()) {
                std::cerr << "can't open " << path << std::endl;
            } else {
                std::istreambuf_iterator<char> it(ifs);
                std::istreambuf_iterator<char> last;
                std::string str(it, last);
                text_area.text = Unicode::Widen(str);
                text_area.cursorPos = text_area.text.size();
                text_area.rebuildGlyphs();
                text_changed = true;
            }
        }
        text = text_area.text.replaced(U"\r", U"").replaced(U"\n", U"").replaced(U" ", U"").replace(U"\t", U"").narrow();
        if (text_changed) {
            bool failed;
            imported_game = import_any_format_processing(text, history_until_now, getData().history_elem, &failed);
        }
        String format_str = language.get("in_out", "format", "error");
        if (text.size() == 0) {
            format_str = language.get("in_out", "format", "egaroucid_recognizes_format");
        } else if (imported_game.format == TEXT_INPUT_FORMAT_GGF) {
            format_str = language.get("in_out", "format", "GGF");
        } else if (imported_game.format == TEXT_INPUT_FORMAT_OTHELLO_QUEST) {
            format_str = language.get("in_out", "format", "othello_quest");
        } else if (imported_game.format == TEXT_INPUT_FORMAT_TRANSCRIPT) {
            format_str = language.get("in_out", "format", "transcript");
        } else if (imported_game.format == TEXT_INPUT_FORMAT_TRANSCRIPT_FROM_THIS_POSITION) {
            format_str = language.get("in_out", "format", "transcript_from_this_position");
        } else if (imported_game.format == TEXT_INPUT_FORMAT_BOARD) {
            format_str = language.get("in_out", "format", "board");
        } else if (imported_game.format == TEXT_INPUT_FORMAT_GENERAL_BOARD_TRANSCRIPT) {
            format_str = language.get("in_out", "format", "board_transcript");
        }
        getData().fonts.font(format_str).draw(13, Arg::topCenter(X_CENTER, sy + 210), getData().colors.white);
        back_button.draw();
        import_button.draw();
        if (back_button.clicked() || KeyEscape.pressed()) {
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        if (imported_game.format != TEXT_INPUT_FORMAT_NONE && text.size()) {
            import_button.enable();
            if (import_button.clicked() || KeyEnter.pressed()) {
                n_history = imported_game.history;
                go_to_main_scene();
            }
        } else {
            import_button.disable();
        }
    }

    void draw() const override {

    }

    void go_to_main_scene() {
        if (!imported_from_position) {
            getData().graph_resources.init();
            getData().graph_resources.nodes[0] = n_history;
            getData().graph_resources.n_discs = getData().graph_resources.nodes[0].back().board.n_discs();
            getData().game_information.init();
        } else {
            getData().graph_resources.nodes[getData().graph_resources.branch] = n_history;
        }
        std::string opening_name, n_opening_name;
        for (int i = 0; i < (int)getData().graph_resources.nodes[getData().graph_resources.branch].size(); ++i) {
            n_opening_name.clear();
            n_opening_name = opening.get(getData().graph_resources.nodes[getData().graph_resources.branch][i].board, getData().graph_resources.nodes[getData().graph_resources.branch][i].player ^ 1);
            if (n_opening_name.size()) {
                opening_name = n_opening_name;
            }
            getData().graph_resources.nodes[getData().graph_resources.branch][i].opening_name = opening_name;
        }
        getData().graph_resources.n_discs = getData().graph_resources.nodes[getData().graph_resources.branch].back().board.n_discs();
        getData().graph_resources.need_init = false;
        getData().history_elem = getData().graph_resources.nodes[getData().graph_resources.branch].back();
        changeScene(U"Main_scene", SCENE_FADE_TIME);
    }
};






class Edit_board : public App::Scene {
private:
    Button back_button;
    Button set_black_button;
    Button set_white_button;
    Radio_button disc_radio;
    bool done;
    bool failed;
    History_elem history_elem;

public:
    Edit_board(const InitData& init) : IScene{ init } {
        set_black_button.init(BUTTON3_VERTICAL_SX, BUTTON3_VERTICAL_1_SY, BUTTON3_VERTICAL_WIDTH, BUTTON3_VERTICAL_HEIGHT, BUTTON3_VERTICAL_RADIUS, language.get("in_out", "import_as_black"), 25, getData().fonts.font, getData().colors.black, getData().colors.white);
        set_white_button.init(BUTTON3_VERTICAL_SX, BUTTON3_VERTICAL_2_SY, BUTTON3_VERTICAL_WIDTH, BUTTON3_VERTICAL_HEIGHT, BUTTON3_VERTICAL_RADIUS, language.get("in_out", "import_as_white"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(BUTTON3_VERTICAL_SX, BUTTON3_VERTICAL_3_SY, BUTTON3_VERTICAL_WIDTH, BUTTON3_VERTICAL_HEIGHT, BUTTON3_VERTICAL_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        done = false;
        failed = false;
        history_elem = getData().history_elem;
        Radio_button_element radio_button_elem;
        disc_radio.init();
        radio_button_elem.init(480, 120, getData().fonts.font, 15, language.get("edit_board", "black"), true);
        disc_radio.push(radio_button_elem);
        radio_button_elem.init(480, 140, getData().fonts.font, 15, language.get("edit_board", "white"), false);
        disc_radio.push(radio_button_elem);
        radio_button_elem.init(480, 160, getData().fonts.font, 15, language.get("edit_board", "empty"), false);
        disc_radio.push(radio_button_elem);

    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        int board_arr[HW2];
        history_elem.board.translate_to_arr(board_arr, history_elem.player);
        for (int cell = 0; cell < HW2; ++cell) {
            int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE;
            int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE;
            Rect cell_region(x, y, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
            if (cell_region.leftPressed()) {
                board_arr[cell] = disc_radio.checked;
            }
        }
        history_elem.board.translate_from_arr(board_arr, history_elem.player);
        if (KeyB.pressed()) {
            disc_radio.checked = BLACK;
        }
        else if (KeyW.pressed()) {
            disc_radio.checked = WHITE;
        }
        else if (KeyE.pressed()) {
            disc_radio.checked = VACANT;
        }
        Scene::SetBackground(getData().colors.green);
        getData().fonts.font(language.get("in_out", "edit_board")).draw(25, 480, 20, getData().colors.white);
        getData().fonts.font(language.get("in_out", "color")).draw(20, 480, 80, getData().colors.white);
        draw_board(getData().fonts, getData().colors, history_elem);
        disc_radio.draw();
        
        // 盤上の石数を計算(偶数なら黒番、奇数なら白番)
        int n_discs = history_elem.board.n_discs();
        bool is_black_turn = (n_discs % 2 == 0);
        
        // ボタンの描画
        set_black_button.draw();
        set_white_button.draw();
        back_button.draw();
        
        // 手番に応じてボタンの周りを白線で囲う
        if (is_black_turn) {
            RoundRect(BUTTON3_VERTICAL_SX - 6, BUTTON3_VERTICAL_1_SY - 6, BUTTON3_VERTICAL_WIDTH + 12, BUTTON3_VERTICAL_HEIGHT + 12, BUTTON3_VERTICAL_RADIUS + 6)
                .drawFrame(2, 0, getData().colors.black);
        } else {
            RoundRect(BUTTON3_VERTICAL_SX - 6, BUTTON3_VERTICAL_2_SY - 6, BUTTON3_VERTICAL_WIDTH + 12, BUTTON3_VERTICAL_HEIGHT + 12, BUTTON3_VERTICAL_RADIUS + 6)
                .drawFrame(2, 0, getData().colors.white);
        }
        
        if (back_button.clicked() || KeyEscape.pressed()) {
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        if (set_black_button.clicked() || (KeyEnter.pressed() && is_black_turn)) {
            process_import(BLACK);
        }
        if (set_white_button.clicked() || (KeyEnter.pressed() && !is_black_turn)) {
            process_import(WHITE);
        }
    }

    void draw() const override {

    }

private:
    void process_import(int player) {
        if (history_elem.player != player) {
            history_elem.board.pass();
        }
        history_elem.player = player;
        history_elem.v = GRAPH_IGNORE_VALUE;
        history_elem.level = -1;
        if (!history_elem.board.is_end() && history_elem.board.get_legal() == 0) {
            history_elem.board.pass();
            history_elem.player ^= 1;
        }
        int n_discs = history_elem.board.n_discs();
        int insert_place = (int)getData().graph_resources.nodes[getData().graph_resources.branch].size();
        int replace_place = -1;
        for (int i = 0; i < (int)getData().graph_resources.nodes[getData().graph_resources.branch].size(); ++i) {
            int node_n_discs = getData().graph_resources.nodes[getData().graph_resources.branch][i].board.n_discs();
            if (node_n_discs == n_discs) {
                replace_place = i;
                insert_place = -1;
                break;
            } else if (node_n_discs > n_discs) {
                insert_place = i;
                break;
            }
        }
        history_elem.policy = -1; // reset last policy
        if (replace_place - 1 >= 0) {
            uint64_t f_discs = getData().graph_resources.nodes[getData().graph_resources.branch][replace_place - 1].board.player | getData().graph_resources.nodes[getData().graph_resources.branch][replace_place - 1].board.opponent;
            uint64_t discs = history_elem.board.player | history_elem.board.opponent;
            if (pop_count_ull(discs ^ f_discs) == 1) {
                int last_policy = ctz(discs ^ f_discs);
                history_elem.policy = last_policy;
            }
        } else if (insert_place - 1 >= 0 && insert_place - 1 < getData().graph_resources.nodes[getData().graph_resources.branch].size()) {
            uint64_t f_discs = getData().graph_resources.nodes[getData().graph_resources.branch][insert_place - 1].board.player | getData().graph_resources.nodes[getData().graph_resources.branch][insert_place - 1].board.opponent;
            uint64_t discs = history_elem.board.player | history_elem.board.opponent;
            if (pop_count_ull(discs ^ f_discs) == 1) {
                int last_policy = ctz(discs ^ f_discs);
                history_elem.policy = last_policy;
            }
        } else {
            for (int i = 0; i < (int)getData().graph_resources.nodes[0].size(); ++i) {
                int node_n_discs = getData().graph_resources.nodes[0][i].board.n_discs();
                if (node_n_discs + 1 == n_discs) {
                    uint64_t f_discs = getData().graph_resources.nodes[0][i].board.player | getData().graph_resources.nodes[0][i].board.opponent;
                    uint64_t discs = history_elem.board.player | history_elem.board.opponent;
                    if (pop_count_ull(discs ^ f_discs) == 1) {
                        int last_policy = ctz(discs ^ f_discs);
                        history_elem.policy = last_policy;
                    }
                }
            }
        }
        if (replace_place != -1) {
            std::cerr << "replace" << std::endl;
            getData().graph_resources.nodes[getData().graph_resources.branch][replace_place] = history_elem;
        } else {
            std::cerr << "insert" << std::endl;
            getData().graph_resources.nodes[getData().graph_resources.branch].insert(getData().graph_resources.nodes[getData().graph_resources.branch].begin() + insert_place, history_elem);
        }
        getData().graph_resources.n_discs = n_discs;
        getData().graph_resources.need_init = false;
        getData().history_elem = getData().graph_resources.nodes[getData().graph_resources.branch].back();
        changeScene(U"Main_scene", SCENE_FADE_TIME);
    }
};

class Import_game : public App::Scene {
private:
    std::vector<Game_abstract> games;
    std::vector<Button> import_buttons;
    std::vector<ImageButton> delete_buttons;
    std::vector<ImageButton> edit_buttons;
    std::vector<ImageButton> folder_delete_buttons;
    Scroll_manager scroll_manager;
    Button back_button;
    Button up_button;
    Button add_folder_button;
    bool failed;
    // Explorer-like folder view
    std::vector<String> folders_display; // includes optional ".." at head
    explorer::PathState explorer_state; // manages current subfolder
    TextAreaEditState new_folder_area;
    Button create_folder_button;
    Button inline_edit_back_button;
    Button inline_edit_ok_button;
    TextAreaEditState folder_rename_area;
    bool renaming_folder = false;
    int renaming_folder_index = -1;
    int selected_folder_index = -1;
    String selected_folder_name;
    String renaming_folder_original_name;
    bool creating_folder = false;

public:
    Import_game(const InitData& init) : IScene{ init } {
        back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        add_folder_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "new_folder"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        create_folder_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "create"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        up_button.init(IMPORT_GAME_SX, IMPORT_GAME_SY - 30, 28, 24, 4, U"↑", 16, getData().fonts.font, getData().colors.white, getData().colors.black);
        inline_edit_back_button.init(0, 0, 80, 30, 8, language.get("common", "back"), 18, getData().fonts.font, getData().colors.white, getData().colors.black);
        inline_edit_ok_button.init(0, 0, 70, 30, 8, language.get("common", "ok"), 18, getData().fonts.font, getData().colors.white, getData().colors.black);
        failed = false;
        // Initialize current dir and load games
        explorer_state.clear();
        enumerate_current_dir();
        load_games();
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        getData().fonts.font(language.get("in_out", "input_game")).draw(25, Arg::center(X_CENTER, 30), getData().colors.white);
        // Current path label
        String path_label = explorer::compose_path_label(U"games/", explorer_state);
        getData().fonts.font(path_label).draw(15, Arg::rightCenter(IMPORT_GAME_SX + IMPORT_GAME_WIDTH, 30), getData().colors.white);
        bool enter_pressed = KeyEnter.down();
        bool escape_pressed = KeyEscape.down();

        if (creating_folder) {
            draw_folder_creation_overlay(enter_pressed, escape_pressed);
            return;
        }

        back_button.draw();
        if (back_button.clicked()) {
            if (renaming_folder) {
                cancel_folder_rename();
            } else {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        } else if (!renaming_folder && escape_pressed) {
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        draw_folder_management_ui();
        if (failed) {
            getData().fonts.font(language.get("in_out", "import_failed")).draw(20, Arg::center(X_CENTER, Y_CENTER), getData().colors.white);
        } else {
            ExplorerFolderInlineConfig inline_cfg{};
            inline_cfg.renaming = renaming_folder;
            inline_cfg.folder_index = renaming_folder ? renaming_folder_index : -1;
            // Always set inline config to show rename buttons
            inline_cfg.text_area = &folder_rename_area;
            inline_cfg.back_button = &inline_edit_back_button;
            inline_cfg.ok_button = &inline_edit_ok_button;
            inline_cfg.on_cancel = [this]() {
                cancel_folder_rename();
            };
            inline_cfg.on_commit = [this](const String& trimmed) {
                return confirm_folder_rename(trimmed);
            };
            const ExplorerFolderInlineConfig* inline_ptr = &inline_cfg;
            auto res = DrawExplorerList(
                folders_display, games, delete_buttons, edit_buttons, scroll_manager, up_button,
                IMPORT_GAME_HEIGHT, IMPORT_GAME_N_GAMES_ON_WINDOW, explorer_state.has_parent(), getData().fonts, getData().colors, getData().resources, language,
                getData().directories.document_dir, explorer_state.subfolder, inline_ptr);
            
            // Draw delete buttons for empty folders
            if (!renaming_folder) {
                int sy = IMPORT_GAME_SY + 8;
                int strt_idx_int = scroll_manager.get_strt_idx_int();
                int parent_offset = explorer_state.has_parent() ? 1 : 0;
                int row_index = parent_offset;  // Start after parent folder row
                
                for (int i = 0; i < (int)folders_display.size(); ++i) {
                    if (row_index >= strt_idx_int && row_index < strt_idx_int + IMPORT_GAME_N_GAMES_ON_WINDOW) {
                        if (is_folder_empty(folders_display[i])) {
                            int display_row = row_index - strt_idx_int;
                            int item_sy = sy + display_row * IMPORT_GAME_HEIGHT;
                            folder_delete_buttons[i].move(IMPORT_GAME_SX + 1, item_sy + 1);
                            folder_delete_buttons[i].draw();
                            if (folder_delete_buttons[i].clicked()) {
                                delete_folder(i);
                                return;
                            }
                        }
                    }
                    row_index++;
                }
            }
            
            if (res.upButtonClicked || res.parentFolderDoubleClicked) {
                if (navigate_to_parent_subfolder()) {
                    return;
                }
            }
            if (res.folderClicked) {
                select_folder(res.clickedFolder);
            }
            if (res.folderDoubleClicked) {
                if (renaming_folder) {
                    cancel_folder_rename();
                }
                explorer_state.navigate_to_child(res.clickedFolder);
                enumerate_current_dir();
                load_games();
                init_scroll_manager();
                return;
            }
            if (res.folderRenameRequested && res.folderRenameIndex >= 0) {
                begin_folder_rename(res.folderRenameIndex);
            }
            if (res.deleteClicked && res.deleteIndex >= 0) {
                delete_game(res.deleteIndex);
            }
            if (res.editClicked && res.editIndex >= 0) {
                edit_game(res.editIndex);
                return;
            }
            if (res.gameDoubleClicked && res.importIndex >= 0) {
                import_game(res.importIndex);
                return;
            }
            if (res.drop_completed) {
                handle_drop(res);
            } else if (res.reorderRequested) {
                handle_reorder(res);
            }
        }
        
        if (renaming_folder && escape_pressed) {
            cancel_folder_rename();
        }
    }

    void draw() const override {

    }

private:
    void init_scroll_manager() {
        int parent_offset = explorer_state.has_parent() ? 1 : 0;  // Add parent folder if not at root
        int total = parent_offset + (int)folders_display.size() + (int)games.size();
        scroll_manager.init(770, IMPORT_GAME_SY + 8, 10, IMPORT_GAME_HEIGHT * IMPORT_GAME_N_GAMES_ON_WINDOW, 20, total, IMPORT_GAME_N_GAMES_ON_WINDOW, IMPORT_GAME_SX, 73, IMPORT_GAME_WIDTH + 10, IMPORT_GAME_HEIGHT * IMPORT_GAME_N_GAMES_ON_WINDOW);
    }
    
    // Check if folder is empty (no subfolders and no games)
    bool is_folder_empty(const String& folder_name) const {
        std::string base_dir = get_root_dir();
        std::string rel_path = explorer_state.subfolder;
        if (!rel_path.empty()) {
            rel_path += "/";
        }
        rel_path += folder_name.narrow();
        
        // Check for subfolders
        std::vector<String> subfolders = explorer::enumerate_subfolders(base_dir, explorer::PathState{rel_path});
        if (!subfolders.empty()) {
            return false;
        }
        
        // Check for games in summary.csv
        String folder_dir = Unicode::Widen(base_dir);
        if (!rel_path.empty()) {
            folder_dir += U"/" + Unicode::Widen(rel_path);
        }
        String csv_path = folder_dir + U"/summary.csv";
        if (FileSystem::Exists(csv_path)) {
            CSV csv{ csv_path };
            if (csv.rows() > 0) {
                return false;
            }
        }
        return true;
    }
    
    // Delete folder (only if empty)
    void delete_folder(int idx) {
        if (idx < 0 || idx >= (int)folders_display.size()) return;
        
        const String& folder_name = folders_display[idx];
        if (!is_folder_empty(folder_name)) {
            return;  // Don't delete non-empty folders
        }
        
        String folder_path = get_base_dir() + folder_name;
        
        // Delete the folder
        if (FileSystem::IsDirectory(folder_path)) {
            FileSystem::Remove(folder_path, AllowUndo::No);
            enumerate_current_dir();
            load_games();
            std::cerr << "Deleted empty folder: " << folder_name << std::endl;
        }
    }

    bool navigate_to_parent_subfolder() {
        if (!explorer_state.navigate_to_parent()) {
            return false;
        }
        if (renaming_folder) {
            cancel_folder_rename();
        }
        enumerate_current_dir();
        load_games();
        init_scroll_manager();
        return true;
    }

    void edit_game(int idx) {
        const String json_path = get_base_dir() + games[idx].filename_date + U".json";
        JSON game_json = JSON::Load(json_path);
        if (not game_json) {
            std::cerr << "can't open game" << std::endl;
            failed = true;
            return;
        }
        
        // Load game information
        if (game_json[GAME_BLACK_PLAYER].getType() == JSONValueType::String) {
            getData().game_information.black_player_name = game_json[GAME_BLACK_PLAYER].getString();
        }
        if (game_json[GAME_WHITE_PLAYER].getType() == JSONValueType::String) {
            getData().game_information.white_player_name = game_json[GAME_WHITE_PLAYER].getString();
        }
        if (game_json[GAME_MEMO].getType() == JSONValueType::String) {
            getData().game_information.memo = game_json[GAME_MEMO].getString();
        }
        // Load date field (YYYY-MM-DD format)
        if (game_json[U"date"].getType() == JSONValueType::String) {
            getData().game_information.date = game_json[U"date"].getString();
        } else {
            // Fallback: generate from filename_date
            getData().game_information.date = games[idx].filename_date.substr(0, 10).replaced(U"_", U"-");
        }
        
        // Mark that a specific game has been loaded and store its location
        getData().game_information.is_game_loaded = true;
        
        // Set game editor info for editing mode
        getData().game_editor_info.return_scene = U"Import_game";
        getData().game_editor_info.is_editing_mode = true;
        getData().game_editor_info.game_date = games[idx].filename_date;
        getData().game_editor_info.subfolder = explorer_state.subfolder;
        getData().game_editor_info.game_info_updated = false;
        
        changeScene(U"Game_editor", SCENE_FADE_TIME);
    }

    void import_game(int idx) {
        const String json_path = get_base_dir() + games[idx].filename_date + U".json";
        if (!load_game_from_json(getData(), opening, json_path, games[idx].filename_date, explorer_state.subfolder)) {
            failed = true;
            return;
        }
        changeScene(U"Main_scene", SCENE_FADE_TIME);
    }

    void delete_game(int idx) {
        if (idx < 0 || idx >= (int)games.size()) {
            std::cerr << "delete_game: invalid index " << idx << std::endl;
            return;
        }
        
        const String json_path = get_base_dir() + games[idx].filename_date + U".json";
        FileSystem::Remove(json_path);

        const String csv_path = get_base_dir() + U"summary.csv";
        
        // CSVファイルの存在確認
        if (!FileSystem::Exists(csv_path)) {
            std::cerr << "CSV file not found: " << csv_path << std::endl;
            return;
        }
        
        CSV csv{ csv_path };
        
        // CSVの読み込み確認
        if (csv.rows() == 0) {
            std::cerr << "Warning: CSV file is empty" << std::endl;
            return;
        }
        
        CSV new_csv;
        
        // games配列はreverseされているので、CSV行インデックスを正しく計算
        int csv_row_to_delete = (int)games.size() - 1 - idx;
        
        for (int i = 0; i < (int)csv.rows(); ++i) {
            if (i != csv_row_to_delete) {
                // CSVの列数をチェックしてから書き込み
                if (csv[i].size() >= 6) {
                    for (int j = 0; j < 6; ++j) {
                        // 文字列の長さをチェックして制限する
                        String data = csv[i][j];
                        if (data.size() > 1000) { // 1000文字制限
                            data = data.substr(0, 1000) + U"...";
                            std::cerr << "Warning: Truncated long string in CSV row " << i << " column " << j << std::endl;
                        }
                        new_csv.write(data);
                    }
                    new_csv.newLine();
                } else {
                    std::cerr << "Warning: CSV row " << i << " has insufficient columns (" << csv[i].size() << ")" << std::endl;
                }
            }
        }
        
        // 一時ファイルに保存してから元ファイルを置き換える
        String temp_csv_path = csv_path + U".tmp";
        new_csv.save(temp_csv_path);
        
        // 一時ファイルの保存が成功したら元ファイルを置き換え
        if (FileSystem::Exists(temp_csv_path)) {
            FileSystem::Remove(csv_path);
            FileSystem::Copy(temp_csv_path, csv_path);
            FileSystem::Remove(temp_csv_path);
        } else {
            std::cerr << "Failed to save temporary CSV file" << std::endl;
            return;
        }

        games.erase(games.begin() + idx);
        import_buttons.erase(import_buttons.begin() + idx);
        delete_buttons.erase(delete_buttons.begin() + idx);
        double strt_idx_double = scroll_manager.get_strt_idx_double();
        init_scroll_manager();
        if ((int)strt_idx_double >= idx) {
            strt_idx_double -= 1.0;
        }
        scroll_manager.set_strt_idx(strt_idx_double);
        std::cerr << "deleted game " << idx << std::endl;
    }

    std::string get_root_dir() const {
        return explorer::build_root_dir_narrow(getData().directories.document_dir, "games");
    }

    // Helper: current base dir (games/ + optional subfolder + '/')
    String get_base_dir() const {
        return explorer::build_current_dir(get_root_dir(), explorer_state);
    }

    // Reload games list from summary.csv in current folder
    void load_games() {
        games.clear();
        import_buttons.clear();
        delete_buttons.clear();
        edit_buttons.clear();
        const String csv_path = get_base_dir() + U"summary.csv";
        const CSV csv{ csv_path };
        if (csv) {
            for (size_t row = 0; row < csv.rows(); ++row) {
                Game_abstract game_abstract;
                game_abstract.filename_date = csv[row][0];
                game_abstract.black_player = csv[row][1];
                game_abstract.white_player = csv[row][2];
                game_abstract.memo = csv[row][3];
                game_abstract.black_score = ParseOr<int32>(csv[row][4], GAME_DISCS_UNDEFINED);
                game_abstract.white_score = ParseOr<int32>(csv[row][5], GAME_DISCS_UNDEFINED);
                // Read game_date from column 6 (7th column), or generate from filename if missing
                if (csv[row].size() >= 7 && !csv[row][6].isEmpty()) {
                    game_abstract.game_date = csv[row][6];
                } else {
                    game_abstract.game_date = game_abstract.filename_date.substr(0, 10).replaced(U"_", U"-");
                }
                games.emplace_back(game_abstract);
            }
        }
        reverse(games.begin(), games.end());
        for (int i = 0; i < (int)games.size(); ++i) {
            Button button;
            button.init(0, 0, IMPORT_GAME_BUTTON_WIDTH, IMPORT_GAME_BUTTON_HEIGHT, IMPORT_GAME_BUTTON_RADIUS, language.get("in_out", "import"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
            import_buttons.emplace_back(button);
        }
        Texture delete_button_image = getData().resources.cross;
        Texture edit_button_image = getData().resources.pencil;
        for (int i = 0; i < (int)games.size(); ++i) {
            ImageButton delete_button;
            delete_button.init(0, 0, 15, delete_button_image);
            delete_buttons.emplace_back(delete_button);
            
            ImageButton edit_button;
            edit_button.init(0, 0, 18, edit_button_image);
            edit_buttons.emplace_back(edit_button);
        }
        init_scroll_manager();
    }

    // Enumerate current directory (folders_display only; parent is shown as a separate up icon)
    void enumerate_current_dir() {
        folders_display.clear();
        folder_delete_buttons.clear();
        std::vector<String> folders = explorer::enumerate_subfolders(get_root_dir(), explorer_state);
        Texture cross_image = getData().resources.cross;
        for (auto& folder : folders) {
            folders_display.emplace_back(folder);
            
            // Add delete button for each folder
            ImageButton delete_btn;
            delete_btn.init(0, 0, 15, cross_image);
            folder_delete_buttons.emplace_back(delete_btn);
        }
        
        if (!selected_folder_name.isEmpty()) {
            selected_folder_index = find_folder_index(selected_folder_name);
        } else {
            selected_folder_index = -1;
        }
        if (selected_folder_index == -1) {
            selected_folder_name.clear();
        }

        if (renaming_folder) {
            int idx = -1;
            if (!renaming_folder_original_name.isEmpty()) {
                idx = find_folder_index(renaming_folder_original_name);
            }
            if (idx >= 0) {
                renaming_folder_index = idx;
            } else if (renaming_folder_index < 0 || renaming_folder_index >= (int)folders_display.size()) {
                cancel_folder_rename();
            }
        }
        
        init_scroll_manager();
    }

    void select_folder(const String& folder_name) {
        selected_folder_index = find_folder_index(folder_name);
        if (selected_folder_index >= 0) {
            selected_folder_name = folders_display[selected_folder_index];
        } else {
            selected_folder_name.clear();
        }
    }

    int find_folder_index(const String& folder_name) const {
        for (int i = 0; i < (int)folders_display.size(); ++i) {
            if (folders_display[i] == folder_name) {
                return i;
            }
        }
        return -1;
    }

    void draw_folder_management_ui() {
        if (renaming_folder) {
            add_folder_button.disable_notransparent();
        } else {
            add_folder_button.enable();
        }
        add_folder_button.draw();
        if (!renaming_folder && add_folder_button.clicked()) {
            creating_folder = true;
            cancel_folder_rename();
            new_folder_area.text.clear();
            new_folder_area.cursorPos = 0;
            new_folder_area.rebuildGlyphs();
            new_folder_area.active = true;
        }
    }

    void draw_folder_creation_overlay(bool enter_pressed, bool escape_pressed) {
        auto& fonts = getData().fonts;
        auto& colors = getData().colors;

        back_button.draw();
        if (back_button.clicked() || escape_pressed) {
            creating_folder = false;
            new_folder_area.text.clear();
            new_folder_area.cursorPos = 0;
            new_folder_area.rebuildGlyphs();
            new_folder_area.active = false;
            return;
        }

        const double center_y = NEW_FOLDER_PANEL_SY + NEW_FOLDER_PANEL_HEIGHT / 2.0;
        const double label_x = IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + NEW_FOLDER_LABEL_INNER_MARGIN;

        fonts.font(language.get("in_out", "new_folder") + U":").draw(20, Arg::leftCenter(label_x, center_y), colors.white);

        Vec2 text_pos{
            IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + NEW_FOLDER_TEXTBOX_OFFSET_X,
            center_y + NEW_FOLDER_TEXTBOX_OFFSET_Y
        };
        SizeF text_size{ NEW_FOLDER_TEXTBOX_WIDTH, NEW_FOLDER_TEXTBOX_HEIGHT };
        SimpleGUI::TextArea(new_folder_area, text_pos, text_size, SimpleGUI::PreferredTextAreaMaxChars);
        gui_list::sanitize_text_area(new_folder_area);

        String folder_name = new_folder_area.text.trimmed();
        bool can_create = gui_list::is_valid_folder_name(folder_name);

        create_folder_button.move(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY);
        if (can_create) {
            create_folder_button.enable();
        } else {
            create_folder_button.disable();
        }
        create_folder_button.draw();

        if (can_create && create_folder_button.clicked()) {
            if (handle_create_folder()) {
                creating_folder = false;
                new_folder_area.active = false;
            }
        }
    }

    bool handle_create_folder() {
        if (renaming_folder) {
            return false;
        }
        String input = gui_list::sanitize_folder_text(new_folder_area.text).trimmed();
        gui_list::sanitize_text_area(new_folder_area);
        if (!gui_list::is_valid_folder_name(input)) {
            return false;
        }
        bool created = gui_list::create_folder_with_initializer(
            get_base_dir(),
            input,
            [](const String& dir) {
                CSV csv;
                csv.save(dir + U"summary.csv");
            }
        );
        if (created) {
            new_folder_area.text.clear();
            new_folder_area.cursorPos = 0;
            new_folder_area.rebuildGlyphs();
            enumerate_current_dir();
            load_games();
            select_folder(input);
        }
        return created;
    }

    void begin_folder_rename(int folder_idx) {
        if (folder_idx < 0 || folder_idx >= (int)folders_display.size()) {
            return;
        }
        renaming_folder = true;
        renaming_folder_index = folder_idx;
        renaming_folder_original_name = folders_display[folder_idx];
        folder_rename_area.text = folders_display[folder_idx];
        folder_rename_area.cursorPos = folder_rename_area.text.size();
        folder_rename_area.rebuildGlyphs();
        folder_rename_area.active = true;
        selected_folder_index = folder_idx;
        selected_folder_name = folders_display[folder_idx];
    }

    void cancel_folder_rename() {
        renaming_folder = false;
        renaming_folder_index = -1;
        renaming_folder_original_name.clear();
        folder_rename_area.text.clear();
        folder_rename_area.cursorPos = 0;
        folder_rename_area.rebuildGlyphs();
        folder_rename_area.active = false;
    }

    bool confirm_folder_rename(const String& target_name) {
        if (!renaming_folder || renaming_folder_index < 0 || renaming_folder_index >= (int)folders_display.size()) {
            return false;
        }
        String trimmed = target_name.trimmed();
        if (!gui_list::is_valid_folder_name(trimmed)) {
            return false;
        }
        String current_name = folders_display[renaming_folder_index];
        if (current_name == trimmed) {
            cancel_folder_rename();
            return true;
        }
        String base_dir = get_base_dir();
        bool renamed = gui_list::rename_folder_in_directory(base_dir, current_name, trimmed);
        if (!renamed) {
            std::cerr << "rename_folder_in_directory failed: base='" << base_dir.narrow()
                      << "' current='" << current_name.narrow() << "' target='" << trimmed.narrow() << "'" << std::endl;
            return false;
        }
        if (renamed) {
            cancel_folder_rename();
            enumerate_current_dir();
            select_folder(trimmed);
        }
        return renamed;
    }

    // Handle drag and drop operations
    void handle_drop(const ExplorerDrawResult& res) {
        if (res.drop_on_parent) {
            // Handle drop on parent folder - move to parent directory
            if (res.is_dragging_game && res.dragged_game_index >= 0 && res.dragged_game_index < (int)games.size()) {
                move_game_to_parent(res.dragged_game_index);
            } else if (res.is_dragging_folder && !res.dragged_folder_name.empty()) {
                move_folder_to_parent(res.dragged_folder_name.narrow());
            }
        } else {
            // Handle normal folder drop
            if (res.is_dragging_game && res.dragged_game_index >= 0 && res.dragged_game_index < (int)games.size()) {
                // Move game to target folder
                move_game_to_folder(res.dragged_game_index, res.drop_target_folder.narrow());
            } else if (res.is_dragging_folder && !res.dragged_folder_name.empty()) {
                // Move folder to target folder
                move_folder_to_folder(res.dragged_folder_name.narrow(), res.drop_target_folder.narrow());
            }
        }
    }

    void handle_reorder(const ExplorerDrawResult& res) {
        if (!res.reorderRequested || !res.is_dragging_game) {
            return;
        }
        if (res.reorderFrom < 0 || res.reorderFrom >= (int)games.size()) {
            return;
        }
        int insert_idx = std::clamp(res.reorderTo, 0, (int)games.size());
        bool changed = gui_list::reorder_parallel(games, res.reorderFrom, insert_idx, import_buttons, delete_buttons, edit_buttons);
        if (!changed) {
            return;
        }
        persist_games_order_to_csv();
    }
    
    // Move a game to a different folder (relative to current subfolder)
    void move_game_to_folder(int game_index, const std::string& target_folder) {
        if (game_index < 0 || game_index >= (int)games.size()) return;
        
        const Game_abstract& game = games[game_index];
        
        // Source and target paths
        String source_base = get_base_dir();
        String target_base = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!explorer_state.subfolder.empty()) {
            target_base += Unicode::Widen(explorer_state.subfolder) + U"/";
        }
        if (!target_folder.empty()) {
            target_base += Unicode::Widen(target_folder) + U"/";
        }
        
        std::cerr << "Moving game (relative):" << std::endl;
        std::cerr << "  Current subfolder: " << explorer_state.subfolder << std::endl;
        std::cerr << "  Target folder: " << target_folder << std::endl;
        std::cerr << "  Source base: " << source_base.narrow() << std::endl;
        std::cerr << "  Target base: " << target_base.narrow() << std::endl;
        
        // Ensure target directory exists
        if (!FileSystem::Exists(target_base)) {
            FileSystem::CreateDirectories(target_base);
        }
        
        // Move JSON file
        String source_json = source_base + game.filename_date + U".json";
        String target_json = target_base + game.filename_date + U".json";
        if (FileSystem::Exists(source_json)) {
            FileSystem::Copy(source_json, target_json);
            FileSystem::Remove(source_json);
        }
        
        // Update CSV files
        remove_game_from_csv(game_index);
        add_game_to_target_csv(game, target_base);
        
        // Refresh current view
        load_games();
        init_scroll_manager();
        
        std::cerr << "Moved game " << game.filename_date.narrow() << " to " << target_folder << std::endl;
    }

    // Move a game to an absolute folder path (from root)
    void move_game_to_absolute_folder(int game_index, const std::string& target_folder) {
        if (game_index < 0 || game_index >= (int)games.size()) return;
        
        const Game_abstract& game = games[game_index];
        
        // Source and target paths
        String source_base = get_base_dir();
        String target_base = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!target_folder.empty()) {
            target_base += Unicode::Widen(target_folder) + U"/";
        }
        
        std::cerr << "Moving game (absolute):" << std::endl;
        std::cerr << "  Current subfolder: " << explorer_state.subfolder << std::endl;
        std::cerr << "  Target folder: " << target_folder << std::endl;
        std::cerr << "  Source base: " << source_base.narrow() << std::endl;
        std::cerr << "  Target base: " << target_base.narrow() << std::endl;
        
        // Ensure target directory exists
        if (!FileSystem::Exists(target_base)) {
            FileSystem::CreateDirectories(target_base);
        }
        
        // Move JSON file
        String source_json = source_base + game.filename_date + U".json";
        String target_json = target_base + game.filename_date + U".json";
        if (FileSystem::Exists(source_json)) {
            FileSystem::Copy(source_json, target_json);
            FileSystem::Remove(source_json);
        }
        
        // Update CSV files
        remove_game_from_csv(game_index);
        add_game_to_target_csv(game, target_base);
        
        // Refresh current view
        load_games();
        init_scroll_manager();
        
        std::cerr << "Moved game " << game.filename_date.narrow() << " to " << target_folder << std::endl;
    }
    
    // Move a folder to a different folder (relative to current subfolder)
    void move_folder_to_folder(const std::string& source_folder, const std::string& target_folder) {
        // Build source path: current subfolder + source_folder
        String source_path = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!explorer_state.subfolder.empty()) {
            source_path += Unicode::Widen(explorer_state.subfolder) + U"/";
        }
        source_path += Unicode::Widen(source_folder);
        
        // Build target parent path: current subfolder + target_folder
        String target_parent = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!explorer_state.subfolder.empty()) {
            target_parent += Unicode::Widen(explorer_state.subfolder) + U"/";
        }
        if (!target_folder.empty()) {
            target_parent += Unicode::Widen(target_folder) + U"/";
        }
        String target_path = target_parent + Unicode::Widen(source_folder);
        
        std::cerr << "Moving folder (relative):" << std::endl;
        std::cerr << "  Current subfolder: " << explorer_state.subfolder << std::endl;
        std::cerr << "  Source folder: " << source_folder << std::endl;
        std::cerr << "  Target folder: " << target_folder << std::endl;
        std::cerr << "  Source path: " << source_path.narrow() << std::endl;
        std::cerr << "  Target path: " << target_path.narrow() << std::endl;
        
        // Check if source and target are different
        if (source_path == target_path) {
            std::cerr << "  Source and target are the same, skipping move" << std::endl;
            return;
        }
        
        // Check if target folder would create a circular reference
        String source_abs = FileSystem::FullPath(source_path);
        String target_abs = FileSystem::FullPath(target_parent);
        if (target_abs.starts_with(source_abs)) {
            std::cerr << "  Cannot move folder into its own subdirectory" << std::endl;
            return;
        }
        
        // Ensure target parent directory exists
        if (!FileSystem::Exists(target_parent)) {
            FileSystem::CreateDirectories(target_parent);
        }
        
        // Move the entire folder
        if (FileSystem::Exists(source_path) && !FileSystem::Exists(target_path)) {
            // Use system command for folder move (more reliable)
            std::string cmd = "move \"" + source_path.narrow() + "\" \"" + target_path.narrow() + "\"";
            int result = system(cmd.c_str());
            
            if (result == 0) {
                std::cerr << "  Successfully moved folder " << source_folder << " to " << target_folder << std::endl;
            } else {
                std::cerr << "  Failed to move folder (error code: " << result << ")" << std::endl;
            }
            
            // Refresh current view
            enumerate_current_dir();
            load_games();
            init_scroll_manager();
        } else {
            if (!FileSystem::Exists(source_path)) {
                std::cerr << "  Source folder does not exist" << std::endl;
            } else if (FileSystem::Exists(target_path)) {
                std::cerr << "  Target folder already exists" << std::endl;
            }
        }
    }

    // Move a folder to an absolute folder path (from root)
    void move_folder_to_absolute_folder(const std::string& source_folder, const std::string& target_folder) {
        // Build source path: current subfolder + source_folder
        String source_path = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!explorer_state.subfolder.empty()) {
            source_path += Unicode::Widen(explorer_state.subfolder) + U"/";
        }
        source_path += Unicode::Widen(source_folder);
        
        // Build target parent path based on target_folder
        String target_parent = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!target_folder.empty()) {
            target_parent += Unicode::Widen(target_folder) + U"/";
        }
        String target_path = target_parent + Unicode::Widen(source_folder);
        
        std::cerr << "Moving folder (absolute):" << std::endl;
        std::cerr << "  Current subfolder: " << explorer_state.subfolder << std::endl;
        std::cerr << "  Source folder: " << source_folder << std::endl;
        std::cerr << "  Target folder: " << target_folder << std::endl;
        std::cerr << "  Source path: " << source_path.narrow() << std::endl;
        std::cerr << "  Target path: " << target_path.narrow() << std::endl;
        
        // Check if source and target are different
        if (source_path == target_path) {
            std::cerr << "  Source and target are the same, skipping move" << std::endl;
            return;
        }
        
        // Check if target folder would create a circular reference
        String source_abs = FileSystem::FullPath(source_path);
        String target_abs = FileSystem::FullPath(target_parent);
        if (target_abs.starts_with(source_abs)) {
            std::cerr << "  Cannot move folder into its own subdirectory" << std::endl;
            return;
        }
        
        // Ensure target parent directory exists
        if (!FileSystem::Exists(target_parent)) {
            FileSystem::CreateDirectories(target_parent);
        }
        
        // Move the entire folder
        if (FileSystem::Exists(source_path) && !FileSystem::Exists(target_path)) {
            // Use system command for folder move (more reliable)
            std::string cmd = "move \"" + source_path.narrow() + "\" \"" + target_path.narrow() + "\"";
            int result = system(cmd.c_str());
            
            if (result == 0) {
                std::cerr << "  Successfully moved folder " << source_folder << " to " << target_folder << std::endl;
            } else {
                std::cerr << "  Failed to move folder (error code: " << result << ")" << std::endl;
            }
            
            // Refresh current view
            enumerate_current_dir();
            load_games();
            init_scroll_manager();
        } else {
            if (!FileSystem::Exists(source_path)) {
                std::cerr << "  Source folder does not exist" << std::endl;
            } else if (FileSystem::Exists(target_path)) {
                std::cerr << "  Target folder already exists" << std::endl;
            }
        }
    }

    // Move a game to parent folder
    void move_game_to_parent(int game_index) {
        if (!explorer_state.has_parent()) return;  // Already at root
        
        // Get parent folder path
        std::string parent_folder = explorer_state.subfolder;
        if (!parent_folder.empty() && parent_folder.back() == '/') parent_folder.pop_back();
        size_t pos = parent_folder.find_last_of('/');
        if (pos == std::string::npos) parent_folder.clear();
        else parent_folder = parent_folder.substr(0, pos);
        
        move_game_to_absolute_folder(game_index, parent_folder);
    }
    
    // Move a folder to parent folder
    void move_folder_to_parent(const std::string& folder_name) {
        if (!explorer_state.has_parent()) return;  // Already at root
        
        // Get parent folder path
        std::string parent_folder = explorer_state.subfolder;
        if (!parent_folder.empty() && parent_folder.back() == '/') parent_folder.pop_back();
        size_t pos = parent_folder.find_last_of('/');
        if (pos == std::string::npos) parent_folder.clear();
        else parent_folder = parent_folder.substr(0, pos);
        
        move_folder_to_absolute_folder(folder_name, parent_folder);
    }
    
    // Remove game from current CSV
    void remove_game_from_csv(int game_index) {
        const String csv_path = get_base_dir() + U"summary.csv";
        CSV csv{ csv_path };
        CSV new_csv;
        
        int csv_row_to_remove = (int)games.size() - 1 - game_index;
        
        for (int i = 0; i < (int)csv.rows(); ++i) {
            if (i != csv_row_to_remove && csv[i].size() >= 6) {
                for (int j = 0; j < 6; ++j) {
                    new_csv.write(csv[i][j]);
                }
                new_csv.newLine();
            }
        }
        new_csv.save(csv_path);
    }
    
    // Add game to target folder's CSV
    void add_game_to_target_csv(const Game_abstract& game, const String& target_base) {
        String target_csv = target_base + U"summary.csv";
        CSV csv{ target_csv };
        
        // Create new CSV with existing data plus new game
        CSV new_csv;
        for (int i = 0; i < (int)csv.rows(); ++i) {
            if (csv[i].size() >= 1) {
                size_t cols = std::min(csv[i].size(), size_t(6));
                for (size_t j = 0; j < cols; ++j) {
                    new_csv.write(csv[i][j]);
                }
                // Add 7th column (game_date) if missing
                if (csv[i].size() < 7) {
                    String old_date = csv[i][0].substr(0, 10).replaced(U"_", U"-");
                    new_csv.write(old_date);
                } else {
                    new_csv.write(csv[i][6]);
                }
                new_csv.newLine();
            }
        }
        
        // Add new game entry
        new_csv.write(game.filename_date);
        new_csv.write(game.black_player);
        new_csv.write(game.white_player);
        new_csv.write(game.memo);
        new_csv.write(game.black_score == GAME_DISCS_UNDEFINED ? U"" : ToString(game.black_score));
        new_csv.write(game.white_score == GAME_DISCS_UNDEFINED ? U"" : ToString(game.white_score));
        new_csv.write(game.game_date);
        new_csv.newLine();
        
        new_csv.save(target_csv);
    }

    void persist_games_order_to_csv() {
        const String csv_path = get_base_dir() + U"summary.csv";
        CSV new_csv;
        for (int i = (int)games.size() - 1; i >= 0; --i) {
            const auto& game = games[i];
            new_csv.write(game.filename_date);
            new_csv.write(game.black_player);
            new_csv.write(game.white_player);
            new_csv.write(game.memo);
            new_csv.write(game.black_score == GAME_DISCS_UNDEFINED ? U"" : ToString(game.black_score));
            new_csv.write(game.white_score == GAME_DISCS_UNDEFINED ? U"" : ToString(game.white_score));
            new_csv.write(game.game_date);
            new_csv.newLine();
        }
        new_csv.save(csv_path);
    }
};


class Import_bitboard : public App::Scene {
private:
    Button single_back_button;
    Button back_button;
    Button import_button;
    TextAreaEditState text_area[2];
    std::string player_string;
    std::string opponent_string;
    Radio_button player_radio;
    Board board;
    bool done;
    bool failed;

public:
    Import_bitboard(const InitData& init) : IScene{ init } {
        single_back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        import_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "import"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        Radio_button_element radio_button_elem;
        player_radio.init();
        radio_button_elem.init(X_CENTER, 70 + SCENE_ICON_WIDTH + 175, getData().fonts.font, 20, language.get("common", "black"), true);
        player_radio.push(radio_button_elem);
        radio_button_elem.init(X_CENTER, 70 + SCENE_ICON_WIDTH + 201, getData().fonts.font, 20, language.get("common", "white"), false);
        player_radio.push(radio_button_elem);
        done = false;
        failed = false;
        text_area[0].active = true;
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        const int icon_width = SCENE_ICON_WIDTH;
        getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
        getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
        int sy = 20 + icon_width + 50;
        if (!done) {
            getData().fonts.font(language.get("in_out", "input_bitboard")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            const int text_area_y[2] = {sy + 40, sy + 90};
            constexpr int text_area_h = 40;
            constexpr int circle_radius = 15;
            for (int i = 0; i < 2; ++i) {
                SimpleGUI::TextArea(text_area[i], Vec2{X_CENTER - 300, text_area_y[i]}, SizeF{600, text_area_h}, SimpleGUI::PreferredTextAreaMaxChars);
                if (player_radio.checked == i) {
                    Circle(X_CENTER + 330, text_area_y[i] + text_area_h / 2, circle_radius).draw(getData().colors.black);
                } else {
                    Circle(X_CENTER + 330, text_area_y[i] + text_area_h / 2, circle_radius).draw(getData().colors.white);
                }
            }
            getData().fonts.font(language.get("in_out", "player")).draw(20, Arg::rightCenter(X_CENTER - 310, text_area_y[0] + text_area_h / 2), getData().colors.white);
            getData().fonts.font(language.get("in_out", "opponent")).draw(20, Arg::rightCenter(X_CENTER - 310, text_area_y[1] + text_area_h / 2), getData().colors.white);
            getData().fonts.font(language.get("in_out", "you_can_paste_with_ctrl_v")).draw(13, Arg::topCenter(X_CENTER, sy + 140), getData().colors.white);
            getData().fonts.font(language.get("in_out", "player")).draw(25, Arg::rightCenter(X_CENTER - 5, 70 + SCENE_ICON_WIDTH + 188), getData().colors.white);
            player_radio.draw();
            for (int i = 0; i < 2; ++i) {
                std::string str = text_area[i].text.narrow();
                if (str.find("\t") != std::string::npos) {
                    text_area[i].active = false;
                    text_area[(i + 1) % 2].active = true;
                    int tab_place = str.find("\t");
                    std::string txt0;
                    for (int j = 0; j < tab_place; ++j) {
                        txt0 += str[j];
                    }
                    std::string txt1;
                    for (int j = tab_place + 1; j < (int)str.size(); ++j) {
                        txt1 += str[j];
                    }
                    text_area[i].text = Unicode::Widen(txt0);
                    text_area[i].cursorPos = text_area[i].text.size();
                    text_area[i].rebuildGlyphs();
                    text_area[(i + 1) % 2].text += Unicode::Widen(txt1);
                    text_area[(i + 1) % 2].cursorPos = text_area[(i + 1) % 2].text.size();
                    text_area[(i + 1) % 2].rebuildGlyphs();
                }
            }
            bool return_pressed = false;
            for (int i = 0; i < 2; ++i) {
                if (text_area[i].text.size()) {
                    if (text_area[i].text[text_area[i].text.size() - 1] == '\n') {
                        return_pressed = true;
                    }
                }
            }
            player_string = text_area[0].text.replaced(U"\r", U"").replaced(U"\n", U"").replaced(U" ", U"").replace(U"\t", U"").narrow();
            opponent_string = text_area[1].text.replaced(U"\r", U"").replaced(U"\n", U"").replaced(U" ", U"").replace(U"\t", U"").narrow();
            back_button.draw();
            import_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
            if (import_button.clicked() || KeyEnter.pressed()) {
                failed = import_bitboard_processing();
                done = true;
            }
        }
        else {
            if (!failed) {
                getData().graph_resources.init();
                History_elem history_elem;
                history_elem.player = player_radio.checked;
                history_elem.board = board;
                getData().graph_resources.nodes[0].emplace_back(history_elem);
                getData().graph_resources.n_discs = board.n_discs();
                getData().game_information.init();
                getData().graph_resources.need_init = false;
                getData().history_elem = getData().graph_resources.nodes[0].back();
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            } else {
                getData().fonts.font(language.get("in_out", "import_failed")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
                single_back_button.draw();
                if (single_back_button.clicked() || KeyEscape.pressed()) {
                    changeScene(U"Main_scene", SCENE_FADE_TIME);
                }
            }
        }
    }

    void draw() const override {

    }

private:
    bool import_bitboard_processing() {
        if (player_string[0] == '0' && player_string[1] == 'x') {
            player_string = player_string.substr(2);
        }
        if (opponent_string[0] == '0' && opponent_string[1] == 'x') {
            opponent_string = opponent_string.substr(2);
        }
        while (player_string.size() < 16) {
            player_string = "0" + player_string;
        }
        while (opponent_string.size() < 16) {
            opponent_string = "0" + opponent_string;
        }
        board.player = 0;
        board.opponent = 0;
        for (int i = 0; i < 16; ++i) {
            player_string[i] |= 0x20; // lower case
            opponent_string[i] |= 0x20; // lower case
            if (player_string[i] < '0' || ('9' < player_string[i] && player_string[i] < 'a') || 'f' < player_string[i]) { // out of range
                std::cerr << "player out of range" << std::endl;
                return true;
            }
            if (opponent_string[i] < '0' || ('9' < opponent_string[i] && opponent_string[i] < 'a') || 'f' < opponent_string[i]) { // out of range
                std::cerr << "opponent out of range" << std::endl;
                return true;
            }
            int p_int = player_string[i] % 87 % 48; // 0-9, a-f hex to decimal
            int o_int = opponent_string[i] % 87 % 48; // 0-9, a-f hex to decimal
            if (p_int & o_int) { // 2 discs on same square
                std::cerr << "both discs on same square" << std::endl;
                return true;
            }
            for (int j = 0; j < 4; ++j) {
                int idx = HW2_M1 - (i * 4 + 3 - j);
                board.player |= (uint64_t)(p_int & 1) << idx;
                board.opponent |= (uint64_t)(o_int & 1) << idx;
                p_int /= 2;
                o_int /= 2;
            }
        }
        if (!board.is_end() && board.get_legal() == 0) {
            board.pass();
        }
        return false;
    }
};

// Load game from JSON and prepare for display
// Returns true on success, false on failure
inline bool load_game_from_json(
    Common_resources& data,
    Opening& opening,
    const String& json_path,
    const String& game_date,
    const std::string& subfolder
) {
    JSON game_json = JSON::Load(json_path);
    if (!game_json) {
        std::cerr << "Failed to load game JSON: " << json_path.narrow() << std::endl;
        return false;
    }
    
    // Load game information
    if (game_json[GAME_BLACK_PLAYER].getType() == JSONValueType::String) {
        data.game_information.black_player_name = game_json[GAME_BLACK_PLAYER].getString();
    }
    if (game_json[GAME_WHITE_PLAYER].getType() == JSONValueType::String) {
        data.game_information.white_player_name = game_json[GAME_WHITE_PLAYER].getString();
    }
    if (game_json[GAME_MEMO].getType() == JSONValueType::String) {
        data.game_information.memo = game_json[GAME_MEMO].getString();
    }
    // Load date field
    if (game_json[U"date"].getType() == JSONValueType::String) {
        data.game_information.date = game_json[U"date"].getString();
    } else {
        data.game_information.date = game_date.substr(0, 10).replaced(U"_", U"-");
    }
    
    // Mark that a specific game has been loaded and store its location
    data.game_information.is_game_loaded = true;
    data.game_editor_info.game_date = game_date;
    data.game_editor_info.subfolder = subfolder;
    
    // Load history
    data.graph_resources.nodes[GRAPH_MODE_NORMAL].clear();
    data.graph_resources.nodes[GRAPH_MODE_INSPECT].clear();
    for (int n_discs = 4; n_discs <= HW2; ++n_discs) {
        String n_discs_str = Format(n_discs);
        History_elem history_elem;
        bool error_found = false;
        if (game_json[n_discs_str][GAME_BOARD_PLAYER].getType() == JSONValueType::Number) {
            history_elem.board.player = game_json[n_discs_str][GAME_BOARD_PLAYER].get<uint64_t>();
        } else {
            error_found = true;
        }
        if (game_json[n_discs_str][GAME_BOARD_OPPONENT].getType() == JSONValueType::Number) {
            history_elem.board.opponent = game_json[n_discs_str][GAME_BOARD_OPPONENT].get<uint64_t>();
        } else {
            error_found = true;
        }
        if (game_json[n_discs_str][GAME_LEVEL].getType() == JSONValueType::Number) {
            history_elem.level = game_json[n_discs_str][GAME_LEVEL].get<int>();
        } else {
            error_found = true;
        }
        if (game_json[n_discs_str][GAME_PLAYER].getType() == JSONValueType::Number) {
            history_elem.player = game_json[n_discs_str][GAME_PLAYER].get<int>();
        } else {
            error_found = true;
        }
        if (game_json[n_discs_str][GAME_POLICY].getType() == JSONValueType::Number) {
            history_elem.policy = game_json[n_discs_str][GAME_POLICY].get<int>();
        } else {
            error_found = true;
        }
        if (game_json[n_discs_str][GAME_NEXT_POLICY].getType() == JSONValueType::Number) {
            history_elem.next_policy = game_json[n_discs_str][GAME_NEXT_POLICY].get<int>();
        } else {
            error_found = true;
        }
        if (game_json[n_discs_str][GAME_VALUE].getType() == JSONValueType::Number) {
            history_elem.v = game_json[n_discs_str][GAME_VALUE].get<int>();
        } else {
            error_found = true;
        }
        if (!error_found) {
            data.graph_resources.nodes[GRAPH_MODE_NORMAL].emplace_back(history_elem);
        }
    }
    
    // Add opening names
    std::string opening_name, n_opening_name;
    for (int i = 0; i < (int)data.graph_resources.nodes[GRAPH_MODE_NORMAL].size(); ++i) {
        n_opening_name.clear();
        n_opening_name = opening.get(data.graph_resources.nodes[GRAPH_MODE_NORMAL][i].board, data.graph_resources.nodes[GRAPH_MODE_NORMAL][i].player ^ 1);
        if (n_opening_name.size()) {
            opening_name = n_opening_name;
        }
        data.graph_resources.nodes[GRAPH_MODE_NORMAL][i].opening_name = opening_name;
    }
    
    // Set up final state
    data.graph_resources.n_discs = data.graph_resources.nodes[GRAPH_MODE_NORMAL].back().board.n_discs();
    data.graph_resources.need_init = false;
    data.history_elem = data.graph_resources.nodes[GRAPH_MODE_NORMAL].back();
    
    return true;
}