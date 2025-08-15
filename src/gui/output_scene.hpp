﻿/*
    Egaroucid Project

    @file output.hpp
        Output scenes
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"
#include "draw.hpp"

class Export_game : public App::Scene {
private:
    Button back_button;
    Button export_main_button;
    Button export_this_board_button;
    TextAreaEditState text_area[3]; // black player, white player, memo
    static constexpr int BLACK_PLAYER_IDX = 0;
    static constexpr int WHITE_PLAYER_IDX = 1;
    static constexpr int MEMO_IDX = 2;
    // Folder picker overlay state (explorer-like under games/)
    bool show_folder_picker = false;
    std::vector<History_elem> pending_history; // history to save after selection
    // current folder path under games/
    std::string subfolder; // final chosen subfolder used for saving
    std::string picker_subfolder; // navigating path during selection
    std::vector<String> save_folders_display; // includes optional ".."
    std::vector<Game_abstract> picker_games; // games in current picker folder
    bool picker_has_parent = false;
    Scroll_manager folder_scroll_manager;
    // new folder UI
    TextAreaEditState new_folder_area;
    Button create_folder_button;
    Button save_here_button;
    Button cancel_picker_button;
    Button up_button;
    Button open_explorer_button;
    // Saving state
    bool is_saving = false;
    bool saving_started = false;


public:
    Export_game(const InitData& init) : IScene{ init } {
        back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        export_main_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "export_main"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        export_this_board_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "export_until_this_board"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
        text_area[BLACK_PLAYER_IDX].active = true;
        text_area[BLACK_PLAYER_IDX].text = getData().game_information.black_player_name;
        text_area[WHITE_PLAYER_IDX].text = getData().game_information.white_player_name;
        text_area[MEMO_IDX].text = getData().game_information.memo;
        for (int i = 0; i < 3; ++i) {
            text_area[i].rebuildGlyphs();
        }
        // init folder picker buttons (positions used only in overlay)
        create_folder_button.init(600, EXPORT_GAME_CREATE_FOLDER_Y_CENTER - 28 / 2, 120, 28, 8, language.get("in_out", "create"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
        save_here_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "save_here"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        cancel_picker_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        up_button.init(IMPORT_GAME_SX, IMPORT_GAME_SY - 30, 28, 24, 4, U"↑", 16, getData().fonts.font, getData().colors.white, getData().colors.black);
        open_explorer_button.init(IMPORT_GAME_SX + IMPORT_GAME_WIDTH - 150, IMPORT_GAME_SY - 30, 150, 24, 5, language.get("in_out", "open_explorer"), 13, getData().fonts.font, getData().colors.white, getData().colors.black);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);

        // Saving mode: handled first
        if (is_saving) {
            getData().fonts.font(language.get("in_out", "saving")).draw(30, Arg::center(X_CENTER, Y_CENTER), getData().colors.white);
            if (!saving_started) {
                saving_started = true; // ensure one frame shows
                return;
            }
            export_game(pending_history);
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
            return;
        }

        if (!show_folder_picker) {
            getData().fonts.font(language.get("in_out", "output_game")).draw(25, Arg::topCenter(X_CENTER, 10), getData().colors.white);
            getData().fonts.font(language.get("in_out", "player_name")).draw(15, Arg::topCenter(X_CENTER, 47), getData().colors.white);
            SimpleGUI::TextArea(text_area[BLACK_PLAYER_IDX], Vec2{X_CENTER - EXPORT_GAME_PLAYER_WIDTH, 70}, SizeF{EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT}, SimpleGUI::PreferredTextAreaMaxChars);
            SimpleGUI::TextArea(text_area[WHITE_PLAYER_IDX], Vec2{X_CENTER, 70}, SizeF{EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT}, SimpleGUI::PreferredTextAreaMaxChars);
            Circle(X_CENTER - EXPORT_GAME_PLAYER_WIDTH - EXPORT_GAME_RADIUS - 20, 70 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.black);
            Circle(X_CENTER + EXPORT_GAME_PLAYER_WIDTH + EXPORT_GAME_RADIUS + 20, 70 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.white);
            // Memo label / counter / textbox (slightly higher and smaller)
            const int memo_label_y = 110;
            const int memo_box_y = 130;
            getData().fonts.font(language.get("in_out", "memo")).draw(15, Arg::topCenter(X_CENTER, memo_label_y), getData().colors.white);
            getData().fonts.font(Format(text_area[MEMO_IDX].text.size()) + U"/" + Format(TEXTBOX_MAX_CHARS) + U" " + language.get("common", "characters")).draw(15, Arg::topRight(X_CENTER + EXPORT_GAME_MEMO_WIDTH / 2, memo_label_y), getData().colors.white);
            SimpleGUI::TextArea(text_area[MEMO_IDX], Vec2{X_CENTER - EXPORT_GAME_MEMO_WIDTH / 2, memo_box_y}, SizeF{EXPORT_GAME_MEMO_WIDTH, EXPORT_GAME_MEMO_HEIGHT}, TEXTBOX_MAX_CHARS);
            // Tab移動: black -> white -> memo -> black（フォルダ入力は使わない）
            auto focus_next_from = [&](int idx) {
                // deactivate current
                text_area[idx].active = false;
                text_area[(idx + 1) % 3].active = true;
            };
            for (int i = 0; i < 3; ++i) {
                std::string str = text_area[i].text.narrow();
                if (str.find('\t') != std::string::npos) {
                    text_area[i].text.replace(U"\t", U"");
                    text_area[i].cursorPos = text_area[i].text.size();
                    text_area[i].rebuildGlyphs();
                    focus_next_from(i);
                }
                if ((str.find('\n') != std::string::npos || str.find('\r') != std::string::npos) && i != MEMO_IDX) {
                    text_area[i].text.replace(U"\r", U"").replace(U"\n", U" ");
                    text_area[i].cursorPos = text_area[i].text.size();
                    text_area[i].rebuildGlyphs();
                }
            }
            getData().game_information.black_player_name = text_area[BLACK_PLAYER_IDX].text;
            getData().game_information.white_player_name = text_area[WHITE_PLAYER_IDX].text;
            getData().game_information.memo = text_area[MEMO_IDX].text;
            back_button.draw();
            export_main_button.draw();
            export_this_board_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
            // Open folder picker after clicking save buttons
            if ( export_main_button.clicked()) {
                pending_history = getData().graph_resources.nodes[0];
                picker_subfolder.clear();
                enumerate_save_dir();
                init_folder_scroll_manager();
                new_folder_area.text.clear();
                new_folder_area.cursorPos = 0;
                new_folder_area.rebuildGlyphs();
                show_folder_picker = true;
            }
            if (export_this_board_button.clicked()) {
                std::vector<History_elem> history;
                int inspect_switch_n_discs = INF;
                if (getData().graph_resources.branch == 1) {
                    if (getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
                        inspect_switch_n_discs = getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs();
                    } else {
                        std::cerr << "no node found in inspect mode" << std::endl;
                    }
                }
                for (History_elem& history_elem : getData().graph_resources.nodes[GRAPH_MODE_NORMAL]) {
                    if (history_elem.board.n_discs() >= inspect_switch_n_discs || history_elem.board.n_discs() > getData().history_elem.board.n_discs()) {
                        break;
                    }
                    history.emplace_back(history_elem);
                }
                if (inspect_switch_n_discs != INF) {
                    for (History_elem& history_elem : getData().graph_resources.nodes[GRAPH_MODE_INSPECT]) {
                        if (history_elem.board.n_discs() > getData().history_elem.board.n_discs()) {
                            break;
                        }
                        history.emplace_back(history_elem);
                    }
                }
                pending_history.swap(history);
                picker_subfolder.clear();
                enumerate_save_dir();
                init_folder_scroll_manager();
                new_folder_area.text.clear();
                new_folder_area.cursorPos = 0;
                new_folder_area.rebuildGlyphs();
                show_folder_picker = true;
            }
        }

        // Folder picker overlay UI
        if (show_folder_picker) {
            // Path label
            getData().fonts.font(language.get("in_out", "save_subfolder")).draw(20, Arg::topCenter(X_CENTER, 10), getData().colors.white);
            String path_label = U"games/" + Unicode::Widen(picker_subfolder);
            getData().fonts.font(path_label).draw(15, Arg::topRight(IMPORT_GAME_SX + IMPORT_GAME_WIDTH, 10), getData().colors.white);

            // List via shared helper (folders only)
            // Dummy variables for unused buttons in folder picker
            std::vector<ImageButton> dummyDeleteBtns; // not used in folder picker
            bool has_parent_folder = !picker_subfolder.empty();
            auto pickRes = DrawExplorerList(
                save_folders_display, picker_games, dummyDeleteBtns,
                folder_scroll_manager, up_button, open_explorer_button, EXPORT_GAME_FOLDER_AREA_HEIGHT, EXPORT_GAME_N_GAMES_ON_WINDOW, 
                has_parent_folder, getData().fonts, getData().colors, getData().resources, language,
                getData().directories.document_dir, picker_subfolder);
            if (pickRes.openExplorerClicked) {
                String path = Unicode::Widen(getData().directories.document_dir) + U"games/";
                if (!picker_subfolder.empty()) {
                    path += Unicode::Widen(picker_subfolder) + U"/";
                }
                System::LaunchFile(path);
                return;
            }
            if (pickRes.upButtonClicked || pickRes.parentFolderDoubleClicked) {
                std::string s = picker_subfolder;
                if (!s.empty() && s.back() == '/') s.pop_back();
                size_t pos = s.find_last_of('/');
                if (pos == std::string::npos) picker_subfolder.clear();
                else picker_subfolder = s.substr(0, pos);
                enumerate_save_dir();
                init_folder_scroll_manager();
                return;
            }
            if (pickRes.folderDoubleClicked) {
                String fname = pickRes.clickedFolder;
                std::cerr << "Folder double-clicked: '" << fname.narrow() << "'" << std::endl;
                std::cerr << "Before: picker_subfolder = '" << picker_subfolder << "'" << std::endl;
                if (!picker_subfolder.empty()) picker_subfolder += "/";
                picker_subfolder += fname.narrow();
                std::cerr << "After: picker_subfolder = '" << picker_subfolder << "'" << std::endl;
                enumerate_save_dir();
                init_folder_scroll_manager();
                return;
            }
            if (pickRes.dropCompleted) {
                handle_picker_drop(pickRes);
            }

            // New folder UI - horizontal layout
            getData().fonts.font(language.get("in_out", "new_folder")).draw(15, Arg::rightCenter(200, EXPORT_GAME_CREATE_FOLDER_Y_CENTER), getData().colors.white);
            SimpleGUI::TextArea(new_folder_area, Vec2{210, EXPORT_GAME_CREATE_FOLDER_Y_CENTER - 30 / 2 - 2}, SizeF{400, 30}, 64);
            
            // Use member create_folder_button instead of temp button
            create_folder_button.move(620, EXPORT_GAME_CREATE_FOLDER_Y_CENTER - 30 / 2);
            create_folder_button.draw();
            if (create_folder_button.clicked()) {
                String s = new_folder_area.text.replaced(U"\r", U"").replaced(U"\n", U"").replaced(U"\\", U"/");
                while (s.size() && s.front() == U'/') s.erase(s.begin());
                while (s.size() && s.back() == U'/') s.pop_back();
                s.replace(U"..", U"");
                if (s.size()) {
                    String base = Unicode::Widen(getData().directories.document_dir) + U"games/" + Unicode::Widen(picker_subfolder);
                    if (base.size() && base.back() != U'/') base += U"/";
                    String target = base + s + U"/";
                    bool created = FileSystem::CreateDirectories(target);
                    if (created) {
                        // Clear the input field after successful creation
                        new_folder_area.text.clear();
                        new_folder_area.cursorPos = 0;
                        new_folder_area.rebuildGlyphs();
                        enumerate_save_dir();
                        init_folder_scroll_manager();
                        std::cerr << "Created folder: " << target.narrow() << std::endl;
                    } else {
                        std::cerr << "Failed to create folder: " << target.narrow() << std::endl;
                    }
                }
            }

            // Action buttons
            cancel_picker_button.draw();
            if (cancel_picker_button.clicked()) {
                show_folder_picker = false;
            }
            save_here_button.draw();
            if (save_here_button.clicked()) {
                std::cerr << "Save here clicked: picker_subfolder = '" << picker_subfolder << "'" << std::endl;
                subfolder = picker_subfolder; // commit selection
                std::cerr << "Committed subfolder = '" << subfolder << "'" << std::endl;
                show_folder_picker = false;
                is_saving = true;
                saving_started = false;
            }
        }
    }

    void draw() const override {

    }

private:
    // Folder picker helpers
    void enumerate_save_dir() {
        save_folders_display.clear();
        picker_has_parent = !picker_subfolder.empty();
        
        // Use the shared utility function
        std::vector<String> folders = enumerate_direct_subdirectories(getData().directories.document_dir, picker_subfolder);
        for (auto& folder : folders) {
            save_folders_display.emplace_back(folder);
        }
        
        // Also load games in current picker folder
        load_picker_games();
    }

    void load_picker_games() {
        picker_games.clear();
        String base = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (picker_subfolder.size()) {
            base += Unicode::Widen(picker_subfolder) + U"/";
        }
        const String csv_path = base + U"summary.csv";
        const CSV csv{ csv_path };
        if (csv) {
            for (size_t row = 0; row < csv.rows(); ++row) {
                Game_abstract game_abstract;
                game_abstract.date = csv[row][0];
                game_abstract.black_player = csv[row][1];
                game_abstract.white_player = csv[row][2];
                game_abstract.memo = csv[row][3];
                game_abstract.black_score = ParseOr<int32>(csv[row][4], GAME_DISCS_UNDEFINED);
                game_abstract.white_score = ParseOr<int32>(csv[row][5], GAME_DISCS_UNDEFINED);
                picker_games.emplace_back(game_abstract);
            }
        }
        reverse(picker_games.begin(), picker_games.end());
    }

    void init_folder_scroll_manager() {
        int parent_offset = picker_has_parent ? 1 : 0;  // Add parent folder if not at root
        int total = parent_offset + (int)save_folders_display.size() + (int)picker_games.size();
        folder_scroll_manager.init(770, IMPORT_GAME_SY + 8, 10, EXPORT_GAME_FOLDER_AREA_HEIGHT * EXPORT_GAME_N_GAMES_ON_WINDOW, 20, total, EXPORT_GAME_N_GAMES_ON_WINDOW, IMPORT_GAME_SX, 73, IMPORT_GAME_WIDTH + 10, EXPORT_GAME_FOLDER_AREA_HEIGHT * EXPORT_GAME_N_GAMES_ON_WINDOW);
    }
    
    // Handle drag and drop operations in folder picker
    void handle_picker_drop(const ExplorerDrawResult& res) {
        if (res.dropOnParent) {
            // Handle drop on parent folder - move to parent directory
            if (res.isDraggingGame && res.draggedGameIndex >= 0 && res.draggedGameIndex < (int)picker_games.size()) {
                move_picker_game_to_parent(res.draggedGameIndex);
            } else if (res.isDraggingFolder && !res.draggedFolderName.empty()) {
                move_picker_folder_to_parent(res.draggedFolderName.narrow());
            }
        } else {
            // Handle normal folder drop
            if (res.isDraggingGame && res.draggedGameIndex >= 0 && res.draggedGameIndex < (int)picker_games.size()) {
                // Move game to target folder
                move_picker_game_to_folder(res.draggedGameIndex, res.dropTargetFolder.narrow());
            } else if (res.isDraggingFolder && !res.draggedFolderName.empty()) {
                // Move folder to target folder
                move_picker_folder_to_folder(res.draggedFolderName.narrow(), res.dropTargetFolder.narrow());
            }
        }
    }
    
    // Move a game to parent folder
    void move_picker_game_to_parent(int game_index) {
        if (picker_subfolder.empty()) return;  // Already at root
        
        // Get parent folder path
        std::string parent_folder = picker_subfolder;
        if (!parent_folder.empty() && parent_folder.back() == '/') parent_folder.pop_back();
        size_t pos = parent_folder.find_last_of('/');
        if (pos == std::string::npos) parent_folder.clear();
        else parent_folder = parent_folder.substr(0, pos);
        
        move_picker_game_to_folder(game_index, parent_folder);
    }
    
    // Move a folder to parent folder
    void move_picker_folder_to_parent(const std::string& folder_name) {
        if (picker_subfolder.empty()) return;  // Already at root
        
        // Get parent folder path
        std::string parent_folder = picker_subfolder;
        if (!parent_folder.empty() && parent_folder.back() == '/') parent_folder.pop_back();
        size_t pos = parent_folder.find_last_of('/');
        if (pos == std::string::npos) parent_folder.clear();
        else parent_folder = parent_folder.substr(0, pos);
        
        move_picker_folder_to_folder(folder_name, parent_folder);
    }
    
    // Move a game to a different folder
    void move_picker_game_to_folder(int game_index, const std::string& target_folder) {
        if (game_index < 0 || game_index >= (int)picker_games.size()) return;
        
        const Game_abstract& game = picker_games[game_index];
        
        // Source and target paths
        String source_base = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!picker_subfolder.empty()) {
            source_base += Unicode::Widen(picker_subfolder) + U"/";
        }
        String target_base = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!target_folder.empty()) {
            target_base += Unicode::Widen(target_folder) + U"/";
        }
        
        // Ensure target directory exists
        if (!FileSystem::Exists(target_base)) {
            FileSystem::CreateDirectories(target_base);
        }
        
        // Move JSON file
        String source_json = source_base + game.date + U".json";
        String target_json = target_base + game.date + U".json";
        if (FileSystem::Exists(source_json)) {
            FileSystem::Copy(source_json, target_json);
            FileSystem::Remove(source_json);
        }
        
        // Update CSV files
        remove_picker_game_from_csv(game_index);
        add_picker_game_to_target_csv(game, target_base);
        
        // Refresh displays
        enumerate_save_dir();
        init_folder_scroll_manager();
        
        std::cerr << "Moved game " << game.date.narrow() << " to " << target_folder << std::endl;
    }
    
    // Move a folder to a different folder
    void move_picker_folder_to_folder(const std::string& folder_name, const std::string& target_folder) {
        String source_folder = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!picker_subfolder.empty()) {
            source_folder += Unicode::Widen(picker_subfolder) + U"/";
        }
        source_folder += Unicode::Widen(folder_name);
        
        String target_parent = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!target_folder.empty()) {
            target_parent += Unicode::Widen(target_folder) + U"/";
        }
        String target_full = target_parent + Unicode::Widen(folder_name);
        
        // Ensure target parent directory exists
        if (!FileSystem::Exists(target_parent)) {
            FileSystem::CreateDirectories(target_parent);
        }
        
        // Move the folder using system command
        if (FileSystem::Exists(source_folder) && !FileSystem::Exists(target_full)) {
            std::string cmd = "move \"" + source_folder.narrow() + "\" \"" + target_full.narrow() + "\"";
            std::system(cmd.c_str());
            
            // Refresh displays
            enumerate_save_dir();
            init_folder_scroll_manager();
            
            std::cerr << "Moved folder " << folder_name << " to " << target_folder << std::endl;
        }
    }
    
    // Remove game from current CSV
    void remove_picker_game_from_csv(int game_index) {
        String source_base = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!picker_subfolder.empty()) {
            source_base += Unicode::Widen(picker_subfolder) + U"/";
        }
        const String csv_path = source_base + U"summary.csv";
        CSV csv{ csv_path };
        CSV new_csv;
        
        int csv_row_to_remove = (int)picker_games.size() - 1 - game_index;
        
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
    void add_picker_game_to_target_csv(const Game_abstract& game, const String& target_base) {
        String target_csv = target_base + U"summary.csv";
        CSV csv{ target_csv };
        
        // Create new CSV with existing data plus new game
        CSV new_csv;
        for (int i = 0; i < (int)csv.rows(); ++i) {
            if (csv[i].size() >= 6) {
                for (int j = 0; j < 6; ++j) {
                    new_csv.write(csv[i][j]);
                }
                new_csv.newLine();
            }
        }
        
        // Add new game entry
        new_csv.write(game.date);
        new_csv.write(game.black_player);
        new_csv.write(game.white_player);
        new_csv.write(game.memo);
        new_csv.write(game.black_score == GAME_DISCS_UNDEFINED ? U"" : ToString(game.black_score));
        new_csv.write(game.white_score == GAME_DISCS_UNDEFINED ? U"" : ToString(game.white_score));
        new_csv.newLine();
        
        new_csv.save(target_csv);
    }

    void export_game(std::vector<History_elem> history) {
        String date = Unicode::Widen(calc_date());
        JSON json;
        json[GAME_DATE] = date;
        json[GAME_BLACK_PLAYER] = getData().game_information.black_player_name;
        json[GAME_WHITE_PLAYER] = getData().game_information.white_player_name;
        json[GAME_MEMO] = getData().game_information.memo;
        int black_discs = GAME_DISCS_UNDEFINED;
        int white_discs = GAME_DISCS_UNDEFINED;
        if (history.back().board.is_end()) {
            if (history.back().player == BLACK) {
                black_discs = history.back().board.count_player();
                white_discs = history.back().board.count_opponent();
            } else {
                black_discs = history.back().board.count_opponent();
                white_discs = history.back().board.count_player();
            }
        }
        json[GAME_BLACK_DISCS] = black_discs;
        json[GAME_WHITE_DISCS] = white_discs;
        for (History_elem history_elem : history) {
            String n_discs = Format(history_elem.board.n_discs());
            json[n_discs][GAME_BOARD_PLAYER] = history_elem.board.player;
            json[n_discs][GAME_BOARD_OPPONENT] = history_elem.board.opponent;
            json[n_discs][GAME_PLAYER] = history_elem.player;
            json[n_discs][GAME_VALUE] = history_elem.v;
            json[n_discs][GAME_LEVEL] = history_elem.level;
            json[n_discs][GAME_POLICY] = history_elem.policy;
            if (history_elem.board.n_discs() < history.back().board.n_discs()) {
                json[n_discs][GAME_NEXT_POLICY] = history_elem.next_policy;
            } else {
                json[n_discs][GAME_NEXT_POLICY] = -1;
            }
        }
        // Build directory path: appdata/games/(subfolder)/
        String base_dir = Unicode::Widen(getData().directories.document_dir) + U"games/";
        String folder = Unicode::Widen(subfolder);
        if (folder.size()) {
            base_dir += folder + U"/";
        }
        FileSystem::CreateDirectories(base_dir);
        const String save_path = base_dir + date + U".json";
        json.save(save_path);

        const String csv_path = base_dir + U"summary.csv";
        CSV csv{ csv_path };
        String memo_summary_all = getData().game_information.memo.replaced(U"\r", U"").replaced(U"\n", U" ");
        String memo_summary;
        for (int i = 0; i < std::min((int)memo_summary_all.size(), GAME_MEMO_SUMMARY_SIZE); ++i) {
            memo_summary += memo_summary_all[i];
        }
        csv.writeRow(date, getData().game_information.black_player_name, getData().game_information.white_player_name, memo_summary, black_discs, white_discs);
        csv.save(csv_path);
    }
};




class Change_screenshot_saving_dir : public App::Scene {
private:
    Button back_button;
    Button default_button;
    Button go_button;
    std::string dir;
    bool is_valid_dir;
    TextAreaEditState text_area;

public:
    Change_screenshot_saving_dir(const InitData& init) : IScene{ init } {
        back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        default_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "use_default"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        go_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        text_area.text = Unicode::Widen(getData().user_settings.screenshot_saving_dir);
        text_area.cursorPos = text_area.text.size();
        text_area.rebuildGlyphs();
        is_valid_dir = FileSystem::Exists(Unicode::Widen(getData().user_settings.screenshot_saving_dir));
        if (is_valid_dir) {
            go_button.enable();
        } else {
            go_button.disable();
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
        int sy = 20 + icon_width + 40;
        getData().fonts.font(language.get("in_out", "change_screenshot_saving_dir")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
        getData().fonts.font(language.get("in_out", "input_screenshot_saving_dir")).draw(15, Arg::topCenter(X_CENTER, sy + 50), getData().colors.white);
        text_area.active = true;
        bool text_changed = SimpleGUI::TextArea(text_area, Vec2{X_CENTER - 300, sy + 80}, SizeF{600, 100}, TEXTBOX_MAX_CHARS);
        bool return_pressed = false;
        if (text_area.text.size()) {
            if (text_area.text[text_area.text.size() - 1] == '\n') {
                return_pressed = true;
            }
        }
        dir = text_area.text.replaced(U"\r", U"").replaced(U"\n", U"").replaced(U"\\", U"/").narrow();
        if (dir.size()) {
            if (dir[dir.size() - 1] != '/') {
                dir += "/";
            }
        }
        if (text_changed) {
            is_valid_dir = FileSystem::Exists(Unicode::Widen(dir));
        }
        if (is_valid_dir) {
            go_button.enable();
        } else {
            getData().fonts.font(language.get("in_out", "directory_not_found")).draw(15, Arg::topCenter(X_CENTER, sy + 190), getData().colors.white);
            go_button.disable();
        }
        back_button.draw();
        if (back_button.clicked() || KeyEscape.pressed()) {
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        default_button.draw();
        if (default_button.clicked()) {
            text_area.text = Unicode::Widen(getData().directories.document_dir + "screenshots/");
            text_area.cursorPos = text_area.text.size();
            text_area.scrollY = 0.0;
            text_area.rebuildGlyphs();
        }
        go_button.draw();
        if (go_button.clicked()) {
            getData().user_settings.screenshot_saving_dir = dir;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
    }

    void draw() const override {

    }
};
