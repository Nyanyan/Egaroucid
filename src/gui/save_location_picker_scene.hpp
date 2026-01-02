/*
    Egaroucid Project

    @file save_location_picker_scene.hpp
        Save location picker scene - folder selection for game saving
    @date 2025
    @author GitHub Copilot
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

// Save Location Picker Scene
// Used to choose a folder location for saving games
class Save_location_picker : public App::Scene {
private:
    Button back_button;
    Button save_here_button;
    Button up_button;
    Button create_folder_button;
    
    // Folder navigation state
    std::string picker_subfolder;
    std::vector<String> save_folders_display;
    std::vector<Game_abstract> picker_games;
    bool picker_has_parent = false;
    
    // Scroll and UI state
    Scroll_manager folder_scroll_manager;
    TextAreaEditState new_folder_area;
    
    // Return scene info
    String return_scene;
    std::vector<History_elem> pending_history;

public:
    Save_location_picker(const InitData& init) : IScene{ init } {
        back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        save_here_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "save_here"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        up_button.init(IMPORT_GAME_SX, IMPORT_GAME_SY - 30, 28, 24, 4, U"↑", 16, getData().fonts.font, getData().colors.white, getData().colors.black);
        create_folder_button.init(620, EXPORT_GAME_CREATE_FOLDER_Y_CENTER - 30 / 2, 120, 30, 8, language.get("in_out", "create"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
        
        // Get return scene and history from getData
        return_scene = getData().game_editor_info.return_scene;
        pending_history = getData().save_location_picker_info.pending_history;
        
        // Initialize folder picker state
        picker_subfolder.clear();
        enumerate_save_dir();
        init_folder_scroll_manager();
        
        new_folder_area.text.clear();
        new_folder_area.cursorPos = 0;
        new_folder_area.rebuildGlyphs();
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);

        // Title
        getData().fonts.font(language.get("in_out", "save_subfolder")).draw(25, Arg::center(X_CENTER, 30), getData().colors.white);
        
        // Path label
        String path_label = U"games/" + Unicode::Widen(picker_subfolder);
        getData().fonts.font(path_label).draw(15, Arg::rightCenter(IMPORT_GAME_SX + IMPORT_GAME_WIDTH, 30), getData().colors.white);

        // List via shared helper (folders only)
        std::vector<ImageButton> dummyDeleteBtns;
        std::vector<ImageButton> dummyEditBtns;
        bool has_parent_folder = !picker_subfolder.empty();
        auto pickRes = DrawExplorerList(
            save_folders_display, picker_games, dummyDeleteBtns, dummyEditBtns,
            folder_scroll_manager, up_button, EXPORT_GAME_FOLDER_AREA_HEIGHT, EXPORT_GAME_N_GAMES_ON_WINDOW, 
            has_parent_folder, getData().fonts, getData().colors, getData().resources, language,
            getData().directories.document_dir, picker_subfolder, nullptr);
        
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
            if (!picker_subfolder.empty()) picker_subfolder += "/";
            picker_subfolder += fname.narrow();
            enumerate_save_dir();
            init_folder_scroll_manager();
            return;
        }
        if (pickRes.drop_completed) {
            handle_picker_drop(pickRes);
        } else if (pickRes.reorderRequested) {
            handle_picker_reorder(pickRes);
        }

        // New folder UI - horizontal layout
        getData().fonts.font(language.get("in_out", "new_folder")).draw(15, Arg::rightCenter(200, EXPORT_GAME_CREATE_FOLDER_Y_CENTER), getData().colors.white);
        SimpleGUI::TextArea(new_folder_area, Vec2{210, EXPORT_GAME_CREATE_FOLDER_Y_CENTER - 30 / 2 - 2}, SizeF{400, 30}, 64);
        
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
                    new_folder_area.text.clear();
                    new_folder_area.cursorPos = 0;
                    new_folder_area.rebuildGlyphs();
                    enumerate_save_dir();
                    init_folder_scroll_manager();
                }
            }
        }

        // Action buttons
        back_button.draw();
        if (back_button.clicked() || KeyEscape.pressed()) {
            // Return to Game_editor to modify game information
            changeScene(U"Game_editor", SCENE_FADE_TIME);
            return;
        }
        
        save_here_button.draw();
        if (save_here_button.clicked()) {
            // Save the game at the selected location
            getData().save_location_picker_info.selected_subfolder = picker_subfolder;
            save_game_here();
            // Return to main scene after saving
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
    }

    void draw() const override {
    }

private:
    void enumerate_save_dir() {
        save_folders_display.clear();
        picker_has_parent = !picker_subfolder.empty();
        
        std::vector<String> folders = enumerate_direct_subdirectories(getData().directories.document_dir, picker_subfolder);
        for (auto& folder : folders) {
            save_folders_display.emplace_back(folder);
        }
        
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
                game_abstract.filename_date = csv[row][0];
                game_abstract.black_player = csv[row][1];
                game_abstract.white_player = csv[row][2];
                game_abstract.memo = csv[row][3];
                game_abstract.black_score = ParseOr<int32>(csv[row][4], GAME_DISCS_UNDEFINED);
                game_abstract.white_score = ParseOr<int32>(csv[row][5], GAME_DISCS_UNDEFINED);
                if (csv[row].size() >= 7 && !csv[row][6].isEmpty()) {
                    game_abstract.game_date = csv[row][6];
                } else {
                    game_abstract.game_date = game_abstract.filename_date.substr(0, 10).replaced(U"_", U"-");
                }
                picker_games.emplace_back(game_abstract);
            }
        }
        reverse(picker_games.begin(), picker_games.end());
    }

    void init_folder_scroll_manager() {
        int parent_offset = picker_has_parent ? 1 : 0;
        int total = parent_offset + (int)save_folders_display.size() + (int)picker_games.size();
        folder_scroll_manager.init(770, IMPORT_GAME_SY + 8, 10, EXPORT_GAME_FOLDER_AREA_HEIGHT * EXPORT_GAME_N_GAMES_ON_WINDOW, 20, total, EXPORT_GAME_N_GAMES_ON_WINDOW, IMPORT_GAME_SX, 73, IMPORT_GAME_WIDTH + 10, EXPORT_GAME_FOLDER_AREA_HEIGHT * EXPORT_GAME_N_GAMES_ON_WINDOW);
    }
    
    void handle_picker_drop(const ExplorerDrawResult& res) {
        if (res.drop_on_parent) {
            if (res.is_dragging_game && res.dragged_game_index >= 0 && res.dragged_game_index < (int)picker_games.size()) {
                move_picker_game_to_parent(res.dragged_game_index);
            } else if (res.is_dragging_folder && !res.dragged_folder_name.empty()) {
                move_picker_folder_to_parent(res.dragged_folder_name.narrow());
            }
        } else {
            if (res.is_dragging_game && res.dragged_game_index >= 0 && res.dragged_game_index < (int)picker_games.size()) {
                move_picker_game_to_folder(res.dragged_game_index, res.drop_target_folder.narrow());
            } else if (res.is_dragging_folder && !res.dragged_folder_name.empty()) {
                move_picker_folder_to_folder(res.dragged_folder_name.narrow(), res.drop_target_folder.narrow());
            }
        }
    }

    void handle_picker_reorder(const ExplorerDrawResult& res) {
        if (!res.reorderRequested || !res.is_dragging_game) {
            return;
        }
        if (res.reorderFrom < 0 || res.reorderFrom >= (int)picker_games.size()) {
            return;
        }
        int insert_idx = std::clamp(res.reorderTo, 0, (int)picker_games.size());
        bool changed = gui_list::reorder_parallel(picker_games, res.reorderFrom, insert_idx);
        if (!changed) {
            return;
        }
        persist_picker_games_order_to_csv();
    }
    
    void move_picker_game_to_parent(int game_index) {
        if (picker_subfolder.empty()) return;
        
        std::string parent_folder = picker_subfolder;
        if (!parent_folder.empty() && parent_folder.back() == '/') parent_folder.pop_back();
        size_t pos = parent_folder.find_last_of('/');
        if (pos == std::string::npos) parent_folder.clear();
        else parent_folder = parent_folder.substr(0, pos);
        
        move_picker_game_to_folder(game_index, parent_folder);
    }
    
    void move_picker_folder_to_parent(const std::string& folder_name) {
        if (picker_subfolder.empty()) return;
        
        std::string parent_folder = picker_subfolder;
        if (!parent_folder.empty() && parent_folder.back() == '/') parent_folder.pop_back();
        size_t pos = parent_folder.find_last_of('/');
        if (pos == std::string::npos) parent_folder.clear();
        else parent_folder = parent_folder.substr(0, pos);
        
        move_picker_folder_to_folder(folder_name, parent_folder);
    }
    
    void move_picker_game_to_folder(int game_index, const std::string& target_folder) {
        if (game_index < 0 || game_index >= (int)picker_games.size()) return;
        
        const Game_abstract& game = picker_games[game_index];
        
        String source_base = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!picker_subfolder.empty()) {
            source_base += Unicode::Widen(picker_subfolder) + U"/";
        }
        String target_base = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!target_folder.empty()) {
            target_base += Unicode::Widen(target_folder) + U"/";
        }
        
        if (!FileSystem::Exists(target_base)) {
            FileSystem::CreateDirectories(target_base);
        }
        
        String source_json = source_base + game.filename_date + U".json";
        String target_json = target_base + game.filename_date + U".json";
        if (FileSystem::Exists(source_json)) {
            FileSystem::Copy(source_json, target_json);
            FileSystem::Remove(source_json);
        }
        
        remove_picker_game_from_csv(game_index);
        add_picker_game_to_target_csv(game, target_base);
        
        enumerate_save_dir();
        init_folder_scroll_manager();
    }
    
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
        
        if (!FileSystem::Exists(target_parent)) {
            FileSystem::CreateDirectories(target_parent);
        }
        
        if (FileSystem::Exists(source_folder) && !FileSystem::Exists(target_full)) {
            std::string cmd = "move \"" + source_folder.narrow() + "\" \"" + target_full.narrow() + "\"";
            std::system(cmd.c_str());
            
            enumerate_save_dir();
            init_folder_scroll_manager();
        }
    }
    
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
    
    void add_picker_game_to_target_csv(const Game_abstract& game, const String& target_base) {
        String target_csv = target_base + U"summary.csv";
        CSV csv{ target_csv };
        
        CSV new_csv;
        for (int i = 0; i < (int)csv.rows(); ++i) {
            if (csv[i].size() >= 1) {
                size_t cols = std::min(csv[i].size(), size_t(6));
                for (size_t j = 0; j < cols; ++j) {
                    new_csv.write(csv[i][j]);
                }
                if (csv[i].size() < 7) {
                    String old_date = csv[i][0].substr(0, 10).replaced(U"_", U"-");
                    new_csv.write(old_date);
                } else {
                    new_csv.write(csv[i][6]);
                }
                new_csv.newLine();
            }
        }
        
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

    void persist_picker_games_order_to_csv() {
        const String csv_path = get_picker_base_dir() + U"summary.csv";
        CSV new_csv;
        for (int i = (int)picker_games.size() - 1; i >= 0; --i) {
            const auto& game = picker_games[i];
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

    String get_picker_base_dir() const {
        String base = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!picker_subfolder.empty()) {
            base += Unicode::Widen(picker_subfolder) + U"/";
        }
        return base;
    }

    void save_game_here() {
        String date = Unicode::Widen(calc_date());
        String base_dir = Unicode::Widen(getData().directories.document_dir) + U"games/";
        String folder = Unicode::Widen(picker_subfolder);
        if (folder.size()) {
            base_dir += folder + U"/";
        }
        
        game_save_helper::save_game_to_file(
            base_dir,
            date,
            getData().game_information.black_player_name,
            getData().game_information.white_player_name,
            getData().game_information.memo,
            pending_history,
            getData().game_information.date
        );
        
        String json_path = base_dir + date + U".json";
        load_game_from_json(getData(), opening, json_path, date, picker_subfolder);
    }
};