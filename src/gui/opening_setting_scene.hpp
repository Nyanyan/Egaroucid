/*
    Egaroucid Project

    @file opening_setting.hpp
        Forced Opening setting
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <Siv3D.hpp>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"


// Opening abstract structure for list display
struct Opening_abstract {
    String transcript;
    double weight;
    bool enabled;  // オンオフ切り替え用
    
    Opening_abstract() : transcript(U""), weight(1.0), enabled(true) {}
    Opening_abstract(const String& t, double w, bool e = true) : transcript(t), weight(w), enabled(e) {}
};


class Opening_setting : public App::Scene {
    private:
        std::vector<Opening_abstract> openings;
        std::vector<ImageButton> delete_buttons;
        std::vector<ImageButton> edit_buttons;
        std::vector<ImageButton> toggle_buttons;  // オンオフ切り替えボタン
        Scroll_manager scroll_manager;
        Button add_button;
        Button ok_button;
        Button back_button;
        Button up_button;
        Button register_button;
        Button update_button;
        bool adding_elem;
        bool editing_elem;
        int editing_index;
        TextAreaEditState text_area[2];
        
        // Explorer-like folder view
        std::vector<String> folders_display;
        bool has_parent = false;
        std::string subfolder;  // current folder (narrow), may be ""
    
    public:
        Opening_setting(const InitData& init) : IScene{ init } {
            add_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("opening_setting", "add"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            ok_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            register_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("opening_setting", "register"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            update_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("opening_setting", "update"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            up_button.init(OPENING_SETTING_SX, OPENING_SETTING_SY - 30, 28, 24, 4, U"↑", 16, getData().fonts.font, getData().colors.white, getData().colors.black);
            
            adding_elem = false;
            editing_elem = false;
            editing_index = -1;
            subfolder.clear();
            enumerate_current_dir();
            load_openings();
        }
    
        void update() override {
            if (System::GetUserActions() & UserAction::CloseButtonClicked) {
                changeScene(U"Close", SCENE_FADE_TIME);
            }
            getData().fonts.font(language.get("opening_setting", "opening_setting")).draw(25, Arg::center(X_CENTER, 30), getData().colors.white);
            
            // Current path label
            String path_label = U"forced_openings/" + Unicode::Widen(subfolder);
            getData().fonts.font(path_label).draw(15, Arg::rightCenter(OPENING_SETTING_SX + OPENING_SETTING_WIDTH, 30), getData().colors.white);
            
            // Handle adding/editing mode
            if (adding_elem || editing_elem) {
                back_button.draw();
                if (back_button.clicked() || KeyEscape.down()) {
                    adding_elem = false;
                    editing_elem = false;
                    editing_index = -1;
                }
                
                std::string transcript = text_area[0].text.narrow();
                std::string weight_str = text_area[1].text.narrow();
                bool can_be_registered = is_valid_transcript(transcript);
                double weight;
                try {
                    weight = stoi(weight_str);
                } catch (const std::invalid_argument& e) {
                    can_be_registered = false;
                } catch (const std::out_of_range& e) {
                    can_be_registered = false;
                }
                
                if (editing_elem) {
                    if (can_be_registered) {
                        update_button.enable();
                    } else {
                        update_button.disable();
                    }
                    update_button.draw();
                    if (update_button.clicked() || (can_be_registered && KeyEnter.down())) {
                        openings[editing_index].transcript = Unicode::Widen(transcript);
                        openings[editing_index].weight = weight;
                        save_openings();
                        editing_elem = false;
                        editing_index = -1;
                    }
                } else {
                    if (can_be_registered) {
                        register_button.enable();
                    } else {
                        register_button.disable();
                    }
                    register_button.draw();
                    if (register_button.clicked() || (can_be_registered && KeyEnter.down())) {
                        add_opening(Unicode::Widen(transcript), weight, true);
                        adding_elem = false;
                    }
                }
            } else {
                // Normal mode
                add_button.draw();
                if (add_button.clicked()) {
                    adding_elem = true;
                    for (int i = 0; i < 2; ++i) {
                        text_area[i].text = U"";
                        if (i == 1) {
                            text_area[i].text = U"1";
                        }
                        text_area[i].cursorPos = text_area[i].text.size();
                        text_area[i].rebuildGlyphs();
                    }
                    text_area[0].active = true;
                    text_area[1].active = false;
                }
                ok_button.draw();
                if (ok_button.clicked() || KeyEnter.down()) {
                    // Save all openings and reload forced_openings
                    save_all_openings_to_forced_openings();
                    getData().graph_resources.need_init = false;
                    changeScene(U"Main_scene", SCENE_FADE_TIME);
                }
            }
            
            // Draw folders and openings list
            draw_openings_list();
            
            // Handle folder navigation
            handle_folder_navigation();
            
            if (!(adding_elem || editing_elem)) {
                scroll_manager.draw();
                scroll_manager.update();
            }
        }
    
        void draw() const override {
    
        }
    
    private:
        void init_scroll_manager() {
            int parent_offset = !subfolder.empty() ? 1 : 0;
            int total = parent_offset + (int)folders_display.size() + (int)openings.size();
            scroll_manager.init(770, OPENING_SETTING_SY + 8, 10, OPENING_SETTING_HEIGHT * OPENING_SETTING_N_GAMES_ON_WINDOW, 20, total, OPENING_SETTING_N_GAMES_ON_WINDOW, OPENING_SETTING_SX, 73, OPENING_SETTING_WIDTH + 10, OPENING_SETTING_HEIGHT * OPENING_SETTING_N_GAMES_ON_WINDOW);
        }
        
        // Get base directory for current folder
        String get_base_dir() const {
            String base = Unicode::Widen(getData().directories.appdata_dir) + U"/forced_openings/";
            if (subfolder.size()) {
                base += Unicode::Widen(subfolder) + U"/";
            }
            return base;
        }
        
        // Load openings from current folder's summary.csv
        void load_openings() {
            openings.clear();
            delete_buttons.clear();
            edit_buttons.clear();
            toggle_buttons.clear();
            
            const String csv_path = get_base_dir() + U"summary.csv";
            const CSV csv{ csv_path };
            if (csv) {
                for (size_t row = 0; row < csv.rows(); ++row) {
                    Opening_abstract opening;
                    opening.transcript = csv[row][0];
                    opening.weight = ParseOr<double>(csv[row][1], 1.0);
                    opening.enabled = ParseOr<bool>(csv[row][2], true);
                    openings.emplace_back(opening);
                }
            }
            
            Texture cross_image = getData().resources.cross;
            Texture edit_image = getData().resources.check;  // Use check icon for edit
            for (int i = 0; i < (int)openings.size(); ++i) {
                ImageButton delete_btn;
                delete_btn.init(0, 0, 15, cross_image);
                delete_buttons.emplace_back(delete_btn);
                
                ImageButton edit_btn;
                edit_btn.init(0, 0, 15, edit_image);
                edit_buttons.emplace_back(edit_btn);
                
                ImageButton toggle_btn;
                toggle_btn.init(0, 0, 15, cross_image);  // Will use different image based on state
                toggle_buttons.emplace_back(toggle_btn);
            }
            
            init_scroll_manager();
        }
        
        // Save openings to current folder's summary.csv
        void save_openings() {
            const String csv_path = get_base_dir() + U"summary.csv";
            CSV csv;
            for (const auto& opening : openings) {
                csv.write(opening.transcript);
                csv.write(Format(opening.weight));
                csv.write(opening.enabled ? U"true" : U"false");
                csv.newLine();
            }
            csv.save(csv_path);
        }
        
        // Add new opening to current folder
        void add_opening(const String& transcript, double weight, bool enabled) {
            Opening_abstract opening(transcript, weight, enabled);
            openings.emplace_back(opening);
            
            ImageButton delete_btn;
            delete_btn.init(0, 0, 15, getData().resources.cross);
            delete_buttons.emplace_back(delete_btn);
            
            ImageButton edit_btn;
            edit_btn.init(0, 0, 15, getData().resources.check);  // Use check icon for edit
            edit_buttons.emplace_back(edit_btn);
            
            ImageButton toggle_btn;
            toggle_btn.init(0, 0, 15, getData().resources.cross);
            toggle_buttons.emplace_back(toggle_btn);
            
            save_openings();
            init_scroll_manager();
        }
        
        // Delete opening
        void delete_opening(int idx) {
            if (idx < 0 || idx >= (int)openings.size()) return;
            
            openings.erase(openings.begin() + idx);
            delete_buttons.erase(delete_buttons.begin() + idx);
            edit_buttons.erase(edit_buttons.begin() + idx);
            toggle_buttons.erase(toggle_buttons.begin() + idx);
            
            save_openings();
            
            double strt_idx_double = scroll_manager.get_strt_idx_double();
            init_scroll_manager();
            if ((int)strt_idx_double >= idx) {
                strt_idx_double -= 1.0;
            }
            scroll_manager.set_strt_idx(strt_idx_double);
            std::cerr << "deleted opening " << idx << std::endl;
        }
        
        // Enumerate current directory
        void enumerate_current_dir() {
            folders_display.clear();
            has_parent = !subfolder.empty();
            
            std::vector<String> folders = enumerate_direct_subdirectories(getData().directories.appdata_dir + "/forced_openings", subfolder);
            for (auto& folder : folders) {
                folders_display.emplace_back(folder);
            }
            
            init_scroll_manager();
        }
        
        // Navigate to folder
        void navigate_to_folder(const String& folder_name) {
            if (subfolder.size()) subfolder += "/";
            subfolder += folder_name.narrow();
            enumerate_current_dir();
            load_openings();
            init_scroll_manager();
        }
        
        // Navigate to parent folder
        void navigate_to_parent() {
            if (subfolder.empty()) return;
            
            std::string s = subfolder;
            if (!s.empty() && s.back() == '/') s.pop_back();
            size_t pos = s.find_last_of('/');
            if (pos == std::string::npos) subfolder.clear();
            else subfolder = s.substr(0, pos);
            enumerate_current_dir();
            load_openings();
            init_scroll_manager();
        }
        
        // Handle folder navigation clicks
        void handle_folder_navigation() {
            static uint64_t last_click_time = 0;
            static String last_clicked_folder;
            uint64_t current_time = Time::GetMillisec();
            constexpr uint64_t DOUBLE_CLICK_TIME_MS = 400;
            
            int sy = OPENING_SETTING_SY + 8;
            int strt_idx_int = scroll_manager.get_strt_idx_int();
            int row_index = 0;
            
            // Parent folder
            if (has_parent) {
                if (row_index >= strt_idx_int && row_index < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                    int display_row = row_index - strt_idx_int;
                    int item_sy = sy + display_row * OPENING_SETTING_HEIGHT;
                    Rect rect(OPENING_SETTING_SX, item_sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
                    if (rect.leftClicked()) {
                        if (current_time - last_click_time < DOUBLE_CLICK_TIME_MS && last_clicked_folder == U"..") {
                            navigate_to_parent();
                            return;
                        }
                        last_click_time = current_time;
                        last_clicked_folder = U"..";
                    }
                }
                row_index++;
            }
            
            // Folders
            for (int i = 0; i < (int)folders_display.size(); ++i) {
                if (row_index >= strt_idx_int && row_index < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                    int display_row = row_index - strt_idx_int;
                    int item_sy = sy + display_row * OPENING_SETTING_HEIGHT;
                    Rect rect(OPENING_SETTING_SX, item_sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
                    if (rect.leftClicked()) {
                        if (current_time - last_click_time < DOUBLE_CLICK_TIME_MS && last_clicked_folder == folders_display[i]) {
                            navigate_to_folder(folders_display[i]);
                            return;
                        }
                        last_click_time = current_time;
                        last_clicked_folder = folders_display[i];
                    }
                }
                row_index++;
            }
        }
        
        // Draw openings list
        void draw_openings_list() {
            int sy = OPENING_SETTING_SY;
            int strt_idx_int = scroll_manager.get_strt_idx_int();
            
            if (adding_elem || editing_elem) {
                if (openings.size() >= OPENING_SETTING_N_GAMES_ON_WINDOW) {
                    strt_idx_int = openings.size() - OPENING_SETTING_N_GAMES_ON_WINDOW + 1;
                }
            }
            
            if (strt_idx_int > 0) {
                getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, sy }, getData().colors.white);
            }
            sy += 8;
            
            int parent_offset = has_parent ? 1 : 0;
            int total_items = parent_offset + (int)folders_display.size() + (int)openings.size();
            
            if (!adding_elem && !editing_elem && total_items == 0) {
                getData().fonts.font(language.get("opening_setting", "no_opening_found")).draw(20, Arg::center(X_CENTER, Y_CENTER), getData().colors.white);
                return;
            }
            
            int row_index = 0;
            
            // Draw parent folder if exists
            if (has_parent) {
                if (row_index >= strt_idx_int && row_index < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                    int display_row = row_index - strt_idx_int;
                    int item_sy = sy + display_row * OPENING_SETTING_HEIGHT;
                    draw_parent_folder_item(item_sy);
                }
                row_index++;
            }
            
            // Draw folders
            for (int i = 0; i < (int)folders_display.size(); ++i) {
                if (row_index >= strt_idx_int && row_index < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                    int display_row = row_index - strt_idx_int;
                    int item_sy = sy + display_row * OPENING_SETTING_HEIGHT;
                    draw_folder_item(folders_display[i], item_sy, i);
                }
                row_index++;
            }
            
            // Draw openings
            for (int i = 0; i < (int)openings.size(); ++i) {
                if (row_index >= strt_idx_int && row_index < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                    int display_row = row_index - strt_idx_int;
                    int item_sy = sy + display_row * OPENING_SETTING_HEIGHT;
                    draw_opening_item(i, item_sy, row_index);
                }
                row_index++;
            }
            
            // Draw input area for adding/editing
            if (adding_elem || editing_elem) {
                int input_row = row_index;
                if (input_row >= strt_idx_int && input_row < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                    int display_row = input_row - strt_idx_int;
                    int item_sy = sy + display_row * OPENING_SETTING_HEIGHT;
                    draw_input_area(item_sy, row_index);
                }
            }
            
            if (strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW < total_items + (adding_elem || editing_elem ? 1 : 0)) {
                getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, 415}, getData().colors.white);
            }
        }
        
        // Draw parent folder item
        void draw_parent_folder_item(int sy) {
            Rect rect(OPENING_SETTING_SX, sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
            rect.draw(getData().colors.dark_green).drawFrame(1.0, getData().colors.white);
            getData().fonts.font(U"📁 ..").draw(20, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 8, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
        }
        
        // Draw folder item
        void draw_folder_item(const String& folder_name, int sy, int idx) {
            Rect rect(OPENING_SETTING_SX, sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
            if (idx % 2) {
                rect.draw(getData().colors.dark_green).drawFrame(1.0, getData().colors.white);
            } else {
                rect.draw(getData().colors.green).drawFrame(1.0, getData().colors.white);
            }
            getData().fonts.font(U"📁 " + folder_name).draw(18, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 8, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
        }
        
        // Draw opening item
        void draw_opening_item(int idx, int sy, int row_index) {
            const auto& opening = openings[idx];
            Rect rect(OPENING_SETTING_SX, sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
            
            if (row_index % 2) {
                rect.draw(getData().colors.dark_green).drawFrame(1.0, getData().colors.white);
            } else {
                rect.draw(getData().colors.green).drawFrame(1.0, getData().colors.white);
            }
            
            if (adding_elem || editing_elem) {
                rect.draw(ColorF{1.0, 1.0, 1.0, 0.5});
            }
            
            if (!(adding_elem || editing_elem)) {
                // Delete button
                delete_buttons[idx].move(OPENING_SETTING_SX + 1, sy + 1);
                delete_buttons[idx].draw();
                if (delete_buttons[idx].clicked()) {
                    delete_opening(idx);
                    return;
                }
                
                // Edit button
                edit_buttons[idx].move(OPENING_SETTING_SX + 20, sy + 1);
                edit_buttons[idx].draw();
                if (edit_buttons[idx].clicked()) {
                    editing_elem = true;
                    editing_index = idx;
                    text_area[0].text = opening.transcript;
                    text_area[1].text = Format(opening.weight);
                    text_area[0].cursorPos = text_area[0].text.size();
                    text_area[1].cursorPos = text_area[1].text.size();
                    text_area[0].rebuildGlyphs();
                    text_area[1].rebuildGlyphs();
                    text_area[0].active = true;
                    text_area[1].active = false;
                    return;
                }
                
                // Toggle enabled/disabled button
                int toggle_x = OPENING_SETTING_SX + OPENING_SETTING_WIDTH - 20;
                Circle toggle_circle(toggle_x, sy + OPENING_SETTING_HEIGHT / 2, 8);
                if (opening.enabled) {
                    toggle_circle.draw(getData().colors.white);
                } else {
                    toggle_circle.drawFrame(2.0, getData().colors.white);
                }
                if (toggle_circle.leftClicked()) {
                    openings[idx].enabled = !openings[idx].enabled;
                    save_openings();
                    return;
                }
            }
            
            // Draw transcript
            String display_text = opening.transcript;
            if (!opening.enabled) {
                display_text = U"[OFF] " + display_text;
            }
            getData().fonts.font(display_text).draw(15, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 45, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
            
            // Draw weight
            getData().fonts.font(language.get("opening_setting", "weight") + U": ").draw(15, Arg::rightCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 90, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
            getData().fonts.font(Format(std::round(opening.weight))).draw(15, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 90, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
        }
        
        // Draw input area for adding/editing
        void draw_input_area(int sy, int row_index) {
            Rect rect(OPENING_SETTING_SX, sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
            if (row_index % 2) {
                rect.draw(getData().colors.dark_green).drawFrame(1.0, getData().colors.white);
            } else {
                rect.draw(getData().colors.green).drawFrame(1.0, getData().colors.white);
            }
            
            SimpleGUI::TextArea(text_area[0], Vec2{OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 8, sy + OPENING_SETTING_HEIGHT / 2 - 17}, SizeF{600, 30}, SimpleGUI::PreferredTextAreaMaxChars);
            getData().fonts.font(language.get("opening_setting", "weight") + U": ").draw(15, Arg::rightCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 70, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
            SimpleGUI::TextArea(text_area[1], Vec2{OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 70, sy + OPENING_SETTING_HEIGHT / 2 - 17}, SizeF{60, 30}, SimpleGUI::PreferredTextAreaMaxChars);
            
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
        }
        
        // Recursively load all openings from all subfolders and save to forced_openings
        void save_all_openings_to_forced_openings() {
            getData().forced_openings.openings.clear();
            load_all_openings_recursive("");
            getData().forced_openings.init();
        }
        
        // Recursively load all enabled openings from a folder and its subfolders
        void load_all_openings_recursive(const std::string& folder_path) {
            String base = Unicode::Widen(getData().directories.appdata_dir) + U"/forced_openings/";
            if (!folder_path.empty()) {
                base += Unicode::Widen(folder_path) + U"/";
            }
            
            // Load openings from current folder
            const String csv_path = base + U"summary.csv";
            const CSV csv{ csv_path };
            if (csv) {
                for (size_t row = 0; row < csv.rows(); ++row) {
                    String transcript = csv[row][0];
                    double weight = ParseOr<double>(csv[row][1], 1.0);
                    bool enabled = ParseOr<bool>(csv[row][2], true);
                    
                    if (enabled) {  // Only add enabled openings
                        getData().forced_openings.openings.emplace_back(std::make_pair(transcript.narrow(), weight));
                    }
                }
            }
            
            // Recursively load from subfolders
            std::vector<String> folders = enumerate_direct_subdirectories(getData().directories.appdata_dir + "/forced_openings", folder_path);
            for (const auto& folder : folders) {
                std::string next_path = folder_path;
                if (!next_path.empty()) next_path += "/";
                next_path += folder.narrow();
                load_all_openings_recursive(next_path);
            }
        }
};

