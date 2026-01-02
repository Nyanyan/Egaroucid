/*
    Egaroucid Project

    @file opening_setting.hpp
        Forced Opening setting
    @date 2021-2026
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
    
    Opening_abstract() : transcript(U""), weight(1.0) {}
    Opening_abstract(const String& t, double w) : transcript(t), weight(w) {}
};

// CSV file structure
struct Opening_csv_file {
    String filename;  // CSV filename without path (e.g., "default.csv")
    bool enabled;     // Whether this CSV file is enabled
    std::vector<Opening_abstract> openings;
    
    Opening_csv_file() : filename(U""), enabled(true) {}
    Opening_csv_file(const String& f, bool e = true) : filename(f), enabled(e) {}
};


class Opening_setting : public App::Scene {
    private:
        std::vector<Opening_csv_file> csv_files;  // CSV files list
        int selected_csv_index;  // Currently selected CSV file for editing (-1 if none)
        std::vector<ImageButton> delete_buttons;
        std::vector<ImageButton> edit_buttons;
        std::vector<ImageButton> csv_toggle_buttons;  // Toggle buttons for CSV files
        Scroll_manager scroll_manager;
        Button add_button;
        Button add_csv_button;  // Button to create new CSV file
        Button ok_button;
        Button back_button;
        Button register_button;
        Button update_button;
        Button create_csv_button;
        bool adding_elem;
        bool editing_elem;
        bool creating_csv;
        int editing_index;
        TextAreaEditState text_area[2];
        TextAreaEditState csv_name_area;
        
        // Drag and drop state for openings within CSV
        struct DragState {
            bool is_dragging = false;
            int dragged_opening_index = -1;
            Vec2 drag_start_pos;
            Vec2 current_mouse_pos;
            bool mouse_was_down = false;
            static constexpr double DRAG_THRESHOLD = 5.0;
            
            void reset() {
                is_dragging = false;
                dragged_opening_index = -1;
            }
        };
        DragState drag_state;
    
    public:
        Opening_setting(const InitData& init) : IScene{ init } {
            add_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("opening_setting", "add"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            add_csv_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "new_folder"), 20, getData().fonts.font, getData().colors.white, getData().colors.black);
            ok_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            register_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("opening_setting", "register"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            update_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "edit"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            create_csv_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "create"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            
            adding_elem = false;
            editing_elem = false;
            creating_csv = false;
            editing_index = -1;
            selected_csv_index = -1;
            load_csv_files();
        }
    
        void update() override {
            if (System::GetUserActions() & UserAction::CloseButtonClicked) {
                changeScene(U"Close", SCENE_FADE_TIME);
            }
            getData().fonts.font(language.get("opening_setting", "opening_setting")).draw(25, Arg::center(X_CENTER, 30), getData().colors.white);
            
            // Current path label
            String path_label = U"forced_openings/";
            if (selected_csv_index >= 0 && selected_csv_index < (int)csv_files.size()) {
                path_label += csv_files[selected_csv_index].filename;
            }
            getData().fonts.font(path_label).draw(15, Arg::rightCenter(OPENING_SETTING_SX + OPENING_SETTING_WIDTH, 30), getData().colors.white);
            
            // Handle CSV creation mode
            if (creating_csv) {
                back_button.draw();
                if (back_button.clicked() || KeyEscape.down()) {
                    creating_csv = false;
                }
                
                std::string csv_name_str = csv_name_area.text.narrow();
                bool can_create = !csv_name_str.empty() && 
                                 csv_name_str.find("/") == std::string::npos && 
                                 csv_name_str.find("\\") == std::string::npos;
                
                // Add .csv extension if not present
                String csv_filename = Unicode::Widen(csv_name_str);
                if (!csv_filename.ends_with(U".csv")) {
                    csv_filename += U".csv";
                }
                
                if (can_create) {
                    create_csv_button.enable();
                } else {
                    create_csv_button.disable();
                }
                create_csv_button.draw();
                if (create_csv_button.clicked() || (can_create && KeyEnter.down())) {
                    if (create_new_csv_file(csv_filename)) {
                        load_csv_files();
                    }
                    creating_csv = false;
                    csv_name_area.text = U"";
                    csv_name_area.cursorPos = 0;
                    csv_name_area.rebuildGlyphs();
                }
                
                // Draw CSV name input area
                int sy = OPENING_SETTING_SY + 8;
                getData().fonts.font(language.get("in_out", "new_folder") + U":").draw(20, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 8, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
                SimpleGUI::TextArea(csv_name_area, Vec2{OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 150, sy + OPENING_SETTING_HEIGHT / 2 - 17}, SizeF{400, 30}, SimpleGUI::PreferredTextAreaMaxChars);
                
            } else if (adding_elem || editing_elem) {
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
                        if (selected_csv_index >= 0 && selected_csv_index < (int)csv_files.size()) {
                            csv_files[selected_csv_index].openings[editing_index].transcript = Unicode::Widen(transcript);
                            csv_files[selected_csv_index].openings[editing_index].weight = weight;
                            save_csv_file(selected_csv_index);
                        }
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
                        add_opening(Unicode::Widen(transcript), weight);
                        adding_elem = false;
                    }
                }
            } else {
                // Normal mode
                bool can_add = (selected_csv_index >= 0 && selected_csv_index < (int)csv_files.size());
                if (can_add) {
                    add_button.enable();
                } else {
                    add_button.disable();
                }
                add_button.draw();
                if (can_add && add_button.clicked()) {
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
                
                add_csv_button.draw();
                if (add_csv_button.clicked()) {
                    creating_csv = true;
                    csv_name_area.text = U"";
                    csv_name_area.cursorPos = 0;
                    csv_name_area.rebuildGlyphs();
                    csv_name_area.active = true;
                }
                
                ok_button.draw();
                if (ok_button.clicked() || KeyEnter.down()) {
                    // Save all openings and reload forced_openings
                    save_all_csv_files();
                    save_all_openings_to_forced_openings();
                    getData().graph_resources.need_init = false;
                    changeScene(U"Main_scene", SCENE_FADE_TIME);
                }
            }
            
            // Handle drag and drop for reordering openings within CSV
            if (!adding_elem && !editing_elem && !creating_csv && selected_csv_index >= 0) {
                handle_drag_and_drop();
            }
            
            // Draw CSV files and openings list
            if (!creating_csv) {
                draw_list();
                
                // Draw dragged item on top
                if (drag_state.is_dragging) {
                    draw_dragged_item();
                }
            }
            
            if (!(adding_elem || editing_elem || creating_csv)) {
                scroll_manager.draw();
                scroll_manager.update();
            }
        }
    
        void draw() const override {
    
        }
    
    private:
        void init_scroll_manager() {
            int total_openings = 0;
            if (selected_csv_index >= 0 && selected_csv_index < (int)csv_files.size()) {
                total_openings = (int)csv_files[selected_csv_index].openings.size();
            }
            int total = (int)csv_files.size() + total_openings;
            scroll_manager.init(770, OPENING_SETTING_SY + 8, 10, OPENING_SETTING_HEIGHT * OPENING_SETTING_N_GAMES_ON_WINDOW, 20, total, OPENING_SETTING_N_GAMES_ON_WINDOW, OPENING_SETTING_SX, 73, OPENING_SETTING_WIDTH + 10, OPENING_SETTING_HEIGHT * OPENING_SETTING_N_GAMES_ON_WINDOW);
        }
        
        // Get base directory for forced_openings
        String get_base_dir() const {
            return Unicode::Widen(getData().directories.document_dir) + U"/forced_openings/";
        }
        
        // Enumerate CSV files in forced_openings directory
        std::vector<String> enumerate_csv_files() {
            std::vector<String> result;
            String base_dir = get_base_dir();
            
            if (!FileSystem::Exists(base_dir)) {
                FileSystem::CreateDirectories(base_dir);
                return result;
            }
            
            Array<FilePath> list = FileSystem::DirectoryContents(base_dir);
            for (const auto& path : list) {
                if (FileSystem::IsFile(path) && path.ends_with(U".csv")) {
                    String filename = FileSystem::FileName(path);
                    result.emplace_back(filename);
                }
            }
            
            std::sort(result.begin(), result.end());
            return result;
        }
        
        // Load openings from current folder's summary.csv
        void load_openings() {
            openings.clear();
            delete_buttons.clear();
            edit_buttons.clear();
            toggle_buttons.clear();
            move_up_buttons.clear();
            move_down_buttons.clear();
            
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
            Texture edit_image = getData().resources.pencil;  // Use pencil icon for edit
            // We'll use buttons for move up/down instead of image buttons
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
                
                // Placeholder for move buttons (will be drawn differently)
                ImageButton move_up_btn;
                move_up_btn.init(0, 0, 15, cross_image);
                move_up_buttons.emplace_back(move_up_btn);
                
                ImageButton move_down_btn;
                move_down_btn.init(0, 0, 15, cross_image);
                move_down_buttons.emplace_back(move_down_btn);
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
            edit_btn.init(0, 0, 15, getData().resources.pencil);  // Use pencil icon for edit
            edit_buttons.emplace_back(edit_btn);
            
            ImageButton toggle_btn;
            toggle_btn.init(0, 0, 15, getData().resources.cross);
            toggle_buttons.emplace_back(toggle_btn);
            
            ImageButton move_up_btn;
            move_up_btn.init(0, 0, 15, getData().resources.cross);
            move_up_buttons.emplace_back(move_up_btn);
            
            ImageButton move_down_btn;
            move_down_btn.init(0, 0, 15, getData().resources.cross);
            move_down_buttons.emplace_back(move_down_btn);
            
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
            move_up_buttons.erase(move_up_buttons.begin() + idx);
            move_down_buttons.erase(move_down_buttons.begin() + idx);
            
            save_openings();
            
            double strt_idx_double = scroll_manager.get_strt_idx_double();
            init_scroll_manager();
            if ((int)strt_idx_double >= idx) {
                strt_idx_double -= 1.0;
            }
            scroll_manager.set_strt_idx(strt_idx_double);
            std::cerr << "deleted opening " << idx << std::endl;
        }
        
        // Swap two openings (for reordering)
        void swap_openings(int idx1, int idx2) {
            if (idx1 < 0 || idx1 >= (int)openings.size() || idx2 < 0 || idx2 >= (int)openings.size()) return;
            if (idx1 == idx2) return;
            
            std::swap(openings[idx1], openings[idx2]);
            std::swap(delete_buttons[idx1], delete_buttons[idx2]);
            std::swap(edit_buttons[idx1], edit_buttons[idx2]);
            std::swap(toggle_buttons[idx1], toggle_buttons[idx2]);
            std::swap(move_up_buttons[idx1], move_up_buttons[idx2]);
            std::swap(move_down_buttons[idx1], move_down_buttons[idx2]);
            
            save_openings();
        }
        
        // Enumerate current directory
        void enumerate_current_dir() {
            folders_display.clear();
            has_parent = !subfolder.empty();
            
            std::string base_dir = getData().directories.document_dir + "/forced_openings";
            std::vector<String> folders = enumerate_subdirectories_generic(base_dir, subfolder);
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
            
            bool is_being_dragged = (drag_state.is_dragging_folder && drag_state.dragged_folder_name == folder_name);
            Color bg_color = is_being_dragged ? getData().colors.yellow.withAlpha(64) : 
                            (idx % 2 ? getData().colors.dark_green : getData().colors.green);
            Color text_color = is_being_dragged ? getData().colors.white.withAlpha(128) : getData().colors.white;
            
            rect.draw(bg_color).drawFrame(1.0, getData().colors.white);
            
            // Handle drag preparation
            bool mouse_is_down = MouseL.pressed();
            bool mouse_was_down = drag_state.mouse_was_down;
            if (mouse_is_down && !mouse_was_down && rect.contains(Cursor::Pos()) && 
                !drag_state.is_dragging && drag_state.dragged_opening_index == -1 && drag_state.dragged_folder_name.empty()) {
                drag_state.dragged_folder_name = folder_name;
                drag_state.drag_start_pos = Cursor::Pos();
            }
            
            getData().fonts.font(U"📁 " + folder_name).draw(18, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 8, sy + OPENING_SETTING_HEIGHT / 2), text_color);
        }
        
        // Draw opening item
        void draw_opening_item(int idx, int sy, int row_index) {
            const auto& opening = openings[idx];
            Rect rect(OPENING_SETTING_SX, sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
            
            bool is_being_dragged = (drag_state.is_dragging_opening && drag_state.dragged_opening_index == idx);
            Color bg_color = is_being_dragged ? getData().colors.yellow.withAlpha(64) : 
                            (row_index % 2 ? getData().colors.dark_green : getData().colors.green);
            Color text_color = is_being_dragged ? getData().colors.white.withAlpha(128) : getData().colors.white;
            
            rect.draw(bg_color).drawFrame(1.0, getData().colors.white);
            
            if (adding_elem || editing_elem) {
                rect.draw(ColorF{1.0, 1.0, 1.0, 0.5});
            }
            
            // Handle drag preparation
            bool mouse_is_down = MouseL.pressed();
            bool mouse_was_down = drag_state.mouse_was_down;
            if (mouse_is_down && !mouse_was_down && rect.contains(Cursor::Pos()) && 
                !drag_state.is_dragging && drag_state.dragged_opening_index == -1 && drag_state.dragged_folder_name.empty() &&
                !(adding_elem || editing_elem)) {
                drag_state.dragged_opening_index = idx;
                drag_state.drag_start_pos = Cursor::Pos();
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
                
                // Move up button (as text button)
                if (idx > 0) {
                    int btn_x = OPENING_SETTING_SX + 40;
                    int btn_y = sy + 1;
                    Rect up_rect(btn_x, btn_y, 15, 15);
                    if (up_rect.mouseOver()) {
                        up_rect.draw(ColorF(0.8, 0.8, 0.8));
                    } else {
                        up_rect.draw(ColorF(0.6, 0.6, 0.6));
                    }
                    getData().fonts.font(U"▲").draw(10, Arg::center(btn_x + 7.5, btn_y + 7.5), getData().colors.black);
                    if (up_rect.leftClicked()) {
                        swap_openings(idx, idx - 1);
                        return;
                    }
                }
                
                // Move down button (as text button)
                if (idx < (int)openings.size() - 1) {
                    int btn_x = OPENING_SETTING_SX + 58;
                    int btn_y = sy + 1;
                    Rect down_rect(btn_x, btn_y, 15, 15);
                    if (down_rect.mouseOver()) {
                        down_rect.draw(ColorF(0.8, 0.8, 0.8));
                    } else {
                        down_rect.draw(ColorF(0.6, 0.6, 0.6));
                    }
                    getData().fonts.font(U"▼").draw(10, Arg::center(btn_x + 7.5, btn_y + 7.5), getData().colors.black);
                    if (down_rect.leftClicked()) {
                        swap_openings(idx, idx + 1);
                        return;
                    }
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
            getData().fonts.font(display_text).draw(15, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 78, sy + OPENING_SETTING_HEIGHT / 2), text_color);
            
            // Draw weight
            getData().fonts.font(language.get("opening_setting", "weight") + U": ").draw(15, Arg::rightCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 90, sy + OPENING_SETTING_HEIGHT / 2), text_color);
            getData().fonts.font(Format(std::round(opening.weight))).draw(15, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 90, sy + OPENING_SETTING_HEIGHT / 2), text_color);
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
        
        // Draw dragged item
        void draw_dragged_item() {
            Vec2 draw_pos = drag_state.current_mouse_pos;
            draw_pos.x -= OPENING_SETTING_WIDTH / 2;
            draw_pos.y -= OPENING_SETTING_HEIGHT / 2;
            
            Rect drag_rect(draw_pos.x, draw_pos.y, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
            drag_rect.draw(getData().colors.yellow.withAlpha(200)).drawFrame(2.0, getData().colors.white);
            
            if (drag_state.is_dragging_folder) {
                getData().fonts.font(U"📁 " + drag_state.dragged_folder_name).draw(18, Arg::leftCenter(draw_pos.x + OPENING_SETTING_LEFT_MARGIN + 8, draw_pos.y + OPENING_SETTING_HEIGHT / 2), getData().colors.black);
            } else if (drag_state.is_dragging_opening && drag_state.dragged_opening_index >= 0 && drag_state.dragged_opening_index < (int)openings.size()) {
                const auto& opening = openings[drag_state.dragged_opening_index];
                String display_text = opening.transcript;
                if (!opening.enabled) {
                    display_text = U"[OFF] " + display_text;
                }
                getData().fonts.font(display_text).draw(15, Arg::leftCenter(draw_pos.x + OPENING_SETTING_LEFT_MARGIN + 8, draw_pos.y + OPENING_SETTING_HEIGHT / 2), getData().colors.black);
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
            String base = Unicode::Widen(getData().directories.document_dir) + U"/forced_openings/";
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
            std::string base_dir = getData().directories.document_dir + "/forced_openings";
            std::vector<String> folders = enumerate_subdirectories_generic(base_dir, folder_path);
            for (const auto& folder : folders) {
                std::string next_path = folder_path;
                if (!next_path.empty()) next_path += "/";
                next_path += folder.narrow();
                load_all_openings_recursive(next_path);
            }
        }
        
        // Handle drag and drop
        OpeningDragResult handle_drag_and_drop() {
            OpeningDragResult res;
            
            // Update mouse state
            drag_state.current_mouse_pos = Cursor::Pos();
            bool mouse_is_down = MouseL.pressed();
            bool mouse_just_pressed = mouse_is_down && !drag_state.mouse_was_down;
            bool mouse_just_released = !mouse_is_down && drag_state.mouse_was_down;
            drag_state.mouse_was_down = mouse_is_down;
            
            // Handle drag end
            if (drag_state.is_dragging && mouse_just_released) {
                // Check if dropping on parent folder
                if (has_parent) {
                    int parent_sy = OPENING_SETTING_SY + 8;
                    Rect parent_rect(OPENING_SETTING_SX, parent_sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
                    if (parent_rect.contains(drag_state.current_mouse_pos)) {
                        res.drop_completed = true;
                        res.drop_on_parent = true;
                        res.is_dragging_opening = drag_state.is_dragging_opening;
                        res.is_dragging_folder = drag_state.is_dragging_folder;
                        res.dragged_opening_index = drag_state.dragged_opening_index;
                        res.dragged_folder_name = drag_state.dragged_folder_name;
                        drag_state.reset();
                        return res;
                    }
                }
                
                // Check if dropping on a folder
                int sy = OPENING_SETTING_SY + 8;
                int parent_offset = has_parent ? 1 : 0;
                int strt_idx_int = scroll_manager.get_strt_idx_int();
                
                for (int folder_idx = 0; folder_idx < (int)folders_display.size(); ++folder_idx) {
                    int row = parent_offset + folder_idx;
                    if (row >= strt_idx_int && row < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                        int display_row = row - strt_idx_int;
                        int item_sy = sy + display_row * OPENING_SETTING_HEIGHT;
                        Rect folder_rect(OPENING_SETTING_SX, item_sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
                        
                        if (folder_rect.contains(drag_state.current_mouse_pos)) {
                            String target_folder = folders_display[folder_idx];
                            // Don't allow dropping folder on itself
                            if (drag_state.is_dragging_folder && drag_state.dragged_folder_name == target_folder) {
                                drag_state.reset();
                                return res;
                            }
                            
                            res.drop_completed = true;
                            res.drop_target_folder = target_folder;
                            res.is_dragging_opening = drag_state.is_dragging_opening;
                            res.is_dragging_folder = drag_state.is_dragging_folder;
                            res.dragged_opening_index = drag_state.dragged_opening_index;
                            res.dragged_folder_name = drag_state.dragged_folder_name;
                            drag_state.reset();
                            return res;
                        }
                    }
                }
                
                drag_state.reset();
            }
            
            // Check for drag start
            if (mouse_is_down && !drag_state.is_dragging) {
                if ((drag_state.dragged_opening_index >= 0 || !drag_state.dragged_folder_name.empty())) {
                    double distance = drag_state.drag_start_pos.distanceFrom(drag_state.current_mouse_pos);
                    if (distance > DragState::DRAG_THRESHOLD) {
                        drag_state.is_dragging = true;
                        if (drag_state.dragged_opening_index >= 0) {
                            drag_state.is_dragging_opening = true;
                        } else {
                            drag_state.is_dragging_folder = true;
                        }
                    }
                }
            }
            
            // Reset drag preparation on mouse release (but not when dragging)
            if (mouse_just_released && !drag_state.is_dragging) {
                drag_state.dragged_opening_index = -1;
                drag_state.dragged_folder_name.clear();
            }
            
            return res;
        }
        
        // Move opening to a different folder (relative to current subfolder)
        void move_opening_to_folder(int opening_index, const std::string& target_folder) {
            if (opening_index < 0 || opening_index >= (int)openings.size()) return;
            
            const Opening_abstract& opening = openings[opening_index];
            
            // Build target path
            String target_base = Unicode::Widen(getData().directories.document_dir) + U"/forced_openings/";
            if (!subfolder.empty()) {
                target_base += Unicode::Widen(subfolder) + U"/";
            }
            if (!target_folder.empty()) {
                target_base += Unicode::Widen(target_folder) + U"/";
            }
            
            // Ensure target directory exists
            if (!FileSystem::Exists(target_base)) {
                FileSystem::CreateDirectories(target_base);
            }
            
            // Load target CSV
            String target_csv_path = target_base + U"summary.csv";
            CSV target_csv{ target_csv_path };
            CSV new_target_csv;
            
            // Copy existing entries
            for (int i = 0; i < (int)target_csv.rows(); ++i) {
                if (target_csv[i].size() >= 3) {
                    for (int j = 0; j < 3; ++j) {
                        new_target_csv.write(target_csv[i][j]);
                    }
                    new_target_csv.newLine();
                }
            }
            
            // Add moved opening
            new_target_csv.write(opening.transcript);
            new_target_csv.write(Format(opening.weight));
            new_target_csv.write(opening.enabled ? U"true" : U"false");
            new_target_csv.newLine();
            new_target_csv.save(target_csv_path);
            
            // Remove from current folder
            delete_opening(opening_index);
            
            std::cerr << "Moved opening to " << target_folder << std::endl;
        }
        
        // Move opening to parent folder
        void move_opening_to_parent(int opening_index) {
            if (subfolder.empty()) return;  // Already at root
            
            // Get parent folder path
            std::string parent_folder = subfolder;
            if (!parent_folder.empty() && parent_folder.back() == '/') parent_folder.pop_back();
            size_t pos = parent_folder.find_last_of('/');
            if (pos == std::string::npos) parent_folder.clear();
            else parent_folder = parent_folder.substr(0, pos);
            
            move_opening_to_absolute_folder(opening_index, parent_folder);
        }
        
        // Move opening to absolute folder path (from root)
        void move_opening_to_absolute_folder(int opening_index, const std::string& target_folder) {
            if (opening_index < 0 || opening_index >= (int)openings.size()) return;
            
            const Opening_abstract& opening = openings[opening_index];
            
            // Build target path
            String target_base = Unicode::Widen(getData().directories.document_dir) + U"/forced_openings/";
            if (!target_folder.empty()) {
                target_base += Unicode::Widen(target_folder) + U"/";
            }
            
            // Ensure target directory exists
            if (!FileSystem::Exists(target_base)) {
                FileSystem::CreateDirectories(target_base);
            }
            
            // Load target CSV
            String target_csv_path = target_base + U"summary.csv";
            CSV target_csv{ target_csv_path };
            CSV new_target_csv;
            
            // Copy existing entries
            for (int i = 0; i < (int)target_csv.rows(); ++i) {
                if (target_csv[i].size() >= 3) {
                    for (int j = 0; j < 3; ++j) {
                        new_target_csv.write(target_csv[i][j]);
                    }
                    new_target_csv.newLine();
                }
            }
            
            // Add moved opening
            new_target_csv.write(opening.transcript);
            new_target_csv.write(Format(opening.weight));
            new_target_csv.write(opening.enabled ? U"true" : U"false");
            new_target_csv.newLine();
            new_target_csv.save(target_csv_path);
            
            // Remove from current folder
            delete_opening(opening_index);
            
            std::cerr << "Moved opening to " << target_folder << std::endl;
        }
        
        // Move folder to target folder (relative to current subfolder)
        void move_folder_to_folder_target(const std::string& source_folder, const std::string& target_folder) {
            // Build source path
            String source_path = Unicode::Widen(getData().directories.document_dir) + U"/forced_openings/";
            if (!subfolder.empty()) {
                source_path += Unicode::Widen(subfolder) + U"/";
            }
            source_path += Unicode::Widen(source_folder);
            
            // Build target parent path
            String target_parent = Unicode::Widen(getData().directories.document_dir) + U"/forced_openings/";
            if (!subfolder.empty()) {
                target_parent += Unicode::Widen(subfolder) + U"/";
            }
            if (!target_folder.empty()) {
                target_parent += Unicode::Widen(target_folder) + U"/";
            }
            
            if (move_folder(source_path, target_parent, Unicode::Widen(source_folder))) {
                enumerate_current_dir();
                load_openings();
            }
        }
        
        // Move folder to parent folder
        void move_folder_to_parent(const std::string& folder_name) {
            if (subfolder.empty()) return;  // Already at root
            
            // Build source path
            String source_path = Unicode::Widen(getData().directories.document_dir) + U"/forced_openings/";
            if (!subfolder.empty()) {
                source_path += Unicode::Widen(subfolder) + U"/";
            }
            source_path += Unicode::Widen(folder_name);
            
            // Get parent folder path
            String parent_path = Unicode::Widen(getData().directories.document_dir) + U"/forced_openings/";
            std::string parent_folder = subfolder;
            if (!parent_folder.empty() && parent_folder.back() == '/') parent_folder.pop_back();
            size_t pos = parent_folder.find_last_of('/');
            if (pos != std::string::npos) {
                parent_path += Unicode::Widen(parent_folder.substr(0, pos)) + U"/";
            }
            
            if (move_folder(source_path, parent_path, Unicode::Widen(folder_name))) {
                enumerate_current_dir();
                load_openings();
            }
        }
};
