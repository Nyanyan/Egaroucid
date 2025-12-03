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
        ImageButton csv_edit_button;  // Edit button for CSV file name
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
        bool editing_csv_name;
        int editing_index;
        int editing_csv_index;
        int saved_scroll_position;  // Save scroll position when editing starts
        TextAreaEditState text_area[2];
        TextAreaEditState csv_name_area;
        TextAreaEditState csv_rename_area;
        
        // Drag and drop state for openings within CSV
        struct DragState {
            bool is_dragging = false;
            int dragged_opening_index = -1;
            int drop_target_index = -1;
            Vec2 drag_start_pos;
            Vec2 current_mouse_pos;
            bool mouse_was_down = false;
            static constexpr double DRAG_THRESHOLD = 5.0;
            
            void reset() {
                is_dragging = false;
                dragged_opening_index = -1;
                drop_target_index = -1;
            }
        };
        DragState drag_state;

        bool is_csv_index_valid(int idx) const {
            return idx >= 0 && idx < static_cast<int>(csv_files.size());
        }

        bool has_selected_csv() const {
            return is_csv_index_valid(selected_csv_index);
        }

        Opening_csv_file* selected_csv() {
            return has_selected_csv() ? &csv_files[selected_csv_index] : nullptr;
        }

        const Opening_csv_file* selected_csv() const {
            return has_selected_csv() ? &csv_files[selected_csv_index] : nullptr;
        }

        String to_display_filename(const String& filename) const {
            return filename.ends_with(U".csv") ? filename.substr(0, filename.size() - 4) : filename;
        }

        bool is_modal_active() const {
            return adding_elem || editing_elem || creating_csv || editing_csv_name;
        }

        bool is_opening_input_active() const {
            return adding_elem || editing_elem;
        }

        void set_text_areas(const String& transcript, const String& weight) {
            text_area[0].text = transcript;
            text_area[1].text = weight;
            for (auto& area : text_area) {
                area.cursorPos = area.text.size();
                area.rebuildGlyphs();
            }
            text_area[0].active = true;
            text_area[1].active = false;
        }

        void reset_text_areas_for_new_entry() {
            set_text_areas(U"", U"1");
        }

        void reset_text_areas_for_opening(const Opening_abstract& opening) {
            set_text_areas(opening.transcript, Format(opening.weight));
        }
    
    public:
        Opening_setting(const InitData& init) : IScene{ init } {
            add_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("opening_setting", "add"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            add_csv_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("opening_setting", "new_category"), 20, getData().fonts.font, getData().colors.white, getData().colors.black);
            ok_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            register_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("opening_setting", "register"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            update_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            create_csv_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "create"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            csv_edit_button.init(0, 0, 15, getData().resources.pencil);
            
            adding_elem = false;
            editing_elem = false;
            creating_csv = false;
            editing_csv_name = false;
            editing_index = -1;
            editing_csv_index = -1;
            selected_csv_index = -1;
            saved_scroll_position = 0;
            load_csv_files();
        }
    
        void update() override {
            if (System::GetUserActions() & UserAction::CloseButtonClicked) {
                changeScene(U"Close", SCENE_FADE_TIME);
            }
            getData().fonts.font(language.get("opening_setting", "opening_setting")).draw(25, Arg::center(X_CENTER, 30), getData().colors.white);
            
            // Current path label
            String path_label = U"forced_openings/";
            if (const auto* csv = selected_csv()) {
                path_label += to_display_filename(csv->filename);
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
                getData().fonts.font(language.get("opening_setting", "category_name") + U":").draw(20, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 8, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
                SimpleGUI::TextArea(csv_name_area, Vec2{OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 150, sy + OPENING_SETTING_HEIGHT / 2 - 17}, SizeF{400, 30}, SimpleGUI::PreferredTextAreaMaxChars);
                
            } else if (editing_csv_name) {
                back_button.draw();
                if (back_button.clicked() || KeyEscape.down()) {
                    editing_csv_name = false;
                    editing_csv_index = -1;
                }
                
                // Remove invalid characters from CSV filename
                // Invalid characters: newline, /, \, :, *, ?, ", <, >, |
                String filtered_text;
                for (auto ch : csv_rename_area.text) {
                    if (ch != U'\n' && ch != U'\r' && 
                        ch != U'/' && ch != U'\\' && 
                        ch != U':' && ch != U'*' && 
                        ch != U'?' && ch != U'"' && 
                        ch != U'<' && ch != U'>' && 
                        ch != U'|') {
                        filtered_text += ch;
                    }
                }
                if (csv_rename_area.text != filtered_text) {
                    csv_rename_area.text = filtered_text;
                    csv_rename_area.cursorPos = std::min((size_t)csv_rename_area.cursorPos, filtered_text.size());
                    csv_rename_area.rebuildGlyphs();
                }
                
                std::string csv_name_str = csv_rename_area.text.narrow();
                bool can_rename = !csv_name_str.empty();
                
                // Add .csv extension if not present
                String new_csv_filename = Unicode::Widen(csv_name_str);
                if (!new_csv_filename.ends_with(U".csv")) {
                    new_csv_filename += U".csv";
                }
                
                if (can_rename) {
                    update_button.enable();
                } else {
                    update_button.disable();
                }
                update_button.draw();
                if (update_button.clicked()) {
                    if (is_csv_index_valid(editing_csv_index)) {
                        auto& csv_file = csv_files[editing_csv_index];
                        // Rename CSV file
                        String old_path = get_base_dir() + csv_file.filename;
                        String new_path = get_base_dir() + new_csv_filename;
                        
                        // Rename file on disk
                        if (FileSystem::Exists(old_path)) {
                            FileSystem::Rename(old_path, new_path);
                        }
                        
                        // Update filename in memory
                        csv_file.filename = new_csv_filename;
                        
                        // Update settings file
                        save_all_csv_files();
                    }
                    editing_csv_name = false;
                    editing_csv_index = -1;
                }
                
                // Draw in the list (handled by draw_list function)
                
            } else if (adding_elem || editing_elem) {
                back_button.draw();
                if (back_button.clicked() || KeyEscape.down()) {
                    bool was_editing = editing_elem;  // Save state before clearing
                    adding_elem = false;
                    editing_elem = false;
                    editing_index = -1;
                    init_scroll_manager();  // Update scroll manager when exiting edit mode
                    if (was_editing) {
                        // Restore scroll position when canceling edit (after init)
                        scroll_manager.set_strt_idx((double)saved_scroll_position);
                    }
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
                        if (auto* csv = selected_csv()) {
                            if (editing_index >= 0 && editing_index < static_cast<int>(csv->openings.size())) {
                                csv->openings[editing_index].transcript = Unicode::Widen(transcript);
                                csv->openings[editing_index].weight = weight;
                                save_csv_file(selected_csv_index);
                            }
                        }
                        editing_elem = false;
                        editing_index = -1;
                        init_scroll_manager();  // Update scroll manager after editing
                        // Restore scroll position after update (after init)
                        scroll_manager.set_strt_idx((double)saved_scroll_position);
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
                        init_scroll_manager();  // Update scroll manager after adding
                    }
                }
            } else {
                // Normal mode
                bool can_add = has_selected_csv();
                if (can_add) {
                    add_button.enable();
                } else {
                    add_button.disable();
                }
                add_button.draw();
                if (can_add && add_button.clicked()) {
                    adding_elem = true;
                    reset_text_areas_for_new_entry();
                    init_scroll_manager();  // Update scroll manager for new row
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
                    // Save all CSV files and reload forced_openings
                    save_all_csv_files();
                    save_all_openings_to_forced_openings();
                    getData().graph_resources.need_init = false;
                    changeScene(U"Main_scene", SCENE_FADE_TIME);
                }
            }
            
            // Handle drag and drop for reordering openings within CSV
            if (!is_modal_active() && has_selected_csv()) {
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
            
            if (!is_modal_active()) {
                scroll_manager.draw();
                scroll_manager.update();
            }
        }
    
        void draw() const override {
    
        }
    
    private:
        void init_scroll_manager() {
            int total_openings = 0;
            if (const auto* csv = selected_csv()) {
                total_openings = static_cast<int>(csv->openings.size());
            }
            int total = (int)csv_files.size() + total_openings;
            
            // Add 1 for input area only if adding (not editing)
            if (adding_elem) {
                total += 1;
            }
            
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
        
        // Load all CSV files and their enabled states
        void load_csv_files() {
            csv_files.clear();
            delete_buttons.clear();
            edit_buttons.clear();
            csv_toggle_buttons.clear();
            
            // Load CSV file list and enabled states from settings file
            String settings_path = get_base_dir() + U"settings.txt";
            std::unordered_map<String, bool> enabled_map;
            
            if (FileSystem::Exists(settings_path)) {
                TextReader reader(settings_path);
                if (reader) {
                    String line;
                    while (reader.readLine(line)) {
                        auto parts = line.split(U'\t');
                        if (parts.size() >= 2) {
                            String filename = parts[0];
                            bool enabled = ParseOr<bool>(parts[1], true);
                            enabled_map[filename] = enabled;
                        }
                    }
                }
            }
            
            // Enumerate all CSV files
            auto filenames = enumerate_csv_files();
            for (const auto& filename : filenames) {
                Opening_csv_file csv_file(filename);
                csv_file.enabled = enabled_map.count(filename) ? enabled_map[filename] : true;
                load_single_csv_file(csv_file);
                csv_files.emplace_back(csv_file);
            }
            
            // Initialize buttons
            Texture cross_image = getData().resources.cross;
            Texture edit_image = getData().resources.pencil;
            
            for (int i = 0; i < (int)csv_files.size(); ++i) {
                ImageButton toggle_btn;
                toggle_btn.init(0, 0, 15, cross_image);
                csv_toggle_buttons.emplace_back(toggle_btn);
            }
            
            init_scroll_manager();
        }
        
        // Load a single CSV file's contents
        void load_single_csv_file(Opening_csv_file& csv_file) {
            csv_file.openings.clear();
            
            String csv_path = get_base_dir() + csv_file.filename;
            const CSV csv{ csv_path };
            if (csv) {
                for (size_t row = 0; row < csv.rows(); ++row) {
                    if (csv[row].size() >= 2) {
                        Opening_abstract opening;
                        opening.transcript = csv[row][0];
                        opening.weight = ParseOr<double>(csv[row][1], 1.0);
                        csv_file.openings.emplace_back(opening);
                    }
                }
            }
        }
        
        // Save a specific CSV file
        void save_csv_file(int csv_index) {
            if (csv_index < 0 || csv_index >= (int)csv_files.size()) return;
            
            const auto& csv_file = csv_files[csv_index];
            String csv_path = get_base_dir() + csv_file.filename;
            
            CSV csv;
            for (const auto& opening : csv_file.openings) {
                csv.write(opening.transcript);
                csv.write(Format(opening.weight));
                csv.newLine();
            }
            csv.save(csv_path);
        }
        
        // Save all CSV files
        void save_all_csv_files() {
            for (int i = 0; i < (int)csv_files.size(); ++i) {
                save_csv_file(i);
            }
            
            // Save enabled states to settings file
            String settings_path = get_base_dir() + U"settings.txt";
            TextWriter writer(settings_path);
            if (writer) {
                for (const auto& csv_file : csv_files) {
                    writer.writeln(csv_file.filename + U"\t" + (csv_file.enabled ? U"true" : U"false"));
                }
            }
        }
        
        // Create a new CSV file
        bool create_new_csv_file(const String& filename) {
            String csv_path = get_base_dir() + filename;
            
            if (FileSystem::Exists(csv_path)) {
                std::cerr << "CSV file already exists: " << filename.narrow() << std::endl;
                return false;
            }
            
            // Create empty CSV file
            CSV csv;
            csv.save(csv_path);
            
            std::cerr << "Created CSV file: " << filename.narrow() << std::endl;
            return true;
        }
        
        // Add new opening to currently selected CSV
        void add_opening(const String& transcript, double weight) {
            auto* csv = selected_csv();
            if (!csv) return;
            
            Opening_abstract opening(transcript, weight);
            csv->openings.emplace_back(opening);
            
            // Add buttons for new opening
            ImageButton delete_btn;
            delete_btn.init(0, 0, 15, getData().resources.cross);
            delete_buttons.emplace_back(delete_btn);
            
            ImageButton edit_btn;
            edit_btn.init(0, 0, 15, getData().resources.pencil);
            edit_buttons.emplace_back(edit_btn);
            
            if (has_selected_csv()) {
                save_csv_file(selected_csv_index);
            }
            init_scroll_manager();
        }
        
        // Delete opening from currently selected CSV
        void delete_opening(int idx) {
            auto* csv = selected_csv();
            if (!csv) return;
            if (idx < 0 || idx >= static_cast<int>(csv->openings.size())) return;
            
            csv->openings.erase(csv->openings.begin() + idx);
            
            // Rebuild buttons for this CSV's openings
            rebuild_opening_buttons();
            
            if (has_selected_csv()) {
                save_csv_file(selected_csv_index);
            }
            
            double strt_idx_double = scroll_manager.get_strt_idx_double();
            init_scroll_manager();
            if ((int)strt_idx_double >= idx) {
                strt_idx_double -= 1.0;
            }
            scroll_manager.set_strt_idx(strt_idx_double);
            std::cerr << "deleted opening " << idx << std::endl;
        }
        
        // Rebuild buttons for currently selected CSV's openings
        void rebuild_opening_buttons() {
            delete_buttons.clear();
            edit_buttons.clear();
            
            const auto* csv = selected_csv();
            if (!csv) return;
            
            Texture cross_image = getData().resources.cross;
            Texture edit_image = getData().resources.pencil;
            
            for (int i = 0; i < (int)csv->openings.size(); ++i) {
                ImageButton delete_btn;
                delete_btn.init(0, 0, 15, cross_image);
                delete_buttons.emplace_back(delete_btn);
                
                ImageButton edit_btn;
                edit_btn.init(0, 0, 15, edit_image);
                edit_buttons.emplace_back(edit_btn);
            }
        }
        
        // Move opening within CSV (for drag and drop)
        void move_opening(int from_idx, int to_idx) {
            auto* csv = selected_csv();
            if (!csv) return;
            
            auto& openings = csv->openings;
            if (from_idx < 0 || from_idx >= (int)openings.size()) return;
            if (to_idx < 0 || to_idx >= (int)openings.size()) return;
            if (from_idx == to_idx) return;
            
            Opening_abstract temp = openings[from_idx];
            openings.erase(openings.begin() + from_idx);
            openings.insert(openings.begin() + to_idx, temp);
            
            if (has_selected_csv()) {
                save_csv_file(selected_csv_index);
            }
        }
        
        // Draw CSV files and openings list
        void draw_list() {
            int strt_idx_int = scroll_manager.get_strt_idx_int();
            
            // First pass: determine if fixed header will be shown
            int row_index = 0;
            int csv_row_for_selected = -1;
            bool fixed_header_shown = false;
            for (int i = 0; i < (int)csv_files.size(); ++i) {
                if (i == selected_csv_index) {
                    csv_row_for_selected = row_index;
                    if (row_index < strt_idx_int) {
                        fixed_header_shown = true;
                    }
                }
                row_index++;
                if (i == selected_csv_index) {
                    if (adding_elem && !editing_elem) {
                        row_index++;
                    }
                    row_index += (int)csv_files[i].openings.size();
                }
            }
            
            // Draw fixed header at top if scrolled (ABOVE OPENING_SETTING_SY)
            if (fixed_header_shown && has_selected_csv()) {
                draw_csv_file_item_fixed_header(selected_csv_index, OPENING_SETTING_SY - OPENING_SETTING_HEADER_HEIGHT);
            }
            
            // Normal elements always start from OPENING_SETTING_SY
            int sy = OPENING_SETTING_SY;
            
            if (strt_idx_int > 0 && !fixed_header_shown) {
                getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, sy }, getData().colors.white);
            }
            sy += 8;
            
            int total_items = (int)csv_files.size();
            if (const auto* csv = selected_csv()) {
                total_items += static_cast<int>(csv->openings.size());
            }
            
            if (!adding_elem && !editing_elem && !editing_csv_name && total_items == 0) {
                getData().fonts.font(language.get("opening_setting", "no_opening_found")).draw(20, Arg::center(X_CENTER, Y_CENTER), getData().colors.white);
                return;
            }
            
            // Second pass: draw items
            row_index = 0;
            csv_row_for_selected = -1;
            
            // Draw CSV files
            for (int i = 0; i < (int)csv_files.size(); ++i) {
                if (i == selected_csv_index) {
                    csv_row_for_selected = row_index;
                }
                
                // Draw CSV file item only if in visible range and not showing fixed header
                if (row_index >= strt_idx_int && row_index < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                    if (!(fixed_header_shown && i == selected_csv_index)) {
                        // Don't draw the CSV file here if it will be shown as fixed header
                        int display_row = row_index - strt_idx_int;
                        int item_sy = sy + display_row * OPENING_SETTING_HEIGHT;
                        draw_csv_file_item(i, item_sy, row_index);
                    }
                }
                row_index++;
                
                // Draw openings if this CSV is selected
                if (i == selected_csv_index) {
                    // Draw input area for adding right after CSV file if adding
                    if (adding_elem && !editing_elem) {
                        bool should_draw = false;
                        int item_sy;
                        
                        if (fixed_header_shown && row_index <= strt_idx_int) {
                            // Fixed header shown and this row is at or before scroll position
                            // Draw at the top of scroll area (right after fixed header)
                            should_draw = true;
                            item_sy = sy;
                        } else if (row_index >= strt_idx_int && row_index < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                            should_draw = true;
                            int display_row = row_index - strt_idx_int;
                            item_sy = sy + display_row * OPENING_SETTING_HEIGHT;
                        }
                        
                        if (should_draw) {
                            draw_input_area(item_sy, row_index);
                        }
                        row_index++;
                    }
                    
                    for (int j = 0; j < (int)csv_files[i].openings.size(); ++j) {
                        bool should_draw = false;
                        int item_sy;
                        
                        if (fixed_header_shown && csv_row_for_selected < strt_idx_int) {
                            // Fixed header is shown (CSV is scrolled out)
                            // All children of the CSV should be drawn from the top
                            int offset_from_csv = row_index - csv_row_for_selected - 1;  // -1 for CSV row itself
                            if (offset_from_csv >= 0) {
                                // Calculate which child is visible in the window
                                int visible_start = strt_idx_int - csv_row_for_selected - 1;  // First visible child index
                                int local_offset = offset_from_csv - visible_start;
                                
                                if (local_offset >= 0 && local_offset < OPENING_SETTING_N_GAMES_ON_WINDOW) {
                                    should_draw = true;
                                    item_sy = sy + local_offset * OPENING_SETTING_HEIGHT;
                                }
                            }
                        } else if (row_index >= strt_idx_int && row_index < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                            should_draw = true;
                            int display_row = row_index - strt_idx_int;
                            item_sy = sy + display_row * OPENING_SETTING_HEIGHT;
                        }
                        
                        if (should_draw) {
                            draw_opening_item(j, item_sy, row_index);
                        }
                        row_index++;
                    }
                }
            }
            
            if (strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW < total_items + (adding_elem ? 1 : 0)) {
                getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, 395}, getData().colors.white);
            }
        }
        
        // Draw CSV file item
        void draw_csv_file_item(int idx, int sy, int row_index) {
            const auto& csv_file = csv_files[idx];
            Rect rect(OPENING_SETTING_SX, sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
            
            bool is_selected = (idx == selected_csv_index);
            Color bg_color;
            if (is_selected) {
                bg_color = getData().colors.darkblue;
            } else if (!csv_file.enabled) {
                // Grayed out for disabled categories
                bg_color = (row_index % 2 ? Color(77, 77, 77) : Color(64, 64, 64));
            } else {
                bg_color = (row_index % 2 ? getData().colors.dark_green : getData().colors.green);
            }
            
            rect.draw(bg_color).drawFrame(1.0, getData().colors.white);
            
            // Text color: dimmed for disabled categories
            Color text_color = csv_file.enabled ? getData().colors.white : Color(128, 128, 128);
            
            // Edit button for CSV file name (top-left corner) - check click first
            bool should_enter_edit_mode = false;
            if (!(adding_elem || editing_elem || editing_csv_name)) {
                // Draw edit button and check if it was clicked
                // Must check click AFTER drawing with the correct position for this item
                csv_edit_button.move(OPENING_SETTING_SX + 1, sy + 1);
                csv_edit_button.draw();
                
                // Create a rect for this specific edit button to check clicks independently
                Rect edit_button_rect(OPENING_SETTING_SX + 1, sy + 1, 15, 15);
                if (edit_button_rect.leftClicked()) {
                    should_enter_edit_mode = true;
                }
            }
            
            // Draw CSV filename (without .csv extension) or edit mode
            if (editing_csv_name && editing_csv_index == idx) {
                // Edit mode: show text input
                SimpleGUI::TextArea(csv_rename_area, Vec2{OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 36, sy + OPENING_SETTING_HEIGHT / 2 - 17}, SizeF{400, 30}, SimpleGUI::PreferredTextAreaMaxChars);
            } else {
                // Normal display mode
                String display_filename = to_display_filename(csv_file.filename);
                
                // Draw folder icon
                int icon_x = OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 8;
                int icon_y = sy + OPENING_SETTING_HEIGHT / 2 - 8;
                getData().resources.folder.resized(16, 16).draw(icon_x, icon_y, text_color);
                
                getData().fonts.font(display_filename).draw(18, Arg::leftCenter(icon_x + 20, sy + OPENING_SETTING_HEIGHT / 2), text_color);
            }
            
            // Handle double-click to select/deselect
            // Only process rect clicks if edit button was NOT clicked
            if (!editing_csv_name && !should_enter_edit_mode) {
                // Check if mouse is on edit button area to exclude it from rect clicks
                Rect edit_button_rect(OPENING_SETTING_SX + 1, sy + 1, 15, 15);
                bool mouse_on_edit_button = edit_button_rect.leftPressed() || edit_button_rect.leftClicked();
                
                if (!mouse_on_edit_button && rect.leftClicked() && !(adding_elem || editing_elem)) {
                    static uint64_t last_click_time = 0;
                    static int last_clicked_csv = -1;
                    uint64_t current_time = Time::GetMillisec();
                    constexpr uint64_t DOUBLE_CLICK_TIME_MS = 400;
                    
                    if (current_time - last_click_time < DOUBLE_CLICK_TIME_MS && last_clicked_csv == idx) {
                        // Double-click: toggle selection
                        if (selected_csv_index == idx) {
                            selected_csv_index = -1;  // Deselect
                        } else {
                            selected_csv_index = idx;  // Select
                            rebuild_opening_buttons();
                        }
                        init_scroll_manager();
                    }
                    last_click_time = current_time;
                    last_clicked_csv = idx;
                }
            }

            // Toggle enabled/disabled with checkbox
            if (!(adding_elem || editing_elem || editing_csv_name)) {
                int checkbox_x = OPENING_SETTING_SX + OPENING_SETTING_WIDTH - 20;
                int checkbox_y = sy + OPENING_SETTING_HEIGHT / 2 - 8;
                Texture checkbox_icon = csv_file.enabled ? getData().resources.checkbox : getData().resources.unchecked;
                checkbox_icon.resized(16, 16).draw(checkbox_x, checkbox_y);
                
                Rect checkbox_rect(checkbox_x, checkbox_y, 16, 16);
                if (checkbox_rect.leftClicked()) {
                    csv_files[idx].enabled = !csv_files[idx].enabled;
                    save_all_csv_files();
                }
            }
            
            // Draw opening count
            String count_text = U"(" + Format((int)csv_file.openings.size()) + U")";
            getData().fonts.font(count_text).draw(15, Arg::rightCenter(OPENING_SETTING_SX + OPENING_SETTING_WIDTH - 30, sy + OPENING_SETTING_HEIGHT / 2), text_color);
            
            // Enter edit mode if button was clicked (at the end, after all drawing)
            if (should_enter_edit_mode) {
                editing_csv_name = true;
                editing_csv_index = idx;
                String display_filename = to_display_filename(csv_file.filename);
                csv_rename_area.text = display_filename;
                csv_rename_area.cursorPos = csv_rename_area.text.size();
                csv_rename_area.rebuildGlyphs();
                csv_rename_area.active = true;
            }
        }
        
        // Draw CSV file item as fixed header (when scrolled out of view)
        // Display only version with smaller height - no editing or toggling
        void draw_csv_file_item_fixed_header(int idx, int sy) {
            const auto& csv_file = csv_files[idx];
            Rect rect(OPENING_SETTING_SX, sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEADER_HEIGHT);
            
            // Always use selected color for fixed header
            Color bg_color = getData().colors.darkblue;
            rect.draw(bg_color).drawFrame(1.0, getData().colors.white);
            
            // Text color
            Color text_color = csv_file.enabled ? getData().colors.white : Color(128, 128, 128);
            
            // Display CSV filename (without .csv extension) - no editing
            String display_filename = to_display_filename(csv_file.filename);
            
            // Draw folder icon (smaller)
            int icon_x = OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 4;
            int icon_y = sy + OPENING_SETTING_HEADER_HEIGHT / 2 - 6;
            getData().resources.folder.resized(12, 12).draw(icon_x, icon_y, text_color);
            
            // Draw filename (smaller font)
            getData().fonts.font(display_filename).draw(14, Arg::leftCenter(icon_x + 16, sy + OPENING_SETTING_HEADER_HEIGHT / 2), text_color);
            
            // Draw opening count (smaller)
            String count_text = U"(" + Format((int)csv_file.openings.size()) + U")";
            getData().fonts.font(count_text).draw(12, Arg::rightCenter(OPENING_SETTING_SX + OPENING_SETTING_WIDTH - 10, sy + OPENING_SETTING_HEADER_HEIGHT / 2), text_color);
        }
        
        // Draw opening item
        void draw_opening_item(int idx, int sy, int row_index) {
            const auto* csv = selected_csv();
            if (!csv) return;
            if (idx < 0 || idx >= static_cast<int>(csv->openings.size())) return;
            const auto& opening = csv->openings[idx];
            
            Rect rect(OPENING_SETTING_SX + 20, sy, OPENING_SETTING_WIDTH - 20, OPENING_SETTING_HEIGHT);
            
            bool is_being_edited = (editing_elem && editing_index == idx);
            bool is_being_dragged = (drag_state.is_dragging && drag_state.dragged_opening_index == idx);
            
            Color bg_color = is_being_dragged ? getData().colors.yellow.withAlpha(64) : 
                            (row_index % 2 ? getData().colors.dark_green : getData().colors.green);
            Color text_color = is_being_dragged ? getData().colors.white.withAlpha(128) : getData().colors.white;
            
            rect.draw(bg_color).drawFrame(1.0, getData().colors.white);
            
            // Grayout if adding or editing other element
            if ((adding_elem || (editing_elem && !is_being_edited))) {
                rect.draw(ColorF{1.0, 1.0, 1.0, 0.5});
            }
            
            // Handle drag preparation
            bool mouse_is_down = MouseL.pressed();
            bool mouse_was_down = drag_state.mouse_was_down;
            if (mouse_is_down && !mouse_was_down && rect.contains(Cursor::Pos()) && 
                !drag_state.is_dragging &&
                !is_opening_input_active()) {
                drag_state.dragged_opening_index = idx;
                drag_state.drag_start_pos = Cursor::Pos();
            }
            
            if (!is_opening_input_active()) {
                // Delete button
                if (idx < (int)delete_buttons.size()) {
                    delete_buttons[idx].move(OPENING_SETTING_SX + 21, sy + 1);
                    delete_buttons[idx].draw();
                    if (delete_buttons[idx].clicked()) {
                        delete_opening(idx);
                        return;
                    }
                }
                
                // Edit button
                if (idx < (int)edit_buttons.size()) {
                    edit_buttons[idx].move(OPENING_SETTING_SX + 40, sy + 1);
                    edit_buttons[idx].draw();
                    if (edit_buttons[idx].clicked()) {
                        // Save current scroll position
                        saved_scroll_position = scroll_manager.get_strt_idx_int();
                        
                        editing_elem = true;
                        editing_index = idx;
                        reset_text_areas_for_opening(opening);
                        return;
                    }
                }
            }
            
            // Draw transcript
            getData().fonts.font(opening.transcript).draw(15, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 78, sy + OPENING_SETTING_HEIGHT / 2), text_color);
            
            // Draw weight
            getData().fonts.font(language.get("opening_setting", "weight") + U": ").draw(15, Arg::rightCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 90, sy + OPENING_SETTING_HEIGHT / 2), text_color);
            getData().fonts.font(Format(std::round(opening.weight))).draw(15, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 90, sy + OPENING_SETTING_HEIGHT / 2), text_color);
            
            // If this element is being edited, draw text boxes on top
            if (is_being_edited) {
                SimpleGUI::TextArea(text_area[0], Vec2{OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 78, sy + OPENING_SETTING_HEIGHT / 2 - 15}, SizeF{480, 30}, SimpleGUI::PreferredTextAreaMaxChars);
                SimpleGUI::TextArea(text_area[1], Vec2{OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 70, sy + OPENING_SETTING_HEIGHT / 2 - 15}, SizeF{60, 30}, SimpleGUI::PreferredTextAreaMaxChars);
                
                // Handle tab key to switch between text areas
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
                        text_area[(i + 1) % 2].text = Unicode::Widen(txt1);
                        text_area[i].rebuildGlyphs();
                        text_area[(i + 1) % 2].rebuildGlyphs();
                    }
                }
            }
        }
        
        // Draw input area for adding/editing
        void draw_input_area(int sy, int row_index) {
            Rect rect(OPENING_SETTING_SX + 20, sy, OPENING_SETTING_WIDTH - 20, OPENING_SETTING_HEIGHT);
            if (row_index % 2) {
                rect.draw(getData().colors.dark_green).drawFrame(1.0, getData().colors.white);
            } else {
                rect.draw(getData().colors.green).drawFrame(1.0, getData().colors.white);
            }
            
            SimpleGUI::TextArea(text_area[0], Vec2{OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 28, sy + OPENING_SETTING_HEIGHT / 2 - 17}, SizeF{580, 30}, SimpleGUI::PreferredTextAreaMaxChars);
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
            const auto* csv = selected_csv();
            if (!csv) return;
            if (drag_state.dragged_opening_index < 0 || drag_state.dragged_opening_index >= static_cast<int>(csv->openings.size())) return;
            
            Vec2 draw_pos = drag_state.current_mouse_pos;
            draw_pos.x -= (OPENING_SETTING_WIDTH - 20) / 2;
            draw_pos.y -= OPENING_SETTING_HEIGHT / 2;
            
            Rect drag_rect(draw_pos.x, draw_pos.y, OPENING_SETTING_WIDTH - 20, OPENING_SETTING_HEIGHT);
            drag_rect.draw(getData().colors.yellow.withAlpha(200)).drawFrame(2.0, getData().colors.white);
            
            const auto& opening = csv->openings[drag_state.dragged_opening_index];
            getData().fonts.font(opening.transcript).draw(15, Arg::leftCenter(draw_pos.x + OPENING_SETTING_LEFT_MARGIN + 8, draw_pos.y + OPENING_SETTING_HEIGHT / 2), getData().colors.black);
        }
        
        // Handle drag and drop
        void handle_drag_and_drop() {
            if (!has_selected_csv()) return;
            
            // Update mouse state
            drag_state.current_mouse_pos = Cursor::Pos();
            bool mouse_is_down = MouseL.pressed();
            bool mouse_just_released = !mouse_is_down && drag_state.mouse_was_down;
            drag_state.mouse_was_down = mouse_is_down;
            
            // Handle drag end
            if (drag_state.is_dragging && mouse_just_released) {
                // Find drop target
                if (drag_state.drop_target_index >= 0) {
                    move_opening(drag_state.dragged_opening_index, drag_state.drop_target_index);
                    rebuild_opening_buttons();
                }
                drag_state.reset();
                return;
            }
            
            // Check for drag start
            if (mouse_is_down && !drag_state.is_dragging && drag_state.dragged_opening_index >= 0) {
                double distance = drag_state.drag_start_pos.distanceFrom(drag_state.current_mouse_pos);
                if (distance > DragState::DRAG_THRESHOLD) {
                    drag_state.is_dragging = true;
                }
            }
            
            // Update drop target while dragging
            if (drag_state.is_dragging) {
                drag_state.drop_target_index = -1;
                
                int sy = OPENING_SETTING_SY + 8;
                int strt_idx_int = scroll_manager.get_strt_idx_int();
                int row_index = 0;
                
                // Skip CSV file rows
                for (int i = 0; i < (int)csv_files.size(); ++i) {
                    row_index++;
                    if (i == selected_csv_index) {
                        // Check openings in selected CSV
                        for (int j = 0; j < (int)csv_files[i].openings.size(); ++j) {
                            if (row_index >= strt_idx_int && row_index < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                                int display_row = row_index - strt_idx_int;
                                int item_sy = sy + display_row * OPENING_SETTING_HEIGHT;
                                Rect rect(OPENING_SETTING_SX + 20, item_sy, OPENING_SETTING_WIDTH - 20, OPENING_SETTING_HEIGHT);
                                
                                if (rect.contains(drag_state.current_mouse_pos) && j != drag_state.dragged_opening_index) {
                                    drag_state.drop_target_index = j;
                                    break;
                                }
                            }
                            row_index++;
                        }
                        break;
                    }
                }
            }
            
            // Reset drag preparation on mouse release (but not when dragging)
            if (mouse_just_released && !drag_state.is_dragging) {
                drag_state.dragged_opening_index = -1;
            }
        }
        
        // Load all enabled openings from all enabled CSV files and save to forced_openings
        void save_all_openings_to_forced_openings() {
            getData().forced_openings.openings.clear();
            
            for (const auto& csv_file : csv_files) {
                // std::cerr << csv_file.filename.narrow() << " " << csv_file.enabled << std::endl;
                if (csv_file.enabled) {  // Only load from enabled CSV files
                    for (const auto& opening : csv_file.openings) {
                        getData().forced_openings.openings.emplace_back(std::make_pair(opening.transcript.narrow(), opening.weight));
                    }
                }
            }
            
            getData().forced_openings.init();
            std::cerr << "Loaded " << getData().forced_openings.openings.size() << " openings from enabled CSV files" << std::endl;
        }
};


