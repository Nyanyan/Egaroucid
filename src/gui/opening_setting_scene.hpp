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
#include <algorithm>
#include <cctype>
#include <string>
#include <stdexcept>
#include <Siv3D.hpp>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"


// Opening abstract structure for list display
struct Opening_abstract {
    String transcript;
    double weight;
    bool enabled;
    
    Opening_abstract() : transcript(U""), weight(1.0), enabled(true) {}
    Opening_abstract(const String& t, double w, bool e = true) : transcript(t), weight(w), enabled(e) {}
};

struct Folder_entry {
    String name;
    std::string relative_path;
    bool enabled;
    
    Folder_entry() : name(U""), relative_path(""), enabled(true) {}
    Folder_entry(const String& n, const std::string& path, bool e = true) : name(n), relative_path(path), enabled(e) {}
};


class Opening_setting : public App::Scene {
private:
    std::vector<Opening_abstract> openings;
    std::vector<ImageButton> delete_buttons;
    std::vector<ImageButton> edit_buttons;
    std::vector<ImageButton> toggle_buttons;
    std::vector<ImageButton> move_up_buttons;
    std::vector<ImageButton> move_down_buttons;
    std::vector<Folder_entry> folders_display;
    Scroll_manager scroll_manager;
    Button add_button;
    Button add_csv_button;  // Button to create new CSV file
    Button ok_button;
    Button back_button;
    Button register_button;
    Button create_csv_button;
    Button inline_edit_back_button;
    Button inline_edit_ok_button;
    bool has_parent;
    std::string subfolder;
    bool adding_elem;
    bool editing_elem;
    bool creating_csv;
    int editing_index;
    TextAreaEditState text_area[2];
    TextAreaEditState csv_name_area;
    bool renaming_folder;
    int renaming_folder_index;
    TextAreaEditState folder_rename_area;
    bool current_folder_effective_enabled;

    struct InlineEditLayout {
        double transcript_x;
        double transcript_width;
        double secondary_x;
        double secondary_width;
        double text_y;
        double field_height;
        double back_x;
        double ok_x;
        double buttons_y;
    };

    InlineEditLayout get_inline_edit_layout(double sy) const {
        InlineEditLayout layout;
        constexpr double control_margin = 10.0;
        layout.field_height = 30.0;
        double row_center = sy + OPENING_SETTING_HEIGHT / 2.0;
        layout.text_y = row_center - layout.field_height / 2.0 - 2.0;
        layout.transcript_x = OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 8;

        double ok_width = inline_edit_ok_button.rect.w;
        double back_width = inline_edit_back_button.rect.w;
        layout.ok_x = OPENING_SETTING_SX + OPENING_SETTING_WIDTH - ok_width - control_margin;
        layout.back_x = layout.ok_x - back_width - control_margin;

        layout.secondary_width = 70.0;
        layout.secondary_x = layout.back_x - layout.secondary_width - control_margin;
        layout.transcript_width = layout.secondary_x - layout.transcript_x - control_margin;
        if (layout.transcript_width < 120.0) {
            layout.transcript_width = 120.0;
        }
        layout.buttons_y = sy + (OPENING_SETTING_HEIGHT - inline_edit_back_button.rect.h) / 2.0;
        return layout;
    }

    bool draw_inline_back_button(const InlineEditLayout& layout) {
        inline_edit_back_button.move((int)layout.back_x, (int)layout.buttons_y);
        inline_edit_back_button.enable();
        inline_edit_back_button.draw();
        return inline_edit_back_button.clicked();
    }

    bool draw_inline_ok_button(const InlineEditLayout& layout, bool can_commit) {
        inline_edit_ok_button.move((int)layout.ok_x, (int)layout.buttons_y);
        if (can_commit) {
            inline_edit_ok_button.enable();
        } else {
            inline_edit_ok_button.disable();
        }
        inline_edit_ok_button.draw();
        return can_commit && inline_edit_ok_button.clicked();
    }

    bool is_locking_bottom_buttons() const {
        return editing_elem || renaming_folder;
    }

    // Drag and drop state for openings within CSV
    struct DragState {
        bool is_dragging = false;
        bool is_dragging_opening = false;
        bool is_dragging_folder = false;
        int dragged_opening_index = -1;
        String dragged_folder_name;
        Vec2 drag_start_pos;
        Vec2 current_mouse_pos;
        bool mouse_was_down = false;
        static constexpr double DRAG_THRESHOLD = 5.0;

        void reset() {
            is_dragging = false;
            is_dragging_opening = false;
            is_dragging_folder = false;
            dragged_opening_index = -1;
            dragged_folder_name.clear();
            drag_start_pos = Vec2{ 0, 0 };
            mouse_was_down = false;
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
        create_csv_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "create"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        inline_edit_back_button.init(0, 0, 80, 30, 5, language.get("common", "back"), 18, getData().fonts.font, getData().colors.white, getData().colors.black);
        inline_edit_ok_button.init(0, 0, 70, 30, 5, language.get("common", "ok"), 18, getData().fonts.font, getData().colors.white, getData().colors.black);

        has_parent = false;
        subfolder.clear();
        adding_elem = false;
        editing_elem = false;
        creating_csv = false;
        editing_index = -1;
        renaming_folder = false;
        renaming_folder_index = -1;
        current_folder_effective_enabled = true;
        enumerate_current_dir();
        load_openings();
    }
    
        void update() override {
            if (System::GetUserActions() & UserAction::CloseButtonClicked) {
                changeScene(U"Close", SCENE_FADE_TIME);
            }
            getData().fonts.font(language.get("opening_setting", "opening_setting")).draw(25, Arg::center(X_CENTER, 30), getData().colors.white);
            
            // Current path label
            String path_label = U"forced_openings/";
            if (!subfolder.empty()) {
                path_label += Unicode::Widen(subfolder) + U"/";
            }
            getData().fonts.font(path_label).draw(15, Arg::rightCenter(OPENING_SETTING_SX + OPENING_SETTING_WIDTH, 30), getData().colors.white);
            
            bool enter_pressed = KeyEnter.down();
            bool rename_textbox_active = renaming_folder && folder_rename_area.active;

            // Handle CSV creation mode
            if (creating_csv) {
                back_button.draw();
                if (back_button.clicked() || KeyEscape.down()) {
                    creating_csv = false;
                }
                
                // Draw CSV name input area
                int sy = OPENING_SETTING_SY + 8;
                getData().fonts.font(language.get("in_out", "new_folder") + U":").draw(20, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 8, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
                SimpleGUI::TextArea(csv_name_area, Vec2{OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 150, sy + OPENING_SETTING_HEIGHT / 2 - 17}, SizeF{400, 30}, SimpleGUI::PreferredTextAreaMaxChars);
                String sanitized = sanitize_folder_text(csv_name_area.text);
                if (sanitized != csv_name_area.text) {
                    size_t old_cursor = csv_name_area.cursorPos;
                    csv_name_area.text = sanitized;
                    csv_name_area.cursorPos = std::min(old_cursor, sanitized.size());
                    csv_name_area.rebuildGlyphs();
                }
                String folder_name = csv_name_area.text.trimmed();
                bool can_create = is_valid_folder_name(folder_name);
                
                if (can_create) {
                    create_csv_button.enable();
                } else {
                    create_csv_button.disable();
                }
                create_csv_button.draw();
                if (create_csv_button.clicked()) {
                    if (create_new_folder(folder_name)) {
                        enumerate_current_dir();
                    }
                    load_openings();
                    creating_csv = false;
                    csv_name_area.text = U"";
                    csv_name_area.cursorPos = 0;
                    csv_name_area.rebuildGlyphs();
                }
                
            } else if (adding_elem) {
                back_button.draw();
                if (back_button.clicked() || KeyEscape.down()) {
                    adding_elem = false;
                }
                
                String transcript_candidate;
                double weight_candidate = 0.0;
                bool can_be_registered = collect_opening_form_payload(transcript_candidate, weight_candidate);
                if (can_be_registered) {
                    register_button.enable();
                } else {
                    register_button.disable();
                }
                register_button.draw();
                if (register_button.clicked() || (can_be_registered && enter_pressed)) {
                    add_opening(transcript_candidate, weight_candidate);
                    adding_elem = false;
                }
            } else {
                // Normal mode
                bool bottom_locked = is_locking_bottom_buttons();
                if (!bottom_locked) {
                    add_button.enable();
                    add_button.draw();
                    if (add_button.clicked()) {
                        adding_elem = true;
                        cancel_folder_rename();
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

                    add_csv_button.enable();
                    add_csv_button.draw();
                    if (add_csv_button.clicked()) {
                        creating_csv = true;
                        csv_name_area.text = U"";
                        csv_name_area.cursorPos = 0;
                        csv_name_area.rebuildGlyphs();
                        csv_name_area.active = true;
                        cancel_folder_rename();
                    }

                    ok_button.draw();
                    if (ok_button.clicked() || (enter_pressed && !rename_textbox_active)) {
                        // Save all openings and reload forced_openings
                        save_openings();
                        save_all_openings_to_forced_openings();
                        getData().graph_resources.need_init = false;
                        changeScene(U"Main_scene", SCENE_FADE_TIME);
                    }
                } else {
                    add_button.disable_notransparent();
                    add_csv_button.disable_notransparent();
                }
            }
            
            // Handle drag and drop for reordering openings within CSV
            if (!adding_elem && !editing_elem && !creating_csv && !renaming_folder) {
                handle_drag_and_drop();
            }

            if (renaming_folder && KeyEscape.down()) {
                cancel_folder_rename();
            }
            if (editing_elem && KeyEscape.down()) {
                cancel_opening_edit();
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
            int total = (has_parent ? 1 : 0) + (int)folders_display.size() + (int)openings.size();
            if (adding_elem || editing_elem) {
                total += 1;
            }
            scroll_manager.init(770, OPENING_SETTING_SY + 8, 10, OPENING_SETTING_HEIGHT * OPENING_SETTING_N_GAMES_ON_WINDOW, 20, total, OPENING_SETTING_N_GAMES_ON_WINDOW, OPENING_SETTING_SX, 73, OPENING_SETTING_WIDTH + 10, OPENING_SETTING_HEIGHT * OPENING_SETTING_N_GAMES_ON_WINDOW);
        }
        
        std::string build_child_relative_path(const std::string& child) const {
            if (subfolder.empty()) {
                return child;
            }
            if (!subfolder.empty() && subfolder.back() == '/') {
                return subfolder + child;
            }
            return subfolder + "/" + child;
        }
        
        String get_folder_state_path(const std::string& relative_path) const {
            String base = Unicode::Widen(getData().directories.document_dir) + U"/forced_openings/";
            if (!relative_path.empty()) {
                base += Unicode::Widen(relative_path);
                if (base.size() && base.back() != U'/') {
                    base += U"/";
                }
            }
            return base + U"folder_state.json";
        }
        
        bool load_folder_enabled_state(const std::string& relative_path) const {
            if (relative_path.empty()) {
                return true;
            }
            String config_path = get_folder_state_path(relative_path);
            if (!FileSystem::Exists(config_path)) {
                return true;
            }
            TextReader reader(config_path);
            if (!reader) {
                return true;
            }
            String line;
            if (!reader.readLine(line)) {
                return true;
            }
            line = line.trimmed();
            std::string lowered = line.narrow();
            std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return !(lowered == "false" || lowered == "0" || lowered == "off");
        }
        
        void save_folder_enabled_state(const std::string& relative_path, bool enabled) {
            if (relative_path.empty()) {
                return;
            }
            String config_path = get_folder_state_path(relative_path);
            TextWriter writer(config_path);
            if (!writer) {
                return;
            }
            writer.write(enabled ? U"true" : U"false");
        }

        bool is_current_folder_locked() const {
            return !current_folder_effective_enabled;
        }

        void refresh_current_folder_effective_state() {
            current_folder_effective_enabled = is_folder_effectively_enabled(subfolder);
        }

        bool is_folder_effectively_enabled(const std::string& relative_path) const {
            if (relative_path.empty()) {
                return true;
            }
            std::string prefix;
            prefix.reserve(relative_path.size());
            size_t start = 0;
            while (start < relative_path.size()) {
                if (!prefix.empty()) {
                    prefix.push_back('/');
                }
                size_t slash = relative_path.find('/', start);
                size_t len = (slash == std::string::npos ? relative_path.size() : slash) - start;
                prefix.append(relative_path, start, len);
                if (!prefix.empty() && !load_folder_enabled_state(prefix)) {
                    return false;
                }
                if (slash == std::string::npos) {
                    break;
                }
                start = slash + 1;
            }
            return true;
        }

        void refresh_directory_views() {
            enumerate_current_dir();
            load_openings();
        }
        
        bool is_forbidden_folder_char(char32 ch) const {
            static constexpr char32 forbidden_chars[] = U"\\/:*?\"<>|";
            if (ch < 0x20) {
                return true;
            }
            for (char32 f : forbidden_chars) {
                if (f == U'\0') {
                    break;
                }
                if (ch == f) {
                    return true;
                }
            }
            return false;
        }
        
        String sanitize_folder_text(const String& text) const {
            String filtered;
            filtered.reserve(text.size());
            for (char32 ch : text) {
                if (!is_forbidden_folder_char(ch)) {
                    filtered.push_back(ch);
                }
            }
            return filtered;
        }
        
        bool is_valid_folder_name(const String& name) const {
            String trimmed = name.trimmed();
            if (trimmed.isEmpty()) {
                return false;
            }
            if (trimmed == U"." || trimmed == U"..") {
                return false;
            }
            return true;
        }
        
        void begin_folder_rename(int idx) {
            if (idx < 0 || idx >= (int)folders_display.size()) {
                return;
            }
            renaming_folder = true;
            renaming_folder_index = idx;
            folder_rename_area.text = folders_display[idx].name;
            folder_rename_area.cursorPos = folder_rename_area.text.size();
            folder_rename_area.rebuildGlyphs();
            folder_rename_area.active = true;
        }
        
        void cancel_folder_rename() {
            renaming_folder = false;
            renaming_folder_index = -1;
            folder_rename_area.text = U"";
            folder_rename_area.cursorPos = 0;
            folder_rename_area.rebuildGlyphs();
            folder_rename_area.active = false;
        }
        
        bool rename_folder_entry(int idx, const String& new_name) {
            if (idx < 0 || idx >= (int)folders_display.size()) {
                return false;
            }
            String trimmed = new_name.trimmed();
            if (!is_valid_folder_name(trimmed)) {
                return false;
            }
            String current_name = folders_display[idx].name;
            if (trimmed == current_name) {
                return true;
            }
            String parent_path = Unicode::Widen(getData().directories.document_dir) + U"/forced_openings/";
            if (!subfolder.empty()) {
                parent_path += Unicode::Widen(subfolder) + U"/";
            }
            String source_path = parent_path + current_name;
            if (!FileSystem::Exists(source_path)) {
                return false;
            }
            if (!move_folder(source_path, parent_path, trimmed)) {
                return false;
            }
            return true;
        }
        
        void confirm_folder_rename() {
            if (!renaming_folder || renaming_folder_index < 0 || renaming_folder_index >= (int)folders_display.size()) {
                return;
            }
            String sanitized = sanitize_folder_text(folder_rename_area.text);
            if (sanitized != folder_rename_area.text) {
                size_t old_cursor = folder_rename_area.cursorPos;
                folder_rename_area.text = sanitized;
                folder_rename_area.cursorPos = std::min(old_cursor, sanitized.size());
                folder_rename_area.rebuildGlyphs();
            }
            String trimmed = folder_rename_area.text.trimmed();
            if (!is_valid_folder_name(trimmed)) {
                return;
            }
            if (rename_folder_entry(renaming_folder_index, trimmed)) {
                cancel_folder_rename();
                enumerate_current_dir();
                load_openings();
            }
        }
        
        void handle_textarea_tab_navigation() {
            for (int i = 0; i < 2; ++i) {
                std::string str = text_area[i].text.narrow();
                size_t tab_place = str.find('\t');
                if (tab_place != std::string::npos) {
                    text_area[i].active = false;
                    text_area[(i + 1) % 2].active = true;
                    std::string txt0 = str.substr(0, tab_place);
                    std::string txt1 = str.substr(tab_place + 1);
                    text_area[i].text = Unicode::Widen(txt0);
                    text_area[i].cursorPos = text_area[i].text.size();
                    text_area[i].rebuildGlyphs();
                    text_area[(i + 1) % 2].text += Unicode::Widen(txt1);
                    text_area[(i + 1) % 2].cursorPos = text_area[(i + 1) % 2].text.size();
                    text_area[(i + 1) % 2].rebuildGlyphs();
                }
            }
        }

        bool collect_opening_form_payload(String& transcript_out, double& weight_out) {
            std::string transcript = text_area[0].text.narrow();
            std::string weight_str = text_area[1].text.narrow();
            if (!is_valid_transcript(transcript)) {
                return false;
            }
            try {
                weight_out = std::stod(weight_str);
            } catch (const std::invalid_argument&) {
                return false;
            } catch (const std::out_of_range&) {
                return false;
            }
            transcript_out = Unicode::Widen(transcript);
            return true;
        }

        void cancel_opening_edit() {
            editing_elem = false;
            editing_index = -1;
        }
        
        // Get base directory for current forced_openings folder
        String get_base_dir() const {
            String base = Unicode::Widen(getData().directories.document_dir) + U"/forced_openings/";
            if (!subfolder.empty()) {
                base += Unicode::Widen(subfolder) + U"/";
            }
            return base;
        }
        
        bool create_new_folder(const String& folder_name) {
            String trimmed = folder_name.trimmed();
            if (trimmed.isEmpty()) {
                return false;
            }
            String target_dir = get_base_dir() + trimmed + U"/";
            if (FileSystem::Exists(target_dir)) {
                return false;
            }
            if (!FileSystem::CreateDirectories(target_dir)) {
                return false;
            }
            CSV csv;
            csv.save(target_dir + U"summary.csv");
            return true;
        }
        
        // Load openings from current folder's summary.csv
        void load_openings() {
            refresh_current_folder_effective_state();
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
        void add_opening(const String& transcript, double weight, bool enabled = true) {
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
        
        void reorder_opening_within_current(int from_idx, int insert_idx) {
            if (from_idx < 0 || from_idx >= (int)openings.size()) {
                return;
            }
            if (insert_idx < 0) {
                insert_idx = 0;
            }
            if (insert_idx > (int)openings.size()) {
                insert_idx = (int)openings.size();
            }
            if (insert_idx == from_idx) {
                return;
            }
            auto opening = openings[from_idx];
            auto del_btn = delete_buttons[from_idx];
            auto edit_btn = edit_buttons[from_idx];
            auto toggle_btn = toggle_buttons[from_idx];
            auto move_up_btn = move_up_buttons[from_idx];
            auto move_down_btn = move_down_buttons[from_idx];
            openings.erase(openings.begin() + from_idx);
            delete_buttons.erase(delete_buttons.begin() + from_idx);
            edit_buttons.erase(edit_buttons.begin() + from_idx);
            toggle_buttons.erase(toggle_buttons.begin() + from_idx);
            move_up_buttons.erase(move_up_buttons.begin() + from_idx);
            move_down_buttons.erase(move_down_buttons.begin() + from_idx);
            if (insert_idx > from_idx) {
                insert_idx -= 1;
            }
            openings.insert(openings.begin() + insert_idx, opening);
            delete_buttons.insert(delete_buttons.begin() + insert_idx, del_btn);
            edit_buttons.insert(edit_buttons.begin() + insert_idx, edit_btn);
            toggle_buttons.insert(toggle_buttons.begin() + insert_idx, toggle_btn);
            move_up_buttons.insert(move_up_buttons.begin() + insert_idx, move_up_btn);
            move_down_buttons.insert(move_down_buttons.begin() + insert_idx, move_down_btn);
            save_openings();
        }
        
        // Enumerate current directory
        void enumerate_current_dir() {
            folders_display.clear();
            has_parent = !subfolder.empty();
            renaming_folder = false;
            renaming_folder_index = -1;
            folder_rename_area.text = U"";
            folder_rename_area.cursorPos = 0;
            folder_rename_area.rebuildGlyphs();
            
            std::string base_dir = getData().directories.document_dir + "/forced_openings";
            std::vector<String> folders = enumerate_subdirectories_generic(base_dir, subfolder);
            for (auto& folder : folders) {
                std::string rel_path = build_child_relative_path(folder.narrow());
                bool is_enabled = load_folder_enabled_state(rel_path);
                folders_display.emplace_back(Folder_entry{ folder, rel_path, is_enabled });
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
                    bool skip_navigation = renaming_folder && renaming_folder_index == i;
                    if (!skip_navigation) {
                        Rect rect(OPENING_SETTING_SX, item_sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
                        bool clicked_row = rect.leftClicked();
                        double checkbox_size = 18.0;
                        double toggle_x = OPENING_SETTING_SX + OPENING_SETTING_WIDTH - 30;
                        RectF toggle_rect(toggle_x, item_sy + (OPENING_SETTING_HEIGHT - checkbox_size) / 2.0, checkbox_size, checkbox_size);
                        double rename_icon_size = 16.0;
                        RectF rename_rect(toggle_rect.x - rename_icon_size - 10.0, item_sy + (OPENING_SETTING_HEIGHT - rename_icon_size) / 2.0, rename_icon_size, rename_icon_size);
                        bool clicking_toggle = toggle_rect.contains(Cursor::Pos()) && MouseL.down();
                        bool clicking_rename = rename_rect.contains(Cursor::Pos()) && MouseL.down();
                        if (clicked_row && !clicking_toggle && !clicking_rename) {
                            if (current_time - last_click_time < DOUBLE_CLICK_TIME_MS && last_clicked_folder == folders_display[i].name) {
                                navigate_to_folder(folders_display[i].name);
                                return;
                            }
                            last_click_time = current_time;
                            last_clicked_folder = folders_display[i].name;
                        }
                    }
                }
                row_index++;
            }
        }
        
        void draw_list() {
            if (!(editing_elem || renaming_folder)) {
                handle_folder_navigation();
            }
            draw_openings_list();
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
            
            // Draw input area for adding
            if (adding_elem) {
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
            const Texture& folder_icon = getData().resources.folder;
            double icon_scale = folder_icon ? (double)(OPENING_SETTING_HEIGHT - 20) / (double)folder_icon.height() : 1.0;
            double icon_x = OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 8;
            if (folder_icon) {
                folder_icon.scaled(icon_scale).draw(Arg::leftCenter(icon_x, sy + OPENING_SETTING_HEIGHT / 2), ColorF(1.0));
            }
            double text_offset = icon_x + (folder_icon ? folder_icon.width() * icon_scale + 10.0 : 0.0);
            getData().fonts.font(U"..").draw(20, Arg::leftCenter(text_offset, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
            if (drag_state.is_dragging && rect.contains(drag_state.current_mouse_pos)) {
                rect.drawFrame(3.0, ColorF(getData().colors.yellow));
            }
            if (editing_elem || renaming_folder) {
                rect.draw(ColorF(0.0, 0.0, 0.0, 0.45));
            }
        }
        
        // Draw folder item
        void draw_folder_item(const Folder_entry& entry, int sy, int idx) {
            Rect rect(OPENING_SETTING_SX, sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
            bool is_being_dragged = (drag_state.is_dragging_folder && drag_state.dragged_folder_name == entry.name);
            bool is_enabled = entry.enabled;
            bool folder_locked = is_current_folder_locked();
            ColorF bg_color = idx % 2 ? ColorF(getData().colors.dark_green) : ColorF(getData().colors.green);
            if (is_being_dragged) {
                bg_color = ColorF(getData().colors.yellow);
                bg_color.a = 0.25;
            }
            if (!is_enabled) {
                bg_color = ColorF(0.25, 0.25, 0.25, 0.85);
            }
            Color text_color = is_being_dragged ? getData().colors.white.withAlpha(128) : getData().colors.white;
            if (!is_enabled) {
                text_color = getData().colors.white.withAlpha(100);
            }
            rect.draw(bg_color).drawFrame(1.0, getData().colors.white);
            if (drag_state.is_dragging_opening && drag_state.is_dragging && !drag_state.is_dragging_folder && rect.contains(drag_state.current_mouse_pos) && !editing_elem && !renaming_folder) {
                rect.drawFrame(3.0, ColorF(getData().colors.yellow));
            }
            bool mouse_down_event = MouseL.down();
            const Texture& folder_icon = getData().resources.folder;
            double icon_scale = folder_icon ? (double)(OPENING_SETTING_HEIGHT - 20) / (double)folder_icon.height() : 1.0;
            double icon_x = OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 8;
            if (folder_icon) {
                ColorF icon_color = is_enabled ? ColorF(1.0) : ColorF(1.0, 0.5);
                folder_icon.scaled(icon_scale).draw(Arg::leftCenter(icon_x, sy + OPENING_SETTING_HEIGHT / 2), icon_color);
            }
            double text_offset = icon_x + (folder_icon ? folder_icon.width() * icon_scale + 10.0 : 0.0);
            bool is_renaming_this = renaming_folder && renaming_folder_index == idx;
            if (is_renaming_this) {
                InlineEditLayout layout = get_inline_edit_layout(sy);
                SimpleGUI::TextArea(folder_rename_area, Vec2{ layout.transcript_x, layout.text_y }, SizeF{ layout.transcript_width, layout.field_height }, SimpleGUI::PreferredTextAreaMaxChars);
                String sanitized = sanitize_folder_text(folder_rename_area.text);
                if (sanitized != folder_rename_area.text) {
                    size_t old_cursor = folder_rename_area.cursorPos;
                    folder_rename_area.text = sanitized;
                    folder_rename_area.cursorPos = std::min(old_cursor, sanitized.size());
                    folder_rename_area.rebuildGlyphs();
                }
                if (draw_inline_back_button(layout)) {
                    cancel_folder_rename();
                    return;
                }

                String trimmed = folder_rename_area.text.trimmed();
                bool can_commit = is_valid_folder_name(trimmed);
                if (draw_inline_ok_button(layout, can_commit)) {
                    confirm_folder_rename();
                    return;
                }
                return;
            } else {
                getData().fonts.font(entry.name).draw(18, Arg::leftCenter(text_offset, sy + OPENING_SETTING_HEIGHT / 2), text_color);
                if (editing_elem || renaming_folder) {
                    rect.draw(ColorF(0.0, 0.0, 0.0, 0.45));
                    return;
                }
                if (folder_locked) {
                    rect.draw(ColorF(0.0, 0.0, 0.0, 0.45));
                }
            }
            int toggle_x = OPENING_SETTING_SX + OPENING_SETTING_WIDTH - 30;
            double checkbox_size = 18.0;
            RectF toggle_rect(toggle_x, sy + (OPENING_SETTING_HEIGHT - checkbox_size) / 2.0, checkbox_size, checkbox_size);
            bool toggle_hovered = toggle_rect.mouseOver();
            const Texture& checked_tex = getData().resources.checkbox;
            const Texture& unchecked_tex = getData().resources.unchecked;
            const Texture& checkbox_tex = is_enabled ? checked_tex : unchecked_tex;
            if (checkbox_tex) {
                checkbox_tex.resized(checkbox_size).draw(toggle_rect.pos, is_enabled ? ColorF(1.0) : ColorF(1.0, 0.5));
            } else {
                if (is_enabled) {
                    toggle_rect.draw(getData().colors.white);
                } else {
                    toggle_rect.drawFrame(2.0, getData().colors.white.withAlpha(90));
                }
            }
            if (toggle_rect.leftClicked()) {
                bool new_state = !entry.enabled;
                folders_display[idx].enabled = new_state;
                save_folder_enabled_state(entry.relative_path, new_state);
                return;
            }
            double rename_icon_size = 16.0;
            double rename_x = toggle_rect.x - rename_icon_size - 10.0;
            RectF rename_rect(rename_x, sy + (OPENING_SETTING_HEIGHT - rename_icon_size) / 2.0, rename_icon_size, rename_icon_size);
            if (!is_renaming_this) {
                const Texture& pencil_tex = getData().resources.pencil;
                if (pencil_tex) {
                    pencil_tex.resized(rename_icon_size).draw(rename_rect.pos, ColorF(1.0));
                } else {
                    rename_rect.draw(getData().colors.white);
                }
                if (rename_rect.leftClicked()) {
                    begin_folder_rename(idx);
                    return;
                }
            }
            if (!is_renaming_this && mouse_down_event && rect.contains(Cursor::Pos()) && 
                !drag_state.is_dragging && drag_state.dragged_opening_index == -1 && drag_state.dragged_folder_name.isEmpty() && !toggle_hovered && !(rename_rect.mouseOver())) {
                drag_state.dragged_folder_name = entry.name;
                drag_state.drag_start_pos = Cursor::Pos();
            }
        }
        
        // Draw opening item
        void draw_opening_item(int idx, int sy, int row_index) {
            const auto& opening = openings[idx];
            Rect rect(OPENING_SETTING_SX, sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
            
            bool is_being_dragged = (drag_state.is_dragging_opening && drag_state.dragged_opening_index == idx);
            bool is_editing_this = editing_elem && editing_index == idx;
            bool folder_locked = is_current_folder_locked();
            bool overlay_noninteractive = adding_elem || renaming_folder || (editing_elem && !is_editing_this);
            ColorF bg_color = row_index % 2 ? ColorF(getData().colors.dark_green) : ColorF(getData().colors.green);
            if (is_being_dragged) {
                bg_color = ColorF(getData().colors.yellow);
                bg_color.a = 0.25;
            }
            if (!opening.enabled) {
                bg_color = ColorF(0.2, 0.2, 0.2, 0.85);
            }
            Color text_color = is_being_dragged ? getData().colors.white.withAlpha(128) : getData().colors.white;
            if (!opening.enabled) {
                text_color = getData().colors.white.withAlpha(100);
            }
            
            rect.draw(bg_color).drawFrame(1.0, getData().colors.white);
            
            // Handle drag preparation
            bool mouse_down_event = MouseL.down();
            if (mouse_down_event && rect.contains(Cursor::Pos()) && 
                !drag_state.is_dragging && drag_state.dragged_opening_index == -1 && drag_state.dragged_folder_name.isEmpty() &&
                !(adding_elem || editing_elem || renaming_folder)) {
                drag_state.dragged_opening_index = idx;
                drag_state.drag_start_pos = Cursor::Pos();
            }
            
            if (!(adding_elem || editing_elem || renaming_folder)) {
                // Delete button
                delete_buttons[idx].move(OPENING_SETTING_SX + 1, sy + 1);
                delete_buttons[idx].draw();
                if (delete_buttons[idx].clicked()) {
                    delete_opening(idx);
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
                int toggle_x = OPENING_SETTING_SX + OPENING_SETTING_WIDTH - 30;
                double checkbox_size = 18.0;
                RectF toggle_rect(toggle_x, sy + (OPENING_SETTING_HEIGHT - checkbox_size) / 2.0, checkbox_size, checkbox_size);
                const Texture& checked_tex = getData().resources.checkbox;
                const Texture& unchecked_tex = getData().resources.unchecked;
                const Texture& checkbox_tex = opening.enabled ? checked_tex : unchecked_tex;
                if (checkbox_tex) {
                    checkbox_tex.resized(checkbox_size).draw(toggle_rect.pos, opening.enabled ? ColorF(1.0) : ColorF(1.0, 0.5));
                } else {
                    if (opening.enabled) {
                        toggle_rect.draw(getData().colors.white);
                    } else {
                        toggle_rect.drawFrame(2.0, getData().colors.white.withAlpha(90));
                    }
                }
                if (toggle_rect.leftClicked()) {
                    openings[idx].enabled = !openings[idx].enabled;
                    save_openings();
                    return;
                }

                double edit_icon_size = 16.0;
                double edit_x = toggle_rect.x - edit_icon_size - 10.0;
                RectF edit_rect(edit_x, sy + (OPENING_SETTING_HEIGHT - edit_icon_size) / 2.0, edit_icon_size, edit_icon_size);
                const Texture& pencil_tex = getData().resources.pencil;
                if (pencil_tex) {
                    pencil_tex.resized(edit_icon_size).draw(edit_rect.pos, ColorF(1.0));
                } else {
                    edit_rect.draw(getData().colors.white);
                }
                if (edit_rect.leftClicked()) {
                    editing_elem = true;
                    cancel_folder_rename();
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
            }
            
            if (is_editing_this) {
                InlineEditLayout layout = get_inline_edit_layout(sy);
                SimpleGUI::TextArea(text_area[0], Vec2{ layout.transcript_x, layout.text_y }, SizeF{ layout.transcript_width, layout.field_height }, SimpleGUI::PreferredTextAreaMaxChars);
                SimpleGUI::TextArea(text_area[1], Vec2{ layout.secondary_x, layout.text_y }, SizeF{ layout.secondary_width, layout.field_height }, SimpleGUI::PreferredTextAreaMaxChars);
                handle_textarea_tab_navigation();

                if (draw_inline_back_button(layout)) {
                    cancel_opening_edit();
                    return;
                }

                String updated_transcript;
                double updated_weight = 0.0;
                bool can_commit = collect_opening_form_payload(updated_transcript, updated_weight);
                if (draw_inline_ok_button(layout, can_commit)) {
                    openings[idx].transcript = updated_transcript;
                    openings[idx].weight = updated_weight;
                    save_openings();
                    cancel_opening_edit();
                    return;
                }
            } else {
                // Draw transcript
                getData().fonts.font(opening.transcript).draw(15, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 78, sy + OPENING_SETTING_HEIGHT / 2), text_color);
                
                // Draw weight
                getData().fonts.font(language.get("opening_setting", "weight") + U": ").draw(15, Arg::rightCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 90, sy + OPENING_SETTING_HEIGHT / 2), text_color);
                getData().fonts.font(Format(std::round(opening.weight))).draw(15, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 90, sy + OPENING_SETTING_HEIGHT / 2), text_color);
            }

            if (drag_state.is_dragging_opening && drag_state.is_dragging && rect.contains(drag_state.current_mouse_pos) && !drag_state.is_dragging_folder) {
                double mid_y = sy + OPENING_SETTING_HEIGHT / 2.0;
                bool draw_top = (drag_state.current_mouse_pos.y < mid_y);
                double line_y = draw_top ? sy + 2.0 : sy + OPENING_SETTING_HEIGHT - 2.0;
                Line line_segment{ Vec2{ OPENING_SETTING_SX + 5.0, line_y }, Vec2{ OPENING_SETTING_SX + OPENING_SETTING_WIDTH - 5.0, line_y } };
                line_segment.draw(4.0, ColorF(getData().colors.yellow));
            }

            if (overlay_noninteractive) {
                rect.draw(ColorF(0.0, 0.0, 0.0, 0.45));
                return;
            }

            if (folder_locked) {
                rect.draw(ColorF(0.0, 0.0, 0.0, 0.45));
            }
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
            handle_textarea_tab_navigation();
        }
        
        // Draw dragged item
        void draw_dragged_item() {
            Vec2 draw_pos = drag_state.current_mouse_pos;
            draw_pos.x -= OPENING_SETTING_WIDTH / 2;
            draw_pos.y -= OPENING_SETTING_HEIGHT / 2;
            
            Rect drag_rect(draw_pos.x, draw_pos.y, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
            drag_rect.draw(getData().colors.yellow.withAlpha(200)).drawFrame(2.0, getData().colors.white);
            
            if (drag_state.is_dragging_folder) {
                const Texture& folder_icon = getData().resources.folder;
                double icon_scale = folder_icon ? (double)(OPENING_SETTING_HEIGHT - 20) / (double)folder_icon.height() : 1.0;
                double icon_x = draw_pos.x + OPENING_SETTING_LEFT_MARGIN + 8;
                if (folder_icon) {
                    folder_icon.scaled(icon_scale).draw(Arg::leftCenter(icon_x, draw_pos.y + OPENING_SETTING_HEIGHT / 2), ColorF(0.2, 0.2, 0.2));
                }
                double text_offset = icon_x + (folder_icon ? folder_icon.width() * icon_scale + 10.0 : 0.0);
                getData().fonts.font(drag_state.dragged_folder_name).draw(18, Arg::leftCenter(text_offset, draw_pos.y + OPENING_SETTING_HEIGHT / 2), getData().colors.black);
            } else if (drag_state.is_dragging_opening && drag_state.dragged_opening_index >= 0 && drag_state.dragged_opening_index < (int)openings.size()) {
                const auto& opening = openings[drag_state.dragged_opening_index];
                getData().fonts.font(opening.transcript).draw(15, Arg::leftCenter(draw_pos.x + OPENING_SETTING_LEFT_MARGIN + 8, draw_pos.y + OPENING_SETTING_HEIGHT / 2), getData().colors.black);
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
            if (!folder_path.empty() && !load_folder_enabled_state(folder_path)) {
                return;
            }
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
                if (!load_folder_enabled_state(next_path)) {
                    continue;
                }
                load_all_openings_recursive(next_path);
            }
        }
        
        // Handle drag and drop
        void handle_drag_and_drop() {
            drag_state.current_mouse_pos = Cursor::Pos();
            bool mouse_is_down = MouseL.pressed();
            bool mouse_just_released = !mouse_is_down && drag_state.mouse_was_down;
            drag_state.mouse_was_down = mouse_is_down;

            if (drag_state.is_dragging && mouse_just_released) {
                int sy = OPENING_SETTING_SY + 8;
                int strt_idx_int = scroll_manager.get_strt_idx_int();
                bool handled = false;
                
                if (has_parent) {
                    int parent_row = 0;
                    if (parent_row >= strt_idx_int && parent_row < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                        int item_sy = sy + (parent_row - strt_idx_int) * OPENING_SETTING_HEIGHT;
                        Rect parent_rect(OPENING_SETTING_SX, item_sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
                        if (parent_rect.contains(drag_state.current_mouse_pos)) {
                            if (drag_state.is_dragging_opening && drag_state.dragged_opening_index >= 0) {
                                move_opening_to_parent(drag_state.dragged_opening_index);
                            } else if (drag_state.is_dragging_folder && !drag_state.dragged_folder_name.isEmpty()) {
                                move_folder_to_parent(drag_state.dragged_folder_name.narrow());
                            }
                            handled = true;
                        }
                    }
                }
                
                if (!handled) {
                    int parent_offset = has_parent ? 1 : 0;
                    for (int folder_idx = 0; folder_idx < (int)folders_display.size(); ++folder_idx) {
                        int row = parent_offset + folder_idx;
                        if (row >= strt_idx_int && row < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                            int item_sy = sy + (row - strt_idx_int) * OPENING_SETTING_HEIGHT;
                            Rect folder_rect(OPENING_SETTING_SX, item_sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
                            if (folder_rect.contains(drag_state.current_mouse_pos)) {
                                String target_folder = folders_display[folder_idx].name;
                                if (drag_state.is_dragging_folder && target_folder == drag_state.dragged_folder_name) {
                                    continue;
                                }
                                if (drag_state.is_dragging_opening && drag_state.dragged_opening_index >= 0) {
                                    move_opening_to_folder(drag_state.dragged_opening_index, target_folder.narrow());
                                } else if (drag_state.is_dragging_folder && !drag_state.dragged_folder_name.isEmpty()) {
                                    move_folder_to_folder_target(drag_state.dragged_folder_name.narrow(), target_folder.narrow());
                                }
                                handled = true;
                                break;
                            }
                        }
                    }
                }
                
                if (!handled && drag_state.is_dragging_opening && drag_state.dragged_opening_index >= 0) {
                    int drop_index = get_drop_index_for_opening(drag_state.current_mouse_pos);
                    if (drop_index != -1) {
                        reorder_opening_within_current(drag_state.dragged_opening_index, drop_index);
                        handled = true;
                    }
                }
                
                drag_state.reset();
                if (handled) {
                    return;
                }
            }
            
            if (mouse_is_down && !drag_state.is_dragging) {
                if ((drag_state.dragged_opening_index >= 0 || !drag_state.dragged_folder_name.isEmpty())) {
                    double distance = drag_state.drag_start_pos.distanceFrom(drag_state.current_mouse_pos);
                    if (distance > DragState::DRAG_THRESHOLD) {
                        drag_state.is_dragging = true;
                        drag_state.is_dragging_opening = (drag_state.dragged_opening_index >= 0);
                        drag_state.is_dragging_folder = !drag_state.dragged_folder_name.isEmpty();
                    }
                }
            }
            
            if (mouse_just_released && !drag_state.is_dragging) {
                drag_state.dragged_opening_index = -1;
                drag_state.dragged_folder_name.clear();
            }
        }
        
    int get_drop_index_for_opening(const Vec2& drop_pos) {
            int sy = OPENING_SETTING_SY + 8;
            int strt_idx_int = scroll_manager.get_strt_idx_int();
            int parent_offset = has_parent ? 1 : 0;
            int row_index = parent_offset + (int)folders_display.size();
            for (int i = 0; i < (int)openings.size(); ++i) {
                if (row_index >= strt_idx_int && row_index < strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW) {
                    int item_sy = sy + (row_index - strt_idx_int) * OPENING_SETTING_HEIGHT;
                    Rect rect(OPENING_SETTING_SX, item_sy, OPENING_SETTING_WIDTH, OPENING_SETTING_HEIGHT);
                    if (rect.contains(drop_pos)) {
                        double mid_y = item_sy + OPENING_SETTING_HEIGHT / 2.0;
                        if (drop_pos.y < mid_y) {
                            return i;
                        } else {
                            return std::min(i + 1, (int)openings.size());
                        }
                    }
                }
                row_index++;
            }
            double list_left = OPENING_SETTING_SX;
            double list_right = OPENING_SETTING_SX + OPENING_SETTING_WIDTH;
            double list_top = sy;
            double list_bottom = sy + OPENING_SETTING_N_GAMES_ON_WINDOW * OPENING_SETTING_HEIGHT;
            if (drop_pos.x >= list_left && drop_pos.x <= list_right && drop_pos.y >= list_top && drop_pos.y <= list_bottom) {
                return (int)openings.size();
            }
            return -1;
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
            refresh_directory_views();
            
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
            refresh_directory_views();
            
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

