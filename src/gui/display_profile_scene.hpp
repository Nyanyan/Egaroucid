/*
    Egaroucid Project

    @file display_profile_scene.hpp
        Display profile scenes
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once

#include <algorithm>
#include <vector>
#include "function/function_all.hpp"

constexpr int DISPLAY_PROFILE_LIST_SX = 40;
constexpr int DISPLAY_PROFILE_LIST_SY = 70;
constexpr int DISPLAY_PROFILE_LIST_WIDTH = 720;
constexpr int DISPLAY_PROFILE_ROW_HEIGHT = 60;
constexpr int DISPLAY_PROFILE_LIST_N_ON_WINDOW = 5;
constexpr int DISPLAY_PROFILE_DOUBLE_CLICK_MS = 350;

struct Display_profile_list_item {
    String file_name;
    String profile_name;
    String path;
};

class Display_profile_load : public App::Scene {
private:
    Button back_button;
    Button new_save_button;
    Button overwrite_button;
    Scroll_manager scroll_manager;
    std::vector<Display_profile_list_item> profiles;
    std::vector<ImageButton> delete_buttons;
    std::vector<ImageButton> edit_buttons;
    String message;
    int last_clicked_idx;
    uint64_t last_clicked_time;

public:
    Display_profile_load(const InitData& init) : IScene{ init } {
        set_scene_ime_enabled(false);
        back_button.init(BUTTON2_1_SX, BUTTON2_SY, BUTTON2_WIDTH, BUTTON2_HEIGHT, BUTTON2_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        new_save_button.init(BUTTON2_2_SX, BUTTON2_SY, BUTTON2_WIDTH, BUTTON2_HEIGHT, BUTTON2_RADIUS, language.get("settings", "profile", "new_save"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        overwrite_button.init(0, 0, 110, 24, 8, language.get("settings", "profile", "overwrite"), 12, getData().fonts.font, getData().colors.white, getData().colors.black);
        last_clicked_idx = -1;
        last_clicked_time = 0;
        refresh_profiles();
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }

        getData().fonts.font(language.get("display", "profile", "profile")).draw(25, Arg::topCenter(X_CENTER, 10), getData().colors.white);
        getData().fonts.font(language.get("settings", "profile", "double_click_to_load")).draw(14, Arg::topCenter(X_CENTER, 45), getData().colors.white);

        draw_profile_list();

        scroll_manager.draw();
        scroll_manager.update();

        back_button.draw();
        if (back_button.clicked() || gui_textarea_ime::escape_down_for_scene_change()) {
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        new_save_button.draw();
        if (new_save_button.clicked()) {
            getData().display_profile_editor_info.init();
            getData().display_profile_editor_info.rename_mode = false;
            changeScene(U"Display_profile_save", SCENE_FADE_TIME);
            return;
        }
        if (!message.isEmpty()) {
            getData().fonts.font(message).draw(14, Arg::topCenter(X_CENTER, 395), getData().colors.white);
        }
    }

    void draw() const override {
    }

private:
    void refresh_profiles() {
        profiles.clear();
        delete_buttons.clear();
        edit_buttons.clear();
        Array<FilePath> files = enumerate_display_profile_files(getData().directories);
        for (const auto& path : files) {
            Display_profile_values values = to_display_profile_values(getData().menu_elements);
            String profile_name;
            load_display_profile_values(path, &values, &profile_name);
            Display_profile_list_item item;
            item.path = path;
            item.file_name = FileSystem::FileName(path);
            item.profile_name = profile_name;
            profiles.emplace_back(item);

            ImageButton delete_button;
            delete_button.init(0, 0, 16, getData().resources.cross);
            delete_buttons.emplace_back(delete_button);
            ImageButton edit_button;
            edit_button.init(0, 0, 16, getData().resources.pencil);
            edit_buttons.emplace_back(edit_button);
        }
        scroll_manager.init(
            770,
            DISPLAY_PROFILE_LIST_SY,
            10,
            DISPLAY_PROFILE_ROW_HEIGHT * DISPLAY_PROFILE_LIST_N_ON_WINDOW,
            20,
            static_cast<int>(profiles.size()),
            DISPLAY_PROFILE_LIST_N_ON_WINDOW,
            DISPLAY_PROFILE_LIST_SX,
            DISPLAY_PROFILE_LIST_SY,
            DISPLAY_PROFILE_LIST_WIDTH + 10,
            DISPLAY_PROFILE_ROW_HEIGHT * DISPLAY_PROFILE_LIST_N_ON_WINDOW);
        shortcut_keys.sync_dynamic_shortcut_keys(&getData().directories);
        shortcut_buttons.clear_invalid_functions();
        mouse_additional_buttons.clear_invalid_functions();
    }

    void draw_profile_list() {
        if (profiles.empty()) {
            getData().fonts.font(language.get("settings", "profile", "no_profile")).draw(16, Arg::topCenter(X_CENTER, DISPLAY_PROFILE_LIST_SY + 30), getData().colors.white);
            return;
        }

        int sy = DISPLAY_PROFILE_LIST_SY;
        int strt_idx = scroll_manager.get_strt_idx_int();
        const int end_idx = std::min(static_cast<int>(profiles.size()), strt_idx + DISPLAY_PROFILE_LIST_N_ON_WINDOW);
        for (int i = strt_idx; i < end_idx; ++i) {
            Rect rect(DISPLAY_PROFILE_LIST_SX, sy, DISPLAY_PROFILE_LIST_WIDTH, DISPLAY_PROFILE_ROW_HEIGHT - 2);
            rect.draw(ColorF(getData().colors.dark_green, 0.8)).drawFrame(1.0, getData().colors.white);
            delete_buttons[i].move(rect.x + 6, rect.y + 6);
            delete_buttons[i].draw();
            edit_buttons[i].move(rect.x + 26, rect.y + 6);
            edit_buttons[i].draw();

            const bool is_current_profile = (profiles[i].file_name.narrow() == getData().settings.display_profile_file);
            const bool is_modified_profile = is_current_profile && is_display_profile_modified(getData().directories, getData().settings, getData().menu_elements);
            if (is_current_profile) {
                rect.drawFrame(3.0, getData().colors.cyan);
            }

            getData().fonts.font(profiles[i].profile_name).draw(18, Arg::leftCenter(rect.x + 50, rect.y + 18), getData().colors.white);
            getData().fonts.font(profiles[i].file_name).draw(12, Arg::leftCenter(rect.x + 50, rect.y + 43), getData().colors.light_green);
            if (is_current_profile) {
                if (is_modified_profile) {
                    const String label = language.get("settings", "profile", "modified_from_profile");
                    getData().fonts.font(label).draw(12, Arg::bottomRight(rect.x + rect.w - 15, rect.y + rect.h - 5), getData().colors.cyan);
                } else {
                    const String label = language.get("settings", "profile", "current");
                    getData().fonts.font(label).draw(12, Arg::rightCenter(rect.x + rect.w - 15, rect.y + rect.h / 2), getData().colors.cyan);
                }
            }

            if (delete_buttons[i].clicked()) {
                FileSystem::Remove(profiles[i].path);
                message.clear();
                refresh_profiles();
                return;
            }
            if (edit_buttons[i].clicked()) {
                getData().display_profile_editor_info.init();
                getData().display_profile_editor_info.rename_mode = true;
                getData().display_profile_editor_info.target_file_name = profiles[i].file_name.narrow();
                getData().display_profile_editor_info.initial_name = profiles[i].profile_name;
                changeScene(U"Display_profile_save", SCENE_FADE_TIME);
                return;
            }
            if (is_modified_profile) {
                overwrite_button.move(rect.x + rect.w - 125, rect.y + 6);
                overwrite_button.enable();
                overwrite_button.draw();
                if (overwrite_button.clicked()) {
                    overwrite_profile(i);
                    return;
                }
            }

            if (rect.leftClicked()) {
                const uint64_t now = tim();
                if (last_clicked_idx == i && (now - last_clicked_time) <= DISPLAY_PROFILE_DOUBLE_CLICK_MS) {
                    if (load_profile(i)) {
                        getData().graph_resources.need_init = false;
                        changeScene(U"Main_scene", SCENE_FADE_TIME);
                        return;
                    }
                }
                last_clicked_idx = i;
                last_clicked_time = now;
            }
            sy += DISPLAY_PROFILE_ROW_HEIGHT;
        }
    }

    bool load_profile(int idx) {
        if (idx < 0 || idx >= static_cast<int>(profiles.size())) {
            return false;
        }
        Display_profile_values values = to_display_profile_values(getData().menu_elements);
        String profile_name;
        if (!load_display_profile_values(profiles[idx].path, &values, &profile_name)) {
            message = language.get("settings", "profile", "load_failed");
            return false;
        }
        apply_display_profile_values(values, &getData().menu_elements);
        apply_display_profile_values(values, &getData().settings);
        getData().settings.display_profile_file = profiles[idx].file_name.narrow();
        getData().settings.display_profile_name = profile_name.narrow();
        message.clear();
        return true;
    }

    void overwrite_profile(int idx) {
        if (idx < 0 || idx >= static_cast<int>(profiles.size())) {
            return;
        }
        const Display_profile_values values = to_display_profile_values(getData().menu_elements);
        const bool saved = save_display_profile_values(profiles[idx].path, values, profiles[idx].profile_name);
        if (saved) {
            getData().settings.display_profile_file = profiles[idx].file_name.narrow();
            getData().settings.display_profile_name = profiles[idx].profile_name.narrow();
            message = language.get("settings", "profile", "saved");
            refresh_profiles();
        } else {
            message = language.get("settings", "profile", "save_failed");
        }
    }
};

class Display_profile_save : public App::Scene {
private:
    Button back_button;
    Button save_button;
    TextAreaEditState name_area;
    String validation_message;
    String status_message;
    bool rename_mode;
    std::string target_file_name;

    void sanitize_profile_name_text() {
        String replaced;
        replaced.reserve(name_area.text.size());
        for (const char32 ch : name_area.text) {
            if (ch == U'\n' || ch == U'\r') {
                replaced.push_back(U' ');
            } else {
                replaced.push_back(ch);
            }
        }
        if (replaced != name_area.text) {
            const size_t cursor = name_area.cursorPos;
            name_area.text = replaced;
            name_area.cursorPos = std::min(cursor, name_area.text.size());
            name_area.rebuildGlyphs();
        }
    }

public:
    Display_profile_save(const InitData& init) : IScene{ init } {
        set_scene_ime_enabled(true);
        rename_mode = getData().display_profile_editor_info.rename_mode;
        target_file_name = getData().display_profile_editor_info.target_file_name;
        back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        save_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("settings", "profile", "save"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        if (rename_mode) {
            name_area.text = getData().display_profile_editor_info.initial_name;
        } else {
            name_area.text.clear();
        }
        name_area.cursorPos = name_area.text.size();
        name_area.rebuildGlyphs();
        name_area.active = true;
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }

        const String title = rename_mode
            ? language.get("display", "profile", "profile") + U" " + language.get("settings", "profile", "edit_name")
            : language.get("display", "profile", "profile") + U" " + language.get("settings", "profile", "new_save");
        getData().fonts.font(title).draw(25, Arg::topCenter(X_CENTER, 10), getData().colors.white);
        getData().fonts.font(language.get("settings", "profile", "name")).draw(18, Arg::leftCenter(100, 130), getData().colors.white);
        text_area_with_ime_candidate_window(name_area, Vec2{ 100, 150 }, SizeF{ 600, 36 }, SimpleGUI::PreferredTextAreaMaxChars);
        sanitize_profile_name_text();

        const String profile_name = name_area.text.trimmed();
        if (profile_name.isEmpty()) {
            save_button.disable();
            validation_message = language.get("settings", "profile", "name_required");
        } else {
            save_button.enable();
            validation_message.clear();
        }

        save_button.draw();
        if (save_button.is_enabled() && save_button.clicked()) {
            save_profile(profile_name);
        }

        back_button.draw();
        if (back_button.clicked() || gui_textarea_ime::escape_down_for_scene_change()) {
            getData().graph_resources.need_init = false;
            changeScene(U"Display_profile_load", SCENE_FADE_TIME);
        }

        if (!validation_message.isEmpty()) {
            getData().fonts.font(validation_message).draw(16, Arg::topCenter(X_CENTER, 210), getData().colors.white);
        } else if (!status_message.isEmpty()) {
            getData().fonts.font(status_message).draw(16, Arg::topCenter(X_CENTER, 210), getData().colors.white);
        }
    }

    void draw() const override {
    }

private:
    void save_profile(const String& profile_name) {
        String path;
        if (rename_mode && !target_file_name.empty()) {
            path = get_display_settings_file_path(getData().directories, Unicode::Widen(target_file_name));
        } else {
            path = generate_unique_display_profile_filepath(getData().directories);
        }
        const Display_profile_values values = to_display_profile_values(getData().menu_elements);
        if (save_display_profile_values(path, values, profile_name)) {
            getData().settings.display_profile_file = FileSystem::FileName(path).narrow();
            getData().settings.display_profile_name = profile_name.narrow();
            status_message = language.get("settings", "profile", "saved");
            getData().graph_resources.need_init = false;
            if (rename_mode) {
                changeScene(U"Display_profile_load", SCENE_FADE_TIME);
            } else {
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
            return;
        }
        status_message = language.get("settings", "profile", "save_failed");
    }
};
