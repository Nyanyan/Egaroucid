/*
    Egaroucid Project

    @file shortcut_button_setting_scene.hpp
        Shortcut button customize scene
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <iostream>
#include <future>
#include "function/function_all.hpp"

constexpr int SHORTCUT_BUTTON_SETTINGS_N_FUNCTIONS_ON_WINDOW = 10;
constexpr int SHORTCUT_BUTTON_SETTINGS_IDX_NOT_CHANGING = -1;

class Shortcut_button_setting : public App::Scene {
private:
    Button ok_button;
    std::vector<Button> change_buttons;
    std::vector<Button> delete_buttons;
    Button assign_button;
    Scroll_manager function_scroll_manager;
    TextAreaEditState search_area;
    int changing_button_idx;
    int selected_function_idx;
    String message;

public:
    Shortcut_button_setting(const InitData& init) : IScene{ init } {
        set_scene_ime_enabled(true);
        shortcut_keys.sync_dynamic_shortcut_keys(&getData().directories);
        shortcut_buttons.clear_invalid_functions();
        mouse_additional_buttons.clear_invalid_functions();
        changing_button_idx = SHORTCUT_BUTTON_SETTINGS_IDX_NOT_CHANGING;
        selected_function_idx = -1;
        ok_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        for (int i = 0; i < SHORTCUT_BUTTON_COUNT; ++i) {
            Button change_button;
            change_button.init(0, 0, 80, 22, 7, language.get("settings", "shortcut_keys", "change"), 12, getData().fonts.font, getData().colors.white, getData().colors.black);
            change_buttons.emplace_back(change_button);
            Button delete_button;
            delete_button.init(0, 0, 80, 22, 7, language.get("settings", "shortcut_keys", "delete"), 12, getData().fonts.font, getData().colors.white, getData().colors.black);
            delete_buttons.emplace_back(delete_button);
        }
        assign_button.init(630, 430, 130, 34, 9, language.get("settings", "shortcut_keys", "assign"), 16, getData().fonts.font, getData().colors.white, getData().colors.black);
        function_scroll_manager.init(770, SHORTCUT_SETTINGS_LIST_SY, 10, 300, 20, (int)shortcut_keys.shortcut_keys.size(), SHORTCUT_BUTTON_SETTINGS_N_FUNCTIONS_ON_WINDOW, SHORTCUT_SETTINGS_LIST_SX, SHORTCUT_SETTINGS_LIST_SY, SHORTCUT_SETTINGS_LIST_WIDTH + 10, 300);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }

        getData().fonts.font(language.get("settings", "shortcut_buttons", "settings")).draw(25, Arg::topCenter(X_CENTER, 10), getData().colors.white);

        if (changing_button_idx == SHORTCUT_BUTTON_SETTINGS_IDX_NOT_CHANGING) {
            search_area.active = false;
            draw_button_assignment_rows();
            ok_button.draw();
            if (ok_button.clicked() || KeyEnter.down()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        } else {
            draw_function_selection_rows();
        }
    }

    void draw() const override {
    }

private:
    void draw_button_assignment_rows() {
        int sy = SHORTCUT_SETTINGS_LIST_SY;
        for (int i = 0; i < SHORTCUT_BUTTON_COUNT; ++i) {
            Rect rect = draw_shortcut_settings_row_background(getData().colors, i, sy);
            String button_label = language.get("settings", "shortcut_buttons", "button") + U" " + Format(i + 1);
            getData().fonts.font(button_label).draw(12, Arg::leftCenter(rect.x + 10, sy + rect.h / 2), getData().colors.white);

            String function_name = shortcut_buttons.get_function(i);
            String function_description = language.get("settings", "shortcut_keys", "not_assigned");
            if (function_name != U"") {
                function_description = get_shortcut_function_description(function_name);
            }
            getData().fonts.font(function_description).draw(13, Arg::leftCenter(rect.x + 200, sy + rect.h / 2), getData().colors.white);

            change_buttons[i].move(680, sy + 4);
            change_buttons[i].draw();
            if (change_buttons[i].clicked()) {
                changing_button_idx = i;
                selected_function_idx = shortcut_buttons.find_function_idx(function_name);
                function_scroll_manager.set_strt_idx(std::max(0, selected_function_idx - SHORTCUT_BUTTON_SETTINGS_N_FUNCTIONS_ON_WINDOW / 2));
            }

            if (function_name != U"") {
                delete_buttons[i].move(590, sy + 4);
                delete_buttons[i].draw();
                if (delete_buttons[i].clicked()) {
                    shortcut_buttons.clear_function(i);
                }
            }

            sy += rect.h;
        }
    }

    void draw_function_selection_rows() {
        if (gui_textarea_ime::escape_down_for_scene_change()) {
            changing_button_idx = SHORTCUT_BUTTON_SETTINGS_IDX_NOT_CHANGING;
            selected_function_idx = -1;
            search_area.active = false;
            message.clear();
            return;
        }

        const bool search_changed = draw_shortcut_settings_search_box(getData().fonts, getData().colors, search_area);
        const std::vector<int> filtered_function_indices = get_filtered_shortcut_function_indices(search_area);
        if (selected_function_idx != -1 && find_shortcut_function_filter_position(filtered_function_indices, selected_function_idx) == -1) {
            selected_function_idx = -1;
        }
        sync_shortcut_settings_scroll_manager(function_scroll_manager, static_cast<int>(filtered_function_indices.size()), SHORTCUT_BUTTON_SETTINGS_N_FUNCTIONS_ON_WINDOW, search_changed);

        int sy = SHORTCUT_SETTINGS_LIST_SY;
        int strt_idx_int = function_scroll_manager.get_strt_idx_int();
        draw_shortcut_settings_scroll_head(getData().fonts, getData().colors, strt_idx_int, sy);
        Rect selected_rect;
        bool selected_rect_found = false;
        for (int filtered_idx = strt_idx_int; filtered_idx < std::min(static_cast<int>(filtered_function_indices.size()), strt_idx_int + SHORTCUT_BUTTON_SETTINGS_N_FUNCTIONS_ON_WINDOW); ++filtered_idx) {
            int function_idx = filtered_function_indices[filtered_idx];
            Rect rect = draw_shortcut_settings_row_background(getData().colors, filtered_idx, sy);
            if (selected_function_idx == function_idx) {
                selected_rect = rect;
                selected_rect_found = true;
            }
            String function_description = get_shortcut_function_description(shortcut_keys.shortcut_keys[function_idx].name);
            getData().fonts.font(function_description).draw(12, Arg::leftCenter(rect.x + 10, sy + rect.h / 2), getData().colors.white);
            if (rect.leftClicked()) {
                selected_function_idx = function_idx;
            }
            sy += rect.h;
        }
        if (selected_rect_found) {
            selected_rect.drawFrame(4.0, getData().colors.cyan);
        }
        if (filtered_function_indices.empty()) {
            draw_shortcut_settings_no_match_message(getData().fonts, getData().colors, SHORTCUT_SETTINGS_LIST_SY);
        }
        draw_shortcut_settings_scroll_tail(getData().fonts, getData().colors, strt_idx_int + SHORTCUT_BUTTON_SETTINGS_N_FUNCTIONS_ON_WINDOW, static_cast<int>(filtered_function_indices.size()), sy);

        assign_button.disable();
        message = filtered_function_indices.empty() ? get_shortcut_settings_fallback_label("no_match", U"No matches") : language.get("settings", "shortcut_buttons", "choose_function_message");
        String selected_function_name;
        bool has_valid_selection = false;
        if (selected_function_idx != -1) {
            assign_button.enable();
            selected_function_name = shortcut_keys.shortcut_keys[selected_function_idx].name;
            has_valid_selection = true;
            message = get_shortcut_function_description(selected_function_name);
            int duplicate_button_idx = shortcut_buttons.find_assigned_button(selected_function_name, changing_button_idx);
            if (duplicate_button_idx != -1) {
                assign_button.disable();
                message = language.get("settings", "shortcut_buttons", "function_duplicate_message") + U" " + Format(duplicate_button_idx + 1);
            }
        }
        assign_button.draw();
        if (has_valid_selection && assign_button.is_enabled() && (assign_button.clicked() || (KeyEnter.down() && !search_area.active))) {
            shortcut_buttons.set_function(changing_button_idx, selected_function_name);
            changing_button_idx = SHORTCUT_BUTTON_SETTINGS_IDX_NOT_CHANGING;
            selected_function_idx = -1;
            search_area.active = false;
            message.clear();
        }
        getData().fonts.font(message).draw(15, Arg::topCenter(X_CENTER, 440), getData().colors.white);
        function_scroll_manager.draw();
        if (!search_area.active) {
            function_scroll_manager.update();
        }
    }
};
