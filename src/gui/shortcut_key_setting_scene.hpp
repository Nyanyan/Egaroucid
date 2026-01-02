/*
    Egaroucid Project

    @file shortcut_key_setting.hpp
        Shortcut key customize scenes
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include "function/function_all.hpp"

constexpr int SHORTCUT_KEY_SETTINGS_N_ON_WINDOW = 10;
constexpr int SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING = -1;

const std::vector<String> allow_multi_input_keys = {
    U"Ctrl",
    U"Shift",
    U"Alt",
#ifdef __APPLE__
    U"Command",
#endif
};

class Shortcut_key_setting : public App::Scene {
private:
    Button default_button;
    Button ok_button;
    Scroll_manager scroll_manager;
    int changing_idx;
    std::vector<String> changed_keys;
    std::vector<Button> change_buttons;
    std::vector<Button> delete_buttons;
    Button assign_button;
    String message;

public:
    Shortcut_key_setting(const InitData& init) : IScene{ init } {
        changing_idx = SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING;
        default_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "use_default"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        ok_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        for (int i = 0; i < (int)shortcut_keys.shortcut_keys.size(); ++i) {
            Button change_button;
            change_button.init(0, 0, 80, 22, 7, language.get("settings", "shortcut_keys", "change"), 12, getData().fonts.font, getData().colors.white, getData().colors.black);
            change_buttons.emplace_back(change_button);
            Button delete_button;
            delete_button.init(0, 0, 80, 22, 7, language.get("settings", "shortcut_keys", "delete"), 12, getData().fonts.font, getData().colors.white, getData().colors.black);
            delete_buttons.emplace_back(delete_button);
        }
        assign_button.init(0, 0, 80, 22, 7, language.get("settings", "shortcut_keys", "assign"), 12, getData().fonts.font, getData().colors.white, getData().colors.black);
        scroll_manager.init(770, 78, 10, 300, 20, (int)shortcut_keys.shortcut_keys.size(), SHORTCUT_KEY_SETTINGS_N_ON_WINDOW, 30, 78, 740 + 10, 300);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        getData().fonts.font(language.get("settings", "shortcut_keys", "settings")).draw(25, Arg::topCenter(X_CENTER, 10), getData().colors.white);
        int sy = 78;
        int strt_idx_int = scroll_manager.get_strt_idx_int();
        if (strt_idx_int > 0) {
            getData().fonts.font(U"︙").draw(15, Arg::bottomCenter(X_CENTER, sy - 6), getData().colors.white);
        }
        bool reset_changing_idx = false;
        for (int i = strt_idx_int; i < std::min((int)shortcut_keys.shortcut_keys.size(), strt_idx_int + SHORTCUT_KEY_SETTINGS_N_ON_WINDOW); ++i) {
            Rect rect;
            rect.y = sy;
            rect.x = 30;
            rect.w = 740;
            rect.h = 30;
            if (i % 2) {
                rect.draw(getData().colors.dark_green).drawFrame(1.0, getData().colors.white);
            } else {
                rect.draw(getData().colors.green).drawFrame(1.0, getData().colors.white);
            }
            String function_name = shortcut_keys.shortcut_keys[i].name;
            String function_description = shortcut_keys.get_shortcut_key_description(function_name);
            getData().fonts.font(function_description).draw(12, Arg::leftCenter(rect.x + 10, sy + rect.h / 2), getData().colors.white);
            String shortcut_key_str;
            if (changing_idx != i) {
                shortcut_key_str = shortcut_keys.get_shortcut_key_str(function_name);
            } else {
                shortcut_key_str = generate_key_str(changed_keys);
            }
            bool shortcut_assigned = true;
            if (shortcut_key_str == U"") {
                shortcut_assigned = false;
                shortcut_key_str = language.get("settings", "shortcut_keys", "not_assigned");
            }
            getData().fonts.font(shortcut_key_str).draw(13, Arg::leftCenter(rect.x + 400, sy + rect.h / 2), getData().colors.white);
            if (changing_idx == SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING) {
                change_buttons[i].move(680, sy + 4);
                change_buttons[i].draw();
                if (change_buttons[i].clicked()) {
                    changing_idx = i;
                    changed_keys = shortcut_keys.shortcut_keys[i].keys;
                }
                if (shortcut_assigned) {
                    delete_buttons[i].move(590, sy + 4);
                    delete_buttons[i].draw();
                    if (delete_buttons[i].clicked()) {
                        shortcut_keys.del(i);
                    }
                }
            } else if (changing_idx == i) {
                if (KeyEscape.down()) {
                    changed_keys.clear();
                    reset_changing_idx = true;
                } else {
                    update_shortcut_key();
                    assign_button.move(680, sy + 4);
                    if (changed_keys.size()) {
                        assign_button.enable();
                        message = language.get("settings", "shortcut_keys", "changing_message");
                        // key valid?
                        int n_other_keys = 0;
                        for (String &key: changed_keys) {
                            if (std::find(allow_multi_input_keys.begin(), allow_multi_input_keys.end(), key) == allow_multi_input_keys.end()) {
                                ++n_other_keys;
                            }
                        }
                        if (n_other_keys == 0) { // Ctrl/Shift/Alt only
                            assign_button.disable();
#ifdef __APPLE__
                            message = language.get("settings", "shortcut_keys", "special_key_error_message_mac");
#else
                            message = language.get("settings", "shortcut_keys", "special_key_error_message");
#endif
                        } else if (n_other_keys > 1) { // too many keys
                            assign_button.disable();
#ifdef __APPLE__
                            message = language.get("settings", "shortcut_keys", "multi_input_error_message_mac");
#else
                            message = language.get("settings", "shortcut_keys", "multi_input_error_message");
#endif
                        }
                        // key duplicate?
                        if (check_duplicate()) {
                            assign_button.disable();
                            message = language.get("settings", "shortcut_keys", "key_duplicate_message") + U": " + get_duplicate_function();
                        }
                        assign_button.draw();
                        if (assign_button.clicked() || (assign_button.is_enabled() && KeyEnter.down())) {
                            shortcut_keys.change(changing_idx, changed_keys);
                            changed_keys.clear();
                            reset_changing_idx = true;
                        }
                    } else { // no keys
                        assign_button.disable();
                        message = language.get("settings", "shortcut_keys", "changing_message");
                    }
                }
            }
            sy += rect.h;
        }
        if (strt_idx_int + SHORTCUT_KEY_SETTINGS_N_ON_WINDOW < (int)shortcut_keys.shortcut_keys.size()) {
            getData().fonts.font(U"︙").draw(15, Arg::topCenter(X_CENTER, sy + 6), getData().colors.white);
        }
        if (changing_idx == SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING) {
            ok_button.draw();
            if (ok_button.clicked() || KeyEnter.down()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
            default_button.draw();
            if (default_button.clicked()) {
                shortcut_keys.set_default();
            }
        }
        if (changing_idx != SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING) {
            getData().fonts.font(message).draw(15, Arg::topCenter(X_CENTER, 440), getData().colors.white);
        }
        if (changing_idx == SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING) {
            scroll_manager.draw();
            scroll_manager.update();
        }
        if (reset_changing_idx) {
            changing_idx = SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING;
        }
    }

    void draw() const override {
    }
private:
    void update_shortcut_key() {
        bool down_found = false;
        std::vector<String> keys = get_all_inputs(&down_found);
        if (down_found) {
            changed_keys = keys;
        }
    }

    bool check_duplicate() {
        if (changed_keys.size() == 0) {
            return false;
        }
        for (int i = 0; i < (int)shortcut_keys.shortcut_keys.size(); ++i) {
            if (i == changing_idx) {
                continue;
            }
            if (changed_keys.size() == shortcut_keys.shortcut_keys[i].keys.size()) {
                bool duplicate = true;
                for (const String &key1: changed_keys) {
                    bool found = false;
                    for (const String &key2: shortcut_keys.shortcut_keys[i].keys) {
                        if (key1 == key2) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        duplicate = false;
                        break;
                    }
                }
                if (duplicate) {
                    return true;
                }
            }
        }
        return false;
    }

    String get_duplicate_function() {
        for (int i = 0; i < (int)shortcut_keys.shortcut_keys.size(); ++i) {
            if (i == changing_idx) {
                continue;
            }
            if (changed_keys.size() == shortcut_keys.shortcut_keys[i].keys.size()) {
                bool duplicate = true;
                for (const String &key1: changed_keys) {
                    bool found = false;
                    for (const String &key2: shortcut_keys.shortcut_keys[i].keys) {
                        if (key1 == key2) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        duplicate = false;
                        break;
                    }
                }
                if (duplicate) {
                    String function_name = shortcut_keys.shortcut_keys[i].name;
                    return shortcut_keys.get_shortcut_key_description(function_name);
                }
            }
        }
        return U"?";
    }
};