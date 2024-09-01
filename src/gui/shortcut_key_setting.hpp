/*
    Egaroucid Project

    @file shortcut_key_setting.hpp
        Shortcut key customize scenes
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include "function/function_all.hpp"

#define SHORTCUT_KEY_SETTINGS_N_ON_WINDOW 10
#define SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING -1

class Shortcut_key_setting : public App::Scene {
private:
    Button ok_button;
    int strt_idx;
    int changing_idx;
    std::vector<String> changed_keys;
    std::vector<Button> change_buttons;
    std::vector<Button> delete_buttons;
    Button assign_button;
    String message;

public:
    Shortcut_key_setting(const InitData& init) : IScene{ init } {
        strt_idx = 0;
        changing_idx = SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING;
        ok_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        for (int i = 0; i < (int)shortcut_keys.shortcut_keys.size(); ++i){
            Button change_button;
            change_button.init(0, 0, 100, 22, 7, language.get("settings", "shortcut_keys", "change"), 13, getData().fonts.font, getData().colors.white, getData().colors.black);
            change_buttons.emplace_back(change_button);
            Button delete_button;
            delete_button.init(0, 0, 100, 22, 7, language.get("settings", "shortcut_keys", "delete"), 13, getData().fonts.font, getData().colors.white, getData().colors.black);
            delete_buttons.emplace_back(delete_button);
        }
        assign_button.init(0, 0, 100, 22, 7, language.get("settings", "shortcut_keys", "assign"), 13, getData().fonts.font, getData().colors.white, getData().colors.black);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        getData().fonts.font(language.get("settings", "shortcut_keys", "settings")).draw(25, Arg::topCenter(X_CENTER, 6), getData().colors.white);
        if (changing_idx != SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING){
            getData().fonts.font(message).draw(15, Arg::topCenter(X_CENTER, 39), getData().colors.white);
        }
        int sy = 80;
        if (strt_idx > 0) {
            getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, sy }, getData().colors.white);
        }
        sy += 6;
        bool reset_changing_idx = false;
        for (int i = strt_idx; i < std::min((int)shortcut_keys.shortcut_keys.size(), strt_idx + SHORTCUT_KEY_SETTINGS_N_ON_WINDOW); ++i) {
            Rect rect;
            rect.y = sy;
            rect.x = 30;
            rect.w = 740;
            rect.h = 30;
            rect.draw(getData().colors.green).drawFrame(1.0, getData().colors.white);
            String function_name = shortcut_keys.shortcut_keys[i].name;
            String function_description = shortcut_keys.get_shortcut_key_description(function_name);
            getData().fonts.font(function_description).draw(13, Arg::leftCenter(rect.x + 10, sy + rect.h / 2), getData().colors.white);
            String shortcut_key_str;
            if (changing_idx != i){
                shortcut_key_str = shortcut_keys.get_shortcut_key_str(function_name);
            } else{
                shortcut_key_str = generate_key_str(changed_keys);
            }
            bool shortcut_assigned = true;
            if (shortcut_key_str == U""){
                shortcut_assigned = false;
                shortcut_key_str = language.get("settings", "shortcut_keys", "not_assigned");
            }
            getData().fonts.font(shortcut_key_str).draw(13, Arg::leftCenter(rect.x + 400, sy + rect.h / 2), getData().colors.white);
            if (changing_idx == SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING){
                change_buttons[i].move(660, sy + 4);
                change_buttons[i].draw();
                if (change_buttons[i].clicked()){
                    changing_idx = i;
                    changed_keys = shortcut_keys.shortcut_keys[i].keys;
                }
                if (shortcut_assigned){
                    delete_buttons[i].move(550, sy + 4);
                    delete_buttons[i].draw();
                    if (delete_buttons[i].clicked()){
                        shortcut_keys.del(i);
                    }
                }
            } else if (changing_idx == i){
                if (KeyEscape.pressed()){
                    changed_keys.clear();
                    reset_changing_idx = true;
                } else{
                    update_shortcut_key();
                    assign_button.move(660, sy + 4);
                    if (check_duplicate()){
                        assign_button.disable();
                        message = language.get("settings", "shortcut_keys", "key_duplicate_message");
                    } else{
                        assign_button.enable();
                        message = language.get("settings", "shortcut_keys", "changing_message");
                    }
                    assign_button.draw();
                    if (assign_button.clicked()){
                        shortcut_keys.change(changing_idx, changed_keys);
                        changed_keys.clear();
                        reset_changing_idx = true;
                    }
                }
            }
            sy += rect.h;
        }
        if (strt_idx + SHORTCUT_KEY_SETTINGS_N_ON_WINDOW < (int)shortcut_keys.shortcut_keys.size() - 1) {
            getData().fonts.font(U"︙").draw(15, Arg::topCenter = Vec2{X_CENTER, 392}, getData().colors.white);
        }
        if (changing_idx == SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING){
            ok_button.draw();
            if (ok_button.clicked() || KeyEnter.pressed()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        }
        if (changing_idx == SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING){
            strt_idx = std::max(0, std::min((int)shortcut_keys.shortcut_keys.size() - 1, strt_idx + (int)Mouse::Wheel()));
        }
        if (reset_changing_idx){
            changing_idx = SHORTCUT_KEY_SETTINGS_IDX_NOT_CHANGING;
        }
    }

    void draw() const override {
    }
private:
    void update_shortcut_key(){
        bool down_found = false;
        std::vector<String> keys = get_all_inputs(&down_found);
        if (down_found){
            changed_keys = keys;
        }
    }

    bool check_duplicate(){
        if (changed_keys.size() == 0){
            return false;
        }
        for (int i = 0; i < (int)shortcut_keys.shortcut_keys.size(); ++i) {
            if (i == changing_idx){
                continue;
            }
            if (changed_keys.size() == shortcut_keys.shortcut_keys[i].keys.size()){
                bool duplicate = true;
                for (const String &key1: changed_keys){
                    bool found = false;
                    for (const String &key2: shortcut_keys.shortcut_keys[i].keys){
                        if (key1 == key2){
                            found = true;
                            break;
                        }
                    }
                    if (!found){
                        duplicate = false;
                        break;
                    }
                }
                if (duplicate){
                    return true;
                }
            }
        }
        return false;
    }
};
