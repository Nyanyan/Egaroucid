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

class Shortcut_key_setting : public App::Scene {
private:
    Button ok_button;
    int strt_idx;

public:
    Shortcut_key_setting(const InitData& init) : IScene{ init } {
        strt_idx = 0;
        ok_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        getData().fonts.font(language.get("settings", "shortcut_keys", "settings")).draw(25, Arg::topCenter(X_CENTER, 10), getData().colors.white);
        std::vector<std::pair<int, Button>> change_buttons;
        int sy = 65;
        if (strt_idx > 0) {
            getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, sy }, getData().colors.white);
        }
        sy += 6;
        for (int i = strt_idx; i < std::min((int)shortcut_keys.shortcut_keys.size(), strt_idx + SHORTCUT_KEY_SETTINGS_N_ON_WINDOW); ++i) {
            Rect rect;
            rect.y = sy;
            rect.x = 30;
            rect.w = 740;
            rect.h = 32;
            rect.draw(getData().colors.green).drawFrame(1.0, getData().colors.white);
            String function_name = shortcut_keys.shortcut_keys[i].name;
            String function_description = shortcut_keys.get_shortcut_key_description(function_name);
            getData().fonts.font(function_description).draw(15, Arg::leftCenter(rect.x + 10, sy + rect.h / 2), getData().colors.white);
            String shortcut_key_str = shortcut_keys.get_shortcut_key_str(function_name);
            if (shortcut_key_str == U""){
                shortcut_key_str = language.get("settings", "shortcut_keys", "not_assigned");
            }
            getData().fonts.font(shortcut_key_str).draw(15, Arg::leftCenter(rect.x + 400, sy + rect.h / 2), getData().colors.white);
            Button button;
            button.init(660, sy + 4, 100, 24, 7, language.get("settings", "shortcut_keys", "change"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
            button.draw();
            change_buttons.emplace_back(std::make_pair(i, button));
            sy += rect.h;
        }
        if (strt_idx + SHORTCUT_KEY_SETTINGS_N_ON_WINDOW < (int)shortcut_keys.shortcut_keys.size() - 1) {
            getData().fonts.font(U"︙").draw(15, Arg::topCenter = Vec2{X_CENTER, 392}, getData().colors.white);
        }
        for (std::pair<int, Button> button_pair : change_buttons) {
            if (button_pair.second.clicked()) {
                
            }
        }
        ok_button.draw();
        if (ok_button.clicked() || KeyEnter.pressed()) {
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        strt_idx = std::max(0, std::min((int)shortcut_keys.shortcut_keys.size() - 1, strt_idx + (int)Mouse::Wheel()));
    }

    void draw() const override {

    }

private:
    
};
