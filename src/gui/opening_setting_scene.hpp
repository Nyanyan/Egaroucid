/*
    Egaroucid Project

    @file opening_setting.hpp
        Forced Opening setting
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <Siv3D.hpp>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"




class Opening_setting : public App::Scene {
    private:
        std::vector<Button> edit_buttons;
        std::vector<ImageButton> delete_buttons;
        Scroll_manager scroll_manager;
        Button add_button;
        Button ok_button;
    
    public:
        Opening_setting(const InitData& init) : IScene{ init } {
            add_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("opening_setting", "add"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            ok_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            Texture delete_button_image = getData().resources.cross;
            for (int i = 0; i < (int)getData().forced_openings.openings.size(); ++i) {
                ImageButton button;
                button.init(0, 0, 15, delete_button_image);
                delete_buttons.emplace_back(button);
            }
            init_scroll_manager();
        }
    
        void update() override {
            if (System::GetUserActions() & UserAction::CloseButtonClicked) {
                changeScene(U"Close", SCENE_FADE_TIME);
            }
            getData().fonts.font(language.get("settings", "play", "opening_setting")).draw(25, Arg::topCenter(X_CENTER, 10), getData().colors.white);
            ok_button.draw();
            if (ok_button.clicked()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
            int sy = OPENING_SETTING_SY;
            int strt_idx_int = scroll_manager.get_strt_idx_int();
            if (strt_idx_int > 0) {
                getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, sy }, getData().colors.white);
            }
            sy += 8;
            for (int i = strt_idx_int; i < std::min((int)getData().forced_openings.openings.size(), strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW); ++i) {
                std::string opening_str = getData().forced_openings.openings[i].first;
                double weight = getData().forced_openings.openings[i].second;
                Rect rect;
                rect.y = sy;
                rect.x = OPENING_SETTING_SX;
                rect.w = OPENING_SETTING_WIDTH;
                rect.h = OPENING_SETTING_HEIGHT;
                if (i % 2) {
                    rect.draw(getData().colors.dark_green).drawFrame(1.0, getData().colors.white);
                } else {
                    rect.draw(getData().colors.green).drawFrame(1.0, getData().colors.white);
                }
                delete_buttons[i].move(OPENING_SETTING_SX + 1, sy + 1);
                delete_buttons[i].draw();
                if (delete_buttons[i].clicked()) {
                    delete_opening(i);
                }
                getData().fonts.font(Unicode::Widen(opening_str)).draw(15, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 10, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
                getData().fonts.font(language.get("opening_setting", "weight") + U": " + Format(std::round(weight))).draw(15, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 100, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
                sy += OPENING_SETTING_HEIGHT;
            }
            if (strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW < (int)getData().forced_openings.openings.size()) {
                getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, 415}, getData().colors.white);
            }
            scroll_manager.draw();
            scroll_manager.update();
        }
    
        void draw() const override {
    
        }
    
    private:
        void init_scroll_manager() {
            scroll_manager.init(770, OPENING_SETTING_SY + 8, 10, OPENING_SETTING_HEIGHT * OPENING_SETTING_N_GAMES_ON_WINDOW, 20, (int)getData().forced_openings.openings.size(), OPENING_SETTING_N_GAMES_ON_WINDOW, OPENING_SETTING_SX, 73, OPENING_SETTING_WIDTH + 10, OPENING_SETTING_HEIGHT * OPENING_SETTING_N_GAMES_ON_WINDOW);
        }

        void delete_opening(int idx) {
            getData().forced_openings.openings.erase(getData().forced_openings.openings.begin() + idx);
            delete_buttons.erase(delete_buttons.begin() + idx);
            double strt_idx_double = scroll_manager.get_strt_idx_double();
            init_scroll_manager();
            if ((int)strt_idx_double >= idx) {
                strt_idx_double -= 1.0;
            }
            scroll_manager.set_strt_idx(strt_idx_double);
            std::cerr << "deleted opening " << idx << std::endl;
        }
};

