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




class Opening_setting : public App::Scene {
    private:
        std::vector<Button> edit_buttons;
        std::vector<ImageButton> delete_buttons;
        Scroll_manager scroll_manager;
        Button add_button;
        Button ok_button;
        Button back_button;
        Button register_button;
        bool adding_elem;
        TextAreaEditState text_area[2];
    
    public:
        Opening_setting(const InitData& init) : IScene{ init } {
            add_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("opening_setting", "add"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            ok_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            register_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("opening_setting", "register"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
            for (int i = 0; i < (int)getData().forced_openings.openings.size(); ++i) {
                ImageButton button;
                button.init(0, 0, 15, getData().resources.cross);
                delete_buttons.emplace_back(button);
            }
            adding_elem = false;
            init_scroll_manager();
        }
    
        void update() override {
            if (System::GetUserActions() & UserAction::CloseButtonClicked) {
                changeScene(U"Close", SCENE_FADE_TIME);
            }
            getData().fonts.font(language.get("opening_setting", "opening_setting")).draw(25, Arg::topCenter(X_CENTER, 10), getData().colors.white);
            if (adding_elem) {
                back_button.draw();
                if (back_button.clicked() || KeyEscape.down()) {
                    adding_elem = false;
                }
                std::string transcript = text_area[0].text.narrow();
                std::string weight_str = text_area[1].text.narrow();
                bool can_be_registered = is_valid_transcript(transcript);
                double weight;
                try {
                    weight = stoi(weight_str);
                } catch (const std::invalid_argument& e) {
                    //std::cerr << "Invalid argument: " << e.what() << std::endl;
                    can_be_registered = false;
                } catch (const std::out_of_range& e) {
                    //std::cerr << "Out of range: " << e.what() << std::endl;
                    can_be_registered = false;
                }
                if (can_be_registered) {
                    register_button.enable();
                } else {
                    register_button.disable();
                }
                register_button.draw();
                if (register_button.clicked() || (can_be_registered && KeyEnter.down())) {
                    getData().forced_openings.add(transcript, weight);
                    ImageButton button;
                    button.init(0, 0, 15, getData().resources.cross);
                    delete_buttons.emplace_back(button);
                    init_scroll_manager();
                    adding_elem = false;
                }
            } else {
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
                    getData().graph_resources.need_init = false;
                    changeScene(U"Main_scene", SCENE_FADE_TIME);
                }
            }
            int sy = OPENING_SETTING_SY;
            int strt_idx_int = scroll_manager.get_strt_idx_int();
            if (adding_elem) {
                if (getData().forced_openings.openings.size() >= OPENING_SETTING_N_GAMES_ON_WINDOW) {
                    strt_idx_int = getData().forced_openings.openings.size() - OPENING_SETTING_N_GAMES_ON_WINDOW + 1;
                }
            }
            if (strt_idx_int > 0) {
                getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, sy }, getData().colors.white);
            }
            sy += 8;
            if (!adding_elem && getData().forced_openings.openings.size() == 0) {
                getData().fonts.font(language.get("opening_setting", "no_opening_found")).draw(20, Arg::center(X_CENTER, Y_CENTER), getData().colors.white);
            }
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
                if (adding_elem) {
                    rect.draw(ColorF{1.0, 1.0, 1.0, 0.5});
                }
                if (!adding_elem) {
                    delete_buttons[i].move(OPENING_SETTING_SX + 1, sy + 1);
                    delete_buttons[i].draw();
                    if (delete_buttons[i].clicked()) {
                        delete_opening(i);
                    }
                }
                getData().fonts.font(Unicode::Widen(opening_str)).draw(15, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 8, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
                getData().fonts.font(language.get("opening_setting", "weight") + U": ").draw(15, Arg::rightCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 70, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
                getData().fonts.font(Format(std::round(weight))).draw(15, Arg::leftCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 70, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
                sy += OPENING_SETTING_HEIGHT;
            }
            if (adding_elem) {
                Rect rect;
                rect.y = sy;
                rect.x = OPENING_SETTING_SX;
                rect.w = OPENING_SETTING_WIDTH;
                rect.h = OPENING_SETTING_HEIGHT;
                if (getData().forced_openings.openings.size() % 2) {
                    rect.draw(getData().colors.dark_green).drawFrame(1.0, getData().colors.white);
                } else {
                    rect.draw(getData().colors.green).drawFrame(1.0, getData().colors.white);
                }
                SimpleGUI::TextArea(text_area[0], Vec2{OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + 8, sy + OPENING_SETTING_HEIGHT / 2 - 17}, SizeF{600, 30}, SimpleGUI::PreferredTextAreaMaxChars);
                getData().fonts.font(language.get("opening_setting", "weight") + U": ").draw(15, Arg::rightCenter(OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 70, sy + OPENING_SETTING_HEIGHT / 2), getData().colors.white);
                SimpleGUI::TextArea(text_area[1], Vec2{OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN + OPENING_SETTING_WIDTH - 70, sy + OPENING_SETTING_HEIGHT / 2 - 17}, SizeF{20, 30}, SimpleGUI::PreferredTextAreaMaxChars);
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
            if (strt_idx_int + OPENING_SETTING_N_GAMES_ON_WINDOW < (int)getData().forced_openings.openings.size()) {
                getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, 415}, getData().colors.white);
            }
            if (!adding_elem) {
                scroll_manager.draw();
                scroll_manager.update();
            }
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

