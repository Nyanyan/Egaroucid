/*
    Egaroucid Project

    @file output.hpp
        Output scenes
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include <algorithm>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"
#include "draw.hpp"

class Export_game : public App::Scene {
private:
    Button back_button;
    Button export_main_button;
    Button export_this_board_button;
    bool first_display = true; // Flag to transition to Game_editor on first frame

public:
    Export_game(const InitData& init) : IScene{ init } {
        back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        export_main_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "export_main"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        export_this_board_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "export_until_this_board"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);

        // First display: immediately go to Game_editor
        if (first_display) {
            first_display = false;
            getData().game_information.date.clear();
            getData().game_editor_info.return_scene = U"Export_game";
            getData().game_editor_info.is_editing_mode = false;
            getData().game_editor_info.game_date.clear();
            getData().game_editor_info.subfolder.clear();
            getData().game_editor_info.game_info_updated = false;
            getData().game_editor_info.export_mode = 0;
            changeScene(U"Game_editor", 0);
            return;
        }

        back_button.draw();
        export_main_button.draw();
        export_this_board_button.draw();
        if (back_button.clicked() || KeyEscape.pressed()) {
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        // Open game editor to input game information
        if (export_main_button.clicked()) {
            // Keep existing player names and memo, only reset date for new game
            getData().game_information.date.clear();
            getData().game_editor_info.return_scene = U"Export_game";
            getData().game_editor_info.is_editing_mode = false;
            getData().game_editor_info.game_date.clear();
            getData().game_editor_info.subfolder.clear();
            getData().game_editor_info.game_info_updated = false;
            getData().game_editor_info.export_mode = 0; // main line
            changeScene(U"Game_editor", SCENE_FADE_TIME);
        }
        if (export_this_board_button.clicked()) {
            // Keep existing player names and memo, only reset date for new game
            getData().game_information.date.clear();
            getData().game_editor_info.return_scene = U"Export_game";
            getData().game_editor_info.is_editing_mode = false;
            getData().game_editor_info.game_date.clear();
            getData().game_editor_info.subfolder.clear();
            getData().game_editor_info.game_info_updated = false;
            getData().game_editor_info.export_mode = 1; // until this board
            changeScene(U"Game_editor", SCENE_FADE_TIME);
        }
    }

    void draw() const override {
    }
};




class Change_screenshot_saving_dir : public App::Scene {
private:
    Button back_button;
    Button default_button;
    Button go_button;
    std::string dir;
    bool is_valid_dir;
    TextAreaEditState text_area;

public:
    Change_screenshot_saving_dir(const InitData& init) : IScene{ init } {
        back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        default_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "use_default"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        go_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        text_area.text = Unicode::Widen(getData().user_settings.screenshot_saving_dir);
        text_area.cursorPos = text_area.text.size();
        text_area.rebuildGlyphs();
        is_valid_dir = FileSystem::Exists(Unicode::Widen(getData().user_settings.screenshot_saving_dir));
        if (is_valid_dir) {
            go_button.enable();
        } else {
            go_button.disable();
        }
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        const int icon_width = SCENE_ICON_WIDTH;
        getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
        getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
        int sy = 20 + icon_width + 40;
        getData().fonts.font(language.get("in_out", "change_screenshot_saving_dir")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
        getData().fonts.font(language.get("in_out", "input_screenshot_saving_dir")).draw(15, Arg::topCenter(X_CENTER, sy + 50), getData().colors.white);
        text_area.active = true;
        bool text_changed = SimpleGUI::TextArea(text_area, Vec2{X_CENTER - 300, sy + 80}, SizeF{600, 100}, TEXTBOX_MAX_CHARS);
        bool return_pressed = false;
        if (text_area.text.size()) {
            if (text_area.text[text_area.text.size() - 1] == '\n') {
                return_pressed = true;
            }
        }
        dir = text_area.text.replaced(U"\r", U"").replaced(U"\n", U"").replaced(U"\\", U"/").narrow();
        if (dir.size()) {
            if (dir[dir.size() - 1] != '/') {
                dir += "/";
            }
        }
        if (text_changed) {
            is_valid_dir = FileSystem::Exists(Unicode::Widen(dir));
        }
        if (is_valid_dir) {
            go_button.enable();
        } else {
            getData().fonts.font(language.get("in_out", "directory_not_found")).draw(15, Arg::topCenter(X_CENTER, sy + 190), getData().colors.white);
            go_button.disable();
        }
        back_button.draw();
        if (back_button.clicked() || KeyEscape.pressed()) {
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        default_button.draw();
        if (default_button.clicked()) {
            text_area.text = Unicode::Widen(getData().directories.document_dir + "screenshots/");
            text_area.cursorPos = text_area.text.size();
            text_area.scrollY = 0.0;
            text_area.rebuildGlyphs();
        }
        go_button.draw();
        if (go_button.clicked()) {
            getData().user_settings.screenshot_saving_dir = dir;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
    }

    void draw() const override {

    }
};