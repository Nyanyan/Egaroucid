/*
    Egaroucid Project

    @file update_check_scene.hpp
        Update Check
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include "load_scene.hpp"

class Update_check : public App::Scene {
private:
    int update_status;
    std::future<int> check_future;
    Button skip_button;
    Button update_button;
    Button back_button;
    String new_version;

public:
    Update_check(const InitData& init) : IScene{ init } {
        skip_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("help", "skip"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        update_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("help", "download"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        update_status = UPDATE_CHECK_NONE;
        check_future = std::async(std::launch::async, check_update, &getData().directories, &new_version);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            if (check_future.valid()) {
                check_future.get();
            }
            changeScene(U"Close", SCENE_FADE_TIME);
            return;
        }
        Scene::SetBackground(getData().colors.green);
        const int icon_width = (LEFT_RIGHT - LEFT_LEFT) / 2;
        getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
        getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
        int sy = 20 + icon_width + 50;
        if (update_status == UPDATE_CHECK_NONE) { // checking updates
            getData().fonts.font(language.get("help", "checking_updates")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            if (check_future.valid()) {
                if (check_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    update_status = check_future.get();
                }
            }
        } else if (update_status == UPDATE_CHECK_UPDATE_FOUND) { // update found
            getData().fonts.font(language.get("help", "new_version_available")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            sy += 35;
            getData().fonts.font(language.get("help", "download?")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            skip_button.draw();
            update_button.draw();
            if (skip_button.clicked() || KeyEscape.down()) {
                changeScene(U"Main_scene", SCENE_FADE_TIME);
                return;
            }
            if (update_button.clicked() || KeyEnter.down()) {
                if (language.get("lang_name") == U"日本語") {
                    System::LaunchBrowser(U"https://www.egaroucid.nyanyan.dev/ja/download/");
                } else {
                    System::LaunchBrowser(U"https://www.egaroucid.nyanyan.dev/en/download/");
                }
                changeScene(U"Close", SCENE_FADE_TIME);
                return;
            }
        } else if (update_status == UPDATE_CHECK_ALREADY_UPDATED) { // already latest version
            getData().fonts.font(language.get("help", "already_latest_version")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            back_button.draw();
            if (back_button.clicked() || KeyEscape.down()) {
                changeScene(U"Main_scene", SCENE_FADE_TIME);
                return;
            }
        } else { // update check failed
            getData().fonts.font(language.get("help", "update_check_failed")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            back_button.draw();
            if (back_button.clicked() || KeyEscape.down()) {
                changeScene(U"Main_scene", SCENE_FADE_TIME);
                return;
            }
        }
    }

    void draw() const override {

    }
};