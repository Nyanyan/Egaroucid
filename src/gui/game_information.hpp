/*
    Egaroucid Project

    @file game_information.hpp
        browse game information
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include "function/function_all.hpp"

class Game_information_scene : public App::Scene {
private:
    Button edit_button;
    Button back_button;
    Scroll_horizontal_manager scroll_manager_black;
    Scroll_horizontal_manager scroll_manager_white;
    Scroll_manager scroll_manager_memo;

public:
    Game_information_scene(const InitData& init) : IScene{ init } {
        back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        edit_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "edit"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        scroll_manager_black.init(X_CENTER - EXPORT_GAME_PLAYER_WIDTH, 80 + EXPORT_GAME_RADIUS * 2, EXPORT_GAME_PLAYER_WIDTH - 10, 10, 20, getData().game_information.black_player_name.size(), 1, X_CENTER - EXPORT_GAME_PLAYER_WIDTH, 80, EXPORT_GAME_PLAYER_WIDTH - 10, EXPORT_GAME_RADIUS * 2 + 10);
        scroll_manager_memo.init(X_CENTER + 300, 140, 10, 250, 20, 20, 1, X_CENTER - 300, 140, 600 + 10, 250);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        getData().fonts.font(language.get("play", "game_information")).draw(25, Arg::topCenter(X_CENTER, 10), getData().colors.white);
        int sy = 78;
        getData().fonts.font(language.get("in_out", "player_name")).draw(15, Arg::topCenter(X_CENTER, 57), getData().colors.white);
        Circle(X_CENTER - EXPORT_GAME_PLAYER_WIDTH - EXPORT_GAME_RADIUS - 20, 80 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.black);
        Circle(X_CENTER + EXPORT_GAME_PLAYER_WIDTH + EXPORT_GAME_RADIUS + 20, 80 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.white);
        
        Rect black_player_rect{X_CENTER - EXPORT_GAME_PLAYER_WIDTH, 80, EXPORT_GAME_PLAYER_WIDTH - 10, EXPORT_GAME_RADIUS * 2};
        black_player_rect.drawFrame(1, getData().colors.white);
        scroll_manager_black.draw();
        scroll_manager_black.update();
        std::cerr << scroll_manager_black.get_strt_idx_int() << std::endl;
        String black_player_display = get_substr(getData().game_information.black_player_name, scroll_manager_black.get_strt_idx_int(), EXPORT_GAME_PLAYER_WIDTH - 14);
        getData().fonts.font(black_player_display).draw(15, Arg::leftCenter(X_CENTER - EXPORT_GAME_PLAYER_WIDTH + 2, 80 + EXPORT_GAME_RADIUS), getData().colors.white);
        
        Rect white_player_rect{X_CENTER + 10, 80, EXPORT_GAME_PLAYER_WIDTH - 10, EXPORT_GAME_RADIUS * 2};
        white_player_rect.drawFrame(1, getData().colors.white);
        getData().fonts.font(getData().game_information.white_player_name).draw(15, white_player_rect.stretched(-3), getData().colors.white);
        getData().fonts.font(language.get("in_out", "memo")).draw(15, Arg::topCenter(X_CENTER, 117), getData().colors.white);
        Rect memo_rect{X_CENTER - 300, 140, 600, 250};
        memo_rect.drawFrame(1, getData().colors.white);

        scroll_manager_memo.draw();
        scroll_manager_memo.update();
        back_button.draw();
        edit_button.draw();
        if (back_button.clicked() || KeyEscape.pressed()){
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        if (edit_button.clicked()){
            changeScene(U"Export_game", SCENE_FADE_TIME);
        }
    }

    void draw() const override {
    }

private:
    String get_substr(String str, int strt_idx, int width){
        String res = str.substr(strt_idx);
        RectF region = getData().fonts.font(res).region(15, Vec2{0, 0});
        while (region.w > width && res.size()){
            res.pop_back();
            region = getData().fonts.font(res).region(15, Vec2{0, 0});
        }
        return res;
    }
};
