/*
    Egaroucid Project

    @file game_information.hpp
        browse game information
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
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
    std::vector<String> memo_lines;

public:
    Game_information_scene(const InitData& init) : IScene{ init } {
        back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        edit_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "edit"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        scroll_manager_black.init(X_CENTER - 300, 80 + 15 * 2, 300 - 10, 10, 20, getData().game_information.black_player_name.size(), 1, X_CENTER - 300, 80, 300 - 10, 15 * 2 + 10);
        scroll_manager_white.init(X_CENTER + 10, 80 + 15 * 2, 300 - 10, 10, 20, getData().game_information.white_player_name.size(), 1, X_CENTER + 10, 80, 300 - 10, 15 * 2 + 10);
        Array<String> lines = getData().game_information.memo.split(U'\n');
        for (String line: lines) {
            int idx = 0;
            while (idx < line.size()) {
                String sub_line = get_substr(line, idx, 600 - 4);
                memo_lines.emplace_back(sub_line);
                idx += sub_line.size();
            }
        }
        scroll_manager_memo.init(X_CENTER + 300, 150, 10, 250, 20, (int)memo_lines.size(), 1, X_CENTER - 300, 140, 600 + 10, 250);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        getData().fonts.font(language.get("play", "game_information")).draw(25, Arg::bottomCenter(X_CENTER, 45), getData().colors.white);
        if (getData().game_information.date.size()) {
            // date
            getData().fonts.font(language.get("in_out", "date") + U": " + getData().game_information.date).draw(15, Arg::bottomRight(X_CENTER + 300, 45), getData().colors.white);
        }
        int sy = 78;
        getData().fonts.font(language.get("in_out", "player_name")).draw(15, Arg::topCenter(X_CENTER, 57), getData().colors.white);
        Circle(X_CENTER - 300 - 15 - 20, 80 + 15, 15).draw(getData().colors.black);
        Circle(X_CENTER + 300 + 15 + 20, 80 + 15, 15).draw(getData().colors.white);
        // black player name
        Rect black_player_rect{X_CENTER - 300, 80, 300 - 10, 15 * 2};
        black_player_rect.drawFrame(1, getData().colors.white);
        scroll_manager_black.draw();
        scroll_manager_black.update();
        String black_player_display = get_substr(getData().game_information.black_player_name, scroll_manager_black.get_strt_idx_int(), 300 - 14);
        getData().fonts.font(black_player_display).draw(15, Arg::leftCenter(X_CENTER - 300 + 2, 80 + 15), getData().colors.white);
        // white player name
        Rect white_player_rect{X_CENTER + 10, 80, 300 - 10, 15 * 2};
        white_player_rect.drawFrame(1, getData().colors.white);
        scroll_manager_white.draw();
        scroll_manager_white.update();
        String white_player_display = get_substr(getData().game_information.white_player_name, scroll_manager_white.get_strt_idx_int(), 300 - 14);
        getData().fonts.font(white_player_display).draw(15, Arg::leftCenter(X_CENTER + 10 + 2, 80 + 15), getData().colors.white);
        // memo
        getData().fonts.font(language.get("in_out", "memo")).draw(15, Arg::topCenter(X_CENTER, 127), getData().colors.white);
        Rect memo_rect{X_CENTER - 300, 150, 600, 250};
        memo_rect.drawFrame(1, getData().colors.white);
        scroll_manager_memo.draw();
        scroll_manager_memo.update();
        String memo_display = get_substr_memo(scroll_manager_memo.get_strt_idx_int(), 250);
        getData().fonts.font(memo_display).draw(15, X_CENTER - 300 + 2, 150, getData().colors.white);
        // buttons
        back_button.draw();
        edit_button.draw();
        if (back_button.clicked() || KeyEscape.pressed()) {
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        if (edit_button.clicked()) {
            // Transition to Game_editor to edit current game information
            getData().game_editor_info.return_scene = U"Game_information_scene";
            getData().game_editor_info.is_editing_mode = getData().game_information.is_game_loaded;
            // Keep game_date and subfolder if a game is loaded (for editing saved game)
            // They are already set in Import_game::import_game()
            if (!getData().game_information.is_game_loaded) {
                getData().game_editor_info.game_date.clear();
                getData().game_editor_info.subfolder.clear();
            }
            getData().game_editor_info.game_info_updated = false;
            changeScene(U"Game_editor", SCENE_FADE_TIME);
        }
    }

    void draw() const override {
    }

private:
    String get_substr(String str, int strt_idx, int width) {
        String res = str.substr(strt_idx);
        RectF region = getData().fonts.font(res).region(15, Vec2{0, 0});
        while (region.w > width && res.size()) {
            res.pop_back();
            region = getData().fonts.font(res).region(15, Vec2{0, 0});
        }
        return res;
    }

    String get_substr_memo(int strt_idx, int height) {
        String res;
        double height_used = 0;
        for (int idx = strt_idx; idx < (int)memo_lines.size(); ++idx) {
            RectF region = getData().fonts.font(memo_lines[idx]).region(15, Vec2{0, 0});
            if (region.h + height_used > (double)height) {
                break;
            }
            res += memo_lines[idx] + U"\n";
            height_used += region.h;
        }
        return res;
    }
};