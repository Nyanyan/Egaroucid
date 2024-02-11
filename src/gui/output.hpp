/*
    Egaroucid Project

    @file output.hpp
        Output scenes
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include <chrono>
#include <time.h>
#include <sstream>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"

class Export_game : public App::Scene {
private:
    Button back_button;
    Button export_main_button;
    Button export_this_board_button;
    TextAreaEditState text_area[3]; // black player, white player, memo
    static constexpr int BLACK_PLAYER_IDX = 0;
    static constexpr int WHITE_PLAYER_IDX = 1;
    static constexpr int MEMO_IDX = 2;


public:
    Export_game(const InitData& init) : IScene{ init } {
        back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        export_main_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "export_main"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        export_this_board_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "export_until_this_board"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
        text_area[BLACK_PLAYER_IDX].active = true;
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        getData().fonts.font(language.get("in_out", "output_game")).draw(25, Arg::topCenter(X_CENTER, 10), getData().colors.white);
        getData().fonts.font(language.get("in_out", "player_name")).draw(20, Arg::topCenter(X_CENTER, 50), getData().colors.white);
        SimpleGUI::TextArea(text_area[BLACK_PLAYER_IDX], Vec2{X_CENTER - EXPORT_GAME_PLAYER_WIDTH, 80}, SizeF{EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT}, SimpleGUI::PreferredTextAreaMaxChars);
        SimpleGUI::TextArea(text_area[WHITE_PLAYER_IDX], Vec2{X_CENTER, 80}, SizeF{EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT}, SimpleGUI::PreferredTextAreaMaxChars);
        Circle(X_CENTER - EXPORT_GAME_PLAYER_WIDTH - EXPORT_GAME_RADIUS - 20, 80 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.black);
        Circle(X_CENTER + EXPORT_GAME_PLAYER_WIDTH + EXPORT_GAME_RADIUS + 20, 80 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.white);
        getData().fonts.font(language.get("in_out", "memo")).draw(20, Arg::topCenter(X_CENTER, 110), getData().colors.white);
        SimpleGUI::TextArea(text_area[MEMO_IDX], Vec2{X_CENTER - EXPORT_GAME_MEMO_WIDTH / 2, 140}, SizeF{EXPORT_GAME_MEMO_WIDTH, EXPORT_GAME_MEMO_HEIGHT}, SimpleGUI::PreferredTextAreaMaxChars);
        for (int i = 0; i < 3; ++i){
            std::string str = text_area[i].text.narrow();
            if (str.find("\t") != std::string::npos){
                text_area[i].active = false;
                text_area[(i + 1) % 3].active = true;
                text_area[i].text.replace(U"\t", U"");
                text_area[i].cursorPos = text_area[i].text.size();
                text_area[i].rebuildGlyphs();
            }
            if ((str.find("\n") != std::string::npos || str.find("\r") != std::string::npos) && i != MEMO_IDX){
                text_area[i].text.replace(U"\r", U"").replace(U"\n", U" ");
                text_area[i].cursorPos = text_area[i].text.size();
                text_area[i].rebuildGlyphs();
            }
        }
        getData().game_information.black_player_name = text_area[BLACK_PLAYER_IDX].text;
        getData().game_information.white_player_name = text_area[WHITE_PLAYER_IDX].text;
        getData().game_information.memo = text_area[MEMO_IDX].text;
        back_button.draw();
        export_main_button.draw();
        export_this_board_button.draw();
        if (back_button.clicked() || KeyEscape.pressed()) {
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        if (export_main_button.clicked()) {
            export_game(getData().graph_resources.nodes[0]);
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        if (export_this_board_button.clicked()) {
            std::vector<History_elem> history;
            int inspect_switch_n_discs = INF;
            if (getData().graph_resources.branch == 1) {
                if (getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
                    inspect_switch_n_discs = getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs();
                }
                else {
                    std::cerr << "no node found in inspect mode" << std::endl;
                }
            }
            for (History_elem& history_elem : getData().graph_resources.nodes[GRAPH_MODE_NORMAL]) {
                if (history_elem.board.n_discs() >= inspect_switch_n_discs || history_elem.board.n_discs() > getData().history_elem.board.n_discs()) {
                    break;
                }
                history.emplace_back(history_elem);
            }
            if (inspect_switch_n_discs != INF) {
                for (History_elem& history_elem : getData().graph_resources.nodes[GRAPH_MODE_INSPECT]) {
                    if (history_elem.board.n_discs() > getData().history_elem.board.n_discs()) {
                        break;
                    }
                    history.emplace_back(history_elem);
                }
            }
            export_game(history);
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
    }

    void draw() const override {

    }

private:

    void export_game(std::vector<History_elem> history) {
        String date = Unicode::Widen(calc_date());
        JSON json;
        json[GAME_DATE] = date;
        json[GAME_BLACK_PLAYER] = getData().game_information.black_player_name;
        json[GAME_WHITE_PLAYER] = getData().game_information.white_player_name;
        json[GAME_MEMO] = getData().game_information.memo;
        int black_discs = GAME_DISCS_UNDEFINED;
        int white_discs = GAME_DISCS_UNDEFINED;
        if (history.back().board.is_end()) {
            if (history.back().player == BLACK) {
                black_discs = history.back().board.count_player();
                white_discs = history.back().board.count_opponent();
            }
            else {
                black_discs = history.back().board.count_opponent();
                white_discs = history.back().board.count_player();
            }
        }
        json[GAME_BLACK_DISCS] = black_discs;
        json[GAME_WHITE_DISCS] = white_discs;
        for (History_elem history_elem : history) {
            String n_discs = Format(history_elem.board.n_discs());
            json[n_discs][GAME_BOARD_PLAYER] = history_elem.board.player;
            json[n_discs][GAME_BOARD_OPPONENT] = history_elem.board.opponent;
            json[n_discs][GAME_PLAYER] = history_elem.player;
            json[n_discs][GAME_VALUE] = history_elem.v;
            json[n_discs][GAME_LEVEL] = history_elem.level;
            json[n_discs][GAME_POLICY] = history_elem.policy;
            if (history_elem.board.n_discs() < history.back().board.n_discs()) {
                json[n_discs][GAME_NEXT_POLICY] = history_elem.next_policy;
            }
            else {
                json[n_discs][GAME_NEXT_POLICY] = -1;
            }
        }
        const String save_path = Unicode::Widen(getData().directories.document_dir) + U"games/" + date + U".json";
        json.save(save_path);

        const String csv_path = Unicode::Widen(getData().directories.document_dir) + U"games/summary.csv";
        CSV csv{ csv_path };
        String memo_summary_all = getData().game_information.memo.replaced(U"\r", U"").replaced(U"\n", U" ");
        String memo_summary;
        for (int i = 0; i < std::min((int)memo_summary_all.size(), GAME_MEMO_SUMMARY_SIZE); ++i) {
            memo_summary += memo_summary_all[i];
        }
        csv.writeRow(date, getData().game_information.black_player_name, getData().game_information.white_player_name, memo_summary, black_discs, white_discs);
        csv.save(csv_path);
    }
};
