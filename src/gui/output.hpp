/*
    Egaroucid Project

    @file output.hpp
        Output scenes
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
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
    bool active_cells[3];


public:
    Export_game(const InitData& init) : IScene{ init } {
        back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        export_main_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "export_main"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        export_this_board_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "export_until_this_board"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
        for (int i = 0; i < 3; ++i) {
            active_cells[i] = false;
        }
        active_cells[0] = true;
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        getData().fonts.font(language.get("in_out", "output_game")).draw(25, Arg::topCenter(X_CENTER, 10), getData().colors.white);
        getData().fonts.font(language.get("in_out", "player_name")).draw(20, Arg::topCenter(X_CENTER, 50), getData().colors.white);
        Rect black_area{ X_CENTER - EXPORT_GAME_PLAYER_WIDTH, 80, EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT };
        Rect white_area{ X_CENTER, 80, EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT };
        Circle(X_CENTER - EXPORT_GAME_PLAYER_WIDTH - EXPORT_GAME_RADIUS - 20, 80 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.black);
        Circle(X_CENTER + EXPORT_GAME_PLAYER_WIDTH + EXPORT_GAME_RADIUS + 20, 80 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.white);
        getData().fonts.font(language.get("in_out", "memo")).draw(20, Arg::topCenter(X_CENTER, 110), getData().colors.white);
        Rect memo_area{ X_CENTER - EXPORT_GAME_MEMO_WIDTH / 2, 140, EXPORT_GAME_MEMO_WIDTH, EXPORT_GAME_MEMO_HEIGHT };
        const String editingText = TextInput::GetEditingText();
        bool active_draw = false;
        if (black_area.leftClicked()) {
            active_cells[0] = true;
            active_cells[1] = false;
            active_cells[2] = false;
        }
        else if (white_area.leftClicked()) {
            active_cells[0] = false;
            active_cells[1] = true;
            active_cells[2] = false;
        }
        else if (memo_area.leftClicked()) {
            active_cells[0] = false;
            active_cells[1] = false;
            active_cells[2] = true;
        }
        if (!active_draw && active_cells[0]) {
            active_draw = true;
            black_area.draw(getData().colors.light_cyan).drawFrame(2, getData().colors.black);
            TextInput::UpdateText(getData().game_information.black_player_name);
            if (KeyControl.pressed() && KeyV.down()) {
                String clip_text;
                Clipboard::GetText(clip_text);
                getData().game_information.black_player_name += clip_text;
            }
            if (getData().game_information.black_player_name.narrow().find('\t') != std::string::npos) {
                for (int i = 0; i < 3; ++i) {
                    if (active_cells[i]) {
                        active_cells[i] = false;
                        active_cells[(i + 1) % 3] = true;
                        break;
                    }
                }
            }
            getData().game_information.black_player_name = getData().game_information.black_player_name.replaced(U"\r", U"").replaced(U"\n", U" ").replaced(U"\t", U"");
            getData().fonts.font(getData().game_information.black_player_name + U'|' + editingText).draw(15, black_area.stretched(-4), getData().colors.black);
        }
        else {
            black_area.draw(getData().colors.white).drawFrame(2, getData().colors.black);
            getData().fonts.font(getData().game_information.black_player_name).draw(15, black_area.stretched(-4), getData().colors.black);
        }
        if (!active_draw && active_cells[1]) {
            active_draw = true;
            white_area.draw(getData().colors.light_cyan).drawFrame(2, getData().colors.black);
            TextInput::UpdateText(getData().game_information.white_player_name);
            if (KeyControl.pressed() && KeyV.down()) {
                String clip_text;
                Clipboard::GetText(clip_text);
                getData().game_information.white_player_name += clip_text;
            }
            if (getData().game_information.white_player_name.narrow().find('\t') != std::string::npos) {
                for (int i = 0; i < 3; ++i) {
                    if (active_cells[i]) {
                        active_cells[i] = false;
                        active_cells[(i + 1) % 3] = true;
                        break;
                    }
                }
            }
            getData().game_information.white_player_name = getData().game_information.white_player_name.replaced(U"\r", U"").replaced(U"\n", U" ").replaced(U"\t", U"");
            getData().fonts.font(getData().game_information.white_player_name + U'|' + editingText).draw(15, white_area.stretched(-4), getData().colors.black);
        }
        else {
            white_area.draw(getData().colors.white).drawFrame(2, getData().colors.black);
            getData().fonts.font(getData().game_information.white_player_name).draw(15, white_area.stretched(-4), getData().colors.black);
        }
        if (!active_draw && active_cells[2]) {
            active_draw = true;
            memo_area.draw(getData().colors.light_cyan).drawFrame(2, getData().colors.black);
            TextInput::UpdateText(getData().game_information.memo);
            if (KeyControl.pressed() && KeyV.down()) {
                String clip_text;
                Clipboard::GetText(clip_text);
                getData().game_information.memo += clip_text;
            }
            if (getData().game_information.memo.narrow().find('\t') != std::string::npos) {
                for (int i = 0; i < 3; ++i) {
                    if (active_cells[i]) {
                        active_cells[i] = false;
                        active_cells[(i + 1) % 3] = true;
                        break;
                    }
                }
            }
            getData().game_information.memo = getData().game_information.memo.replaced(U"\t", U"");
            getData().fonts.font(getData().game_information.memo + U'|' + editingText).draw(15, memo_area.stretched(-4), getData().colors.black);
            
        }
        else {
            memo_area.draw(getData().colors.white).drawFrame(2, getData().colors.black);
            getData().fonts.font(getData().game_information.memo).draw(15, memo_area.stretched(-4), getData().colors.black);
        }
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
            if (getData().graph_resources.put_mode == 1) {
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
#ifdef _WIN64
    int get_localtime(tm* a, time_t* b) {
        return localtime_s(a, b);
    }
#else
    int get_localtime(tm* a, time_t* b) {
        a = localtime(b);
        return 0;
    }
#endif

    std::string calc_date() {
        time_t now;
        tm newtime;
        time(&now);
        int err = get_localtime(&newtime, &now);
        std::stringstream sout;
        std::string year = std::to_string(newtime.tm_year + 1900);
        sout << std::setfill('0') << std::setw(2) << newtime.tm_mon + 1;
        std::string month = sout.str();
        sout.str("");
        sout.clear(std::stringstream::goodbit);
        sout << std::setfill('0') << std::setw(2) << newtime.tm_mday;
        std::string day = sout.str();
        sout.str("");
        sout.clear(std::stringstream::goodbit);
        sout << std::setfill('0') << std::setw(2) << newtime.tm_hour;
        std::string hour = sout.str();
        sout.str("");
        sout.clear(std::stringstream::goodbit);
        sout << std::setfill('0') << std::setw(2) << newtime.tm_min;
        std::string minute = sout.str();
        sout.str("");
        sout.clear(std::stringstream::goodbit);
        sout << std::setfill('0') << std::setw(2) << newtime.tm_sec;
        std::string second = sout.str();
        return year + "_" + month + "_" + day + "_" + hour + "_" + minute + "_" + second;
    }

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
        const String save_path = Unicode::Widen(getData().directories.document_dir) + U"Egaroucid/games/" + date + U".json";
        json.save(save_path);

        const String csv_path = Unicode::Widen(getData().directories.document_dir) + U"Egaroucid/games/summary.csv";
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
