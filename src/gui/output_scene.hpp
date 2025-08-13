/*
    Egaroucid Project

    @file output.hpp
        Output scenes
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
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
    // Subfolder input (under games/)
    TextAreaEditState folder_area; // relative path under games/
    std::string subfolder; // cached sanitized folder string


public:
    Export_game(const InitData& init) : IScene{ init } {
        back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        export_main_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "export_main"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        export_this_board_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "export_until_this_board"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
        text_area[BLACK_PLAYER_IDX].active = true;
        text_area[BLACK_PLAYER_IDX].text = getData().game_information.black_player_name;
        text_area[WHITE_PLAYER_IDX].text = getData().game_information.white_player_name;
        text_area[MEMO_IDX].text = getData().game_information.memo;
        for (int i = 0; i < 3; ++i) {
            text_area[i].rebuildGlyphs();
        }
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        getData().fonts.font(language.get("in_out", "output_game")).draw(25, Arg::topCenter(X_CENTER, 10), getData().colors.white);
        getData().fonts.font(language.get("in_out", "player_name")).draw(15, Arg::topCenter(X_CENTER, 47), getData().colors.white);
        SimpleGUI::TextArea(text_area[BLACK_PLAYER_IDX], Vec2{X_CENTER - EXPORT_GAME_PLAYER_WIDTH, 70}, SizeF{EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT}, SimpleGUI::PreferredTextAreaMaxChars);
        SimpleGUI::TextArea(text_area[WHITE_PLAYER_IDX], Vec2{X_CENTER, 70}, SizeF{EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT}, SimpleGUI::PreferredTextAreaMaxChars);
        Circle(X_CENTER - EXPORT_GAME_PLAYER_WIDTH - EXPORT_GAME_RADIUS - 20, 70 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.black);
        Circle(X_CENTER + EXPORT_GAME_PLAYER_WIDTH + EXPORT_GAME_RADIUS + 20, 70 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.white);
        // Memo label / counter / textbox (slightly higher and smaller)
        const int memo_label_y = 110;
        const int memo_box_y = 130;
        getData().fonts.font(language.get("in_out", "memo")).draw(15, Arg::topCenter(X_CENTER, memo_label_y), getData().colors.white);
        getData().fonts.font(Format(text_area[MEMO_IDX].text.size()) + U"/" + Format(TEXTBOX_MAX_CHARS) + U" " + language.get("common", "characters")).draw(15, Arg::topRight(X_CENTER + EXPORT_GAME_MEMO_WIDTH / 2, memo_label_y), getData().colors.white);
        SimpleGUI::TextArea(text_area[MEMO_IDX], Vec2{X_CENTER - EXPORT_GAME_MEMO_WIDTH / 2, memo_box_y}, SizeF{EXPORT_GAME_MEMO_WIDTH, EXPORT_GAME_MEMO_HEIGHT}, TEXTBOX_MAX_CHARS);
        // Subfolder input UI (place just above buttons area to avoid overlap)
        const int folder_label_y = BUTTON3_SY - 86;
        const int folder_box_y   = BUTTON3_SY - 60;
        getData().fonts.font(U"保存先サブフォルダ (games/ 以下)").draw(15, Arg::topCenter(X_CENTER, folder_label_y), getData().colors.white);
        SimpleGUI::TextArea(folder_area, Vec2{ X_CENTER - EXPORT_GAME_MEMO_WIDTH / 2, folder_box_y }, SizeF{ EXPORT_GAME_MEMO_WIDTH, 26 }, TEXTBOX_MAX_CHARS);
        // Tab navigation across 4 fields: black -> white -> memo -> folder -> black
        auto focus_next_from = [&](int idx) {
            // deactivate current
            text_area[idx].active = false;
            if (idx == MEMO_IDX) {
                folder_area.active = true;
            } else {
                text_area[(idx + 1) % 3].active = true;
            }
        };
        for (int i = 0; i < 3; ++i) {
            std::string str = text_area[i].text.narrow();
            if (str.find('\t') != std::string::npos) {
                text_area[i].text.replace(U"\t", U"");
                text_area[i].cursorPos = text_area[i].text.size();
                text_area[i].rebuildGlyphs();
                focus_next_from(i);
            }
            if ((str.find('\n') != std::string::npos || str.find('\r') != std::string::npos) && i != MEMO_IDX) {
                text_area[i].text.replace(U"\r", U"").replace(U"\n", U" ");
                text_area[i].cursorPos = text_area[i].text.size();
                text_area[i].rebuildGlyphs();
            }
        }
        // Folder field: sanitize newline, Tab moves to BLACK field
        {
            std::string fstr = folder_area.text.narrow();
            bool tab_found = (fstr.find('\t') != std::string::npos);
            if (tab_found) {
                folder_area.text.replace(U"\t", U"");
            }
            if (fstr.find('\n') != std::string::npos || fstr.find('\r') != std::string::npos) {
                folder_area.text.replace(U"\r", U"").replace(U"\n", U" ");
            }
            if (tab_found) {
                folder_area.active = false;
                text_area[BLACK_PLAYER_IDX].active = true;
            }
            folder_area.cursorPos = folder_area.text.size();
            folder_area.rebuildGlyphs();
        }
        getData().game_information.black_player_name = text_area[BLACK_PLAYER_IDX].text;
        getData().game_information.white_player_name = text_area[WHITE_PLAYER_IDX].text;
        getData().game_information.memo = text_area[MEMO_IDX].text;
        // Sanitize folder input each frame
        {
            String s = folder_area.text.replaced(U"\r", U"").replaced(U"\n", U"").replaced(U"\\", U"/");
            // trim leading and trailing '/'
            while (s.size() && s.front() == U'/') s.erase(s.begin());
            while (s.size() && s.back() == U'/') s.pop_back();
            // very naive '..' removal for safety
            s.replace(U"..", U"");
            subfolder = s.narrow();
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
            if (getData().graph_resources.branch == 1) {
                if (getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
                    inspect_switch_n_discs = getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs();
                } else {
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
            } else {
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
            } else {
                json[n_discs][GAME_NEXT_POLICY] = -1;
            }
        }
        // Build directory path: appdata/games/(subfolder)/
        String base_dir = Unicode::Widen(getData().directories.document_dir) + U"games/";
        String folder = Unicode::Widen(subfolder);
        if (folder.size()) {
            base_dir += folder + U"/";
        }
        FileSystem::CreateDirectories(base_dir);
        const String save_path = base_dir + date + U".json";
        json.save(save_path);

        const String csv_path = base_dir + U"summary.csv";
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
