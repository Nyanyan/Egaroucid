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
    // Folder picker overlay state (explorer-like under games/)
    bool show_folder_picker = false;
    std::vector<History_elem> pending_history; // history to save after selection
    // current folder path under games/
    std::string subfolder; // final chosen subfolder used for saving
    std::string picker_subfolder; // navigating path during selection
    std::vector<String> save_folders_display; // includes optional ".."
    bool picker_has_parent = false;
    Scroll_manager folder_scroll_manager;
    // new folder UI
    TextAreaEditState new_folder_area;
    Button create_folder_button;
    Button save_here_button;
    Button cancel_picker_button;
    // Saving state
    bool is_saving = false;
    bool saving_started = false;


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
        // init folder picker buttons (positions used only in overlay)
        create_folder_button.init(600, EXPORT_GAME_CREATE_FOLDER_Y_CENTER - 28 / 2, 120, 28, 8, language.get("in_out", "create"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
        save_here_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "save_here"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        cancel_picker_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        // Saving mode: handled first
        if (is_saving) {
            Scene::SetBackground(getData().colors.green);
            getData().fonts.font(language.get("in_out", "saving")).draw(30, Arg::center(X_CENTER, Y_CENTER), getData().colors.white);
            if (!saving_started) {
                saving_started = true; // ensure one frame shows
                return;
            }
            export_game(pending_history);
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
            return;
        }

        // Folder picker mode: draw exclusively and return
        if (show_folder_picker) {
            Scene::SetBackground(getData().colors.green);
            // Path label
            String path_label = U"games/" + Unicode::Widen(picker_subfolder);
            getData().fonts.font(language.get("in_out", "save_subfolder")).draw(20, Arg::topCenter(X_CENTER, 10), getData().colors.white);
            getData().fonts.font(path_label).draw(15, Arg::topCenter(X_CENTER, 40), getData().colors.white);
            // List
            int sy = IMPORT_GAME_SY;
            int strt_idx_int = folder_scroll_manager.get_strt_idx_int();
            if (strt_idx_int > 0) {
                getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, sy }, getData().colors.white);
            }
            sy += 8;
            for (int row = strt_idx_int; row < std::min((int)save_folders_display.size(), strt_idx_int + IMPORT_GAME_N_GAMES_ON_WINDOW); ++row) {
                Rect rect;
                rect.y = sy;
                rect.x = IMPORT_GAME_SX;
                rect.w = IMPORT_GAME_WIDTH;
                rect.h = IMPORT_GAME_HEIGHT;
                if (row % 2) {
                    rect.draw(getData().colors.green).drawFrame(1.0, getData().colors.white);
                } else {
                    rect.draw(getData().colors.dark_green).drawFrame(1.0, getData().colors.white);
                }
                String fname = save_folders_display[row];
                getData().fonts.font(fname).draw(15, IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, sy + IMPORT_GAME_PLAYER_HEIGHT / 2, getData().colors.white);
                if (Rect(IMPORT_GAME_SX, sy, IMPORT_GAME_WIDTH, IMPORT_GAME_HEIGHT).leftClicked()) {
                    if (fname == U"..") {
                        if (!picker_subfolder.empty()) {
                            std::string s = picker_subfolder;
                            if (s.back() == '/') s.pop_back();
                            size_t pos = s.find_last_of('/');
                            if (pos == std::string::npos) picker_subfolder.clear();
                            else picker_subfolder = s.substr(0, pos);
                        }
                    } else {
                        if (!picker_subfolder.empty()) picker_subfolder += "/";
                        picker_subfolder += fname.narrow();
                    }
                    enumerate_save_dir();
                    init_folder_scroll_manager();
                    return;
                }
                sy += IMPORT_GAME_HEIGHT;
            }
            if (folder_scroll_manager.get_strt_idx_int() + IMPORT_GAME_N_GAMES_ON_WINDOW < (int)save_folders_display.size()) {
                getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, sy }, getData().colors.white);
            }
            folder_scroll_manager.draw();
            folder_scroll_manager.update();

            // New folder UI (label + textarea + button in one horizontal row)
            const int labelX = IMPORT_GAME_SX;
            const int labelW = 140;   // fixed column width for the label
            const int rowTop = BUTTON3_SY - 115; // align with textarea top

            // Label
            getData().fonts.font(language.get("in_out", "new_folder")).draw(15, Arg::rightCenter(IMPORT_GAME_SX + 200, EXPORT_GAME_CREATE_FOLDER_Y_CENTER), getData().colors.white);
            // TextArea to the right of the label
            SimpleGUI::TextArea(new_folder_area, Vec2{ IMPORT_GAME_SX + 205, EXPORT_GAME_CREATE_FOLDER_Y_CENTER - 18 }, SizeF{ 300, 26 }, 64);

            // Button to the right of the textarea
            create_folder_button.draw();
            if (create_folder_button.clicked()) {
                String s = new_folder_area.text.replaced(U"\r", U"").replaced(U"\n", U"").replaced(U"\\", U"/");
                while (s.size() && s.front() == U'/') s.erase(s.begin());
                while (s.size() && s.back() == U'/') s.pop_back();
                s.replace(U"..", U"");
                if (s.size()) {
                    String base = Unicode::Widen(getData().directories.document_dir) + U"games/" + Unicode::Widen(picker_subfolder);
                    if (base.size() && base.back() != U'/') base += U"/";
                    String target = base + s + U"/";
                    FileSystem::CreateDirectories(target);
                    new_folder_area.text.clear();
                    new_folder_area.cursorPos = 0;
                    new_folder_area.rebuildGlyphs();
                    enumerate_save_dir();
                    init_folder_scroll_manager();
                }
            }

            // Action buttons
            cancel_picker_button.draw();
            if (cancel_picker_button.clicked() || KeyEscape.pressed()) {
                show_folder_picker = false;
                return;
            }
            save_here_button.draw();
            if (save_here_button.clicked()) {
                subfolder = picker_subfolder; // commit selection
                show_folder_picker = false;
                is_saving = true;
                saving_started = false;
                return;
            }
            return;
        }
        // Saving mode: full-screen, hide all other UI
        if (is_saving) {
            Scene::SetBackground(getData().colors.green);
            getData().fonts.font(language.get("in_out", "saving")).draw(30, Arg::center(X_CENTER, Y_CENTER), getData().colors.white);
            if (!saving_started) {
                saving_started = true; // show at least one frame
                return;
            }
            export_game(pending_history);
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
            return;
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
        // Tab移動: black -> white -> memo -> black（フォルダ入力は使わない）
        auto focus_next_from = [&](int idx) {
            // deactivate current
            text_area[idx].active = false;
            text_area[(idx + 1) % 3].active = true;
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
        getData().game_information.black_player_name = text_area[BLACK_PLAYER_IDX].text;
        getData().game_information.white_player_name = text_area[WHITE_PLAYER_IDX].text;
        getData().game_information.memo = text_area[MEMO_IDX].text;
        back_button.draw();
        export_main_button.draw();
        export_this_board_button.draw();
        if (!show_folder_picker) {
            if (back_button.clicked() || KeyEscape.pressed()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        } else {
            if (KeyEscape.pressed()) {
                show_folder_picker = false;
            }
        }
        // Open folder picker after clicking save buttons
        if (!show_folder_picker && export_main_button.clicked()) {
            pending_history = getData().graph_resources.nodes[0];
            picker_subfolder.clear();
            enumerate_save_dir();
            init_folder_scroll_manager();
            new_folder_area.text.clear();
            new_folder_area.cursorPos = 0;
            new_folder_area.rebuildGlyphs();
            show_folder_picker = true;
        }
        if (!show_folder_picker && export_this_board_button.clicked()) {
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
            pending_history.swap(history);
            picker_subfolder.clear();
            enumerate_save_dir();
            init_folder_scroll_manager();
            new_folder_area.text.clear();
            new_folder_area.cursorPos = 0;
            new_folder_area.rebuildGlyphs();
            show_folder_picker = true;
        }

        // Folder picker overlay UI
        if (show_folder_picker) {
            // Dim background
            Rect{0, 0, Scene::Width(), Scene::Height()}.draw(ColorF{0.0, 0.0, 0.0, 0.4});
            // Panel
            const int panelX = IMPORT_GAME_SX - 10;
            const int panelY = 60;
            const int panelW = IMPORT_GAME_WIDTH + 20;
            const int panelH = IMPORT_GAME_HEIGHT * IMPORT_GAME_N_GAMES_ON_WINDOW + 120;
            Rect{panelX, panelY, panelW, panelH}.draw(getData().colors.dark_green).drawFrame(2, 0, getData().colors.white);
            // Path label
            String path_label = U"games/" + Unicode::Widen(picker_subfolder);
            getData().fonts.font(path_label).draw(15, Arg::topCenter(X_CENTER, panelY + 8), getData().colors.white);
            // List
            int sy = IMPORT_GAME_SY;
            int strt_idx_int = folder_scroll_manager.get_strt_idx_int();
            if (strt_idx_int > 0) {
                getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, sy }, getData().colors.white);
            }
            sy += 8;
            for (int row = strt_idx_int; row < std::min((int)save_folders_display.size(), strt_idx_int + IMPORT_GAME_N_GAMES_ON_WINDOW); ++row) {
                Rect rect;
                rect.y = sy;
                rect.x = IMPORT_GAME_SX;
                rect.w = IMPORT_GAME_WIDTH;
                rect.h = IMPORT_GAME_HEIGHT;
                if (row % 2) {
                    rect.draw(getData().colors.green).drawFrame(1.0, getData().colors.white);
                } else {
                    rect.draw(getData().colors.dark_green).drawFrame(1.0, getData().colors.white);
                }
                String fname = save_folders_display[row];
                getData().fonts.font(fname).draw(15, IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, sy + IMPORT_GAME_PLAYER_HEIGHT / 2, getData().colors.white);
                if (Rect(IMPORT_GAME_SX, sy, IMPORT_GAME_WIDTH, IMPORT_GAME_HEIGHT).leftClicked()) {
                    if (fname == U"..") {
                        if (!picker_subfolder.empty()) {
                            std::string s = picker_subfolder;
                            if (s.back() == '/') s.pop_back();
                            size_t pos = s.find_last_of('/');
                            if (pos == std::string::npos) picker_subfolder.clear();
                            else picker_subfolder = s.substr(0, pos);
                        }
                    } else {
                        if (!picker_subfolder.empty()) picker_subfolder += "/";
                        picker_subfolder += fname.narrow();
                    }
                    enumerate_save_dir();
                    init_folder_scroll_manager();
                    return;
                }
                sy += IMPORT_GAME_HEIGHT;
            }
            if (folder_scroll_manager.get_strt_idx_int() + IMPORT_GAME_N_GAMES_ON_WINDOW < (int)save_folders_display.size()) {
                getData().fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, panelY + panelH - 60 }, getData().colors.white);
            }
            folder_scroll_manager.draw();
            folder_scroll_manager.update();

            // New folder UI
            getData().fonts.font(language.get("in_out", "new_folder")).draw(13, IMPORT_GAME_SX, BUTTON3_SY - 140, getData().colors.white);
            SimpleGUI::TextArea(new_folder_area, Vec2{ IMPORT_GAME_SX, BUTTON3_SY - 115 }, SizeF{ 200, 26 }, 64);
            create_folder_button.draw();
            if (create_folder_button.clicked()) {
                String s = new_folder_area.text.replaced(U"\r", U"").replaced(U"\n", U"").replaced(U"\\", U"/");
                while (s.size() && s.front() == U'/') s.erase(s.begin());
                while (s.size() && s.back() == U'/') s.pop_back();
                s.replace(U"..", U"");
                if (s.size()) {
                    String base = Unicode::Widen(getData().directories.document_dir) + U"games/" + Unicode::Widen(picker_subfolder);
                    if (base.size() && base.back() != U'/') base += U"/";
                    String target = base + s + U"/";
                    FileSystem::CreateDirectories(target);
                    new_folder_area.text.clear();
                    new_folder_area.cursorPos = 0;
                    new_folder_area.rebuildGlyphs();
                    enumerate_save_dir();
                    init_folder_scroll_manager();
                }
            }

            // Action buttons
            cancel_picker_button.draw();
            if (cancel_picker_button.clicked()) {
                show_folder_picker = false;
            }
            save_here_button.draw();
            if (save_here_button.clicked()) {
                subfolder = picker_subfolder; // commit selection
                show_folder_picker = false;
                is_saving = true;
                saving_started = false;
            }
        }
    }

    void draw() const override {

    }

private:
    // Folder picker helpers
    void enumerate_save_dir() {
        save_folders_display.clear();
        picker_has_parent = !picker_subfolder.empty();
        if (picker_has_parent) save_folders_display.emplace_back(U"..");
        String base = Unicode::Widen(getData().directories.document_dir) + U"games/" + Unicode::Widen(picker_subfolder);
        if (base.size() && base.back() != U'/') base += U"/";
        Array<FilePath> list = FileSystem::DirectoryContents(base);
        Array<String> real_folders;
        for (const auto& path : list) {
            if (FileSystem::IsDirectory(path)) {
                String name = path;
                while (name.size() && (name.back() == U'/' || name.back() == U'\\')) name.pop_back();
                size_t pos = name.lastIndexOf(U'/');
                if (pos == String::npos) pos = name.lastIndexOf(U'\\');
                if (pos != String::npos) name = name.substr(pos + 1);
                if (name.size()) real_folders.emplace_back(name);
            }
        }
        std::sort(real_folders.begin(), real_folders.end());
        for (auto& n : real_folders) save_folders_display.emplace_back(n);
    }

    void init_folder_scroll_manager() {
        int total = (int)save_folders_display.size();
        folder_scroll_manager.init(770, IMPORT_GAME_SY + 8, 10, IMPORT_GAME_HEIGHT * IMPORT_GAME_N_GAMES_ON_WINDOW, 20, total, IMPORT_GAME_N_GAMES_ON_WINDOW, IMPORT_GAME_SX, 73, IMPORT_GAME_WIDTH + 10, IMPORT_GAME_HEIGHT * IMPORT_GAME_N_GAMES_ON_WINDOW);
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
