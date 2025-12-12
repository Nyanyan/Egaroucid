/*
    Egaroucid Project

    @file game_editor_scene.hpp
        Game editor scene
    @date 2021-2025
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

// Game save helper functions
namespace game_save_helper {
    // Save game to JSON file and update CSV summary
    inline void save_game_to_file(
        const String& base_dir,
        const String& date,
        const String& black_player_name,
        const String& white_player_name,
        const String& memo,
        const std::vector<History_elem>& history
    ) {
        JSON json;
        json[GAME_DATE] = date;
        json[GAME_BLACK_PLAYER] = black_player_name;
        json[GAME_WHITE_PLAYER] = white_player_name;
        json[GAME_MEMO] = memo;
        
        int black_discs = GAME_DISCS_UNDEFINED;
        int white_discs = GAME_DISCS_UNDEFINED;
        Board last_board = history.back().board;
        if (last_board.is_end()) {
            if (history.back().player == BLACK) {
                black_discs = last_board.count_player();
                white_discs = last_board.count_opponent();
            } else {
                black_discs = last_board.count_opponent();
                white_discs = last_board.count_player();
            }
        }
        json[GAME_BLACK_DISCS] = black_discs;
        json[GAME_WHITE_DISCS] = white_discs;
        
        for (const History_elem& history_elem : history) {
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
        
        FileSystem::CreateDirectories(base_dir);
        const String save_path = base_dir + date + U".json";
        json.save(save_path);

        const String csv_path = base_dir + U"summary.csv";
        CSV csv{ csv_path };
        String memo_summary_all = memo.replaced(U"\r", U"").replaced(U"\n", U" ");
        String memo_summary;
        for (int i = 0; i < std::min((int)memo_summary_all.size(), GAME_MEMO_SUMMARY_SIZE); ++i) {
            memo_summary += memo_summary_all[i];
        }
        csv.writeRow(date, black_player_name, white_player_name, memo_summary, black_discs, white_discs);
        csv.save(csv_path);
    }

    // Update existing game in CSV (for editing existing games)
    inline void update_game_in_csv(
        const String& csv_path,
        const String& date,
        const String& black_player_name,
        const String& white_player_name,
        const String& memo,
        int black_discs,
        int white_discs
    ) {
        CSV csv{ csv_path };
        CSV new_csv;
        
        bool found = false;
        for (size_t i = 0; i < csv.rows(); ++i) {
            if (csv[i].size() >= 6) {
                if (csv[i][0] == date) {
                    // Update this row
                    String memo_summary_all = memo.replaced(U"\r", U"").replaced(U"\n", U" ");
                    String memo_summary;
                    for (int j = 0; j < std::min((int)memo_summary_all.size(), GAME_MEMO_SUMMARY_SIZE); ++j) {
                        memo_summary += memo_summary_all[j];
                    }
                    new_csv.write(date);
                    new_csv.write(black_player_name);
                    new_csv.write(white_player_name);
                    new_csv.write(memo_summary);
                    new_csv.write(black_discs == GAME_DISCS_UNDEFINED ? U"" : ToString(black_discs));
                    new_csv.write(white_discs == GAME_DISCS_UNDEFINED ? U"" : ToString(white_discs));
                    new_csv.newLine();
                    found = true;
                } else {
                    // Keep existing row
                    for (int j = 0; j < 6; ++j) {
                        new_csv.write(csv[i][j]);
                    }
                    new_csv.newLine();
                }
            }
        }
        
        new_csv.save(csv_path);
    }
}

class Game_editor : public App::Scene {
private:
    Button back_button;
    Button ok_button;
    TextAreaEditState text_area[3]; // black player, white player, memo
    static constexpr int BLACK_PLAYER_IDX = 0;
    static constexpr int WHITE_PLAYER_IDX = 1;
    static constexpr int MEMO_IDX = 2;
    
    // Return scene info
    String return_scene;
    
    // Editing mode: true = editing existing game, false = new game (from Export_game)
    bool is_editing_mode;
    String existing_game_date;  // for editing mode
    std::string existing_game_subfolder;  // for editing mode

public:
    Game_editor(const InitData& init) : IScene{ init } {
        back_button.init(BUTTON2_1_SX, BUTTON2_SY, BUTTON2_WIDTH, BUTTON2_HEIGHT, BUTTON2_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        ok_button.init(BUTTON2_2_SX, BUTTON2_SY, BUTTON2_WIDTH, BUTTON2_HEIGHT, BUTTON2_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        
        text_area[BLACK_PLAYER_IDX].active = true;
        text_area[BLACK_PLAYER_IDX].text = getData().game_information.black_player_name;
        text_area[WHITE_PLAYER_IDX].text = getData().game_information.white_player_name;
        text_area[MEMO_IDX].text = getData().game_information.memo;
        for (int i = 0; i < 3; ++i) {
            text_area[i].rebuildGlyphs();
        }
        
        // Get return scene info from game_editor_info
        return_scene = getData().game_editor_info.return_scene;
        is_editing_mode = getData().game_editor_info.is_editing_mode;
        existing_game_date = getData().game_editor_info.game_date;
        existing_game_subfolder = getData().game_editor_info.subfolder;
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);

        getData().fonts.font(language.get("in_out", is_editing_mode ? "edit_game" : "output_game")).draw(25, Arg::topCenter(X_CENTER, 10), getData().colors.white);
        getData().fonts.font(language.get("in_out", "player_name")).draw(15, Arg::topCenter(X_CENTER, 47), getData().colors.white);
        SimpleGUI::TextArea(text_area[BLACK_PLAYER_IDX], Vec2{X_CENTER - EXPORT_GAME_PLAYER_WIDTH, 70}, SizeF{EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT}, SimpleGUI::PreferredTextAreaMaxChars);
        SimpleGUI::TextArea(text_area[WHITE_PLAYER_IDX], Vec2{X_CENTER, 70}, SizeF{EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT}, SimpleGUI::PreferredTextAreaMaxChars);
        Circle(X_CENTER - EXPORT_GAME_PLAYER_WIDTH - EXPORT_GAME_RADIUS - 20, 70 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.black);
        Circle(X_CENTER + EXPORT_GAME_PLAYER_WIDTH + EXPORT_GAME_RADIUS + 20, 70 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.white);
        
        // Memo label / counter / textbox
        const int memo_label_y = 110;
        const int memo_box_y = 130;
        getData().fonts.font(language.get("in_out", "memo")).draw(15, Arg::topCenter(X_CENTER, memo_label_y), getData().colors.white);
        getData().fonts.font(Format(text_area[MEMO_IDX].text.size()) + U"/" + Format(TEXTBOX_MAX_CHARS) + U" " + language.get("common", "characters")).draw(15, Arg::topRight(X_CENTER + EXPORT_GAME_MEMO_WIDTH / 2, memo_label_y), getData().colors.white);
        SimpleGUI::TextArea(text_area[MEMO_IDX], Vec2{X_CENTER - EXPORT_GAME_MEMO_WIDTH / 2, memo_box_y}, SizeF{EXPORT_GAME_MEMO_WIDTH, EXPORT_GAME_MEMO_HEIGHT}, TEXTBOX_MAX_CHARS);
        
        // Tab移動: black -> white -> memo -> black
        auto focus_next_from = [&](int idx) {
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
        ok_button.draw();
        
        if (back_button.clicked() || KeyEscape.pressed()) {
            changeScene(return_scene, SCENE_FADE_TIME);
        }
        
        // Only allow Enter to submit if memo field is not active (to allow newlines in memo)
        bool can_submit_with_enter = KeyEnter.pressed() && !text_area[MEMO_IDX].active;
        
        if (ok_button.clicked() || can_submit_with_enter) {
            if (is_editing_mode) {
                // Update existing game
                save_edited_game();
            } else {
                // New game from Export_game - need folder picker
                getData().game_editor_info.game_info_updated = true;
            }
            changeScene(return_scene, SCENE_FADE_TIME);
        }
    }

    void draw() const override {
    }

private:
    void save_edited_game() {
        // Build base directory
        String base_dir = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!existing_game_subfolder.empty()) {
            base_dir += Unicode::Widen(existing_game_subfolder) + U"/";
        }
        
        // Load existing game history from JSON
        const String json_path = base_dir + existing_game_date + U".json";
        JSON game_json = JSON::Load(json_path);
        if (!game_json) {
            std::cerr << "Failed to load game JSON: " << json_path.narrow() << std::endl;
            return;
        }
        
        // Reconstruct history from JSON
        std::vector<History_elem> history;
        for (int n_discs = 4; n_discs <= HW2; ++n_discs) {
            String n_discs_str = Format(n_discs);
            if (game_json[n_discs_str]) {
                History_elem history_elem;
                if (game_json[n_discs_str][GAME_BOARD_PLAYER].getType() == JSONValueType::Number) {
                    history_elem.board.player = game_json[n_discs_str][GAME_BOARD_PLAYER].get<uint64_t>();
                } else {
                    break;
                }
                if (game_json[n_discs_str][GAME_BOARD_OPPONENT].getType() == JSONValueType::Number) {
                    history_elem.board.opponent = game_json[n_discs_str][GAME_BOARD_OPPONENT].get<uint64_t>();
                } else {
                    break;
                }
                if (game_json[n_discs_str][GAME_PLAYER].getType() == JSONValueType::Number) {
                    history_elem.player = game_json[n_discs_str][GAME_PLAYER].get<int>();
                } else {
                    break;
                }
                if (game_json[n_discs_str][GAME_VALUE].getType() == JSONValueType::Number) {
                    history_elem.v = game_json[n_discs_str][GAME_VALUE].get<int>();
                } else {
                    break;
                }
                if (game_json[n_discs_str][GAME_LEVEL].getType() == JSONValueType::Number) {
                    history_elem.level = game_json[n_discs_str][GAME_LEVEL].get<int>();
                } else {
                    break;
                }
                if (game_json[n_discs_str][GAME_POLICY].getType() == JSONValueType::Number) {
                    history_elem.policy = game_json[n_discs_str][GAME_POLICY].get<int>();
                } else {
                    break;
                }
                if (n_discs < HW2) {
                    if (game_json[n_discs_str][GAME_NEXT_POLICY].getType() == JSONValueType::Number) {
                        history_elem.next_policy = game_json[n_discs_str][GAME_NEXT_POLICY].get<int>();
                    } else {
                        history_elem.next_policy = -1;
                    }
                } else {
                    history_elem.next_policy = -1;
                }
                history.emplace_back(history_elem);
            } else {
                break;
            }
        }
        
        // Save updated JSON only (do not call save_game_to_file to avoid CSV duplication)
        JSON updated_json;
        updated_json[U"black_player_name"] = getData().game_information.black_player_name;
        updated_json[U"white_player_name"] = getData().game_information.white_player_name;
        updated_json[U"memo"] = getData().game_information.memo;
        
        // Add history data
        for (const History_elem& elem : history) {
            int n_discs = elem.board.n_discs();
            String n_discs_str = Format(n_discs);
            updated_json[n_discs_str][GAME_BOARD_PLAYER] = elem.board.player;
            updated_json[n_discs_str][GAME_BOARD_OPPONENT] = elem.board.opponent;
            updated_json[n_discs_str][GAME_PLAYER] = elem.player;
            updated_json[n_discs_str][GAME_VALUE] = elem.v;
            updated_json[n_discs_str][GAME_LEVEL] = elem.level;
            updated_json[n_discs_str][GAME_POLICY] = elem.policy;
            if (n_discs < HW2) {
                updated_json[n_discs_str][GAME_NEXT_POLICY] = elem.next_policy;
            }
        }
        
        if (!updated_json.save(json_path)) {
            std::cerr << "Failed to save updated game JSON: " << json_path.narrow() << std::endl;
            return;
        }
        
        // Update CSV only (do not append)
        int black_discs = GAME_DISCS_UNDEFINED;
        int white_discs = GAME_DISCS_UNDEFINED;
        Board last_board = history.back().board;
        if (last_board.is_end()) {
            if (history.back().player == BLACK) {
                black_discs = last_board.count_player();
                white_discs = last_board.count_opponent();
            } else {
                black_discs = last_board.count_opponent();
                white_discs = last_board.count_player();
            }
        }
        
        game_save_helper::update_game_in_csv(
            base_dir + U"summary.csv",
            existing_game_date,
            getData().game_information.black_player_name,
            getData().game_information.white_player_name,
            getData().game_information.memo,
            black_discs,
            white_discs
        );
        
        std::cerr << "Game edited and saved: " << existing_game_date.narrow() << std::endl;
    }
};
