/*
    Egaroucid Project

    @file game_editor_scene.hpp
        Game editor scene
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

// Game save helper functions
namespace game_save_helper {
    // Save game to JSON file and update CSV summary
    inline void save_game_to_file(
        const String& base_dir,
        const String& filename_date,
        const String& black_player_name,
        const String& white_player_name,
        const String& memo,
        const std::vector<History_elem>& history,
        const String& game_date = U""
    ) {
        JSON json;
        json[GAME_DATE] = filename_date;
        json[GAME_BLACK_PLAYER] = black_player_name;
        json[GAME_WHITE_PLAYER] = white_player_name;
        json[GAME_MEMO] = memo;
        json[U"date"] = game_date.isEmpty() ? filename_date.substr(0, 10).replaced(U"_", U"-") : game_date;
        
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
        const String save_path = base_dir + filename_date + U".json";
        json.save(save_path);

        const String csv_path = base_dir + U"summary.csv";
        CSV csv{ csv_path };
        String memo_summary_all = memo.replaced(U"\r", U"").replaced(U"\n", U" ");
        String memo_summary;
        for (int i = 0; i < std::min((int)memo_summary_all.size(), GAME_MEMO_SUMMARY_SIZE); ++i) {
            memo_summary += memo_summary_all[i];
        }
        String date_for_csv = game_date.isEmpty() ? filename_date.substr(0, 10).replaced(U"_", U"-") : game_date;
        csv.writeRow(filename_date, black_player_name, white_player_name, memo_summary, black_discs, white_discs, date_for_csv);
        csv.save(csv_path);
    }

    // Update existing game in CSV (for editing existing games)
    inline void update_game_in_csv(
        const String& csv_path,
        const String& filename_date,
        const String& black_player_name,
        const String& white_player_name,
        const String& memo,
        int black_discs,
        int white_discs,
        const String& game_date
    ) {
        CSV csv{ csv_path };
        CSV new_csv;
        
        bool found = false;
        for (size_t i = 0; i < csv.rows(); ++i) {
            if (csv[i].size() >= 1) {
                if (csv[i][0] == filename_date) {
                    // Update this row
                    String memo_summary_all = memo.replaced(U"\r", U"").replaced(U"\n", U" ");
                    String memo_summary;
                    for (int j = 0; j < std::min((int)memo_summary_all.size(), GAME_MEMO_SUMMARY_SIZE); ++j) {
                        memo_summary += memo_summary_all[j];
                    }
                    new_csv.write(filename_date);
                    new_csv.write(black_player_name);
                    new_csv.write(white_player_name);
                    new_csv.write(memo_summary);
                    new_csv.write(black_discs == GAME_DISCS_UNDEFINED ? U"" : ToString(black_discs));
                    new_csv.write(white_discs == GAME_DISCS_UNDEFINED ? U"" : ToString(white_discs));
                    new_csv.write(game_date);
                    new_csv.newLine();
                    found = true;
                } else {
                    // Keep existing row - copy up to 6 columns, and add date column if missing
                    size_t cols = std::min(csv[i].size(), size_t(6));
                    for (size_t j = 0; j < cols; ++j) {
                        new_csv.write(csv[i][j]);
                    }
                    // If 7th column (date) doesn't exist, generate from filename
                    if (csv[i].size() < 7) {
                        String old_date = csv[i][0].substr(0, 10).replaced(U"_", U"-");
                        new_csv.write(old_date);
                    } else {
                        new_csv.write(csv[i][6]);
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
    Button export_main_button;     // For new game save: save main line
    TextAreaEditState text_area[4]; // black player, white player, memo, date
    static constexpr int BLACK_PLAYER_IDX = 0;
    static constexpr int WHITE_PLAYER_IDX = 1;
    static constexpr int MEMO_IDX = 2;
    static constexpr int DATE_IDX = 3;
    
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
        export_main_button.init(BUTTON2_2_SX, BUTTON2_SY, BUTTON2_WIDTH, BUTTON2_HEIGHT, BUTTON2_RADIUS, language.get("in_out", "export_main"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        
        text_area[BLACK_PLAYER_IDX].active = true;
        text_area[BLACK_PLAYER_IDX].text = getData().game_information.black_player_name;
        text_area[WHITE_PLAYER_IDX].text = getData().game_information.white_player_name;
        text_area[MEMO_IDX].text = getData().game_information.memo;
        
        // Initialize date field: use existing date or current date in YYYY-MM-DD format
        if (getData().game_information.date.isEmpty()) {
            const DateTime now = DateTime::Now();
            text_area[DATE_IDX].text = Format(now.year, U"-", Pad(now.month, { 2, U'0' }), U"-", Pad(now.day, { 2, U'0' }));
        } else {
            text_area[DATE_IDX].text = getData().game_information.date;
        }
        
        for (int i = 0; i < 4; ++i) {
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
        
        // Date label / textbox (below player names)
        const int date_box_y = 47;
        getData().fonts.font(language.get("in_out", "date") + U": ").draw(15, Arg::rightCenter(X_CENTER, date_box_y + EXPORT_GAME_DATE_HEIGHT / 2), getData().colors.white);
        SimpleGUI::TextArea(text_area[DATE_IDX], Vec2{X_CENTER, date_box_y}, SizeF{EXPORT_GAME_DATE_WIDTH, EXPORT_GAME_DATE_HEIGHT}, 30);

        // Player name label / textboxes
        const int player_label_y = 85;
        const int player_box_y = 108;
        getData().fonts.font(language.get("in_out", "player_name")).draw(15, Arg::topCenter(X_CENTER, player_label_y), getData().colors.white);
        SimpleGUI::TextArea(text_area[BLACK_PLAYER_IDX], Vec2{X_CENTER - EXPORT_GAME_PLAYER_WIDTH, player_box_y}, SizeF{EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT}, SimpleGUI::PreferredTextAreaMaxChars);
        SimpleGUI::TextArea(text_area[WHITE_PLAYER_IDX], Vec2{X_CENTER, player_box_y}, SizeF{EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT}, SimpleGUI::PreferredTextAreaMaxChars);
        Circle(X_CENTER - EXPORT_GAME_PLAYER_WIDTH - EXPORT_GAME_RADIUS - 20, player_box_y + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.black);
        Circle(X_CENTER + EXPORT_GAME_PLAYER_WIDTH + EXPORT_GAME_RADIUS + 20, player_box_y + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.white);
        
        // Memo label / counter / textbox
        const int memo_label_y = 143;
        const int memo_box_y = 163;
        getData().fonts.font(language.get("in_out", "memo")).draw(15, Arg::topCenter(X_CENTER, memo_label_y), getData().colors.white);
        getData().fonts.font(Format(text_area[MEMO_IDX].text.size()) + U"/" + Format(TEXTBOX_MAX_CHARS) + U" " + language.get("common", "characters")).draw(15, Arg::topRight(X_CENTER + EXPORT_GAME_MEMO_WIDTH / 2, memo_label_y), getData().colors.white);
        SimpleGUI::TextArea(text_area[MEMO_IDX], Vec2{X_CENTER - EXPORT_GAME_MEMO_WIDTH / 2, memo_box_y}, SizeF{EXPORT_GAME_MEMO_WIDTH, EXPORT_GAME_MEMO_HEIGHT}, TEXTBOX_MAX_CHARS);
        
        // Tab移動: black -> white -> date -> memo -> black
        auto focus_next_from = [&](int idx) {
            text_area[idx].active = false;
            text_area[(idx + 1) % 4].active = true;
        };
        for (int i = 0; i < 4; ++i) {
            std::string str = text_area[i].text.narrow();
            if (str.find('\t') != std::string::npos) {
                text_area[i].text.replace(U"\t", U"");
                text_area[i].cursorPos = text_area[i].text.size();
                text_area[i].rebuildGlyphs();
                focus_next_from(i);
            }
            // Remove newlines from all fields except memo
            if ((str.find('\n') != std::string::npos || str.find('\r') != std::string::npos) && i != MEMO_IDX) {
                text_area[i].text.replace(U"\r", U"").replace(U"\n", U" ");
                text_area[i].cursorPos = text_area[i].text.size();
                text_area[i].rebuildGlyphs();
            }
        }
        
        getData().game_information.black_player_name = text_area[BLACK_PLAYER_IDX].text;
        getData().game_information.white_player_name = text_area[WHITE_PLAYER_IDX].text;
        getData().game_information.memo = text_area[MEMO_IDX].text;
        getData().game_information.date = text_area[DATE_IDX].text;
        
        
        if (is_editing_mode) {
            back_button.draw();
            // Editing existing game: show OK button
            ok_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                // Clear game information only when not returning to Game_information_scene
                if (return_scene != U"Game_information_scene") {
                    getData().game_information.init();
                }
                changeScene(return_scene, SCENE_FADE_TIME);
            }
        } else {
            // New game save: show two export buttons
            back_button.draw();
            export_main_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                if (return_scene == U"Game_information_scene") {
                    // Returning to Game_information_scene
                    changeScene(return_scene, SCENE_FADE_TIME);
                } else {
                    // Returning from Export_game: go to Main_scene
                    getData().graph_resources.need_init = false;
                    changeScene(U"Main_scene", SCENE_FADE_TIME);
                }
            }
        }
        
        if (is_editing_mode) {
            // Only allow Enter to submit if no text field is active
            bool can_submit_with_enter = KeyEnter.pressed() && 
                !text_area[BLACK_PLAYER_IDX].active && 
                !text_area[WHITE_PLAYER_IDX].active && 
                !text_area[DATE_IDX].active && 
                !text_area[MEMO_IDX].active;
            
            if (ok_button.clicked() || can_submit_with_enter) {
                // Update existing game
                save_edited_game();
                changeScene(return_scene, SCENE_FADE_TIME);
            }
        } else {
            // New game save mode
            if (export_main_button.clicked()) {
                // Prepare history for save location picker
                if (getData().game_editor_info.export_mode == 0) {
                    // Main line
                    getData().save_location_picker_info.pending_history = getData().graph_resources.nodes[0];
                } else {
                    // Until this board
                    std::vector<History_elem> history;
                    int inspect_switch_n_discs = INF;
                    if (getData().graph_resources.branch == 1) {
                        if (getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
                            inspect_switch_n_discs = getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs();
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
                    getData().save_location_picker_info.pending_history.swap(history);
                }
                // Transition to Save_location_picker scene
                getData().game_editor_info.return_scene = U"Game_editor";
                changeScene(U"Save_location_picker", SCENE_FADE_TIME);
            }
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
        
        // Update only player names, memo, and date in the existing JSON
        game_json[GAME_BLACK_PLAYER] = getData().game_information.black_player_name;
        game_json[GAME_WHITE_PLAYER] = getData().game_information.white_player_name;
        game_json[GAME_MEMO] = getData().game_information.memo;
        game_json[U"date"] = getData().game_information.date;
        
        // Keep all other fields (history, scores, etc.) unchanged
        
        if (!game_json.save(json_path)) {
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
            white_discs,
            getData().game_information.date
        );
        
        std::cerr << "Game edited and saved: " << existing_game_date.narrow() << std::endl;
    }
};