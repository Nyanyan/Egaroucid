/*
    Egaroucid Project

    @file input.hpp
        Input scenes
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <ctime>
#include <unordered_set>
#ifdef _WIN32
#include <Windows.h>
#endif
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"
#include "draw.hpp"

// Forward declaration
inline bool load_game_from_json(
    Common_resources& data,
    Opening& opening,
    const String& json_path,
    const String& game_date,
    const std::string& subfolder
);

class Import_text : public App::Scene {
private:
    Button back_button;
    Button import_button;
    bool done;
    std::string text;
    std::vector<History_elem> n_history;
    std::vector<History_elem> history_until_now;
    bool imported_from_position;
    TextAreaEditState text_area;
    int text_input_format;
    Game_import_t imported_game;

public:
    Import_text(const InitData& init) : IScene{ init } {
        set_scene_ime_enabled(true);
        back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        import_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "import"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);done = false;
        import_button.disable();
        imported_from_position = false;
        text_input_format = TEXT_INPUT_FORMAT_NONE;
        text.clear();
        for (History_elem history_elem : getData().graph_resources.nodes[getData().graph_resources.branch]) {
            if (getData().history_elem.board.n_discs() >= history_elem.board.n_discs()) {
                history_until_now.emplace_back(history_elem);
            }
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
        int sy = 20 + icon_width + 30;
        getData().fonts.font(language.get("in_out", "input_text")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
        text_area.active = true;
        bool text_changed = text_area_with_ime_candidate_window(text_area, Vec2{X_CENTER - 300, sy + 40}, SizeF{600, 150}, INPUT_STR_MAX_SIZE);
        getData().fonts.font(language.get("in_out", "you_can_paste_with_ctrl_v") + U" / " + language.get("in_out", "you_can_drag_and_drop_game_file")).draw(13, Arg::topCenter(X_CENTER, sy + 195), getData().colors.white);
        bool return_pressed = false;
        if (text_area.text.size()) {
            if (text_area.text[text_area.text.size() - 1] == '\n') {
                return_pressed = true;
            }
        }
        if (DragDrop::HasNewFilePaths()) {
            std::string path = DragDrop::GetDroppedFilePaths()[0].path.narrow();
            std::ifstream ifs(path);
            if (ifs.fail()) {
                std::cerr << "can't open " << path << std::endl;
            } else {
                std::istreambuf_iterator<char> it(ifs);
                std::istreambuf_iterator<char> last;
                std::string str(it, last);
                text_area.text = Unicode::Widen(str);
                text_area.cursorPos = text_area.text.size();
                text_area.rebuildGlyphs();
                text_changed = true;
            }
        }
        text = text_area.text.replaced(U"\r", U"").replaced(U"\n", U"").replaced(U" ", U"").replace(U"\t", U"").narrow();
        if (text_changed) {
            bool failed;
            imported_game = import_any_format_processing(text, history_until_now, getData().history_elem, &failed);
        }
        String format_str = language.get("in_out", "format", "error");
        if (text.size() == 0) {
            format_str = language.get("in_out", "format", "egaroucid_recognizes_format");
        } else if (imported_game.format == TEXT_INPUT_FORMAT_GGF) {
            format_str = language.get("in_out", "format", "GGF");
        } else if (imported_game.format == TEXT_INPUT_FORMAT_OTHELLO_QUEST) {
            format_str = language.get("in_out", "format", "othello_quest");
        } else if (imported_game.format == TEXT_INPUT_FORMAT_TRANSCRIPT) {
            format_str = language.get("in_out", "format", "transcript");
        } else if (imported_game.format == TEXT_INPUT_FORMAT_TRANSCRIPT_FROM_THIS_POSITION) {
            format_str = language.get("in_out", "format", "transcript_from_this_position");
        } else if (imported_game.format == TEXT_INPUT_FORMAT_BOARD) {
            format_str = language.get("in_out", "format", "board");
        } else if (imported_game.format == TEXT_INPUT_FORMAT_GENERAL_BOARD_TRANSCRIPT) {
            format_str = language.get("in_out", "format", "board_transcript");
        }
        getData().fonts.font(format_str).draw(13, Arg::topCenter(X_CENTER, sy + 210), getData().colors.white);
        back_button.draw();
        import_button.draw();
        if (back_button.clicked() || gui_textarea_ime::escape_pressed_for_scene_change()) {
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        if (imported_game.format != TEXT_INPUT_FORMAT_NONE && text.size()) {
            import_button.enable();
            if (import_button.clicked() || KeyEnter.pressed()) {
                n_history = imported_game.history;
                go_to_main_scene();
            }
        } else {
            import_button.disable();
        }
    }

    void draw() const override {

    }

    void go_to_main_scene() {
        if (!imported_from_position) {
            getData().graph_resources.init();
            getData().graph_resources.nodes[0] = n_history;
            getData().graph_resources.n_discs = getData().graph_resources.nodes[0].back().board.n_discs();
            getData().game_information.init();
        } else {
            getData().graph_resources.nodes[getData().graph_resources.branch] = n_history;
        }
        std::string opening_name, n_opening_name;
        for (int i = 0; i < (int)getData().graph_resources.nodes[getData().graph_resources.branch].size(); ++i) {
            n_opening_name.clear();
            n_opening_name = opening.get(getData().graph_resources.nodes[getData().graph_resources.branch][i].board, getData().graph_resources.nodes[getData().graph_resources.branch][i].player ^ 1);
            if (n_opening_name.size()) {
                opening_name = n_opening_name;
            }
            getData().graph_resources.nodes[getData().graph_resources.branch][i].opening_name = opening_name;
        }
        update_xot_start_n_discs(&getData().graph_resources);
        getData().graph_resources.n_discs = getData().graph_resources.nodes[getData().graph_resources.branch].back().board.n_discs();
        getData().graph_resources.need_init = false;
        getData().history_elem = getData().graph_resources.nodes[getData().graph_resources.branch].back();
        changeScene(U"Main_scene", SCENE_FADE_TIME);
    }
};






class Import_othello_quest : public App::Scene {
private:
    static constexpr int GAMES_PER_PAGE = 5;
    static constexpr int OQ_TABLE_X = 24;
    static constexpr int OQ_TABLE_Y = 112;
    static constexpr int OQ_TABLE_W = WINDOW_SIZE_X - OQ_TABLE_X * 2;
    static constexpr int OQ_TABLE_HEADER_H = 24;
    static constexpr int OQ_TABLE_ROW_H = 55;
    static constexpr int OQ_DATE_W = 128;
    static constexpr int OQ_PLAYER_W = 225;
    static constexpr int OQ_SCORE_W = 66;
    static constexpr int OQ_PLAYER_H = 24;
    static constexpr const char* OQ_BASE_URL = "http://questgames.net";

    enum class Detail_state {
        None,
        Loading,
        Ready,
        Failed
    };

    struct Oq_game {
        String id;
        String created;
        String black_name;
        String white_name;
        String black_id;
        String white_id;
        String black_rating;
        String white_rating;
        String black_rank;
        String white_rank;
        String list_status;
        int length = 0;
        Detail_state detail_state = Detail_state::None;
        String status_raw;
        String result_label;
        String score_label;
        String error;
        int black_score = GAME_DISCS_UNDEFINED;
        int white_score = GAME_DISCS_UNDEFINED;
        bool friend_match = false;
        std::string transcript;
        std::vector<History_elem> history;
    };

    struct Detail_task {
        int game_idx = -1;
        AsyncHTTPTask task;
    };

    struct Oq_cache {
        bool has_result = false;
        std::string username;
        int mode = 0;
        std::vector<Oq_game> games;
        int current_page = 0;
        int selected_idx = -1;
        std::unordered_set<int> selected_indices;
        int selection_anchor_idx = -1;
        int next_prefetch_idx = 0;
        String status_message;
    };

    inline static std::array<Oq_cache, 3> oq_caches;

    Button back_button;
    Button save_button;
    Button search_button;
    Button select_all_button;
    Button refresh_button;
    TextEditState username_area;
    Radio_button mode_radio;
    AsyncHTTPTask list_task;
    std::vector<Detail_task> detail_tasks;
    std::vector<Oq_game> games;
    int current_page;
    int selected_idx;
    std::unordered_set<int> selected_indices;
    int selection_anchor_idx;
    int next_prefetch_idx;
    int last_clicked_idx;
    uint64_t last_click_time;
    bool list_loading;
    bool user_status_message;
    String status_message;
    std::string searched_username;
    int searched_mode;

public:
    Import_othello_quest(const InitData& init) : IScene{ init } {
        set_scene_ime_enabled(true);
        back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        save_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "save_to_game_library"), 23, getData().fonts.font, getData().colors.white, getData().colors.black);
        save_button.disable();
        search_button.init(X_CENTER + 245, 70, 105, 34, 10, U"Search", 17, getData().fonts.font, getData().colors.white, getData().colors.black);
        select_all_button.init(22, 18, 120, 30, 8, U"Select all", 14, getData().fonts.font, getData().colors.white, getData().colors.black);
        refresh_button.init(156, 18, 100, 30, 8, U"Refresh", 14, getData().fonts.font, getData().colors.white, getData().colors.black);

        username_area.active = true;
        current_page = 0;
        selected_idx = -1;
        selected_indices.clear();
        selection_anchor_idx = -1;
        next_prefetch_idx = 0;
        last_clicked_idx = -1;
        last_click_time = 0;
        list_loading = false;
        user_status_message = false;
        searched_mode = getData().user_settings.othello_quest_mode;
        status_message.clear();

        Radio_button_element radio_button_elem;
        mode_radio.init();
        radio_button_elem.init(X_CENTER - 125, 132, getData().fonts.font, 17, U"5 min", true);
        mode_radio.push(radio_button_elem);
        radio_button_elem.init(X_CENTER - 25, 132, getData().fonts.font, 17, U"1 min", false);
        mode_radio.push(radio_button_elem);
        radio_button_elem.init(X_CENTER + 75, 132, getData().fonts.font, 17, U"XOT", false);
        mode_radio.push(radio_button_elem);
        mode_radio.checked = std::clamp(searched_mode, 0, 2);

        if (!getData().user_settings.othello_quest_username.empty()) {
            username_area.text = Unicode::Widen(getData().user_settings.othello_quest_username);
            username_area.cursorPos = username_area.text.size();
            if (!restore_cached_result(getData().user_settings.othello_quest_username, mode_radio.checked)) {
                start_search();
            }
        }
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }

        update_http_tasks();
        consume_game_library_save_result();

        Scene::SetBackground(getData().colors.green);
        const bool result_page = !searched_username.empty();
        const int label_x = X_CENTER - 158;
        const int input_x = X_CENTER - 132;
        const int input_w = 335;
        const int username_y = 178;
        const int mode_y = result_page ? 18 : 250;
        if (!result_page) {
            getData().fonts.font(language.get("in_out", "input_othello_quest")).draw(27, Arg::topCenter(X_CENTER, 82), getData().colors.white);
            getData().fonts.font(language.get("in_out", "othello_quest_username")).draw(20, Arg::rightCenter(label_x, username_y + 22), getData().colors.white);
            text_box_with_ime_candidate_window(username_area, Vec2{input_x, username_y + 5}, input_w, 40, true, false);
            search_button.rect.x = input_x + input_w + 18;
            search_button.rect.y = username_y + 2;
            search_button.rect.w = 112;
            search_button.rect.h = 42;
            search_button.draw();
            if (search_button.clicked() || username_area.enterKey) {
                start_search();
            }
        }
        const int previous_mode = mode_radio.checked;
        draw_mode_buttons(result_page, mode_y);
        if (result_page && previous_mode != mode_radio.checked) {
            start_search();
        }

        draw_games(result_page);
        draw_result_buttons(result_page);
        handle_result_list_shortcuts(result_page);

        if (!result_page) {
            back_button.move((WINDOW_SIZE_X - GO_BACK_BUTTON_WIDTH) / 2, GO_BACK_BUTTON_SY);
        } else {
            back_button.move(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY);
        }
        back_button.draw();
        if (result_page) {
            if (can_save_any_selected()) {
                save_button.enable();
            } else {
                save_button.disable();
            }
            save_button.draw();
        }
        if (back_button.clicked() || gui_textarea_ime::escape_pressed_for_scene_change()) {
            write_cache();
            if (result_page) {
                return_to_search_form();
            } else {
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        }
        if (result_page && save_button.clicked()) {
            save_selected_game_to_library();
        }
    }

    void draw() const override {

    }

private:
    void set_system_status(const String& message) {
        if (!user_status_message) {
            status_message = message;
        }
    }

    void set_user_status(const String& message) {
        user_status_message = true;
        status_message = message;
    }

    void consume_game_library_save_result() {
        String& result_message = getData().game_library_save_request_info.result_message;
        if (result_message.empty()) {
            return;
        }
        set_user_status(result_message);
        result_message.clear();
        write_cache();
    }

    void start_search(bool force_refresh = false) {
        String account_text = remove_oq_account_whitespace(username_area.text);
        if (account_text.empty()) {
            set_user_status(U"Input an Othello Quest user name.");
            return;
        }

        const String invalid_char = first_invalid_oq_account_char(account_text);
        if (!invalid_char.empty()) {
            set_user_status(U"Invalid character \"" + invalid_char + U"\". Please correct it.");
            return;
        }

        if (username_area.text != account_text) {
            username_area.text = account_text;
            username_area.cursorPos = username_area.text.size();
        }

        searched_username = account_text.narrow();
        for (char& c : searched_username) {
            c = (char)std::tolower((unsigned char)c);
        }
        searched_mode = mode_radio.checked;
        getData().user_settings.othello_quest_username = searched_username;
        getData().user_settings.othello_quest_mode = searched_mode;
        user_status_message = false;
        if (force_refresh) {
            clear_cached_result(searched_username, searched_mode);
        } else if (restore_cached_result(searched_username, searched_mode)) {
            return;
        }
        games.clear();
        detail_tasks.clear();
        current_page = 0;
        selected_idx = -1;
        selected_indices.clear();
        selection_anchor_idx = -1;
        last_clicked_idx = -1;
        last_click_time = 0;
        next_prefetch_idx = 0;
        list_loading = true;
        set_system_status(force_refresh ? U"Refreshing game list..." : U"Loading game list...");
        list_task = SimpleHTTP::GetAsync(Unicode::Widen(build_list_url()), oq_headers());
    }

    void update_http_tasks() {
        if (list_loading && list_task && list_task.isReady()) {
            list_loading = false;
            if (!list_task.getResponse().isOK()) {
                set_user_status(U"Failed to load game list.");
            } else {
                JSON json = list_task.getAsJSON();
                if (!parse_game_list(json)) {
                    set_user_status(U"Failed to parse game list.");
                } else if (games.empty()) {
                    set_user_status(U"No games found.");
                } else {
                    set_system_status(U"Loading details 1-" + Format(std::min(GAMES_PER_PAGE, (int)games.size())) + U"...");
                    start_next_detail_batch();
                }
            }
            write_cache();
        }

        for (int i = 0; i < (int)detail_tasks.size();) {
            Detail_task& detail_task = detail_tasks[i];
            if (!detail_task.task.isReady()) {
                ++i;
                continue;
            }
            const int game_idx = detail_task.game_idx;
            if (0 <= game_idx && game_idx < (int)games.size()) {
                if (!detail_task.task.getResponse().isOK()) {
                    games[game_idx].detail_state = Detail_state::Failed;
                    games[game_idx].error = U"HTTP " + Format(detail_task.task.getResponse().getStatusCodeInt());
                } else {
                    JSON detail_json = detail_task.task.getAsJSON();
                    parse_game_detail(game_idx, detail_json);
                }
                write_cache();
            }
            detail_tasks.erase(detail_tasks.begin() + i);
        }

        if (!list_loading && detail_tasks.empty()) {
            if (next_prefetch_idx < (int)games.size()) {
                start_next_detail_batch();
            } else if (!games.empty()) {
                if (!user_status_message) {
                    status_message = U"All details loaded.";
                }
                write_cache();
            }
        }
    }

    void start_next_detail_batch() {
        if (next_prefetch_idx >= (int)games.size()) {
            set_system_status(U"All details loaded.");
            return;
        }
        const int batch_start = next_prefetch_idx;
        const int batch_end = std::min(batch_start + GAMES_PER_PAGE, (int)games.size());
        for (int idx = batch_start; idx < batch_end; ++idx) {
            if (games[idx].id.empty() || games[idx].detail_state != Detail_state::None) {
                continue;
            }
            games[idx].detail_state = Detail_state::Loading;
            Detail_task detail_task;
            detail_task.game_idx = idx;
            detail_task.task = SimpleHTTP::GetAsync(Unicode::Widen(build_detail_url(games[idx].id.narrow())), oq_headers());
            detail_tasks.emplace_back(std::move(detail_task));
        }
        next_prefetch_idx = batch_end;
        set_system_status(U"Loading details " + Format(batch_start + 1) + U"-" + Format(batch_end) + U"...");
        write_cache();
    }

    bool parse_game_list(JSON json) {
        JSON game_array = json[U"games"];
        if (!game_array.isArray() && json.isArray()) {
            game_array = json;
        }
        if (!game_array.isArray()) {
            return false;
        }
        for (size_t i = 0; i < game_array.size(); ++i) {
            JSON item = game_array[i];
            if (!item.isObject()) {
                continue;
            }
            Oq_game game;
            game.id = json_string(item[U"id"]);
            game.created = json_string(item[U"created"]);
            game.list_status = json_string(item[U"finalStatus"]);
            game.length = json_int(item[U"length"]);
            JSON players = item[U"players"];
            if (players.isArray() && players.size() >= 2) {
                fill_player_summary(players[0], &game.black_name, &game.black_rating, &game.black_rank, &game.black_id);
                fill_player_summary(players[1], &game.white_name, &game.white_rating, &game.white_rank, &game.white_id);
                game.friend_match = both_ratings_unchanged(players[0], players[1]);
            }
            if (!game.id.empty()) {
                games.emplace_back(game);
            }
        }
        return true;
    }

    void parse_game_detail(int game_idx, JSON detail_json) {
        if (game_idx < 0 || game_idx >= (int)games.size() || !detail_json.isObject()) {
            return;
        }
        Oq_game& game = games[game_idx];
        JSON players = detail_json[U"players"];
        if (players.isArray() && players.size() >= 2) {
            fill_player_summary(players[0], &game.black_name, &game.black_rating, &game.black_rank, &game.black_id);
            fill_player_summary(players[1], &game.white_name, &game.white_rating, &game.white_rank, &game.white_id);
            game.friend_match = both_ratings_unchanged(players[0], players[1]);
        }

        std::string transcript = extract_start_pos(detail_json[U"position"][U"startPos"]);
        String terminal_status;
        JSON moves = detail_json[U"position"][U"moves"];
        if (moves.isArray()) {
            for (size_t i = 0; i < moves.size(); ++i) {
                JSON move = moves[i];
                const String move_text = json_string(move[U"m"]);
                if (is_coord(move_text)) {
                    transcript += normalized_coord(move_text);
                }
                const String status = json_string(move[U"s"]);
                if (!status.empty()) {
                    terminal_status = status;
                }
            }
        }
        if (terminal_status.empty()) {
            terminal_status = game.list_status;
        }
        game.status_raw = terminal_status;
        game.transcript = transcript;

        History_elem start_history_elem;
        start_history_elem.reset();
        std::vector<History_elem> history;
        history.emplace_back(start_history_elem);
        bool failed = false;
        history = import_transcript_processing(history, start_history_elem, transcript, &failed);
        if (failed || history.empty()) {
            game.detail_state = Detail_state::Failed;
            game.error = U"Transcript replay failed.";
            return;
        }
        game.history = history;
        game.detail_state = Detail_state::Ready;
        fill_result_from_history(&game);
    }

    void fill_result_from_history(Oq_game* game) {
        const String raw_upper = game->status_raw.uppercased();
        String ending = U"Score";
        if (raw_upper.includes(U"RESIGN")) {
            ending = U"Resign";
        } else if (raw_upper.includes(U"TIMEUP") || raw_upper.includes(U"TIMEOUT")) {
            ending = U"Timeout";
        } else if (raw_upper.includes(U"DISCONNECT")) {
            ending = U"Disconnect";
        } else if (raw_upper.substr(0, 5) != U"SCORE") {
            ending = U"Unknown";
        }

        const History_elem& last = game->history.back();
        int black_count = last.player == BLACK ? last.board.count_player() : last.board.count_opponent();
        int white_count = last.player == WHITE ? last.board.count_player() : last.board.count_opponent();
        int black_score = black_count;
        int white_score = white_count;

        if (ending != U"Score") {
            const int terminal_player = last.player;
            int loser = -1;
            if (raw_upper.substr(0, 5) == U"LOSE:") {
                loser = terminal_player;
            } else if (raw_upper.substr(0, 4) == U"WIN:") {
                loser = terminal_player ^ 1;
            }
            if (loser == BLACK) {
                black_score = 0;
                white_score = HW2;
            } else if (loser == WHITE) {
                black_score = HW2;
                white_score = 0;
            }
            game->black_score = black_score;
            game->white_score = white_score;
            game->score_label = Format(black_score) + U"-" + Format(white_score);
            game->result_label = ending + U" (" + game->status_raw + U")";
            return;
        }

        if (black_count > white_count) {
            black_score = HW2 - white_count;
        } else if (white_count > black_count) {
            white_score = HW2 - black_count;
        } else if (black_count + white_count < HW2) {
            black_score = HW2 / 2;
            white_score = HW2 / 2;
        }
        game->black_score = black_score;
        game->white_score = white_score;
        game->score_label = Format(black_score) + U"-" + Format(white_score);
        game->result_label = U"Score (" + game->status_raw + U")";
    }

    void draw_games(bool result_page) {
        const int table_x = OQ_TABLE_X;
        const int table_y = result_page ? 74 : OQ_TABLE_Y;
        const int table_w = OQ_TABLE_W;
        const int row_h = OQ_TABLE_ROW_H;
        const int header_h = OQ_TABLE_HEADER_H;
        auto fit_text = [&](String text, int font_size, double width) {
            if (getData().fonts.font(text).region(font_size, 0, 0).w <= width) {
                return text;
            }
            while (text.size() > 3 && getData().fonts.font(text + U"...").region(font_size, 0, 0).w > width) {
                text.pop_back();
            }
            return text + U"...";
        };
        const int n_pages = games.empty() ? 1 : ((int)games.size() + GAMES_PER_PAGE - 1) / GAMES_PER_PAGE;
        current_page = std::clamp(current_page, 0, n_pages - 1);

        if (games.empty()) {
            if (!status_message.empty()) {
                getData().fonts.font(status_message).draw(15, Arg::topCenter(X_CENTER, result_page ? 118 : 320), getData().colors.white);
            }
            return;
        }

        const Rect header_rect(table_x, table_y, table_w, header_h);
        header_rect.draw(getData().colors.dark_green).drawFrame(1.0, getData().colors.white);

        const Rect prev_rect(table_x + 2, table_y + 2, 34, header_h - 4);
        const Rect next_rect(table_x + table_w - 36, table_y + 2, 34, header_h - 4);
        const bool can_prev = current_page > 0;
        const bool can_next = current_page + 1 < n_pages;
        draw_page_arrow(prev_rect, U"◀", can_prev);
        draw_page_arrow(next_rect, U"▶", can_next);
        if (can_prev && prev_rect.leftClicked()) {
            --current_page;
            write_cache();
        }
        if (can_next && next_rect.leftClicked()) {
            ++current_page;
            write_cache();
        }

        getData().fonts.font(U"Page " + Format(current_page + 1) + U" / " + Format(n_pages) + U"   " + Format(games.size()) + U" games").draw(12, Arg::center(X_CENTER, table_y + header_h / 2), getData().colors.white);
        getData().fonts.font(status_message).draw(11, table_x, table_y - 17, ColorF(getData().colors.white, 0.85));

        const int start_idx = current_page * GAMES_PER_PAGE;
        for (int row = 0; row < GAMES_PER_PAGE; ++row) {
            const int idx = start_idx + row;
            const int y = table_y + header_h + row * row_h;
            Rect row_rect(table_x, y, table_w, row_h);
            Color row_color = row % 2 ? getData().colors.dark_green : getData().colors.green;
            row_rect.draw(row_color).drawFrame(1.0, getData().colors.white);
            if (idx >= (int)games.size()) {
                continue;
            }
            if (selected_indices.count(idx)) {
                row_rect.draw(ColorF(getData().colors.white, 0.18)).drawFrame(2, Palette::Cyan);
            }
            if (row_rect.mouseOver()) {
                Cursor::RequestStyle(CursorStyle::Hand);
                row_rect.drawFrame(2, Palette::Cyan);
            }
            if (row_rect.leftClicked()) {
                const uint64_t now = tim();
                if (last_clicked_idx == idx && now - last_click_time < 500) {
                    selected_indices.clear();
                    selected_idx = idx;
                    selected_indices.insert(idx);
                    selection_anchor_idx = idx;
                    write_cache();
                    import_selected_game();
                    return;
                }
                handle_result_selection_click(idx, KeyControl.pressed(), KeyShift.pressed());
                write_cache();
                last_clicked_idx = idx;
                last_click_time = now;
            }
            const Oq_game& game = games[idx];
            const int date_w = OQ_DATE_W;
            const int player_w = OQ_PLAYER_W;
            const int score_w = OQ_SCORE_W;
            const int date_x = table_x + 12;
            const int black_x = table_x + date_w;
            const int score_x = black_x + player_w;
            const int white_x = score_x + score_w;
            const int upper_center_y = y + OQ_PLAYER_H / 2 + 3;

            getData().fonts.font(fit_text(display_date(game.created), 13, date_w - 18)).draw(13, date_x, y + 7, getData().colors.white);
            draw_player_box(Rect(black_x, y + 3, player_w, OQ_PLAYER_H), fit_text(player_label(game.black_name, game.black_rating, game.black_rank), 15, player_w - 6), game, BLACK, upper_center_y);
            draw_score_label(score_x, score_w, upper_center_y, game);
            draw_player_box(Rect(white_x, y + 3, player_w, OQ_PLAYER_H), fit_text(player_label(game.white_name, game.white_rating, game.white_rank), 15, player_w - 6), game, WHITE, upper_center_y);

            getData().fonts.font(fit_text(user_result_summary(game), 11, table_x + table_w - black_x - 12)).draw(11, black_x, y + 34, ColorF(getData().colors.white, 0.78));
        }
    }

    void handle_result_selection_click(int idx, bool ctrl_pressed, bool shift_pressed) {
        if (idx < 0 || idx >= (int)games.size()) {
            return;
        }

        if (shift_pressed && 0 <= selection_anchor_idx && selection_anchor_idx < (int)games.size()) {
            if (!ctrl_pressed) {
                selected_indices.clear();
            }
            const int first = std::min(selection_anchor_idx, idx);
            const int last = std::max(selection_anchor_idx, idx);
            for (int selection_idx = first; selection_idx <= last; ++selection_idx) {
                selected_indices.insert(selection_idx);
            }
            selected_idx = idx;
            return;
        }

        if (ctrl_pressed) {
            if (selected_indices.count(idx)) {
                selected_indices.erase(idx);
                if (selected_idx == idx) {
                    selected_idx = selected_indices.empty() ? -1 : *selected_indices.begin();
                }
            } else {
                selected_indices.insert(idx);
                selected_idx = idx;
            }
            selection_anchor_idx = idx;
            return;
        }

        selected_indices.clear();
        selected_indices.insert(idx);
        selected_idx = idx;
        selection_anchor_idx = idx;
    }

    void draw_page_arrow(const Rect& rect, const String& arrow, bool enabled) {
        if (enabled && rect.mouseOver()) {
            Cursor::RequestStyle(CursorStyle::Hand);
            rect.draw(ColorF(getData().colors.white, 0.16));
        }
        getData().fonts.font(arrow).draw(16, Arg::center(rect.center()), enabled ? ColorF(getData().colors.white) : ColorF(getData().colors.white, 0.35));
    }

    void draw_result_buttons(bool result_page) {
        if (!result_page || games.empty()) {
            return;
        }

        const bool all_selected = (int)selected_indices.size() == (int)games.size();
        select_all_button.str = all_selected ? U"Deselect all" : U"Select all";
        select_all_button.font_size = update_font_size_overfull(select_all_button.font, select_all_button.str, 14, select_all_button.rect.h, select_all_button.rect.w);
        select_all_button.draw();
        if (select_all_button.clicked()) {
            if (all_selected) {
                selected_indices.clear();
                selected_idx = -1;
                selection_anchor_idx = -1;
            } else {
                selected_indices.clear();
                for (int i = 0; i < (int)games.size(); ++i) {
                    selected_indices.insert(i);
                }
                selected_idx = 0;
                selection_anchor_idx = 0;
            }
            write_cache();
        }

        refresh_button.font_size = update_font_size_overfull(refresh_button.font, refresh_button.str, 14, refresh_button.rect.h, refresh_button.rect.w);
        if (list_loading) {
            refresh_button.disable();
        } else {
            refresh_button.enable();
        }
        refresh_button.draw();
        if (refresh_button.clicked()) {
            start_search(true);
        }
    }

    void handle_result_list_shortcuts(bool result_page) {
        if (!result_page || games.empty() || !KeyControl.pressed()) {
            return;
        }
        if (KeyA.down()) {
            selected_indices.clear();
            for (int i = 0; i < (int)games.size(); ++i) {
                selected_indices.insert(i);
            }
            selected_idx = 0;
            selection_anchor_idx = 0;
            write_cache();
        }
    }

    void draw_mode_buttons(bool compact, int y) {
        const Array<String> labels = { U"5 min", U"1 min", U"XOT" };
        const int button_w = compact ? 86 : 105;
        const int button_h = compact ? 30 : 34;
        const int button_gap = compact ? 8 : 10;
        const int total_w = button_w * 3 + button_gap * 2;
        const int base_x = compact ? X_CENTER - total_w / 2 : X_CENTER - 132;
        const int fs = compact ? 15 : 17;

        if (!compact) {
            getData().fonts.font(language.get("in_out", "othello_quest_mode")).draw(18, Arg::rightCenter(base_x - 26, y + button_h / 2), getData().colors.white);
        }

        for (int i = 0; i < 3; ++i) {
            s3d::RoundRect rect;
            rect.x = base_x + i * (button_w + button_gap);
            rect.y = y;
            rect.w = button_w;
            rect.h = button_h;
            rect.r = 8;
            const bool selected = mode_radio.checked == i;
            const bool hovered = rect.mouseOver();
            if (hovered) {
                Cursor::RequestStyle(CursorStyle::Hand);
            }

            if (selected) {
                rect.draw(getData().colors.white);
                getData().fonts.font(labels[i]).drawAt(fs, rect.x + rect.w / 2, rect.y + rect.h / 2, getData().colors.black);
            } else {
                rect.draw(ColorF(getData().colors.dark_green, hovered ? 0.95 : 0.78)).drawFrame(1.0, ColorF(getData().colors.white, hovered ? 0.95 : 0.65));
                getData().fonts.font(labels[i]).drawAt(fs, rect.x + rect.w / 2, rect.y + rect.h / 2, getData().colors.white);
            }

            if (rect.leftClicked()) {
                mode_radio.checked = i;
            }
        }
    }

    void return_to_search_form() {
        write_cache();
        searched_username.clear();
        searched_mode = mode_radio.checked;
        games.clear();
        detail_tasks.clear();
        current_page = 0;
        selected_idx = -1;
        selected_indices.clear();
        selection_anchor_idx = -1;
        last_clicked_idx = -1;
        last_click_time = 0;
        next_prefetch_idx = 0;
        list_loading = false;
        status_message.clear();
        username_area.active = true;
    }

    int winner_of(const Oq_game& game) const {
        if (game.black_score == GAME_DISCS_UNDEFINED || game.white_score == GAME_DISCS_UNDEFINED) {
            return -1;
        }
        if (game.black_score > game.white_score) {
            return IMPORT_GAME_WINNER_BLACK;
        }
        if (game.black_score < game.white_score) {
            return IMPORT_GAME_WINNER_WHITE;
        }
        return IMPORT_GAME_WINNER_DRAW;
    }

    void draw_player_box(const Rect& rect, const String& player, const Oq_game& game, int player_color, int center_y) {
        const int winner = winner_of(game);
        if (winner == IMPORT_GAME_WINNER_DRAW) {
            rect.draw(getData().colors.chocolate);
        } else if ((winner == IMPORT_GAME_WINNER_BLACK && player_color == BLACK) || (winner == IMPORT_GAME_WINNER_WHITE && player_color == WHITE)) {
            rect.draw(getData().colors.darkred);
        } else if ((winner == IMPORT_GAME_WINNER_BLACK && player_color == WHITE) || (winner == IMPORT_GAME_WINNER_WHITE && player_color == BLACK)) {
            rect.draw(getData().colors.darkblue);
        }
        if (player_color == BLACK) {
            getData().fonts.font(player).draw(15, Arg::rightCenter(rect.x + rect.w - 2, center_y), getData().colors.white);
        } else {
            getData().fonts.font(player).draw(15, Arg::leftCenter(rect.x + 2, center_y), getData().colors.white);
        }
    }

    void draw_score_label(int x, int width, int center_y, const Oq_game& game) {
        String black_score = U"??";
        String white_score = U"??";
        if (game.black_score != GAME_DISCS_UNDEFINED && game.white_score != GAME_DISCS_UNDEFINED) {
            black_score = Format(game.black_score);
            white_score = Format(game.white_score);
        }
        const double hyphen_w = getData().fonts.font(U"-").region(15, Vec2{0, 0}).w;
        getData().fonts.font(black_score).draw(15, Arg::rightCenter(x + width / 2 - hyphen_w / 2 - 1, center_y), getData().colors.white);
        getData().fonts.font(U"-").draw(15, Arg::center(x + width / 2, center_y), getData().colors.white);
        getData().fonts.font(white_score).draw(15, Arg::leftCenter(x + width / 2 + hyphen_w / 2 + 1, center_y), getData().colors.white);
    }

    String user_result_summary(const Oq_game& game) const {
        String user = searched_user_label(game);
        String summary;
        if (game.detail_state == Detail_state::Loading) {
            summary = user + U" loading";
        } else if (game.detail_state == Detail_state::Failed) {
            summary = user + U" failed to load";
        } else if (game.detail_state != Detail_state::Ready || game.black_score == GAME_DISCS_UNDEFINED || game.white_score == GAME_DISCS_UNDEFINED) {
            summary = user + U" queued";
        } else {
            int user_score = 0;
            int opponent_score = 0;
            if (searched_user_is_black(game)) {
                user_score = game.black_score;
                opponent_score = game.white_score;
            } else {
                user_score = game.white_score;
                opponent_score = game.black_score;
            }
            if (user_score == opponent_score) {
                summary = user + U" draw";
            } else {
                const bool user_won = user_score > opponent_score;
                const String reason = result_reason(game);
                if (reason.empty()) {
                    summary = user + (user_won ? U" won by " : U" loss by ") + Format(std::abs(user_score - opponent_score));
                } else {
                    summary = user + (user_won ? U" won by " : U" loss by ") + reason;
                }
            }
        }
        if (game.friend_match) {
            summary += U"   friend match";
        }
        return summary;
    }

    String searched_user_label(const Oq_game& game) const {
        if (searched_user_is_black(game) && !game.black_name.empty()) {
            return game.black_name;
        }
        if (searched_user_is_white(game) && !game.white_name.empty()) {
            return game.white_name;
        }
        return Unicode::Widen(searched_username);
    }

    bool searched_user_is_black(const Oq_game& game) const {
        return player_matches_search(game.black_id) || player_matches_search(game.black_name);
    }

    bool searched_user_is_white(const Oq_game& game) const {
        return player_matches_search(game.white_id) || player_matches_search(game.white_name);
    }

    bool player_matches_search(const String& value) const {
        return !value.empty() && value.lowercased().narrow() == searched_username;
    }

    static String result_reason(const Oq_game& game) {
        const String raw_upper = game.status_raw.uppercased();
        if (raw_upper.includes(U"RESIGN")) {
            return U"resign";
        }
        if (raw_upper.includes(U"TIMEUP") || raw_upper.includes(U"TIMEOUT")) {
            return U"timeout";
        }
        if (raw_upper.includes(U"DISCONNECT")) {
            return U"disconnect";
        }
        return U"";
    }

    bool can_use_selected() const {
        return 0 <= selected_idx && selected_idx < (int)games.size() && games[selected_idx].detail_state == Detail_state::Ready && !games[selected_idx].transcript.empty();
    }

    bool can_save_any_selected() const {
        for (int idx : selected_indices) {
            if (0 <= idx && idx < (int)games.size() && games[idx].detail_state == Detail_state::Ready && !games[idx].transcript.empty()) {
                return true;
            }
        }
        return false;
    }

    int selected_count() const {
        int count = 0;
        for (int idx : selected_indices) {
            if (0 <= idx && idx < (int)games.size()) {
                ++count;
            }
        }
        return count;
    }

    void import_selected_game() {
        if (!can_use_selected()) {
            return;
        }
        write_cache();
        Oq_game& game = games[selected_idx];
        apply_selected_game_information(game);
        getData().graph_resources.init();
        getData().graph_resources.nodes[GRAPH_MODE_NORMAL] = game.history;
        getData().graph_resources.n_discs = game.history.back().board.n_discs();
        std::string opening_name, n_opening_name;
        for (int i = 0; i < (int)getData().graph_resources.nodes[GRAPH_MODE_NORMAL].size(); ++i) {
            n_opening_name.clear();
            n_opening_name = opening.get(getData().graph_resources.nodes[GRAPH_MODE_NORMAL][i].board, getData().graph_resources.nodes[GRAPH_MODE_NORMAL][i].player ^ 1);
            if (n_opening_name.size()) {
                opening_name = n_opening_name;
            }
            getData().graph_resources.nodes[GRAPH_MODE_NORMAL][i].opening_name = opening_name;
        }
        update_xot_start_n_discs(&getData().graph_resources);
        getData().graph_resources.need_init = false;
        getData().history_elem = getData().graph_resources.nodes[GRAPH_MODE_NORMAL].back();
        changeScene(U"Main_scene", SCENE_FADE_TIME);
    }

    void save_selected_game_to_library() {
        if (!can_save_any_selected()) {
            if (selected_count() == 0) {
                set_user_status(U"Select a game to save.");
            } else {
                set_user_status(U"Selected game details are still loading.");
            }
            write_cache();
            return;
        }
        write_cache();
        std::vector<int> indices(selected_indices.begin(), selected_indices.end());
        std::sort(indices.begin(), indices.end());
        Game_library_save_request_info& save_request = getData().game_library_save_request_info;
        save_request.init();
        save_request.active = true;
        save_request.return_scene = U"Import_othello_quest";
        save_request.initial_subfolder = "othello_quest/" + oq_mode_folder(searched_mode).narrow();
        for (int idx : indices) {
            if (0 <= idx && idx < (int)games.size() && games[idx].detail_state == Detail_state::Ready && !games[idx].transcript.empty()) {
                Oq_game& game = games[idx];
                Game_library_pending_save pending;
                pending.filename_stem = make_oq_filename_stem(game);
                pending.black_player_name = game.black_name;
                pending.white_player_name = game.white_name;
                pending.memo = user_result_summary(game);
                pending.history = game.history;
                pending.game_date = display_date(game.created).substr(0, 10);
                pending.black_score = game.black_score;
                pending.white_score = game.white_score;
                save_request.pending_games.emplace_back(std::move(pending));
            }
        }
        if (save_request.pending_games.empty()) {
            save_request.init();
            set_user_status(U"Selected game details are still loading.");
            write_cache();
            return;
        }
        changeScene(U"Game_library", SCENE_FADE_TIME);
    }

    void apply_selected_game_information(const Oq_game& game) {
        getData().game_information.init();
        getData().game_information.black_player_name = game.black_name;
        getData().game_information.white_player_name = game.white_name;
        getData().game_information.date = display_date(game.created).substr(0, 10);
        getData().game_information.memo = U"Othello Quest\nmode: " + mode_label(searched_mode) + U"\nid: " + game.id + U"\nstatus: " + game.status_raw + U"\nresult: " + game.result_label + U"\nscore: " + game.score_label;
    }

    bool save_game_to_oq_library(Oq_game& game) {
        const String base_dir = oq_library_base_dir();
        if (oq_library_has_duplicate_history(base_dir, game.history)) {
            return false;
        }
        apply_selected_game_information(game);
        const String memo = user_result_summary(game);
        getData().game_information.memo = memo;
        game_save_helper::save_game_to_file(
            base_dir,
            make_oq_filename_date(game),
            game.black_name,
            game.white_name,
            memo,
            game.history,
            display_date(game.created).substr(0, 10),
            game.black_score,
            game.white_score
        );
        return true;
    }

    String oq_library_base_dir() const {
        String base = Unicode::Widen(getData().directories.document_dir) + U"games/othello_quest/" + oq_mode_folder(searched_mode) + U"/";
        FileSystem::CreateDirectories(base);
        return base;
    }

    String make_oq_filename_date(const Oq_game& game) const {
        const String stem = make_oq_filename_stem(game);
        const String base = oq_library_base_dir();
        String candidate = stem;
        for (int suffix = 1; FileSystem::Exists(base + candidate + U".json") && suffix < 1000; ++suffix) {
            candidate = stem + U"_" + Format(suffix);
        }
        return candidate;
    }

    String make_oq_filename_stem(const Oq_game& game) const {
        String stem = display_date(game.created).replaced(U"-", U"_").replaced(U":", U"_").replaced(U" ", U"_");
        if (!game.id.empty()) {
            stem += U"_" + game.id;
        }
        return stem;
    }

    bool oq_library_has_duplicate_history(const String& base_dir, const std::vector<History_elem>& history) const {
        if (history.empty() || !FileSystem::IsDirectory(base_dir)) {
            return false;
        }
        const String signature = history_signature(history);
        for (const auto& path : FileSystem::DirectoryContents(base_dir)) {
            if (FileSystem::IsDirectory(path) || FileSystem::Extension(path).lowercased() != U"json") {
                continue;
            }
            JSON json = JSON::Load(path);
            if (!json) {
                continue;
            }
            if (json_history_signature(json) == signature) {
                return true;
            }
        }
        return false;
    }

    static String history_signature(const std::vector<History_elem>& history) {
        String signature;
        for (const History_elem& elem : history) {
            signature += Format(elem.board.n_discs()) + U":";
            signature += Format(elem.board.player) + U":";
            signature += Format(elem.board.opponent) + U":";
            signature += Format(elem.player) + U";";
        }
        return signature;
    }

    static String json_history_signature(JSON json) {
        String signature;
        for (int n_discs = 4; n_discs <= HW2; ++n_discs) {
            const String key = Format(n_discs);
            if (!json[key].isObject()) {
                continue;
            }
            if (json[key][GAME_BOARD_PLAYER].getType() != JSONValueType::Number ||
                json[key][GAME_BOARD_OPPONENT].getType() != JSONValueType::Number ||
                json[key][GAME_PLAYER].getType() != JSONValueType::Number) {
                continue;
            }
            signature += key + U":";
            signature += Format(json[key][GAME_BOARD_PLAYER].get<uint64_t>()) + U":";
            signature += Format(json[key][GAME_BOARD_OPPONENT].get<uint64_t>()) + U":";
            signature += Format(json[key][GAME_PLAYER].get<int>()) + U";";
        }
        return signature;
    }

    static String oq_mode_folder(int mode) {
        if (mode == 1) {
            return U"1min";
        }
        if (mode == 2) {
            return U"xot";
        }
        return U"5min";
    }

    bool restore_cached_result(const std::string& username, int mode) {
        const int cache_mode = std::clamp(mode, 0, 2);
        Oq_cache& cache = oq_caches[cache_mode];
        if (!cache.has_result || cache.username != username || cache.mode != cache_mode) {
            return false;
        }

        searched_username = cache.username;
        searched_mode = cache.mode;
        mode_radio.checked = searched_mode;
        username_area.text = Unicode::Widen(searched_username);
        username_area.cursorPos = username_area.text.size();
        games = cache.games;
        current_page = cache.current_page;
        selected_idx = cache.selected_idx;
        selected_indices = cache.selected_indices;
        selection_anchor_idx = cache.selection_anchor_idx;
        next_prefetch_idx = cache.next_prefetch_idx;
        status_message = cache.status_message;
        detail_tasks.clear();
        list_loading = false;
        last_clicked_idx = -1;
        last_click_time = 0;

        normalize_restored_detail_state();
        const int n_pages = games.empty() ? 1 : ((int)games.size() + GAMES_PER_PAGE - 1) / GAMES_PER_PAGE;
        current_page = std::clamp(current_page, 0, n_pages - 1);
        if (selected_idx >= (int)games.size()) {
            selected_idx = -1;
        }
        for (auto it = selected_indices.begin(); it != selected_indices.end();) {
            if (*it < 0 || *it >= (int)games.size()) {
                it = selected_indices.erase(it);
            } else {
                ++it;
            }
        }
        if (selected_idx == -1 && !selected_indices.empty()) {
            selected_idx = *selected_indices.begin();
        }
        if (selection_anchor_idx < 0 || selection_anchor_idx >= (int)games.size()) {
            selection_anchor_idx = selected_idx;
        }
        if (status_message.empty()) {
            status_message = games.empty() ? U"No games found." : U"Loaded from cache.";
        }
        getData().user_settings.othello_quest_username = searched_username;
        getData().user_settings.othello_quest_mode = searched_mode;
        if (!games.empty() && next_prefetch_idx < (int)games.size()) {
            start_next_detail_batch();
        }
        return true;
    }

    void clear_cached_result(const std::string& username, int mode) {
        const int cache_mode = std::clamp(mode, 0, 2);
        Oq_cache& cache = oq_caches[cache_mode];
        if (cache.has_result && cache.username == username && cache.mode == cache_mode) {
            cache = Oq_cache{};
        }
    }

    void write_cache() {
        if (searched_username.empty() || list_loading || searched_mode < 0 || searched_mode > 2) {
            return;
        }

        Oq_cache& cache = oq_caches[searched_mode];
        cache.has_result = true;
        cache.username = searched_username;
        cache.mode = searched_mode;
        cache.games = games;
        int retry_from = (int)cache.games.size();
        for (int i = 0; i < (int)cache.games.size(); ++i) {
            if (cache.games[i].detail_state == Detail_state::Loading) {
                cache.games[i].detail_state = Detail_state::None;
                cache.games[i].error.clear();
                retry_from = std::min(retry_from, i);
            } else if (cache.games[i].detail_state == Detail_state::None) {
                retry_from = std::min(retry_from, i);
            }
        }
        cache.current_page = current_page;
        cache.selected_idx = selected_idx;
        cache.selected_indices = selected_indices;
        cache.selection_anchor_idx = selection_anchor_idx;
        cache.next_prefetch_idx = std::clamp(next_prefetch_idx, 0, (int)cache.games.size());
        if (retry_from < (int)cache.games.size()) {
            cache.next_prefetch_idx = std::min(cache.next_prefetch_idx, retry_from);
        }
        cache.status_message = status_message;
    }

    void normalize_restored_detail_state() {
        int retry_from = (int)games.size();
        for (int i = 0; i < (int)games.size(); ++i) {
            if (games[i].detail_state == Detail_state::Loading) {
                games[i].detail_state = Detail_state::None;
                games[i].error.clear();
                retry_from = std::min(retry_from, i);
            } else if (games[i].detail_state == Detail_state::None) {
                retry_from = std::min(retry_from, i);
            }
        }
        next_prefetch_idx = std::clamp(next_prefetch_idx, 0, (int)games.size());
        if (retry_from < (int)games.size()) {
            next_prefetch_idx = std::min(next_prefetch_idx, retry_from);
        }
    }

    static HashTable<String, String> oq_headers() {
        return HashTable<String, String>{{U"User-Agent", U"egaroucid-othello-quest-import/1.0"}};
    }

    std::string build_list_url() const {
        return std::string(OQ_BASE_URL) + "/games/" + mode_endpoint(searched_mode) + "/" + searched_username + ".json";
    }

    static std::string build_detail_url(const std::string& game_id) {
        return std::string(OQ_BASE_URL) + "/game/" + game_id + ".json";
    }

    static const char* mode_endpoint(int mode) {
        if (mode == 1) {
            return "reversi1";
        }
        if (mode == 2) {
            return "reversix";
        }
        return "reversi";
    }

    static String mode_label(int mode) {
        if (mode == 1) {
            return U"1 min";
        }
        if (mode == 2) {
            return U"XOT";
        }
        return U"5 min";
    }

    static void fill_player_summary(JSON player, String* name, String* rating, String* rank, String* id = nullptr) {
        const String player_id = json_string(player[U"id"]);
        if (id != nullptr) {
            *id = player_id;
        }
        String player_name = json_string(player[U"name"]);
        if (player_name.empty()) {
            player_name = player_id;
        }
        *name = player_name;
        if (player[U"newR"].isNumber()) {
            *rating = Format((int)std::round(player[U"newR"].get<double>()));
        } else if (player[U"oldR"].isNumber()) {
            *rating = Format((int)std::round(player[U"oldR"].get<double>()));
        }
        if (player[U"newD"].isNumber()) {
            *rank = rank_label(player[U"newD"].get<int>());
        } else if (player[U"oldD"].isNumber()) {
            *rank = rank_label(player[U"oldD"].get<int>());
        }
    }

    static bool both_ratings_unchanged(JSON black, JSON white) {
        double black_delta = 0.0;
        double white_delta = 0.0;
        return rating_delta(black, &black_delta) && rating_delta(white, &white_delta) && std::abs(black_delta) < 0.0001 && std::abs(white_delta) < 0.0001;
    }

    static bool rating_delta(JSON player, double* delta) {
        if (!player[U"oldR"].isNumber() || !player[U"newR"].isNumber()) {
            return false;
        }
        *delta = player[U"newR"].get<double>() - player[U"oldR"].get<double>();
        return true;
    }

    static String rank_label(int rank) {
        if (rank >= 0) {
            return Format(rank + 1) + U"D";
        }
        return Format(-rank) + U"K";
    }

    static String player_label(const String& name, const String& rating, const String& rank) {
        String res = name;
        if (!rating.empty() || !rank.empty()) {
            res += U" ";
            if (!rating.empty()) {
                res += rating;
            }
            if (!rank.empty()) {
                res += U" " + rank;
            }
        }
        return res;
    }

    static String detail_result_text(const Oq_game& game) {
        if (game.detail_state == Detail_state::Ready) {
            return game.score_label;
        }
        if (game.detail_state == Detail_state::Loading) {
            return U"Loading...";
        }
        if (game.detail_state == Detail_state::Failed) {
            return U"Failed";
        }
        return U"Queued";
    }

    static String detail_status_text(const Oq_game& game) {
        if (game.detail_state == Detail_state::Ready) {
            return game.result_label;
        }
        if (game.detail_state == Detail_state::Failed) {
            return game.error;
        }
        if (!game.list_status.empty()) {
            return U"list: " + game.list_status;
        }
        return U"";
    }

    static String display_date(const String& created) {
        std::tm utc_tm{};
        if (parse_oq_created_utc(created, &utc_tm)) {
#if defined(_WIN32)
            std::time_t utc_time = _mkgmtime(&utc_tm);
#else
            std::time_t utc_time = timegm(&utc_tm);
#endif
            if (utc_time != (std::time_t)-1) {
                std::tm local_tm{};
#if defined(_WIN32)
                if (localtime_s(&local_tm, &utc_time) == 0) {
#else
                if (localtime_r(&utc_time, &local_tm) != nullptr) {
#endif
                    return Format(local_tm.tm_year + 1900) + U"-" + two_digits(local_tm.tm_mon + 1) + U"-" + two_digits(local_tm.tm_mday);
                }
            }
        }
        if (created.size() >= 10) {
            return created.substr(0, 10);
        }
        return created;
    }

    static bool parse_oq_created_utc(const String& created, std::tm* utc_tm) {
        if (created.size() < 20 || created[4] != U'-' || created[7] != U'-' || created[10] != U'T' || created[13] != U':' || created[16] != U':') {
            return false;
        }
        int year = 0;
        int month = 0;
        int day = 0;
        int hour = 0;
        int minute = 0;
        int second = 0;
        if (!parse_fixed_int(created, 0, 4, &year) || !parse_fixed_int(created, 5, 2, &month) || !parse_fixed_int(created, 8, 2, &day) ||
            !parse_fixed_int(created, 11, 2, &hour) || !parse_fixed_int(created, 14, 2, &minute) || !parse_fixed_int(created, 17, 2, &second)) {
            return false;
        }
        bool utc_mark = false;
        for (size_t i = 19; i < created.size(); ++i) {
            if (created[i] == U'Z') {
                utc_mark = true;
                break;
            }
        }
        if (!utc_mark) {
            return false;
        }
        utc_tm->tm_year = year - 1900;
        utc_tm->tm_mon = month - 1;
        utc_tm->tm_mday = day;
        utc_tm->tm_hour = hour;
        utc_tm->tm_min = minute;
        utc_tm->tm_sec = second;
        utc_tm->tm_isdst = 0;
        return true;
    }

    static bool parse_fixed_int(const String& text, size_t start, size_t length, int* value) {
        if (start + length > text.size()) {
            return false;
        }
        int res = 0;
        for (size_t i = start; i < start + length; ++i) {
            if (text[i] < U'0' || U'9' < text[i]) {
                return false;
            }
            res = res * 10 + (int)(text[i] - U'0');
        }
        *value = res;
        return true;
    }

    static String two_digits(int value) {
        return (value < 10 ? U"0" : U"") + Format(value);
    }

    static bool is_coord(const String& move_text) {
        if (move_text.size() != 2) {
            return false;
        }
        const char32 x = std::tolower((char)move_text[0]);
        const char32 y = move_text[1];
        return U'a' <= x && x <= U'h' && U'1' <= y && y <= U'8';
    }

    static std::string extract_start_pos(JSON start_pos_json) {
        String start_pos = json_string(start_pos_json);
        std::string res;
        for (int i = 0; i + 1 < (int)start_pos.size(); ++i) {
            String maybe = start_pos.substr(i, 2);
            if (is_coord(maybe)) {
                res += normalized_coord(maybe);
                ++i;
            }
        }
        return res;
    }

    static std::string normalized_coord(const String& move_text) {
        std::string res = move_text.narrow();
        if (!res.empty()) {
            res[0] = (char)std::tolower((unsigned char)res[0]);
        }
        return res;
    }

    static String json_string(JSON json) {
        if (json.isString()) {
            return json.getString();
        }
        return U"";
    }

    static int json_int(JSON json) {
        if (json.isNumber()) {
            return json.get<int>();
        }
        return 0;
    }

    static bool is_oq_account_whitespace(char32 ch) {
        return ch == U' ' || ch == U'\t' || ch == U'\r' || ch == U'\n' || ch == U'　';
    }

    static bool is_oq_account_char(char32 ch) {
        return (U'a' <= ch && ch <= U'z') || (U'A' <= ch && ch <= U'Z') || (U'0' <= ch && ch <= U'9') || ch == U'_';
    }

    static String remove_oq_account_whitespace(const String& text) {
        String res;
        for (const char32 ch : text) {
            if (!is_oq_account_whitespace(ch)) {
                res.push_back(ch);
            }
        }
        return res;
    }

    static String first_invalid_oq_account_char(const String& text) {
        for (const char32 ch : text) {
            if (!is_oq_account_char(ch)) {
                String res;
                res.push_back(ch);
                return res;
            }
        }
        return U"";
    }

    static std::string trim_copy(std::string s) {
        while (!s.empty() && std::isspace((unsigned char)s.front())) {
            s.erase(s.begin());
        }
        while (!s.empty() && std::isspace((unsigned char)s.back())) {
            s.pop_back();
        }
        return s;
    }
};


class Edit_board : public App::Scene {
private:
    Button back_button;
    Button set_black_button;
    Button set_white_button;
    Radio_button disc_radio;
    bool done;
    bool failed;
    bool update_future_positions;
    History_elem history_elem;

    struct Import_place {
        int insert_place;
        int replace_place;
    };

    struct Future_move {
        int policy;
        int destination_idx;
    };

public:
    Edit_board(const InitData& init) : IScene{ init } {
        set_scene_ime_enabled(false);
        set_black_button.init(BUTTON3_VERTICAL_SX, BUTTON3_VERTICAL_1_SY, BUTTON3_VERTICAL_WIDTH, BUTTON3_VERTICAL_HEIGHT, BUTTON3_VERTICAL_RADIUS, language.get("in_out", "import_as_black"), 25, getData().fonts.font, getData().colors.black, getData().colors.white);
        set_white_button.init(BUTTON3_VERTICAL_SX, BUTTON3_VERTICAL_2_SY, BUTTON3_VERTICAL_WIDTH, BUTTON3_VERTICAL_HEIGHT, BUTTON3_VERTICAL_RADIUS, language.get("in_out", "import_as_white"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(BUTTON3_VERTICAL_SX, BUTTON3_VERTICAL_3_SY, BUTTON3_VERTICAL_WIDTH, BUTTON3_VERTICAL_HEIGHT, BUTTON3_VERTICAL_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        done = false;
        failed = false;
        update_future_positions = false;
        history_elem = getData().history_elem;
        Radio_button_element radio_button_elem;
        disc_radio.init();
        radio_button_elem.init(480, 120, getData().fonts.font, 15, language.get("edit_board", "black"), true);
        disc_radio.push(radio_button_elem);
        radio_button_elem.init(480, 140, getData().fonts.font, 15, language.get("edit_board", "white"), false);
        disc_radio.push(radio_button_elem);
        radio_button_elem.init(480, 160, getData().fonts.font, 15, language.get("edit_board", "empty"), false);
        disc_radio.push(radio_button_elem);

    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        int board_arr[HW2];
        history_elem.board.translate_to_arr(board_arr, history_elem.player);
        for (int cell = 0; cell < HW2; ++cell) {
            int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE;
            int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE;
            Rect cell_region(x, y, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
            if (cell_region.leftPressed()) {
                board_arr[cell] = disc_radio.checked;
            }
        }
        history_elem.board.translate_from_arr(board_arr, history_elem.player);
        if (KeyB.pressed()) {
            disc_radio.checked = BLACK;
        }
        else if (KeyW.pressed()) {
            disc_radio.checked = WHITE;
        }
        else if (KeyE.pressed()) {
            disc_radio.checked = VACANT;
        }
        Scene::SetBackground(getData().colors.green);
        getData().fonts.font(language.get("in_out", "edit_board")).draw(25, 480, 20, getData().colors.white);
        getData().fonts.font(language.get("in_out", "color")).draw(20, 480, 80, getData().colors.white);
        draw_board(getData().fonts, getData().colors, history_elem);
        disc_radio.draw();
        const bool future_update_available = can_update_following_positions(BLACK) || can_update_following_positions(WHITE);
        draw_future_update_checkbox(future_update_available);
        
        // 盤上の石数を計算(偶数なら黒番、奇数なら白番)
        int n_discs = history_elem.board.n_discs();
        bool is_black_turn = (n_discs % 2 == 0);
        
        // ボタンの描画
        set_black_button.draw();
        set_white_button.draw();
        back_button.draw();
        
        // 手番に応じてボタンの周りを白線で囲う
        if (is_black_turn) {
            s3d::RoundRect(BUTTON3_VERTICAL_SX - 6, BUTTON3_VERTICAL_1_SY - 6, BUTTON3_VERTICAL_WIDTH + 12, BUTTON3_VERTICAL_HEIGHT + 12, BUTTON3_VERTICAL_RADIUS + 6)
                .drawFrame(2, 0, getData().colors.black);
        } else {
            s3d::RoundRect(BUTTON3_VERTICAL_SX - 6, BUTTON3_VERTICAL_2_SY - 6, BUTTON3_VERTICAL_WIDTH + 12, BUTTON3_VERTICAL_HEIGHT + 12, BUTTON3_VERTICAL_RADIUS + 6)
                .drawFrame(2, 0, getData().colors.white);
        }
        
        if (back_button.clicked() || gui_textarea_ime::escape_pressed_for_scene_change()) {
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        if (set_black_button.clicked() || (KeyEnter.pressed() && is_black_turn)) {
            process_import(BLACK);
        }
        if (set_white_button.clicked() || (KeyEnter.pressed() && !is_black_turn)) {
            process_import(WHITE);
        }
    }

    void draw() const override {

    }

private:
    void draw_future_update_checkbox(bool enabled) {
        constexpr int checkbox_x = 480;
        constexpr int checkbox_y = 195;
        constexpr int checkbox_size = 18;
        constexpr int font_size = 15;
        constexpr double gap = 8.0;
        const String label = language.get("edit_board", "update_following_positions");
        const double label_width = getData().fonts.font(label).region(font_size, Vec2{0, 0}).w;
        const RectF checkbox_rect(checkbox_x, checkbox_y, checkbox_size, checkbox_size);
        const RectF hit_rect(checkbox_x, checkbox_y - 4.0, checkbox_size + gap + label_width, checkbox_size + 8.0);
        if (!enabled) {
            update_future_positions = false;
        } else if (hit_rect.mouseOver()) {
            Cursor::RequestStyle(CursorStyle::Hand);
        }
        const double alpha = enabled ? 1.0 : 0.35;
        const Texture& checkbox_tex = update_future_positions ? getData().resources.checkbox : getData().resources.unchecked;
        if (checkbox_tex) {
            checkbox_tex.resized(checkbox_size).draw(checkbox_x, checkbox_y, ColorF{1.0, alpha});
        } else if (update_future_positions) {
            checkbox_rect.draw(ColorF(getData().colors.white, alpha));
        } else {
            checkbox_rect.drawFrame(2.0, ColorF(getData().colors.white, alpha));
        }
        getData().fonts.font(label).draw(font_size, Arg::leftCenter(checkbox_x + checkbox_size + gap, checkbox_y + checkbox_size / 2.0), ColorF(getData().colors.white, enabled ? 1.0 : 0.45));
        if (enabled && hit_rect.leftClicked()) {
            update_future_positions = !update_future_positions;
        }
    }

    History_elem make_import_history_elem(int player) {
        History_elem elem = history_elem;
        if (elem.player != player) {
            elem.board.pass();
        }
        elem.player = player;
        elem.v = GRAPH_IGNORE_VALUE;
        elem.level = -1;
        if (!elem.board.is_end() && elem.board.get_legal() == 0) {
            elem.board.pass();
            elem.player ^= 1;
        }
        return elem;
    }

    Import_place find_import_place(int n_discs) {
        Import_place place;
        place.insert_place = (int)getData().graph_resources.nodes[getData().graph_resources.branch].size();
        place.replace_place = -1;
        for (int i = 0; i < (int)getData().graph_resources.nodes[getData().graph_resources.branch].size(); ++i) {
            int node_n_discs = getData().graph_resources.nodes[getData().graph_resources.branch][i].board.n_discs();
            if (node_n_discs == n_discs) {
                place.replace_place = i;
                place.insert_place = -1;
                break;
            } else if (node_n_discs > n_discs) {
                place.insert_place = i;
                break;
            }
        }
        return place;
    }

    int infer_last_policy(const History_elem& elem, const Import_place& place) {
        const int branch = getData().graph_resources.branch;
        const std::vector<History_elem>& nodes = getData().graph_resources.nodes[branch];
        int previous_idx = -1;
        if (place.replace_place - 1 >= 0) {
            previous_idx = place.replace_place - 1;
        } else if (place.insert_place - 1 >= 0 && place.insert_place - 1 < (int)nodes.size()) {
            previous_idx = place.insert_place - 1;
        }
        if (previous_idx != -1) {
            uint64_t f_discs = nodes[previous_idx].board.player | nodes[previous_idx].board.opponent;
            uint64_t discs = elem.board.player | elem.board.opponent;
            if (pop_count_ull(discs ^ f_discs) == 1) {
                return ctz(discs ^ f_discs);
            }
            return -1;
        }
        for (int i = 0; i < (int)getData().graph_resources.nodes[GRAPH_MODE_NORMAL].size(); ++i) {
            int node_n_discs = getData().graph_resources.nodes[GRAPH_MODE_NORMAL][i].board.n_discs();
            if (node_n_discs + 1 == elem.board.n_discs()) {
                uint64_t f_discs = getData().graph_resources.nodes[GRAPH_MODE_NORMAL][i].board.player | getData().graph_resources.nodes[GRAPH_MODE_NORMAL][i].board.opponent;
                uint64_t discs = elem.board.player | elem.board.opponent;
                if (pop_count_ull(discs ^ f_discs) == 1) {
                    return ctz(discs ^ f_discs);
                }
            }
        }
        return -1;
    }

    bool apply_history_move(Board* board, int* player, int policy) {
        if (!is_valid_policy(policy) || board->is_end()) {
            return false;
        }
        if (board->get_legal() == 0ULL) {
            board->pass();
            *player ^= 1;
            if (board->get_legal() == 0ULL) {
                return false;
            }
        }
        if ((board->get_legal() & (1ULL << policy)) == 0ULL) {
            return false;
        }
        Flip flip;
        calc_flip(&flip, board, policy);
        board->move_board(&flip);
        *player ^= 1;
        if (board->get_legal() == 0ULL) {
            board->pass();
            *player ^= 1;
        }
        return true;
    }

    bool collect_following_moves(int start_n_discs, std::vector<Future_move>* moves) {
        moves->clear();
        const int branch = getData().graph_resources.branch;
        const std::vector<History_elem>& nodes = getData().graph_resources.nodes[branch];
        int idx = 0;
        while (idx < (int)nodes.size() && nodes[idx].board.n_discs() < start_n_discs) {
            ++idx;
        }
        if (idx >= (int)nodes.size()) {
            return true;
        }
        int current_idx = idx;
        if (nodes[idx].board.n_discs() == start_n_discs + 1) {
            if (!is_valid_policy(nodes[idx].policy)) {
                return false;
            }
            moves->emplace_back(Future_move{nodes[idx].policy, idx});
        } else if (nodes[idx].board.n_discs() != start_n_discs) {
            return false;
        }
        while (current_idx < (int)nodes.size()) {
            int destination_idx = current_idx + 1;
            int policy = nodes[current_idx].next_policy;
            if (!is_valid_policy(policy)) {
                if (destination_idx < (int)nodes.size() &&
                    nodes[destination_idx].board.n_discs() == nodes[current_idx].board.n_discs() + 1 &&
                    is_valid_policy(nodes[destination_idx].policy)) {
                    policy = nodes[destination_idx].policy;
                } else {
                    break;
                }
            }
            if (destination_idx >= (int)nodes.size() || nodes[destination_idx].board.n_discs() != nodes[current_idx].board.n_discs() + 1) {
                return false;
            }
            moves->emplace_back(Future_move{policy, destination_idx});
            current_idx = destination_idx;
        }
        return true;
    }

    bool build_following_positions(const History_elem& start_elem, std::vector<History_elem>* rebuilt_nodes) {
        std::vector<Future_move> following_moves;
        if (!collect_following_moves(start_elem.board.n_discs(), &following_moves) || following_moves.empty()) {
            return false;
        }
        std::vector<History_elem> rebuilt;
        rebuilt.emplace_back(start_elem);
        Board board = start_elem.board.copy();
        int player = start_elem.player;
        for (const Future_move& move : following_moves) {
            if (!apply_history_move(&board, &player, move.policy)) {
                return false;
            }
            rebuilt.back().next_policy = move.policy;
            History_elem next_elem = getData().graph_resources.nodes[getData().graph_resources.branch][move.destination_idx];
            next_elem.board = board;
            next_elem.player = player;
            next_elem.policy = move.policy;
            next_elem.next_policy = -1;
            next_elem.v = GRAPH_IGNORE_VALUE;
            next_elem.level = -1;
            next_elem.opening_name.clear();
            rebuilt.emplace_back(next_elem);
            board = next_elem.board.copy();
            player = next_elem.player;
        }
        rebuilt.back().next_policy = -1;
        if (rebuilt_nodes != nullptr) {
            *rebuilt_nodes = rebuilt;
        }
        return true;
    }

    bool can_update_following_positions(int player) {
        History_elem elem = make_import_history_elem(player);
        Import_place place = find_import_place(elem.board.n_discs());
        elem.policy = infer_last_policy(elem, place);
        return build_following_positions(elem, nullptr);
    }

    void replace_or_insert_single_node(const History_elem& elem, const Import_place& place) {
        if (place.replace_place != -1) {
            std::cerr << "replace" << std::endl;
            getData().graph_resources.nodes[getData().graph_resources.branch][place.replace_place] = elem;
        } else {
            std::cerr << "insert" << std::endl;
            getData().graph_resources.nodes[getData().graph_resources.branch].insert(getData().graph_resources.nodes[getData().graph_resources.branch].begin() + place.insert_place, elem);
        }
    }

    void replace_or_insert_rebuilt_nodes(const std::vector<History_elem>& rebuilt_nodes, const Import_place& place) {
        std::vector<History_elem>& nodes = getData().graph_resources.nodes[getData().graph_resources.branch];
        int start_idx = place.replace_place != -1 ? place.replace_place : place.insert_place;
        nodes.erase(nodes.begin() + start_idx, nodes.end());
        nodes.insert(nodes.end(), rebuilt_nodes.begin(), rebuilt_nodes.end());
    }

    void process_import(int player) {
        history_elem = make_import_history_elem(player);
        int n_discs = history_elem.board.n_discs();
        Import_place place = find_import_place(n_discs);
        history_elem.policy = infer_last_policy(history_elem, place);
        std::vector<History_elem> rebuilt_nodes;
        bool update_future = update_future_positions && build_following_positions(history_elem, &rebuilt_nodes);
        if (update_future) {
            replace_or_insert_rebuilt_nodes(rebuilt_nodes, place);
        } else {
            replace_or_insert_single_node(history_elem, place);
        }
        getData().graph_resources.n_discs = n_discs;
        update_xot_start_n_discs(&getData().graph_resources);
        getData().graph_resources.need_init = false;
        getData().history_elem = getData().graph_resources.nodes[getData().graph_resources.branch].back();
        changeScene(U"Main_scene", SCENE_FADE_TIME);
    }
};

class Game_library : public App::Scene {
private:
    std::vector<Game_abstract> games;
    std::vector<Button> import_buttons;
    std::vector<ImageButton> delete_buttons;
    std::vector<ImageButton> edit_buttons;
    std::vector<ImageButton> folder_delete_buttons;
    Scroll_manager scroll_manager;
    Button back_button;
    Button up_button;
    Button add_folder_button;
    Button save_here_button;
    Button select_all_button;
    Button copy_button;
    Button cut_button;
    Button paste_button;
    Button delete_selected_button;
    bool failed;
    // Explorer-like folder view
    std::vector<String> folders_display; // includes optional ".." at head
    explorer::PathState explorer_state; // manages current subfolder
    TextEditState new_folder_area;
    Button create_folder_button;
    Button inline_edit_back_button;
    Button inline_edit_ok_button;
    TextEditState folder_rename_area;
    bool renaming_folder = false;
    int renaming_folder_index = -1;
    int selected_folder_index = -1;
    String selected_folder_name;
    String renaming_folder_original_name;
    bool creating_folder = false;
    std::unordered_set<int> selected_folder_indices;
    std::unordered_set<int> selected_game_indices;
    int selection_anchor_row = -1;
    bool clipboard_cut = false;

    struct Clipboard_item {
        bool folder = false;
        String source_path;
        Game_abstract game;
        String name;
    };
    std::vector<Clipboard_item> clipboard_items;

    String shortcut_label(const String& label, const String& shortcut) const {
        return label + U" (" + shortcut + U")";
    }

    void sync_last_opened_subfolder() {
        if (is_save_request_active()) {
            return;
        }
        getData().user_settings.input_game_last_subfolder = explorer_state.subfolder;
    }

    bool is_save_request_active() const {
        const Game_library_save_request_info& request = getData().game_library_save_request_info;
        return request.active && !request.pending_games.empty();
    }

    void ensure_game_library_system_folders() const {
        const String games_root = Unicode::Widen(getData().directories.document_dir) + U"games/";
        ensure_summary_folder(games_root);
        ensure_summary_folder(games_root + U"othello_quest/");
        ensure_summary_folder(games_root + U"othello_quest/5min/");
        ensure_summary_folder(games_root + U"othello_quest/1min/");
        ensure_summary_folder(games_root + U"othello_quest/xot/");
        ensure_summary_folder(games_root + U"recycle_bin/");
    }

    static void ensure_summary_folder(const String& dir) {
        FileSystem::CreateDirectories(dir);
        const String csv_path = dir + U"summary.csv";
        if (!FileSystem::Exists(csv_path)) {
            CSV csv;
            csv.save(csv_path);
        }
    }

    bool in_recycle_bin() const {
        return explorer_state.subfolder == "recycle_bin" || explorer_state.subfolder.starts_with("recycle_bin/");
    }

    bool recycle_bin_visible() const {
        return getData().menu_elements.enable_recycle_bin;
    }

    void refresh_after_recycle_bin_visibility_change() {
        clear_selection();
        if (!recycle_bin_visible() && in_recycle_bin()) {
            explorer_state.clear();
        }
        sync_last_opened_subfolder();
        enumerate_current_dir();
        load_games();
        init_scroll_manager();
    }

    void draw_recycle_bin_checkbox() {
        const String label = language.get("in_out", "enable_recycle_bin");
        constexpr int checkbox_size = 18;
        int font_size = 13;
        double label_width = getData().fonts.font(label).region(font_size, Vec2{ 0, 0 }).w;
        while (font_size > 10 && label_width > 220.0) {
            --font_size;
            label_width = getData().fonts.font(label).region(font_size, Vec2{ 0, 0 }).w;
        }

        const double gap = 8.0;
        const double total_width = checkbox_size + gap + label_width;
        const double x = WINDOW_SIZE_X - 24.0 - total_width;
        const double y = 9.0;
        const RectF checkbox_rect(x, y, checkbox_size, checkbox_size);
        const RectF hit_rect(x, y - 4.0, total_width, checkbox_size + 8.0);

        if (hit_rect.mouseOver()) {
            Cursor::RequestStyle(CursorStyle::Hand);
        }

        const bool enabled = recycle_bin_visible();
        const Texture& checkbox_tex = enabled ? getData().resources.checkbox : getData().resources.unchecked;
        if (checkbox_tex) {
            checkbox_tex.resized(checkbox_size).draw(checkbox_rect.pos);
        } else if (enabled) {
            checkbox_rect.draw(getData().colors.white);
        } else {
            checkbox_rect.drawFrame(2.0, getData().colors.white.withAlpha(120));
        }
        getData().fonts.font(label).draw(font_size, Arg::leftCenter(x + checkbox_size + gap, y + checkbox_size / 2.0), getData().colors.white);

        if (hit_rect.leftClicked()) {
            getData().menu_elements.enable_recycle_bin = !enabled;
            refresh_after_recycle_bin_visibility_change();
        }
    }

    bool is_protected_system_folder(const String& folder_name) const {
        if (explorer_state.subfolder.empty()) {
            return folder_name == U"othello_quest" || folder_name == U"recycle_bin";
        }
        if (explorer_state.subfolder == "othello_quest") {
            return folder_name == U"5min" || folder_name == U"1min" || folder_name == U"xot";
        }
        return false;
    }

    void restore_last_opened_subfolder() {
        explorer_state.clear();
        explorer_state.subfolder = getData().user_settings.input_game_last_subfolder;
        if (!recycle_bin_visible() && in_recycle_bin()) {
            explorer_state.clear();
            getData().user_settings.input_game_last_subfolder.clear();
            return;
        }
        if (explorer_state.subfolder.empty()) {
            return;
        }
        String root_dir = explorer::build_root_dir(getData().directories.document_dir, "games");
        String current_dir = explorer::build_current_dir(root_dir, explorer_state);
        if (!FileSystem::IsDirectory(current_dir)) {
            explorer_state.clear();
            getData().user_settings.input_game_last_subfolder.clear();
        }
    }

public:
    Game_library(const InitData& init) : IScene{ init } {
        set_scene_ime_enabled(true);
        back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        add_folder_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "new_folder"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        save_here_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "save_here"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        create_folder_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "create"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        up_button.init(IMPORT_GAME_SX, IMPORT_GAME_SY - 30, 28, 24, 4, U"↑", 16, getData().fonts.font, getData().colors.white, getData().colors.black);
        select_all_button.init(24, 36, 150, 28, 8, shortcut_label(language.get("in_out", "game_library_select_all"), U"Ctrl+A"), 12, getData().fonts.font, getData().colors.white, getData().colors.black);
        delete_selected_button.init(182, 36, 160, 28, 8, language.get("in_out", "game_library_delete"), 12, getData().fonts.font, getData().colors.white, getData().colors.black);
        copy_button.init(WINDOW_SIZE_X - 330, 36, 100, 28, 8, shortcut_label(language.get("in_out", "game_library_copy"), U"Ctrl+C"), 12, getData().fonts.font, getData().colors.white, getData().colors.black);
        cut_button.init(WINDOW_SIZE_X - 222, 36, 100, 28, 8, shortcut_label(language.get("in_out", "game_library_cut"), U"Ctrl+X"), 12, getData().fonts.font, getData().colors.white, getData().colors.black);
        paste_button.init(WINDOW_SIZE_X - 114, 36, 90, 28, 8, shortcut_label(language.get("in_out", "game_library_paste"), U"Ctrl+V"), 12, getData().fonts.font, getData().colors.white, getData().colors.black);
        inline_edit_back_button.init(0, 0, 80, 30, 8, language.get("common", "back"), 18, getData().fonts.font, getData().colors.white, getData().colors.black);
        inline_edit_ok_button.init(0, 0, 70, 30, 8, language.get("common", "ok"), 18, getData().fonts.font, getData().colors.white, getData().colors.black);
        failed = false;
        ensure_game_library_system_folders();
        if (is_save_request_active()) {
            explorer_state.clear();
            explorer_state.subfolder = getData().game_library_save_request_info.initial_subfolder;
            String root_dir = explorer::build_root_dir(getData().directories.document_dir, "games");
            String current_dir = explorer::build_current_dir(root_dir, explorer_state);
            if (!FileSystem::IsDirectory(current_dir)) {
                explorer_state.clear();
            }
        } else {
            restore_last_opened_subfolder();
        }
        sync_last_opened_subfolder();
        enumerate_current_dir();
        load_games();
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        getData().fonts.font(language.get("in_out", "game_library")).draw(22, Arg::center(X_CENTER, 22), getData().colors.white);
        String path_label;
        if (explorer_state.has_parent()) {
            path_label = explorer::compose_path_label(U"games/", explorer_state);
        }
        bool enter_pressed = KeyEnter.down();
        bool escape_pressed = gui_textarea_ime::escape_down_for_scene_change();

        if (creating_folder) {
            draw_folder_creation_overlay(enter_pressed, escape_pressed);
            return;
        }

        draw_recycle_bin_checkbox();

        update_bottom_button_layout();
        back_button.draw();
        if (back_button.clicked()) {
            if (renaming_folder) {
                cancel_folder_rename();
            } else if (is_save_request_active()) {
                cancel_save_request_and_return();
            } else {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        } else if (!renaming_folder && escape_pressed) {
            if (is_save_request_active()) {
                cancel_save_request_and_return();
                return;
            }
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
        handle_batch_action_shortcuts();
        draw_batch_action_buttons();
        draw_folder_management_ui();
        if (failed) {
            getData().fonts.font(language.get("in_out", "import_failed")).draw(20, Arg::center(X_CENTER, Y_CENTER), getData().colors.white);
        } else {
            ExplorerFolderInlineConfig inline_cfg{};
            inline_cfg.renaming = renaming_folder;
            inline_cfg.folder_index = renaming_folder ? renaming_folder_index : -1;
            // Always set inline config to show rename buttons
            inline_cfg.text_area = &folder_rename_area;
            inline_cfg.back_button = &inline_edit_back_button;
            inline_cfg.ok_button = &inline_edit_ok_button;
            inline_cfg.on_cancel = [this]() {
                cancel_folder_rename();
            };
            inline_cfg.on_commit = [this](const String& trimmed) {
                return confirm_folder_rename(trimmed);
            };
            const ExplorerFolderInlineConfig* inline_ptr = &inline_cfg;
            auto res = DrawExplorerList(
                folders_display, games, delete_buttons, edit_buttons, scroll_manager, up_button,
                IMPORT_GAME_HEIGHT, IMPORT_GAME_N_GAMES_ON_WINDOW, explorer_state.has_parent(), getData().fonts, getData().colors, getData().resources, language,
                getData().directories.document_dir, explorer_state.subfolder, inline_ptr, &selected_folder_indices, &selected_game_indices, path_label);
            
            // Draw delete buttons for empty folders
            if (!renaming_folder) {
                int sy = IMPORT_GAME_SY + 8;
                int strt_idx_int = scroll_manager.get_strt_idx_int();
                int parent_offset = explorer_state.has_parent() ? 1 : 0;
                int row_index = parent_offset;  // Start after parent folder row
                
                for (int i = 0; i < (int)folders_display.size(); ++i) {
                    if (row_index >= strt_idx_int && row_index < strt_idx_int + IMPORT_GAME_N_GAMES_ON_WINDOW) {
                        if (i < (int)folder_delete_buttons.size() && !is_protected_system_folder(folders_display[i]) && is_folder_empty(folders_display[i])) {
                            int display_row = row_index - strt_idx_int;
                            int item_sy = sy + display_row * IMPORT_GAME_HEIGHT;
                            folder_delete_buttons[i].move(IMPORT_GAME_SX + 1, item_sy + 1);
                            folder_delete_buttons[i].enable();
                            folder_delete_buttons[i].draw();
                            if (folder_delete_buttons[i].clicked()) {
                                delete_folder(i);
                                return;
                            }
                        }
                    }
                    row_index++;
                }
                draw_selection_highlights_foreground();
            }
            
            if (res.upButtonClicked || res.parentFolderDoubleClicked) {
                if (navigate_to_parent_subfolder()) {
                    return;
                }
            }
            if (res.backgroundClicked) {
                clear_selection();
            }
            if (res.folderClicked) {
                handle_selection_click(selection_row_for_folder(res.folderRenameIndex), KeyControl.pressed(), KeyShift.pressed());
            }
            if (res.folderDoubleClicked) {
                if (renaming_folder) {
                    cancel_folder_rename();
                }
                clear_selection();
                explorer_state.navigate_to_child(res.clickedFolder);
                sync_last_opened_subfolder();
                enumerate_current_dir();
                load_games();
                init_scroll_manager();
                return;
            }
            if (res.folderRenameRequested && res.folderRenameIndex >= 0) {
                begin_folder_rename(res.folderRenameIndex);
            }
            if (res.deleteClicked && res.deleteIndex >= 0) {
                delete_game(res.deleteIndex);
            }
            if (res.gameClicked && res.clickedGameIndex >= 0) {
                handle_selection_click(selection_row_for_game(res.clickedGameIndex), KeyControl.pressed(), KeyShift.pressed());
            }
            if (res.editClicked && res.editIndex >= 0) {
                edit_game(res.editIndex);
                return;
            }
            if (res.gameDoubleClicked && res.importIndex >= 0) {
                import_game(res.importIndex);
                return;
            }
            if (res.drop_completed) {
                handle_drop(res);
            } else if (res.reorderRequested) {
                handle_reorder(res);
            }
        }
        
        if (renaming_folder && escape_pressed) {
            cancel_folder_rename();
        }
        clear_selection_on_empty_space_press();
    }

    void draw() const override {

    }

private:
    void update_bottom_button_layout() {
        if (is_save_request_active()) {
            back_button.move(BUTTON3_1_SX, BUTTON3_SY);
            save_here_button.move(BUTTON3_2_SX, BUTTON3_SY);
            add_folder_button.move(BUTTON3_3_SX, BUTTON3_SY);
        } else {
            back_button.move(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY);
            add_folder_button.move(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY);
        }
    }

    static bool button_contains(const Button& button, const Vec2& pos) {
        return RectF(button.rect.x, button.rect.y, button.rect.w, button.rect.h).contains(pos);
    }

    RectF recycle_bin_checkbox_hit_rect() const {
        const String label = language.get("in_out", "enable_recycle_bin");
        constexpr int checkbox_size = 18;
        int font_size = 13;
        double label_width = getData().fonts.font(label).region(font_size, Vec2{ 0, 0 }).w;
        while (font_size > 10 && label_width > 220.0) {
            --font_size;
            label_width = getData().fonts.font(label).region(font_size, Vec2{ 0, 0 }).w;
        }
        constexpr double gap = 8.0;
        const double total_width = checkbox_size + gap + label_width;
        const double x = WINDOW_SIZE_X - 24.0 - total_width;
        const double y = 9.0;
        return RectF(x, y - 4.0, total_width, checkbox_size + 8.0);
    }

    bool explorer_element_row_contains(const Vec2& pos) const {
        const RectF list_bounds(
            IMPORT_GAME_SX,
            IMPORT_GAME_SY + 8,
            IMPORT_GAME_WIDTH,
            IMPORT_GAME_HEIGHT * IMPORT_GAME_N_GAMES_ON_WINDOW
        );
        if (!list_bounds.contains(pos)) {
            return false;
        }
        const int local_row = static_cast<int>((pos.y - list_bounds.y) / IMPORT_GAME_HEIGHT);
        const int row = scroll_manager.get_strt_idx_int() + local_row;
        const int parent_offset = explorer_state.has_parent() ? 1 : 0;
        const int total_rows = parent_offset + static_cast<int>(folders_display.size()) + static_cast<int>(games.size());
        return 0 <= row && row < total_rows;
    }

    bool explorer_control_contains(const Vec2& pos) const {
        if (button_contains(back_button, pos) ||
            button_contains(add_folder_button, pos) ||
            button_contains(up_button, pos) ||
            button_contains(select_all_button, pos) ||
            button_contains(delete_selected_button, pos) ||
            button_contains(copy_button, pos) ||
            button_contains(cut_button, pos) ||
            button_contains(paste_button, pos) ||
            recycle_bin_checkbox_hit_rect().contains(pos)) {
            return true;
        }
        if (is_save_request_active() && button_contains(save_here_button, pos)) {
            return true;
        }
        const RectF scrollbar_rect(770, IMPORT_GAME_SY + 8, 10, IMPORT_GAME_HEIGHT * IMPORT_GAME_N_GAMES_ON_WINDOW);
        return scrollbar_rect.contains(pos);
    }

    void clear_selection_on_empty_space_press() {
        if (renaming_folder || creating_folder || !MouseL.down()) {
            return;
        }
        const Vec2 pos = Cursor::Pos();
        if (explorer_element_row_contains(pos) || explorer_control_contains(pos)) {
            return;
        }
        clear_selection();
    }

    void cancel_save_request_and_return() {
        String return_scene = getData().game_library_save_request_info.return_scene;
        getData().game_library_save_request_info.init();
        changeScene(return_scene.empty() ? U"Import_othello_quest" : return_scene, SCENE_FADE_TIME);
    }

    void init_scroll_manager() {
        int parent_offset = explorer_state.has_parent() ? 1 : 0;  // Add parent folder if not at root
        int total = parent_offset + (int)folders_display.size() + (int)games.size();
        scroll_manager.init(770, IMPORT_GAME_SY + 8, 10, IMPORT_GAME_HEIGHT * IMPORT_GAME_N_GAMES_ON_WINDOW, 20, total, IMPORT_GAME_N_GAMES_ON_WINDOW, IMPORT_GAME_SX, 73, IMPORT_GAME_WIDTH + 10, IMPORT_GAME_HEIGHT * IMPORT_GAME_N_GAMES_ON_WINDOW);
    }

    void clear_selection() {
        explorer::clear_selection(selected_folder_indices, selected_game_indices, selection_anchor_row);
        selected_folder_index = -1;
        selected_folder_name.clear();
    }

    void prune_selection() {
        explorer::prune_selection(selected_folder_indices, selected_game_indices, selection_anchor_row, (int)folders_display.size(), (int)games.size());
    }

    bool has_selection() const {
        return explorer::has_selection(selected_folder_indices, selected_game_indices);
    }

    bool can_delete_selected_items() const {
        if (!has_selection()) {
            return false;
        }
        for (int idx : selected_folder_indices) {
            if (idx < 0 || idx >= (int)folders_display.size()) {
                return false;
            }
            if (is_protected_system_folder(folders_display[idx]) || !is_folder_empty(folders_display[idx])) {
                return false;
            }
        }
        for (int idx : selected_game_indices) {
            if (idx < 0 || idx >= (int)games.size()) {
                return false;
            }
        }
        return true;
    }

    bool all_items_selected() const {
        return explorer::all_items_selected(selected_folder_indices, selected_game_indices, (int)folders_display.size(), (int)games.size());
    }

    int selection_row_for_folder(int idx) const {
        return explorer::selection_row_for_folder(idx, (int)folders_display.size());
    }

    int selection_row_for_game(int idx) const {
        return explorer::selection_row_for_item(idx, (int)folders_display.size(), (int)games.size());
    }

    bool is_valid_selection_row(int row) const {
        return explorer::is_valid_selection_row(row, (int)folders_display.size(), (int)games.size());
    }

    void clear_selection_sets() {
        selected_folder_indices.clear();
        selected_game_indices.clear();
    }

    void add_selection_row(int row) {
        explorer::add_selection_row(selected_folder_indices, selected_game_indices, row, (int)folders_display.size(), (int)games.size());
    }

    void remove_selection_row(int row) {
        explorer::remove_selection_row(selected_folder_indices, selected_game_indices, row, (int)folders_display.size(), (int)games.size());
    }

    bool selection_row_selected(int row) const {
        return explorer::selection_row_selected(selected_folder_indices, selected_game_indices, row, (int)folders_display.size(), (int)games.size());
    }

    void update_focused_selection_row(int row) {
        if (row >= 0 && row < (int)folders_display.size()) {
            selected_folder_index = row;
            selected_folder_name = folders_display[row];
        } else {
            selected_folder_index = -1;
            selected_folder_name.clear();
        }
    }

    void handle_selection_click(int row, bool ctrl_pressed, bool shift_pressed) {
        if (!is_valid_selection_row(row)) {
            return;
        }

        explorer::handle_selection_click(
            selected_folder_indices,
            selected_game_indices,
            selection_anchor_row,
            row,
            (int)folders_display.size(),
            (int)games.size(),
            ctrl_pressed,
            shift_pressed
        );
        update_focused_selection_row(row);
    }

    void select_all_items() {
        explorer::select_all_items(selected_folder_indices, selected_game_indices, selection_anchor_row, (int)folders_display.size(), (int)games.size());
    }

    void handle_batch_action_shortcuts() {
        if (renaming_folder || creating_folder || !KeyControl.pressed()) {
            return;
        }
        if (KeyA.down()) {
            select_all_items();
            return;
        }
        if (KeyC.down() && has_selection()) {
            set_clipboard(false);
            return;
        }
        if (KeyX.down() && has_selection()) {
            set_clipboard(true);
            return;
        }
        if (KeyV.down() && !clipboard_items.empty()) {
            paste_clipboard();
        }
    }

    void draw_selection_highlights_foreground() {
        draw_explorer_selection_highlights_foreground(
            scroll_manager,
            explorer_state.has_parent(),
            (int)folders_display.size(),
            (int)games.size(),
            selected_folder_indices,
            selected_game_indices,
            IMPORT_GAME_SX,
            IMPORT_GAME_SY + 8,
            IMPORT_GAME_WIDTH,
            IMPORT_GAME_HEIGHT,
            IMPORT_GAME_N_GAMES_ON_WINDOW,
            getData().colors.white
        );
    }

    void draw_batch_action_buttons() {
        prune_selection();
        const bool all_selected = all_items_selected();
        select_all_button.str = all_selected ?
            language.get("in_out", "game_library_deselect_all") :
            shortcut_label(language.get("in_out", "game_library_select_all"), U"Ctrl+A");
        select_all_button.font_size = update_font_size_overfull(select_all_button.font, select_all_button.str, 12, select_all_button.rect.h, select_all_button.rect.w);
        select_all_button.draw();
        if (select_all_button.clicked()) {
            if (all_selected) {
                clear_selection();
            } else {
                select_all_items();
            }
        }

        delete_selected_button.str = language.get("in_out", in_recycle_bin() ? "game_library_permanently_delete" : "game_library_delete");
        delete_selected_button.font_size = update_font_size_overfull(delete_selected_button.font, delete_selected_button.str, 12, delete_selected_button.rect.h, delete_selected_button.rect.w);
        copy_button.str = shortcut_label(language.get("in_out", "game_library_copy"), U"Ctrl+C");
        copy_button.font_size = update_font_size_overfull(copy_button.font, copy_button.str, 12, copy_button.rect.h, copy_button.rect.w);
        cut_button.str = shortcut_label(language.get("in_out", "game_library_cut"), U"Ctrl+X");
        cut_button.font_size = update_font_size_overfull(cut_button.font, cut_button.str, 12, cut_button.rect.h, cut_button.rect.w);
        paste_button.str = shortcut_label(language.get("in_out", "game_library_paste"), U"Ctrl+V");
        paste_button.font_size = update_font_size_overfull(paste_button.font, paste_button.str, 12, paste_button.rect.h, paste_button.rect.w);

        const bool any_selected = has_selection();
        const bool can_delete_selected = can_delete_selected_items();
        if (any_selected) {
            copy_button.enable();
            cut_button.enable();
        } else {
            copy_button.disable();
            cut_button.disable();
        }
        if (can_delete_selected) {
            delete_selected_button.enable();
        } else {
            delete_selected_button.disable();
        }
        if (clipboard_items.empty()) {
            paste_button.disable();
        } else {
            paste_button.enable();
        }

        copy_button.draw();
        cut_button.draw();
        paste_button.draw();
        delete_selected_button.draw();
        if (copy_button.clicked()) {
            set_clipboard(false);
        }
        if (cut_button.clicked()) {
            set_clipboard(true);
        }
        if (paste_button.clicked()) {
            paste_clipboard();
        }
        if (can_delete_selected && delete_selected_button.clicked()) {
            delete_selected_items();
        }
    }
    
    // Check if folder is empty (no subfolders and no games)
    bool is_folder_empty(const String& folder_name) const {
        std::string base_dir = get_root_dir();
        std::string rel_path = explorer_state.subfolder;
        if (!rel_path.empty()) {
            rel_path += "/";
        }
        rel_path += folder_name.narrow();
        
        // Check for subfolders
        std::vector<String> subfolders = explorer::enumerate_subfolders(base_dir, explorer::PathState{rel_path});
        if (!subfolders.empty()) {
            return false;
        }
        
        // Check for games in summary.csv
        String folder_dir = Unicode::Widen(base_dir);
        if (!rel_path.empty()) {
            folder_dir += U"/" + Unicode::Widen(rel_path);
        }
        String csv_path = folder_dir + U"/summary.csv";
        if (FileSystem::Exists(csv_path)) {
            CSV csv{ csv_path };
            if (csv.rows() > 0) {
                return false;
            }
        }
        return true;
    }
    
    // Delete folder (only if empty)
    void delete_folder(int idx) {
        if (idx < 0 || idx >= (int)folders_display.size()) return;
        
        const String& folder_name = folders_display[idx];
        if (is_protected_system_folder(folder_name)) {
            return;
        }
        if (!is_folder_empty(folder_name)) {
            return;  // Don't delete non-empty folders
        }
        
        selected_folder_indices.clear();
        selected_game_indices.clear();
        selected_folder_indices.insert(idx);
        delete_selected_items();
    }

    bool navigate_to_parent_subfolder() {
        if (!explorer_state.navigate_to_parent()) {
            return false;
        }
        sync_last_opened_subfolder();
        if (renaming_folder) {
            cancel_folder_rename();
        }
        enumerate_current_dir();
        load_games();
        init_scroll_manager();
        clear_selection();
        return true;
    }

    void edit_game(int idx) {
        const String json_path = get_base_dir() + games[idx].filename_date + U".json";
        JSON game_json = JSON::Load(json_path);
        if (not game_json) {
            std::cerr << "can't open game" << std::endl;
            failed = true;
            return;
        }
        
        // Load game information
        if (game_json[GAME_BLACK_PLAYER].getType() == JSONValueType::String) {
            getData().game_information.black_player_name = game_json[GAME_BLACK_PLAYER].getString();
        }
        if (game_json[GAME_WHITE_PLAYER].getType() == JSONValueType::String) {
            getData().game_information.white_player_name = game_json[GAME_WHITE_PLAYER].getString();
        }
        if (game_json[GAME_MEMO].getType() == JSONValueType::String) {
            getData().game_information.memo = game_json[GAME_MEMO].getString();
        }
        // Load date field (YYYY-MM-DD format)
        if (game_json[U"date"].getType() == JSONValueType::String) {
            getData().game_information.date = game_json[U"date"].getString();
        } else {
            // Fallback: generate from filename_date
            getData().game_information.date = games[idx].filename_date.substr(0, 10).replaced(U"_", U"-");
        }
        
        // Mark that a specific game has been loaded and store its location
        getData().game_information.is_game_loaded = true;
        
        // Set game editor info for editing mode
        getData().game_editor_info.return_scene = U"Game_library";
        getData().game_editor_info.is_editing_mode = true;
        getData().game_editor_info.game_date = games[idx].filename_date;
        getData().game_editor_info.subfolder = explorer_state.subfolder;
        getData().game_editor_info.game_info_updated = false;
        
        changeScene(U"Game_editor", SCENE_FADE_TIME);
    }

    void import_game(int idx) {
        const String json_path = get_base_dir() + games[idx].filename_date + U".json";
        if (!load_game_from_json(getData(), opening, json_path, games[idx].filename_date, explorer_state.subfolder)) {
            failed = true;
            return;
        }
        changeScene(U"Main_scene", SCENE_FADE_TIME);
    }

    void delete_game(int idx) {
        if (idx < 0 || idx >= (int)games.size()) {
            std::cerr << "delete_game: invalid index " << idx << std::endl;
            return;
        }
        selected_folder_indices.clear();
        selected_game_indices.clear();
        selected_game_indices.insert(idx);
        delete_selected_items();
    }

    std::string get_root_dir() const {
        return explorer::build_root_dir_narrow(getData().directories.document_dir, "games");
    }

    // Helper: current base dir (games/ + optional subfolder + '/')
    String get_base_dir() const {
        return explorer::build_current_dir(get_root_dir(), explorer_state);
    }

    // Reload games list from summary.csv in current folder
    void load_games() {
        games.clear();
        import_buttons.clear();
        delete_buttons.clear();
        edit_buttons.clear();
        const String csv_path = get_base_dir() + U"summary.csv";
        const CSV csv{ csv_path };
        if (csv) {
            for (size_t row = 0; row < csv.rows(); ++row) {
                if (csv[row].size() < 6) {
                    continue;
                }
                Game_abstract game_abstract;
                game_abstract.filename_date = csv[row][0];
                game_abstract.black_player = csv[row][1];
                game_abstract.white_player = csv[row][2];
                game_abstract.memo = csv[row][3];
                game_abstract.black_score = ParseOr<int32>(csv[row][4], GAME_DISCS_UNDEFINED);
                game_abstract.white_score = ParseOr<int32>(csv[row][5], GAME_DISCS_UNDEFINED);
                // Read game_date from column 6 (7th column), or generate from filename if missing
                if (csv[row].size() >= 7 && !csv[row][6].isEmpty()) {
                    game_abstract.game_date = csv[row][6];
                } else {
                    game_abstract.game_date = game_abstract.filename_date.substr(0, 10).replaced(U"_", U"-");
                }
                games.emplace_back(game_abstract);
            }
        }
        reverse(games.begin(), games.end());
        for (int i = 0; i < (int)games.size(); ++i) {
            Button button;
            button.init(0, 0, IMPORT_GAME_BUTTON_WIDTH, IMPORT_GAME_BUTTON_HEIGHT, IMPORT_GAME_BUTTON_RADIUS, language.get("in_out", "import"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
            import_buttons.emplace_back(button);
        }
        Texture delete_button_image = getData().resources.cross;
        Texture edit_button_image = getData().resources.pencil;
        for (int i = 0; i < (int)games.size(); ++i) {
            ImageButton delete_button;
            delete_button.init(0, 0, 15, delete_button_image);
            delete_buttons.emplace_back(delete_button);
            
            ImageButton edit_button;
            edit_button.init(0, 0, 18, edit_button_image);
            edit_buttons.emplace_back(edit_button);
        }
        init_scroll_manager();
    }

    // Enumerate current directory (folders_display only; parent is shown as a separate up icon)
    void enumerate_current_dir() {
        folders_display.clear();
        folder_delete_buttons.clear();
        std::vector<String> folders = explorer::enumerate_subfolders(get_root_dir(), explorer_state);
        Texture cross_image = getData().resources.cross;
        for (auto& folder : folders) {
            if (explorer_state.subfolder.empty() && folder == U"recycle_bin" && !recycle_bin_visible()) {
                continue;
            }
            folders_display.emplace_back(folder);
            
            // Add delete button for each folder
            ImageButton delete_btn;
            delete_btn.init(0, 0, 15, cross_image);
            folder_delete_buttons.emplace_back(delete_btn);
        }
        
        if (!selected_folder_name.isEmpty()) {
            selected_folder_index = find_folder_index(selected_folder_name);
        } else {
            selected_folder_index = -1;
        }
        if (selected_folder_index == -1) {
            selected_folder_name.clear();
        }

        if (renaming_folder) {
            int idx = -1;
            if (!renaming_folder_original_name.isEmpty()) {
                idx = find_folder_index(renaming_folder_original_name);
            }
            if (idx >= 0) {
                renaming_folder_index = idx;
            } else if (renaming_folder_index < 0 || renaming_folder_index >= (int)folders_display.size()) {
                cancel_folder_rename();
            }
        }
        
        init_scroll_manager();
    }

    void select_folder(const String& folder_name) {
        selected_folder_index = find_folder_index(folder_name);
        if (selected_folder_index >= 0) {
            selected_folder_name = folders_display[selected_folder_index];
        } else {
            selected_folder_name.clear();
        }
    }

    int find_folder_index(const String& folder_name) const {
        for (int i = 0; i < (int)folders_display.size(); ++i) {
            if (folders_display[i] == folder_name) {
                return i;
            }
        }
        return -1;
    }

    void draw_folder_management_ui() {
        if (is_save_request_active()) {
            if (renaming_folder) {
                save_here_button.disable_notransparent();
            } else {
                save_here_button.enable();
            }
            save_here_button.draw();
            if (!renaming_folder && save_here_button.clicked()) {
                save_pending_games_here();
                return;
            }
        }

        if (renaming_folder) {
            add_folder_button.disable_notransparent();
        } else {
            add_folder_button.enable();
        }
        add_folder_button.draw();
        if (!renaming_folder && add_folder_button.clicked()) {
            creating_folder = true;
            cancel_folder_rename();
            new_folder_area.text.clear();
            new_folder_area.cursorPos = 0;
            new_folder_area.active = true;
        }
    }

    void draw_folder_creation_overlay(bool enter_pressed, bool escape_pressed) {
        auto& fonts = getData().fonts;
        auto& colors = getData().colors;

        back_button.draw();
        if (back_button.clicked() || escape_pressed) {
            creating_folder = false;
            new_folder_area.text.clear();
            new_folder_area.cursorPos = 0;
            new_folder_area.active = false;
            return;
        }

        const double center_y = NEW_FOLDER_PANEL_SY + NEW_FOLDER_PANEL_HEIGHT / 2.0;
        const double label_x = IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + NEW_FOLDER_LABEL_INNER_MARGIN;

        fonts.font(language.get("in_out", "new_folder") + U":").draw(20, Arg::leftCenter(label_x, center_y), colors.white);

        Vec2 text_pos{
            IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + NEW_FOLDER_TEXTBOX_OFFSET_X,
            center_y + NEW_FOLDER_TEXTBOX_OFFSET_Y
        };
        SizeF text_size{ NEW_FOLDER_TEXTBOX_WIDTH, NEW_FOLDER_TEXTBOX_HEIGHT };
        text_box_with_ime_candidate_window(new_folder_area, text_pos, text_size.x, SimpleGUI::PreferredTextAreaMaxChars);
        gui_list::sanitize_text_area(new_folder_area);

        String folder_name = new_folder_area.text.trimmed();
        bool can_create = gui_list::is_valid_folder_name(folder_name);

        create_folder_button.move(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY);
        if (can_create) {
            create_folder_button.enable();
        } else {
            create_folder_button.disable();
        }
        create_folder_button.draw();

        if (can_create && create_folder_button.clicked()) {
            if (handle_create_folder()) {
                creating_folder = false;
                new_folder_area.active = false;
            }
        }
    }

    bool handle_create_folder() {
        if (renaming_folder) {
            return false;
        }
        String input = gui_list::sanitize_folder_text(new_folder_area.text).trimmed();
        gui_list::sanitize_text_area(new_folder_area);
        if (!gui_list::is_valid_folder_name(input)) {
            return false;
        }
        bool created = gui_list::create_folder_with_initializer(
            get_base_dir(),
            input,
            [](const String& dir) {
                CSV csv;
                csv.save(dir + U"summary.csv");
            }
        );
        if (created) {
            new_folder_area.text.clear();
            new_folder_area.cursorPos = 0;
            enumerate_current_dir();
            load_games();
            handle_selection_click(selection_row_for_folder(find_folder_index(input)), false, false);
        }
        return created;
    }

    void begin_folder_rename(int folder_idx) {
        if (folder_idx < 0 || folder_idx >= (int)folders_display.size()) {
            return;
        }
        if (is_protected_system_folder(folders_display[folder_idx])) {
            return;
        }
        renaming_folder = true;
        renaming_folder_index = folder_idx;
        renaming_folder_original_name = folders_display[folder_idx];
        folder_rename_area.text = folders_display[folder_idx];
        folder_rename_area.cursorPos = folder_rename_area.text.size();
        folder_rename_area.active = true;
        selected_folder_index = folder_idx;
        selected_folder_name = folders_display[folder_idx];
    }

    void cancel_folder_rename() {
        renaming_folder = false;
        renaming_folder_index = -1;
        renaming_folder_original_name.clear();
        folder_rename_area.text.clear();
        folder_rename_area.cursorPos = 0;
        folder_rename_area.active = false;
    }

    bool confirm_folder_rename(const String& target_name) {
        if (!renaming_folder || renaming_folder_index < 0 || renaming_folder_index >= (int)folders_display.size()) {
            return false;
        }
        String trimmed = target_name.trimmed();
        if (!gui_list::is_valid_folder_name(trimmed)) {
            return false;
        }
        String current_name = folders_display[renaming_folder_index];
        if (current_name == trimmed) {
            cancel_folder_rename();
            return true;
        }
        String base_dir = get_base_dir();
        bool renamed = gui_list::rename_folder_in_directory(base_dir, current_name, trimmed);
        if (!renamed) {
            std::cerr << "rename_folder_in_directory failed: base='" << base_dir.narrow()
                      << "' current='" << current_name.narrow() << "' target='" << trimmed.narrow() << "'" << std::endl;
            return false;
        }
        if (renamed) {
            cancel_folder_rename();
            enumerate_current_dir();
            handle_selection_click(selection_row_for_folder(find_folder_index(trimmed)), false, false);
        }
        return renamed;
    }

    // Handle drag and drop operations
    void handle_drop(const ExplorerDrawResult& res) {
        const String target_base = drop_target_base_dir(res);
        const bool selected_drag = dragged_item_is_selected(res);
        if (target_base.empty()) {
            return;
        }

        if (selected_drag) {
            move_selected_items_to_base(target_base);
        } else if (res.is_dragging_game && res.dragged_game_index >= 0 && res.dragged_game_index < (int)games.size()) {
            move_single_game_to_base(res.dragged_game_index, target_base);
        } else if (res.is_dragging_folder && !res.dragged_folder_name.empty()) {
            move_single_folder_to_base(res.dragged_folder_name, target_base);
        }
    }

    std::string parent_subfolder() const {
        std::string parent = explorer_state.subfolder;
        explorer::trim_trailing_slash(parent);
        size_t pos = parent.find_last_of('/');
        if (pos == std::string::npos) {
            parent.clear();
        } else {
            parent = parent.substr(0, pos);
        }
        return parent;
    }

    String base_dir_for_subfolder(const std::string& subfolder) const {
        explorer::PathState target_state;
        target_state.subfolder = subfolder;
        return explorer::build_current_dir(get_root_dir(), target_state);
    }

    String drop_target_base_dir(const ExplorerDrawResult& res) const {
        if (res.drop_on_parent) {
            if (!explorer_state.has_parent()) {
                return U"";
            }
            return base_dir_for_subfolder(parent_subfolder());
        }
        if (res.drop_target_folder.empty()) {
            return U"";
        }
        return get_base_dir() + res.drop_target_folder + U"/";
    }

    bool dragged_item_is_selected(const ExplorerDrawResult& res) const {
        if (res.is_dragging_game) {
            return selected_game_indices.count(res.dragged_game_index) != 0;
        }
        if (res.is_dragging_folder) {
            const int idx = find_folder_index(res.dragged_folder_name);
            return idx >= 0 && selected_folder_indices.count(idx) != 0;
        }
        return false;
    }

    void refresh_after_library_moves() {
        enumerate_current_dir();
        load_games();
        clear_selection();
    }

    void move_selected_items_to_base(const String& target_base) {
        bool changed = false;

        std::vector<int> game_indices(selected_game_indices.begin(), selected_game_indices.end());
        std::sort(game_indices.begin(), game_indices.end());
        for (int idx : game_indices) {
            changed = move_game_to_base(idx, target_base) || changed;
        }

        std::vector<int> folder_indices(selected_folder_indices.begin(), selected_folder_indices.end());
        std::sort(folder_indices.begin(), folder_indices.end());
        for (int idx : folder_indices) {
            if (idx >= 0 && idx < (int)folders_display.size()) {
                changed = move_folder_to_base(folders_display[idx], target_base) || changed;
            }
        }

        if (changed) {
            refresh_after_library_moves();
        }
    }

    void move_single_game_to_base(int game_index, const String& target_base) {
        if (move_game_to_base(game_index, target_base)) {
            refresh_after_library_moves();
        }
    }

    void move_single_folder_to_base(const String& folder_name, const String& target_base) {
        if (move_folder_to_base(folder_name, target_base)) {
            refresh_after_library_moves();
        }
    }

    bool move_game_to_base(int game_index, const String& target_base) {
        if (game_index < 0 || game_index >= (int)games.size() || target_base.empty()) {
            return false;
        }

        ensure_summary_folder(target_base);
        const Game_abstract game = games[game_index];
        const String source_json = get_base_dir() + game.filename_date + U".json";
        String target_name = game.filename_date;
        String target_json = target_base + target_name + U".json";
        if (FileSystem::Exists(target_json)) {
            target_json = make_unique_child_path(target_base, target_name, false);
            if (target_json.empty()) {
                return false;
            }
            target_name = FileSystem::BaseName(target_json).replaced(U".json", U"");
        }
        if (FileSystem::FullPath(source_json) == FileSystem::FullPath(target_json)) {
            return false;
        }
        if (!FileSystem::Exists(source_json)) {
            return false;
        }
        if (move_path_fast(source_json, target_json) || FileSystem::Copy(source_json, target_json)) {
            add_game_to_target_csv(game_with_filename(game, target_name), target_base);
            remove_game_record_by_filename(game.filename_date, source_json);
            if (FileSystem::Exists(source_json)) {
                FileSystem::Remove(source_json);
            }
            return true;
        }
        return false;
    }

    bool move_folder_to_base(const String& folder_name, const String& target_base) {
        if (folder_name.empty() || target_base.empty() || is_protected_system_folder(folder_name)) {
            return false;
        }

        const String source = get_base_dir() + folder_name;
        const String normalized_target_base = gui_list::normalize_directory_base(target_base);
        const String target = normalized_target_base + folder_name;
        if (!FileSystem::IsDirectory(source) || FileSystem::Exists(target)) {
            return false;
        }
        if (FileSystem::FullPath(source) == FileSystem::FullPath(target)) {
            return false;
        }

        auto with_trailing_separator = [](String path) {
            if (!path.ends_with(U"/") && !path.ends_with(U"\\")) {
                path += U"/";
            }
            return path;
        };
        const String source_abs = with_trailing_separator(FileSystem::FullPath(source));
        const String target_base_abs = with_trailing_separator(FileSystem::FullPath(normalized_target_base));
        if (target_base_abs.starts_with(source_abs)) {
            return false;
        }

        FileSystem::CreateDirectories(normalized_target_base);
        if (move_path_fast(source, target)) {
            return true;
        }
        if (FileSystem::Copy(source, target)) {
            FileSystem::Remove(source, AllowUndo::No);
            return true;
        }
        return false;
    }

    void handle_reorder(const ExplorerDrawResult& res) {
        if (!res.reorderRequested || !res.is_dragging_game) {
            return;
        }
        if (res.reorderFrom < 0 || res.reorderFrom >= (int)games.size()) {
            return;
        }
        int insert_idx = std::clamp(res.reorderTo, 0, (int)games.size());
        bool changed = gui_list::reorder_parallel(games, res.reorderFrom, insert_idx, import_buttons, delete_buttons, edit_buttons);
        if (!changed) {
            return;
        }
        persist_games_order_to_csv();
    }

    String make_unique_child_path(const String& target_dir, const String& name, bool folder) const {
        String candidate = target_dir + name + (folder ? U"/" : U".json");
        if (!FileSystem::Exists(candidate)) {
            return candidate;
        }
        for (int i = 1; i < 1000; ++i) {
            String suffix = U" (" + Format(i) + U")";
            candidate = target_dir + name + suffix + (folder ? U"/" : U".json");
            if (!FileSystem::Exists(candidate)) {
                return candidate;
            }
        }
        return U"";
    }

    static String strip_trailing_separator(String path) {
        while (!path.isEmpty() && (path.ends_with(U"/") || path.ends_with(U"\\"))) {
            path.pop_back();
        }
        return path;
    }

    static String normalized_directory_compare_path(String path) {
        path = strip_trailing_separator(FileSystem::FullPath(path)).replaced(U"\\", U"/").lowercased();
        if (!path.ends_with(U"/")) {
            path += U"/";
        }
        return path;
    }

    static bool is_same_or_inside_directory(const String& path, const String& directory) {
        if (path.isEmpty() || directory.isEmpty()) {
            return false;
        }
        const String normalized_path = normalized_directory_compare_path(path);
        const String normalized_directory = normalized_directory_compare_path(directory);
        return normalized_path == normalized_directory || normalized_path.starts_with(normalized_directory);
    }

    Game_abstract game_with_filename(const Game_abstract& game, const String& filename_date) const {
        Game_abstract copied = game;
        copied.filename_date = filename_date;
        return copied;
    }

    void set_clipboard(bool cut) {
        clipboard_items.clear();
        clipboard_cut = cut;
        std::vector<int> folders(selected_folder_indices.begin(), selected_folder_indices.end());
        std::sort(folders.begin(), folders.end());
        for (int idx : folders) {
            if (idx < 0 || idx >= (int)folders_display.size()) {
                continue;
            }
            if (cut && is_protected_system_folder(folders_display[idx])) {
                continue;
            }
            Clipboard_item item;
            item.folder = true;
            item.name = folders_display[idx];
            item.source_path = get_base_dir() + item.name;
            clipboard_items.emplace_back(item);
        }

        std::vector<int> game_indices(selected_game_indices.begin(), selected_game_indices.end());
        std::sort(game_indices.begin(), game_indices.end());
        for (int idx : game_indices) {
            if (idx < 0 || idx >= (int)games.size()) {
                continue;
            }
            Clipboard_item item;
            item.folder = false;
            item.game = games[idx];
            item.name = games[idx].filename_date;
            item.source_path = get_base_dir() + games[idx].filename_date + U".json";
            clipboard_items.emplace_back(item);
        }
    }

    void paste_clipboard() {
        if (clipboard_items.empty()) {
            return;
        }
        const String target_base = get_base_dir();
        for (const Clipboard_item& item : clipboard_items) {
            if (item.folder) {
                paste_folder_item(item, target_base);
            } else {
                paste_game_item(item, target_base);
            }
        }
        if (clipboard_cut) {
            clipboard_items.clear();
            clipboard_cut = false;
        }
        enumerate_current_dir();
        load_games();
        clear_selection();
    }

    void paste_folder_item(const Clipboard_item& item, const String& target_base) {
        if (item.source_path.empty() || item.name.empty()) {
            return;
        }
        const String source = strip_trailing_separator(item.source_path);
        if (!FileSystem::IsDirectory(source)) {
            return;
        }

        FileSystem::CreateDirectories(target_base);
        String target = strip_trailing_separator(target_base + item.name);
        if (clipboard_cut) {
            if (FileSystem::FullPath(source) == FileSystem::FullPath(target)) {
                return;
            }
            if (FileSystem::Exists(target)) {
                target = strip_trailing_separator(make_unique_child_path(target_base, item.name, true));
                if (target.empty()) {
                    return;
                }
            }
            if (is_same_or_inside_directory(target, source)) {
                return;
            }
            if (move_path_fast(source, target) || FileSystem::Copy(source, target)) {
                if (FileSystem::Exists(source)) {
                    FileSystem::Remove(source, AllowUndo::No);
                }
            }
        } else {
            if (FileSystem::Exists(target) || FileSystem::FullPath(source) == FileSystem::FullPath(target)) {
                target = strip_trailing_separator(make_unique_child_path(target_base, item.name, true));
            }
            if (target.empty() || is_same_or_inside_directory(target, source)) {
                return;
            }
            FileSystem::Copy(source, target);
        }
    }

    void paste_game_item(const Clipboard_item& item, const String& target_base) {
        if (item.source_path.empty() || item.name.empty()) {
            return;
        }
        String target_name = item.game.filename_date;
        String target_json = target_base + target_name + U".json";
        if (!clipboard_cut && FileSystem::Exists(target_json)) {
            target_json = make_unique_child_path(target_base, target_name, false);
            if (target_json.empty()) {
                return;
            }
            target_name = FileSystem::BaseName(target_json).replaced(U".json", U"");
        }
        if (FileSystem::FullPath(item.source_path) == FileSystem::FullPath(target_json)) {
            return;
        }
        if (FileSystem::Copy(item.source_path, target_json)) {
            add_game_to_target_csv(game_with_filename(item.game, target_name), target_base);
            if (clipboard_cut) {
                remove_game_record_by_filename(item.game.filename_date, item.source_path);
                FileSystem::Remove(item.source_path);
            }
        }
    }

    void remove_game_record_by_filename(const String& filename_date, const String& source_json_path) {
        String source_base = FileSystem::ParentPath(source_json_path);
        if (!source_base.ends_with(U"/") && !source_base.ends_with(U"\\")) {
            source_base += U"/";
        }
        const String csv_path = source_base + U"summary.csv";
        CSV csv{ csv_path };
        CSV new_csv;
        for (int i = 0; i < (int)csv.rows(); ++i) {
            if (csv[i].size() >= 1 && csv[i][0] == filename_date) {
                continue;
            }
            for (size_t j = 0; j < csv[i].size(); ++j) {
                new_csv.write(csv[i][j]);
            }
            new_csv.newLine();
        }
        new_csv.save(csv_path);
    }

    String recycle_session_dir() const {
        const String root = Unicode::Widen(getData().directories.document_dir) + U"games/recycle_bin/";
        FileSystem::CreateDirectories(root);
        DateTime now = DateTime::Now();
        String stem = U"{:04}-{:02}-{:02} {:02}：{:02}"_fmt(now.year, now.month, now.day, now.hour, now.minute);
        String dir = root + stem + U"/";
        for (int suffix = 1; FileSystem::Exists(dir) && suffix < 1000; ++suffix) {
            dir = root + stem + U" (" + Format(suffix) + U")/";
        }
        ensure_summary_folder(dir);
        return dir;
    }

    bool recycle_bin_enabled() const {
        return getData().menu_elements.enable_recycle_bin;
    }

    bool should_permanently_delete() const {
        return in_recycle_bin() || !recycle_bin_enabled();
    }

    static bool move_path_fast(const String& source, const String& target) {
        if (source.empty() || target.empty() || !FileSystem::Exists(source) || FileSystem::Exists(target)) {
            return false;
        }
#ifdef _WIN32
        return MoveFileExW(source.toWstr().c_str(), target.toWstr().c_str(), MOVEFILE_COPY_ALLOWED) != 0;
#else
        std::string cmd = "mv \"" + source.narrow() + "\" \"" + target.narrow() + "\" >/dev/null";
        return system(cmd.c_str()) == 0;
#endif
    }

    void move_game_to_recycle(int idx, const String& recycle_dir) {
        if (idx < 0 || idx >= (int)games.size()) {
            return;
        }
        const Game_abstract game = games[idx];
        const String source_json = get_base_dir() + game.filename_date + U".json";
        String target_json = recycle_dir + game.filename_date + U".json";
        String target_name = game.filename_date;
        if (FileSystem::Exists(target_json)) {
            target_json = make_unique_child_path(recycle_dir, game.filename_date, false);
            if (target_json.empty()) {
                return;
            }
            target_name = FileSystem::BaseName(target_json).replaced(U".json", U"");
        }
        if (move_path_fast(source_json, target_json) || FileSystem::Copy(source_json, target_json)) {
            add_game_to_target_csv(game_with_filename(game, target_name), recycle_dir);
            remove_game_record_by_filename(game.filename_date, source_json);
            if (FileSystem::Exists(source_json)) {
                FileSystem::Remove(source_json);
            }
        }
    }

    void move_folder_to_recycle(int idx, const String& recycle_dir) {
        if (idx < 0 || idx >= (int)folders_display.size()) {
            return;
        }
        if (is_protected_system_folder(folders_display[idx])) {
            return;
        }
        const String source = get_base_dir() + folders_display[idx];
        String target = recycle_dir + folders_display[idx];
        if (FileSystem::FullPath(source) == FileSystem::FullPath(target)) {
            return;
        }
        if (FileSystem::Exists(target)) {
            target = make_unique_child_path(recycle_dir, folders_display[idx], true);
            if (target.empty()) {
                return;
            }
        }
        if (FileSystem::IsDirectory(source)) {
            if (move_path_fast(source, target)) {
                return;
            }
            if (FileSystem::Copy(source, target)) {
                FileSystem::Remove(source, AllowUndo::No);
            }
        }
    }

    void delete_selected_items() {
        if (!can_delete_selected_items()) {
            return;
        }
        const bool permanent = should_permanently_delete();
        String recycle_dir;
        if (!permanent) {
            recycle_dir = recycle_session_dir();
        }

        std::vector<int> game_indices(selected_game_indices.begin(), selected_game_indices.end());
        std::sort(game_indices.rbegin(), game_indices.rend());
        for (int idx : game_indices) {
            if (idx >= 0 && idx < (int)games.size()) {
                if (permanent) {
                    String json_path = get_base_dir() + games[idx].filename_date + U".json";
                    FileSystem::Remove(json_path);
                    remove_game_record_by_filename(games[idx].filename_date, json_path);
                } else {
                    move_game_to_recycle(idx, recycle_dir);
                }
            }
        }

        std::vector<int> folder_indices(selected_folder_indices.begin(), selected_folder_indices.end());
        std::sort(folder_indices.rbegin(), folder_indices.rend());
        for (int idx : folder_indices) {
            if (idx >= 0 && idx < (int)folders_display.size()) {
                if (is_protected_system_folder(folders_display[idx])) {
                    continue;
                }
                if (permanent) {
                    String folder_path = get_base_dir() + folders_display[idx];
                    if (FileSystem::IsDirectory(folder_path)) {
                        FileSystem::Remove(folder_path, AllowUndo::No);
                    }
                } else {
                    move_folder_to_recycle(idx, recycle_dir);
                }
            }
        }
        enumerate_current_dir();
        load_games();
        clear_selection();
    }
    
    // Move a game to a different folder (relative to current subfolder)
    void move_game_to_folder(int game_index, const std::string& target_folder) {
        if (game_index < 0 || game_index >= (int)games.size()) return;

        const Game_abstract game = games[game_index];
        String source_base = get_base_dir();
        String target_base = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!explorer_state.subfolder.empty()) {
            target_base += Unicode::Widen(explorer_state.subfolder) + U"/";
        }
        if (!target_folder.empty()) {
            target_base += Unicode::Widen(target_folder) + U"/";
        }

        if (!FileSystem::Exists(target_base)) {
            FileSystem::CreateDirectories(target_base);
        }

        const String source_json = source_base + game.filename_date + U".json";
        String target_name = game.filename_date;
        String target_json = target_base + target_name + U".json";
        if (FileSystem::Exists(target_json)) {
            target_json = make_unique_child_path(target_base, target_name, false);
            if (target_json.empty()) {
                return;
            }
            target_name = FileSystem::BaseName(target_json).replaced(U".json", U"");
        }
        if (FileSystem::FullPath(source_json) == FileSystem::FullPath(target_json)) {
            return;
        }
        if (!FileSystem::Exists(source_json) || !FileSystem::Copy(source_json, target_json)) {
            return;
        }
        FileSystem::Remove(source_json);

        remove_game_from_csv(game_index);
        add_game_to_target_csv(game_with_filename(game, target_name), target_base);

        load_games();
        init_scroll_manager();
    }

    // Move a game to an absolute folder path (from root)
    void move_game_to_absolute_folder(int game_index, const std::string& target_folder) {
        if (game_index < 0 || game_index >= (int)games.size()) return;

        const Game_abstract game = games[game_index];
        String source_base = get_base_dir();
        String target_base = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!target_folder.empty()) {
            target_base += Unicode::Widen(target_folder) + U"/";
        }

        if (!FileSystem::Exists(target_base)) {
            FileSystem::CreateDirectories(target_base);
        }

        const String source_json = source_base + game.filename_date + U".json";
        String target_name = game.filename_date;
        String target_json = target_base + target_name + U".json";
        if (FileSystem::Exists(target_json)) {
            target_json = make_unique_child_path(target_base, target_name, false);
            if (target_json.empty()) {
                return;
            }
            target_name = FileSystem::BaseName(target_json).replaced(U".json", U"");
        }
        if (FileSystem::FullPath(source_json) == FileSystem::FullPath(target_json)) {
            return;
        }
        if (!FileSystem::Exists(source_json) || !FileSystem::Copy(source_json, target_json)) {
            return;
        }
        FileSystem::Remove(source_json);

        remove_game_from_csv(game_index);
        add_game_to_target_csv(game_with_filename(game, target_name), target_base);

        load_games();
        init_scroll_manager();
    }
    
    // Move a folder to a different folder (relative to current subfolder)
    void move_folder_to_folder(const std::string& source_folder, const std::string& target_folder) {
        // Build source path: current subfolder + source_folder
        String source_path = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!explorer_state.subfolder.empty()) {
            source_path += Unicode::Widen(explorer_state.subfolder) + U"/";
        }
        source_path += Unicode::Widen(source_folder);
        
        // Build target parent path: current subfolder + target_folder
        String target_parent = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!explorer_state.subfolder.empty()) {
            target_parent += Unicode::Widen(explorer_state.subfolder) + U"/";
        }
        if (!target_folder.empty()) {
            target_parent += Unicode::Widen(target_folder) + U"/";
        }
        String target_path = target_parent + Unicode::Widen(source_folder);
        
        // Check if source and target are different
        if (source_path == target_path) {
            return;
        }
        
        // Check if target folder would create a circular reference
        String source_abs = FileSystem::FullPath(source_path);
        String target_abs = FileSystem::FullPath(target_parent);
        if (target_abs.starts_with(source_abs)) {
            return;
        }
        
        // Ensure target parent directory exists
        if (!FileSystem::Exists(target_parent)) {
            FileSystem::CreateDirectories(target_parent);
        }
        
        // Move the entire folder
        if (FileSystem::Exists(source_path) && !FileSystem::Exists(target_path)) {
            // Use system command for folder move (more reliable)
#ifdef _WIN32
            std::string cmd = "move \"" + source_path.narrow() + "\" \"" + target_path.narrow() + "\"";
#else
            std::string cmd = "mv \"" + source_path.narrow() + "\" \"" + target_path.narrow() + "\"";
#endif
            system(cmd.c_str());
            
            // Refresh current view
            enumerate_current_dir();
            load_games();
            init_scroll_manager();
        }
    }

    // Move a folder to an absolute folder path (from root)
    void move_folder_to_absolute_folder(const std::string& source_folder, const std::string& target_folder) {
        // Build source path: current subfolder + source_folder
        String source_path = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!explorer_state.subfolder.empty()) {
            source_path += Unicode::Widen(explorer_state.subfolder) + U"/";
        }
        source_path += Unicode::Widen(source_folder);
        
        // Build target parent path based on target_folder
        String target_parent = Unicode::Widen(getData().directories.document_dir) + U"games/";
        if (!target_folder.empty()) {
            target_parent += Unicode::Widen(target_folder) + U"/";
        }
        String target_path = target_parent + Unicode::Widen(source_folder);
        
        // Check if source and target are different
        if (source_path == target_path) {
            return;
        }
        
        // Check if target folder would create a circular reference
        String source_abs = FileSystem::FullPath(source_path);
        String target_abs = FileSystem::FullPath(target_parent);
        if (target_abs.starts_with(source_abs)) {
            return;
        }
        
        // Ensure target parent directory exists
        if (!FileSystem::Exists(target_parent)) {
            FileSystem::CreateDirectories(target_parent);
        }
        
        // Move the entire folder
        if (FileSystem::Exists(source_path) && !FileSystem::Exists(target_path)) {
            // Use system command for folder move (more reliable)
#ifdef _WIN32
            std::string cmd = "move \"" + source_path.narrow() + "\" \"" + target_path.narrow() + "\"";
#else
            std::string cmd = "mv \"" + source_path.narrow() + "\" \"" + target_path.narrow() + "\"";
#endif
            system(cmd.c_str());
            
            // Refresh current view
            enumerate_current_dir();
            load_games();
            init_scroll_manager();
        }
    }

    // Move a game to parent folder
    void move_game_to_parent(int game_index) {
        if (!explorer_state.has_parent()) return;  // Already at root
        
        // Get parent folder path
        std::string parent_folder = explorer_state.subfolder;
        if (!parent_folder.empty() && parent_folder.back() == '/') parent_folder.pop_back();
        size_t pos = parent_folder.find_last_of('/');
        if (pos == std::string::npos) parent_folder.clear();
        else parent_folder = parent_folder.substr(0, pos);
        
        move_game_to_absolute_folder(game_index, parent_folder);
    }
    
    // Move a folder to parent folder
    void move_folder_to_parent(const std::string& folder_name) {
        if (!explorer_state.has_parent()) return;  // Already at root
        
        // Get parent folder path
        std::string parent_folder = explorer_state.subfolder;
        if (!parent_folder.empty() && parent_folder.back() == '/') parent_folder.pop_back();
        size_t pos = parent_folder.find_last_of('/');
        if (pos == std::string::npos) parent_folder.clear();
        else parent_folder = parent_folder.substr(0, pos);
        
        move_folder_to_absolute_folder(folder_name, parent_folder);
    }
    
    // Remove game from current CSV
    void remove_game_from_csv(int game_index) {
        const String csv_path = get_base_dir() + U"summary.csv";
        CSV csv{ csv_path };
        CSV new_csv;
        
        int csv_row_to_remove = (int)games.size() - 1 - game_index;
        
        for (int i = 0; i < (int)csv.rows(); ++i) {
            if (i == csv_row_to_remove || csv[i].size() < 1) {
                continue;
            }
            for (size_t j = 0; j < csv[i].size(); ++j) {
                new_csv.write(csv[i][j]);
            }
            new_csv.newLine();
        }
        new_csv.save(csv_path);
    }
    
    void save_pending_games_here() {
        Game_library_save_request_info& request = getData().game_library_save_request_info;
        if (!is_save_request_active()) {
            return;
        }
        const String target_base = get_base_dir();
        ensure_summary_folder(target_base);

        int saved = 0;
        int skipped = 0;
        for (const Game_library_pending_save& pending : request.pending_games) {
            if (pending.history.empty() || current_dir_has_duplicate_history(target_base, pending.history)) {
                ++skipped;
                continue;
            }
            const String filename = make_unique_pending_filename(target_base, pending.filename_stem);
            game_save_helper::save_game_to_file(
                target_base,
                filename,
                pending.black_player_name,
                pending.white_player_name,
                pending.memo,
                pending.history,
                pending.game_date,
                pending.black_score,
                pending.white_score
            );
            ++saved;
        }

        String message;
        if (saved > 0) {
            message = U"Saved " + Format(saved) + U" game" + (saved == 1 ? U"" : U"s") + U" to Game Library.";
            if (skipped > 0) {
                message += U" Skipped " + Format(skipped) + U" duplicate" + (skipped == 1 ? U"" : U"s") + U".";
            }
        } else {
            message = U"No new games to save here.";
        }

        const String return_scene = request.return_scene.empty() ? U"Import_othello_quest" : request.return_scene;
        request.init();
        request.result_message = message;
        changeScene(return_scene, SCENE_FADE_TIME);
    }

    String make_unique_pending_filename(const String& target_base, const String& filename_stem) const {
        const String stem = filename_stem.empty() ? U"othello_quest_game" : filename_stem;
        String candidate = stem;
        for (int suffix = 1; FileSystem::Exists(target_base + candidate + U".json") && suffix < 1000; ++suffix) {
            candidate = stem + U"_" + Format(suffix);
        }
        return candidate;
    }

    bool current_dir_has_duplicate_history(const String& base_dir, const std::vector<History_elem>& history) const {
        if (history.empty() || !FileSystem::IsDirectory(base_dir)) {
            return false;
        }
        const String signature = game_library_history_signature(history);
        for (const auto& path : FileSystem::DirectoryContents(base_dir)) {
            if (FileSystem::IsDirectory(path) || FileSystem::Extension(path).lowercased() != U"json") {
                continue;
            }
            JSON json = JSON::Load(path);
            if (!json) {
                continue;
            }
            if (json_game_library_history_signature(json) == signature) {
                return true;
            }
        }
        return false;
    }

    static String game_library_history_signature(const std::vector<History_elem>& history) {
        String signature;
        for (const History_elem& elem : history) {
            signature += Format(elem.board.n_discs()) + U":";
            signature += Format(elem.board.player) + U":";
            signature += Format(elem.board.opponent) + U":";
            signature += Format(elem.player) + U";";
        }
        return signature;
    }

    static String json_game_library_history_signature(JSON json) {
        String signature;
        for (int n_discs = 4; n_discs <= HW2; ++n_discs) {
            const String key = Format(n_discs);
            if (!json[key].isObject()) {
                continue;
            }
            if (json[key][GAME_BOARD_PLAYER].getType() != JSONValueType::Number ||
                json[key][GAME_BOARD_OPPONENT].getType() != JSONValueType::Number ||
                json[key][GAME_PLAYER].getType() != JSONValueType::Number) {
                continue;
            }
            signature += key + U":";
            signature += Format(json[key][GAME_BOARD_PLAYER].get<uint64_t>()) + U":";
            signature += Format(json[key][GAME_BOARD_OPPONENT].get<uint64_t>()) + U":";
            signature += Format(json[key][GAME_PLAYER].get<int>()) + U";";
        }
        return signature;
    }

    // Add game to target folder's CSV
    void add_game_to_target_csv(const Game_abstract& game, const String& target_base) {
        String target_csv = target_base + U"summary.csv";
        CSV csv{ target_csv };
        
        // Create new CSV with existing data plus new game
        CSV new_csv;
        for (int i = 0; i < (int)csv.rows(); ++i) {
            if (csv[i].size() >= 1) {
                size_t cols = std::min(csv[i].size(), size_t(6));
                for (size_t j = 0; j < cols; ++j) {
                    new_csv.write(csv[i][j]);
                }
                // Add 7th column (game_date) if missing
                if (csv[i].size() < 7) {
                    String old_date = csv[i][0].substr(0, 10).replaced(U"_", U"-");
                    new_csv.write(old_date);
                } else {
                    new_csv.write(csv[i][6]);
                }
                new_csv.newLine();
            }
        }
        
        // Add new game entry
        new_csv.write(game.filename_date);
        new_csv.write(game.black_player);
        new_csv.write(game.white_player);
        new_csv.write(game.memo);
        new_csv.write(game.black_score == GAME_DISCS_UNDEFINED ? U"" : ToString(game.black_score));
        new_csv.write(game.white_score == GAME_DISCS_UNDEFINED ? U"" : ToString(game.white_score));
        new_csv.write(game.game_date);
        new_csv.newLine();
        
        new_csv.save(target_csv);
    }

    void persist_games_order_to_csv() {
        const String csv_path = get_base_dir() + U"summary.csv";
        CSV new_csv;
        for (int i = (int)games.size() - 1; i >= 0; --i) {
            const auto& game = games[i];
            new_csv.write(game.filename_date);
            new_csv.write(game.black_player);
            new_csv.write(game.white_player);
            new_csv.write(game.memo);
            new_csv.write(game.black_score == GAME_DISCS_UNDEFINED ? U"" : ToString(game.black_score));
            new_csv.write(game.white_score == GAME_DISCS_UNDEFINED ? U"" : ToString(game.white_score));
            new_csv.write(game.game_date);
            new_csv.newLine();
        }
        new_csv.save(csv_path);
    }
};


class Import_bitboard : public App::Scene {
private:
    Button single_back_button;
    Button back_button;
    Button import_button;
    TextEditState text_area[2];
    std::string player_string;
    std::string opponent_string;
    Radio_button player_radio;
    Board board;
    bool done;
    bool failed;

public:
    Import_bitboard(const InitData& init) : IScene{ init } {
        set_scene_ime_enabled(true);
        single_back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        import_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "import"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        Radio_button_element radio_button_elem;
        player_radio.init();
        radio_button_elem.init(X_CENTER, 70 + SCENE_ICON_WIDTH + 175, getData().fonts.font, 20, language.get("common", "black"), true);
        player_radio.push(radio_button_elem);
        radio_button_elem.init(X_CENTER, 70 + SCENE_ICON_WIDTH + 201, getData().fonts.font, 20, language.get("common", "white"), false);
        player_radio.push(radio_button_elem);
        done = false;
        failed = false;
        text_area[0].active = true;
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        const int icon_width = SCENE_ICON_WIDTH;
        getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
        getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
        int sy = 20 + icon_width + 50;
        if (!done) {
            const bool text_box_active_before = text_area[0].active || text_area[1].active;
            getData().fonts.font(language.get("in_out", "input_bitboard")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            const int text_area_y[2] = {sy + 40, sy + 90};
            constexpr int text_area_h = 40;
            constexpr int circle_radius = 15;
            for (int i = 0; i < 2; ++i) {
                text_box_with_ime_candidate_window(text_area[i], Vec2{X_CENTER - 300, text_area_y[i] + 2}, 600, SimpleGUI::PreferredTextAreaMaxChars);
                if (player_radio.checked == i) {
                    Circle(X_CENTER + 330, text_area_y[i] + text_area_h / 2, circle_radius).draw(getData().colors.black);
                } else {
                    Circle(X_CENTER + 330, text_area_y[i] + text_area_h / 2, circle_radius).draw(getData().colors.white);
                }
            }
            getData().fonts.font(language.get("in_out", "player")).draw(20, Arg::rightCenter(X_CENTER - 310, text_area_y[0] + text_area_h / 2), getData().colors.white);
            getData().fonts.font(language.get("in_out", "opponent")).draw(20, Arg::rightCenter(X_CENTER - 310, text_area_y[1] + text_area_h / 2), getData().colors.white);
            getData().fonts.font(language.get("in_out", "you_can_paste_with_ctrl_v")).draw(13, Arg::topCenter(X_CENTER, sy + 140), getData().colors.white);
            getData().fonts.font(language.get("in_out", "player")).draw(25, Arg::rightCenter(X_CENTER - 5, 70 + SCENE_ICON_WIDTH + 188), getData().colors.white);
            player_radio.draw();
            for (int i = 0; i < 2; ++i) {
                if (text_area[i].tabKey) {
                    text_area[i].active = false;
                    text_area[(i + 1) % 2].active = true;
                }
                std::string str = text_area[i].text.narrow();
                if (str.find("\t") != std::string::npos) {
                    text_area[i].active = false;
                    text_area[(i + 1) % 2].active = true;
                    int tab_place = str.find("\t");
                    std::string txt0;
                    for (int j = 0; j < tab_place; ++j) {
                        txt0 += str[j];
                    }
                    std::string txt1;
                    for (int j = tab_place + 1; j < (int)str.size(); ++j) {
                        txt1 += str[j];
                    }
                    text_area[i].text = Unicode::Widen(txt0);
                    text_area[i].cursorPos = text_area[i].text.size();
                    text_area[(i + 1) % 2].text += Unicode::Widen(txt1);
                    text_area[(i + 1) % 2].cursorPos = text_area[(i + 1) % 2].text.size();
                }
            }
            player_string = text_area[0].text.replaced(U"\r", U"").replaced(U"\n", U"").replaced(U" ", U"").replace(U"\t", U"").narrow();
            opponent_string = text_area[1].text.replaced(U"\r", U"").replaced(U"\n", U"").replaced(U" ", U"").replace(U"\t", U"").narrow();
            back_button.draw();
            import_button.draw();
            if (back_button.clicked() || gui_textarea_ime::escape_pressed_for_scene_change()) {
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
            if (import_button.clicked() || (KeyEnter.pressed() && !text_box_active_before)) {
                failed = import_bitboard_processing();
                done = true;
            }
        }
        else {
            if (!failed) {
                getData().graph_resources.init();
                History_elem history_elem;
                history_elem.player = player_radio.checked;
                history_elem.board = board;
                getData().graph_resources.nodes[0].emplace_back(history_elem);
                getData().graph_resources.n_discs = board.n_discs();
                getData().game_information.init();
                update_xot_start_n_discs(&getData().graph_resources);
                getData().graph_resources.need_init = false;
                getData().history_elem = getData().graph_resources.nodes[0].back();
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            } else {
                getData().fonts.font(language.get("in_out", "import_failed")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
                single_back_button.draw();
                if (single_back_button.clicked() || gui_textarea_ime::escape_pressed_for_scene_change()) {
                    changeScene(U"Main_scene", SCENE_FADE_TIME);
                }
            }
        }
    }

    void draw() const override {

    }

private:
    bool import_bitboard_processing() {
        if (player_string[0] == '0' && player_string[1] == 'x') {
            player_string = player_string.substr(2);
        }
        if (opponent_string[0] == '0' && opponent_string[1] == 'x') {
            opponent_string = opponent_string.substr(2);
        }
        while (player_string.size() < 16) {
            player_string = "0" + player_string;
        }
        while (opponent_string.size() < 16) {
            opponent_string = "0" + opponent_string;
        }
        board.player = 0;
        board.opponent = 0;
        for (int i = 0; i < 16; ++i) {
            player_string[i] |= 0x20; // lower case
            opponent_string[i] |= 0x20; // lower case
            if (player_string[i] < '0' || ('9' < player_string[i] && player_string[i] < 'a') || 'f' < player_string[i]) { // out of range
                std::cerr << "player out of range" << std::endl;
                return true;
            }
            if (opponent_string[i] < '0' || ('9' < opponent_string[i] && opponent_string[i] < 'a') || 'f' < opponent_string[i]) { // out of range
                std::cerr << "opponent out of range" << std::endl;
                return true;
            }
            int p_int = player_string[i] % 87 % 48; // 0-9, a-f hex to decimal
            int o_int = opponent_string[i] % 87 % 48; // 0-9, a-f hex to decimal
            if (p_int & o_int) { // 2 discs on same square
                std::cerr << "both discs on same square" << std::endl;
                return true;
            }
            for (int j = 0; j < 4; ++j) {
                int idx = HW2_M1 - (i * 4 + 3 - j);
                board.player |= (uint64_t)(p_int & 1) << idx;
                board.opponent |= (uint64_t)(o_int & 1) << idx;
                p_int /= 2;
                o_int /= 2;
            }
        }
        if (!board.is_end() && board.get_legal() == 0) {
            board.pass();
        }
        return false;
    }
};

// Load game from JSON and prepare for display
// Returns true on success, false on failure
inline bool load_game_from_json(
    Common_resources& data,
    Opening& opening,
    const String& json_path,
    const String& game_date,
    const std::string& subfolder
) {
    JSON game_json = JSON::Load(json_path);
    if (!game_json) {
        std::cerr << "Failed to load game JSON: " << json_path.narrow() << std::endl;
        return false;
    }
    
    // Load game information
    if (game_json[GAME_BLACK_PLAYER].getType() == JSONValueType::String) {
        data.game_information.black_player_name = game_json[GAME_BLACK_PLAYER].getString();
    }
    if (game_json[GAME_WHITE_PLAYER].getType() == JSONValueType::String) {
        data.game_information.white_player_name = game_json[GAME_WHITE_PLAYER].getString();
    }
    if (game_json[GAME_MEMO].getType() == JSONValueType::String) {
        data.game_information.memo = game_json[GAME_MEMO].getString();
    }
    // Load date field
    if (game_json[U"date"].getType() == JSONValueType::String) {
        data.game_information.date = game_json[U"date"].getString();
    } else {
        data.game_information.date = game_date.substr(0, 10).replaced(U"_", U"-");
    }
    
    // Mark that a specific game has been loaded and store its location
    data.game_information.is_game_loaded = true;
    data.game_editor_info.game_date = game_date;
    data.game_editor_info.subfolder = subfolder;
    
    // Load history
    data.graph_resources.nodes[GRAPH_MODE_NORMAL].clear();
    data.graph_resources.nodes[GRAPH_MODE_INSPECT].clear();
    for (int n_discs = 4; n_discs <= HW2; ++n_discs) {
        String n_discs_str = Format(n_discs);
        History_elem history_elem;
        bool error_found = false;
        if (game_json[n_discs_str][GAME_BOARD_PLAYER].getType() == JSONValueType::Number) {
            history_elem.board.player = game_json[n_discs_str][GAME_BOARD_PLAYER].get<uint64_t>();
        } else {
            error_found = true;
        }
        if (game_json[n_discs_str][GAME_BOARD_OPPONENT].getType() == JSONValueType::Number) {
            history_elem.board.opponent = game_json[n_discs_str][GAME_BOARD_OPPONENT].get<uint64_t>();
        } else {
            error_found = true;
        }
        if (game_json[n_discs_str][GAME_LEVEL].getType() == JSONValueType::Number) {
            history_elem.level = game_json[n_discs_str][GAME_LEVEL].get<int>();
        } else {
            error_found = true;
        }
        if (game_json[n_discs_str][GAME_PLAYER].getType() == JSONValueType::Number) {
            history_elem.player = game_json[n_discs_str][GAME_PLAYER].get<int>();
        } else {
            error_found = true;
        }
        if (game_json[n_discs_str][GAME_POLICY].getType() == JSONValueType::Number) {
            history_elem.policy = game_json[n_discs_str][GAME_POLICY].get<int>();
        } else {
            error_found = true;
        }
        if (game_json[n_discs_str][GAME_NEXT_POLICY].getType() == JSONValueType::Number) {
            history_elem.next_policy = game_json[n_discs_str][GAME_NEXT_POLICY].get<int>();
        } else {
            error_found = true;
        }
        if (game_json[n_discs_str][GAME_VALUE].getType() == JSONValueType::Number) {
            history_elem.v = game_json[n_discs_str][GAME_VALUE].get<int>();
        } else {
            error_found = true;
        }
        if (game_json[n_discs_str][GAME_BLACK_TIME_MSEC].getType() == JSONValueType::Number) {
            history_elem.black_time_msec = game_json[n_discs_str][GAME_BLACK_TIME_MSEC].get<int64_t>();
        } else {
            history_elem.black_time_msec = 0;
        }
        if (game_json[n_discs_str][GAME_WHITE_TIME_MSEC].getType() == JSONValueType::Number) {
            history_elem.white_time_msec = game_json[n_discs_str][GAME_WHITE_TIME_MSEC].get<int64_t>();
        } else {
            history_elem.white_time_msec = 0;
        }
        if (game_json[n_discs_str][GAME_IS_RANDOM_GENERATED_POSITION].getType() == JSONValueType::Bool) {
            history_elem.is_random_generated_position = game_json[n_discs_str][GAME_IS_RANDOM_GENERATED_POSITION].get<bool>();
        } else {
            history_elem.is_random_generated_position = false;
        }
        if (!error_found) {
            data.graph_resources.nodes[GRAPH_MODE_NORMAL].emplace_back(history_elem);
        }
    }
    
    // Add opening names
    std::string opening_name, n_opening_name;
    for (int i = 0; i < (int)data.graph_resources.nodes[GRAPH_MODE_NORMAL].size(); ++i) {
        n_opening_name.clear();
        n_opening_name = opening.get(data.graph_resources.nodes[GRAPH_MODE_NORMAL][i].board, data.graph_resources.nodes[GRAPH_MODE_NORMAL][i].player ^ 1);
        if (n_opening_name.size()) {
            opening_name = n_opening_name;
        }
        data.graph_resources.nodes[GRAPH_MODE_NORMAL][i].opening_name = opening_name;
    }
    
    if (data.graph_resources.nodes[GRAPH_MODE_NORMAL].empty()) {
        std::cerr << "Game JSON contains no valid history nodes: " << json_path.narrow() << std::endl;
        return false;
    }

    // Set up final state
    data.graph_resources.n_discs = data.graph_resources.nodes[GRAPH_MODE_NORMAL].back().board.n_discs();
    update_xot_start_n_discs(&data.graph_resources);
    data.graph_resources.need_init = false;
    data.history_elem = data.graph_resources.nodes[GRAPH_MODE_NORMAL].back();
    
    return true;
}
