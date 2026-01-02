/*
    Egaroucid Project

    @file main_scene.hpp
        Main scene for GUI
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"
#include "draw.hpp"
#include "screen_shot.hpp"

constexpr double HINT_PRIORITY = 0.49;

bool compare_value_cell(std::pair<int, int>& a, std::pair<int, int>& b) {
    return a.first > b.first;
}

bool compare_hint_info(Hint_info& a, Hint_info& b) {
    if (a.type != b.type) {
        return a.type > b.type;
    }
    return a.value > b.value;
}

Umigame_result get_umigame(Board board, int player, int depth) {
    return calculate_umigame(&board, player, depth);
}

int get_book_accuracy(Board board) {
    return calculate_book_accuracy(&board);
}

class Main_scene : public App::Scene {
private:
    Graph graph;
    Move_board_button_status move_board_button_status;
    AI_status ai_status;
    Button start_game_button;
    Button pass_button;
    bool need_start_game_button;
    Umigame_status umigame_status;
    Book_accuracy_status book_accuracy_status;
    bool changing_scene;
    int taking_screen_shot_state;
    bool pausing_in_pass;
    bool putting_1_move_by_ai;
    int umigame_value_depth_before;
    String shortcut_key;
    String shortcut_key_pressed;
public:
    std::string principal_variation;

public:
    void init_main_scene() {
        // for (char c = 0; c < 127; ++c) {
        //     String s = Unicode::Widen(std::string(1, c));
        //     double size = (double)getData().fonts.font(s).region(100, Point{ 0, 0 }).w / 100.0;
        //     std::cerr << (int)c << " " << c << " " << size << std::endl;
        // }
        // for (char c = 0; c < 127; ++c) {
        //     String s = Unicode::Widen(std::string(1, c));
        //     double size = (double)getData().fonts.font(s).region(100, Point{ 0, 0 }).w / 100.0;
        //     std::cerr << size << ", ";
        // }
        // std::cerr << std::endl;
        // uint64_t strt = tim();
        std::cerr << "main scene loading" << std::endl;
        getData().menu = create_menu(&getData().menu_elements, &getData().resources, getData().fonts.font, getData().settings.lang_name);
        graph.sx = GRAPH_SX;
        graph.sy = GRAPH_SY;
        graph.size_x = GRAPH_WIDTH;
        graph.size_y = GRAPH_HEIGHT;
        if (getData().graph_resources.need_init) {
            getData().game_information.init();
            getData().graph_resources.init();
            getData().graph_resources.nodes[getData().graph_resources.branch].emplace_back(getData().history_elem);
        }
        start_game_button.init(START_GAME_BUTTON_SX, START_GAME_BUTTON_SY, START_GAME_BUTTON_WIDTH, START_GAME_BUTTON_HEIGHT, START_GAME_BUTTON_RADIUS, language.get("play", "start_game"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
        pass_button.init(PASS_BUTTON_SX, PASS_BUTTON_SY, PASS_BUTTON_WIDTH, PASS_BUTTON_HEIGHT, PASS_BUTTON_RADIUS, language.get("play", "pass"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
        need_start_game_button_calculation();
        changing_scene = false;
        taking_screen_shot_state = 0;
        pausing_in_pass = false;
        putting_1_move_by_ai = false;
        umigame_value_depth_before = 0;
        shortcut_key = SHORTCUT_KEY_UNDEFINED;
        shortcut_key_pressed = SHORTCUT_KEY_UNDEFINED;
        std::cerr << "main scene loaded" << std::endl;
        // std::cerr << tim() - strt << " ms" << std::endl;
    }

    Main_scene(const InitData& init) : IScene{ init } {
        init_main_scene();
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            stop_calculating();
            changing_scene = true;
            changeScene(U"Close", SCENE_FADE_TIME);
            return;
        }

        Scene::SetBackground(getData().colors.green);

        // multi threading
        if (getData().menu_elements.n_threads - 1 != thread_pool.size()) {
            stop_calculating();
            thread_pool.resize(getData().menu_elements.n_threads - 1);
            std::cerr << "thread pool resized to " << thread_pool.size() << std::endl;
            resume_calculating();
        }

        // hash resize
#if USE_CHANGEABLE_HASH_LEVEL
        if (getData().menu_elements.hash_level != global_hash_level) {
            int n_hash_level = getData().menu_elements.hash_level;
            if (n_hash_level != global_hash_level) {
                stop_calculating();
                if (!hash_resize(global_hash_level, n_hash_level, true)) {
                    std::cerr << "hash resize failed. use default level" << std::endl;
                    hash_resize(DEFAULT_HASH_LEVEL, DEFAULT_HASH_LEVEL, true);
                    getData().menu_elements.hash_level = DEFAULT_HASH_LEVEL;
                    global_hash_level = DEFAULT_HASH_LEVEL;
                } else {
                    global_hash_level = n_hash_level;
                }
                resume_calculating();
            }
        }
#endif

        // init
        getData().graph_resources.delta = 0;

        // shortcut
        shortcut_keys.check_shortcut_key(&shortcut_key, &shortcut_key_pressed);
        // if (shortcut_key != SHORTCUT_KEY_UNDEFINED) {
        //     std::cerr << "shortcut key found: " << shortcut_key.narrow() << std::endl;
        // }

        // opening
        update_opening();

        // analyze
        if (ai_status.analyzing) {
            analyze_get_task();
        }

        // random board generator
        if (ai_status.random_board_generator_calculating) {
            check_random_board_generater();
        }

        // move
        bool ai_should_move =
            !need_start_game_button &&
            !getData().history_elem.board.is_end() && 
            getData().graph_resources.branch == GRAPH_MODE_NORMAL &&
            ((getData().history_elem.player == BLACK && getData().menu_elements.ai_put_black) || (getData().history_elem.player == WHITE && getData().menu_elements.ai_put_white)) &&
            getData().history_elem.board.n_discs() == getData().graph_resources.nodes[GRAPH_MODE_NORMAL][getData().graph_resources.nodes[GRAPH_MODE_NORMAL].size() - 1].board.n_discs() && 
            !pausing_in_pass;
        ai_should_move |= 
            !getData().history_elem.board.is_end() && 
            putting_1_move_by_ai;
        bool ignore_move = 
            (getData().book_information.changing != BOOK_CHANGE_NO_CELL) || 
            ai_status.analyzing || 
            ai_status.random_board_generator_calculating;
        if (need_start_game_button) {
            need_start_game_button_calculation();
            if (getData().menu.active()) {
                start_game_button.disable_notransparent();
            } else {
                start_game_button.enable();
            }
            start_game_button.draw();
            if (!getData().menu.active() && (start_game_button.clicked() || shortcut_key == U"start_game")) {
                need_start_game_button = false;
                stop_calculating();
                resume_calculating();
            }
        }
        if (pausing_in_pass) {
            bool ai_to_move = 
                (getData().history_elem.player == BLACK && getData().menu_elements.ai_put_black) || 
                (getData().history_elem.player == WHITE && getData().menu_elements.ai_put_white);
            if (!ai_to_move) {
                pausing_in_pass = false;
            } else {
                if (getData().menu.active()) {
                    pass_button.disable_notransparent();
                } else {
                    pass_button.enable();
                }
                pass_button.draw();
                if (!getData().menu.active() && (pass_button.clicked() || (pass_button.is_enabled() && shortcut_key == U"pass"))) {
                    pausing_in_pass = false;
                }
            }
        }
        if (!ignore_move) {
            if (ai_should_move) {
                ai_move();
            } else if (!getData().menu.active() && !need_start_game_button && !pausing_in_pass) {
                interact_move();
            }
        }

        // graph move
        bool graph_interact_ignore = ai_status.analyzing || ai_status.random_board_generator_calculating || ai_should_move;
        if (!ignore_move && !graph_interact_ignore && !getData().menu.active()) {
            interact_graph();
        }
        update_n_discs();

        // local strategy drawing
        bool local_strategy_ignore = ai_should_move || ai_status.analyzing || ai_status.random_board_generator_calculating || need_start_game_button || pausing_in_pass || changing_scene;
        if (ai_status.local_strategy_done_level > 0 && getData().menu_elements.show_ai_focus && !local_strategy_ignore) {
            draw_local_strategy();
        }

        // board drawing
        bool use_transcript_board = getData().menu_elements.show_play_ordering && getData().menu_elements.play_ordering_transcript_format;
        if (use_transcript_board) {
            draw_transcript_board(getData().fonts, getData().colors, getData().history_elem, getData().graph_resources, false);
        } else {
            draw_board(getData().fonts, getData().colors, getData().history_elem);
        }

        // book modifying by right-clicking
        bool changing_book_by_right_click = false;
        if (!ai_should_move && !need_start_game_button && getData().menu_elements.change_book_by_right_click) {
            change_book_by_right_click();
            changing_book_by_right_click = (getData().book_information.changing != BOOK_CHANGE_NO_CELL);
        }

        // draw on discs
        // last move drawing
        if (getData().menu_elements.show_last_move) {
            draw_last_move();
        }

        // stable drawing
        if (getData().menu_elements.show_stable_discs) {
            draw_stable_discs();
        }

        // play ordering drawing
        if (getData().menu_elements.show_play_ordering) {
            draw_play_ordering();
        }

        // draw on cells
        // next move drawing
        if (getData().menu_elements.show_next_move) {
            draw_next_move();
        }

        uint64_t legal_ignore = 0ULL;

        // hint calculating
        bool hint_ignore = ai_should_move || ai_status.analyzing || ai_status.random_board_generator_calculating || (need_start_game_button && !getData().menu_elements.show_value_when_ai_calculating) || pausing_in_pass || changing_scene;
        bool show_value_ai_turn = ai_should_move && getData().menu_elements.show_value_when_ai_calculating && getData().menu_elements.use_disc_hint;
        if (!hint_ignore || show_value_ai_turn) {
            if (getData().menu_elements.use_disc_hint) {
                if ((ai_status.hint_calculating || ai_status.hint_calculated) && getData().menu_elements.n_disc_hint > ai_status.n_hint_display) {
                    stop_calculating();
                    resume_calculating();
                }
                if (!ai_status.hint_calculating && !ai_status.hint_calculated) {
                    hint_calculate();
                } else {
                    try_hint_get();
                    legal_ignore = draw_hint(getData().menu_elements.use_book && getData().menu_elements.show_book_accuracy);
                }
            }
        }
        //  else if (show_value_ai_turn) {
        //     legal_ignore = draw_hint_tt(getData().menu_elements.use_book, getData().menu_elements.use_book && getData().menu_elements.show_book_accuracy);
        // }

        // principal variation calculating
        // bool pv_ignore = ai_should_move || ai_status.analyzing || need_start_game_button || changing_scene;
        bool pv_ignore = need_start_game_button || changing_scene;
        if (!pv_ignore && getData().menu_elements.show_principal_variation) {
            if (!ai_status.pv_calculating && !ai_status.pv_calculated) {
                pv_calculate();
            } else if (ai_status.pv_calculating && !ai_status.pv_calculated) {
                try_pv_get();
            }
        }

        // local strategy calculating
        if (getData().menu_elements.show_ai_focus && !local_strategy_ignore) {
            if (!ai_status.local_strategy_calculating && !ai_status.local_strategy_calculated) {
                local_strategy_calculate();
            } else if (ai_status.local_strategy_calculating && !ai_status.local_strategy_calculated) {
                try_local_strategy_get();
            }
        }
        if (ai_status.analyzing || ai_status.random_board_generator_calculating) {
            principal_variation = "";
        }

        // local strategy policy calculating
        if (getData().menu_elements.show_ai_focus && !local_strategy_ignore) {
            if (!ai_status.local_strategy_policy_calculating && !ai_status.local_strategy_policy_calculated) {
                local_strategy_policy_calculate();
            } else if (ai_status.local_strategy_policy_calculating && !ai_status.local_strategy_policy_calculated) {
                try_local_strategy_policy_get();
            }
        }

        // legal drawing
        if (getData().menu_elements.show_legal && !pausing_in_pass) {
            draw_legal(legal_ignore);
        }

        // book information drawing
        if (getData().menu_elements.use_book) {
            // book accuracy drawing
            if (getData().menu_elements.show_book_accuracy && (!hint_ignore || show_value_ai_turn)) {
                draw_book_n_lines(legal_ignore);
                if (book_accuracy_status.book_accuracy_calculated) {
                    draw_book_accuracy(legal_ignore);
                } else {
                    calculate_book_accuracy();
                }
            }

            // umigame calculating / drawing
            if (getData().menu_elements.use_umigame_value && (!hint_ignore || show_value_ai_turn)) {
                if (umigame_value_depth_before != getData().menu_elements.umigame_value_depth) {
                    umigame_status.umigame_calculated = false;
                    umigame_status.umigame_calculating = false;
                    umigame.delete_all();
                    umigame_value_depth_before = getData().menu_elements.umigame_value_depth;
                }
                if (umigame_status.umigame_calculated) {
                    draw_umigame(legal_ignore);
                } else {
                    calculate_umigame();
                }
            }
        }

        // graph drawing
        graph.draw(getData().graph_resources.nodes[0], getData().graph_resources.nodes[1], getData().graph_resources.n_discs, getData().menu_elements.show_graph, getData().menu_elements.level, getData().fonts.font, getData().menu_elements.change_color_type, getData().menu_elements.show_graph_sum_of_loss, getData().menu_elements.show_endgame_error);

        // info drawing
        int playing_mode = PLAYING_MODE_NONE;
        if ((getData().menu_elements.ai_put_black || getData().menu_elements.ai_put_white) && !need_start_game_button) {
            if (getData().history_elem == getData().graph_resources.nodes[0].back()) {
                playing_mode = PLAYING_MODE_PLAYING;
            } else {
                playing_mode = PLAYING_MODE_ANALYZING;
            }
        }
        draw_info(getData().colors, getData().history_elem, getData().fonts, getData().menu_elements, pausing_in_pass, principal_variation, getData().forced_openings.is_forced(getData().history_elem.board), playing_mode);

        // draw local strategy policy
        if (ai_status.local_strategy_policy_done_level > 0 && getData().menu_elements.show_ai_focus && !local_strategy_ignore && !getData().menu.active()) {
            draw_local_strategy_policy();
        }

        // opening on cell drawing
        if (getData().menu_elements.show_opening_on_cell && !getData().menu.active()) {
            draw_opening_on_cell();
        }

        // menu
        bool draw_menu_flag = (taking_screen_shot_state == 0) && !changing_book_by_right_click;
        if (draw_menu_flag) {
            getData().menu.draw();
            menu_game();
            menu_setting();
            menu_display();
            menu_manipulate();
            menu_in_out();
            menu_book();
            menu_help();
            menu_language();
        }

        // laser pointer
        if (getData().menu_elements.show_laser_pointer && Cursor::Pos().y >= 0) {
            Cursor::RequestStyle(CursorStyle::Hidden);
            getData().resources.laser_pointer.scaled(30.0 / getData().resources.laser_pointer.width()).drawAt(Cursor::Pos());
        }

        // screen shot
        if (taking_screen_shot_state == 1) {
            taking_screen_shot_state = 2;
        } else if (taking_screen_shot_state == 2) {
            ScreenCapture::RequestCurrentFrame();
            taking_screen_shot_state = 3;
        } else if (taking_screen_shot_state == 3) {
            std::string transcript = get_transcript(getData().graph_resources, getData().history_elem);
            take_screen_shot(getData().window_state.window_scale, getData().user_settings.screenshot_saving_dir, transcript);
            taking_screen_shot_state = 0;
        }
    }

    void draw() const override {

    }

private:
    void reset_ai() {
        ai_status.ai_thinking = false;
        if (ai_status.ai_future.valid()) {
            ai_status.ai_future.get();
        }
    }

    void reset_hint() {
        ai_status.hint_calculating = false;
        ai_status.hint_calculated = false;
        if (ai_status.hint_future.valid()) {
            ai_status.hint_future.get();
        }
    }

    void reset_pv() {
        ai_status.pv_calculating = false;
        ai_status.pv_calculated = false;
        if (ai_status.pv_future.valid()) {
            ai_status.pv_future.get();
        }
        principal_variation = "";
    }

    void reset_local_strategy() {
        ai_status.local_strategy_calculating = false;
        ai_status.local_strategy_calculated = false;
        if (ai_status.local_strategy_future.valid()) {
            ai_status.local_strategy_future.get();
        }
    }

    void reset_local_strategy_policy() {
        ai_status.local_strategy_policy_calculating = false;
        ai_status.local_strategy_policy_calculated = false;
        if (ai_status.local_strategy_policy_future.valid()) {
            ai_status.local_strategy_policy_future.get();
        }
    }

    void reset_analyze() {
        ai_status.analyzing = false;
        ai_status.analyze_task_stack.clear();
        for (int i = 0; i < ANALYZE_SIZE; ++i) {
            if (ai_status.analyze_future[i].valid()) {
                ai_status.analyze_future[i].get();
            }
        }
    }

    void reset_book_additional_features() {
        umigame_status.umigame_calculated = false;
        umigame_status.umigame_calculating = false;
        book_accuracy_status.book_accuracy_calculated = false;
        book_accuracy_status.book_accuracy_calculating = false;
    }

    void reset_random_board_generator() {
        ai_status.random_board_generator_calculating = false;
        if (ai_status.random_board_generator_future.valid()) {
            ai_status.random_board_generator_future.get();
        }
    }

    void stop_calculating() {
        std::cerr << "terminating calculation" << std::endl;
        global_searching = false;
        if (ai_status.ai_future.valid()) {
            std::cerr << "terminating AI" << std::endl;
            ai_status.ai_future.get();
        }
        if (ai_status.hint_future.valid()) {
            std::cerr << "terminating hint" << std::endl;
            ai_status.hint_future.get();
        }
        if (ai_status.pv_future.valid()) {
            std::cerr << "terminating pv" << std::endl;
            ai_status.pv_future.get();
        }
        std::cerr << "terminating analyze AI" << std::endl;
        for (int i = 0; i < ANALYZE_SIZE; ++i) {
            if (ai_status.analyze_future[i].valid()) {
                ai_status.analyze_future[i].get();
            }
        }
        std::cerr << "terminating umigame value" << std::endl;
        for (int i = 0; i < HW2; ++i) {
            if (umigame_status.umigame_future[i].valid()) {
                umigame_status.umigame_future[i].get();
            }
        }
        std::cerr << "terminating book accuracy" << std::endl;
        for (int i = 0; i < HW2; ++i) {
            if (book_accuracy_status.book_accuracy_future[i].valid()) {
                book_accuracy_status.book_accuracy_future[i].get();
            }
        }
        std::cerr << "calculation terminated" << std::endl;
        reset_ai();
        reset_hint();
        reset_pv();
        reset_local_strategy();
        reset_local_strategy_policy();
        reset_analyze();
        reset_book_additional_features();
        reset_random_board_generator();
        std::cerr << "reset all calculations" << std::endl;
    }

    void resume_calculating() {
        global_searching = true;
    }

    void menu_game() {
        if (getData().menu_elements.start_game || shortcut_key == U"new_game") {
            stop_calculating();
            getData().history_elem.reset();
            getData().graph_resources.init();
            getData().graph_resources.nodes[getData().graph_resources.branch].emplace_back(getData().history_elem);
            getData().game_information.init();
            getData().menu_elements.ai_put_black = false;
            getData().menu_elements.ai_put_white = false;
            resume_calculating();
            need_start_game_button_calculation();
            pausing_in_pass = false;
        }
        if (getData().menu_elements.start_game_human_black || shortcut_key == U"new_game_human_black") {
            stop_calculating();
            getData().history_elem.reset();
            getData().graph_resources.init();
            getData().graph_resources.nodes[getData().graph_resources.branch].emplace_back(getData().history_elem);
            getData().game_information.init();
            getData().menu_elements.ai_put_black = false;
            getData().menu_elements.ai_put_white = true;
            resume_calculating();
            need_start_game_button_calculation();
            pausing_in_pass = false;
        }
        if (getData().menu_elements.start_game_human_white || shortcut_key == U"new_game_human_white") {
            stop_calculating();
            getData().history_elem.reset();
            getData().graph_resources.init();
            getData().graph_resources.nodes[getData().graph_resources.branch].emplace_back(getData().history_elem);
            getData().game_information.init();
            getData().menu_elements.ai_put_black = true;
            getData().menu_elements.ai_put_white = false;
            resume_calculating();
            need_start_game_button_calculation();
            pausing_in_pass = false;
        }
        if (getData().menu_elements.start_selfplay || shortcut_key == U"new_selfplay") {
            stop_calculating();
            getData().history_elem.reset();
            getData().graph_resources.init();
            getData().graph_resources.nodes[getData().graph_resources.branch].emplace_back(getData().history_elem);
            getData().game_information.init();
            getData().menu_elements.ai_put_black = true;
            getData().menu_elements.ai_put_white = true;
            resume_calculating();
            need_start_game_button_calculation();
            pausing_in_pass = false;
        }
        if ((getData().menu_elements.analyze || shortcut_key == U"analyze") && !ai_status.ai_thinking && !ai_status.analyzing) {
            stop_calculating();
            resume_calculating();
            init_analyze();
        }
        if (getData().menu_elements.game_information || shortcut_key == U"game_information") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Game_information_scene", SCENE_FADE_TIME);
            return;
        }
    }

    void menu_setting() {
        if (shortcut_key == U"use_book") {
            getData().menu_elements.use_book = !getData().menu_elements.use_book;
        }
        if (shortcut_key == U"accept_ai_loss") {
            getData().menu_elements.accept_ai_loss = !getData().menu_elements.accept_ai_loss;
        }
        if (shortcut_key == U"ai_put_black") {
            getData().menu_elements.ai_put_black = !getData().menu_elements.ai_put_black;
        }
        if (shortcut_key == U"ai_put_white") {
            getData().menu_elements.ai_put_white = !getData().menu_elements.ai_put_white;
        }
        if (shortcut_key == U"pause_when_pass") {
            getData().menu_elements.pause_when_pass = !getData().menu_elements.pause_when_pass;
        }
        if (shortcut_key == U"force_specified_openings") {
            getData().menu_elements.force_specified_openings = !getData().menu_elements.force_specified_openings;
        }
        if (getData().menu_elements.opening_setting || shortcut_key == U"opening_setting") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Opening_setting", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.shortcut_key_setting || shortcut_key == U"shortcut_key_setting") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Shortcut_key_setting", SCENE_FADE_TIME);
            return;
        }
    }

    void menu_display() {
        // on cells
        if (shortcut_key == U"show_legal") {
            getData().menu_elements.show_legal = !getData().menu_elements.show_legal;
        }
        if (shortcut_key == U"show_disc_hint") {
            getData().menu_elements.use_disc_hint = !getData().menu_elements.use_disc_hint;
        }
        if (shortcut_key == U"show_hint_level") {
            getData().menu_elements.show_hint_level = !getData().menu_elements.show_hint_level;
        }
        if (shortcut_key == U"show_value_when_ai_calculating") {
            getData().menu_elements.show_value_when_ai_calculating = !getData().menu_elements.show_value_when_ai_calculating;
        }
        if (shortcut_key == U"show_umigame_value") {
            getData().menu_elements.use_umigame_value = !getData().menu_elements.use_umigame_value;
        }
        if (shortcut_key == U"show_opening_on_cell") {
            getData().menu_elements.show_opening_on_cell = !getData().menu_elements.show_opening_on_cell;
        }
        if (shortcut_key == U"show_next_move") {
            getData().menu_elements.show_next_move = !getData().menu_elements.show_next_move;
        }
        if (shortcut_key == U"show_book_accuracy") {
            getData().menu_elements.show_book_accuracy = !getData().menu_elements.show_book_accuracy;
        }
        if (shortcut_key == U"hint_colorize") {
            getData().menu_elements.hint_colorize = !getData().menu_elements.hint_colorize;
        }
        // on discs
        if (shortcut_key == U"show_last_move") {
            getData().menu_elements.show_last_move = !getData().menu_elements.show_last_move;
        }
        if (shortcut_key == U"show_stable_discs") {
            getData().menu_elements.show_stable_discs = !getData().menu_elements.show_stable_discs;
        }
        if (shortcut_key == U"show_play_ordering") {
            getData().menu_elements.show_play_ordering = !getData().menu_elements.show_play_ordering;
        }
        if (shortcut_key == U"play_ordering_board_format") {
            getData().menu_elements.play_ordering_board_format = true;
            getData().menu_elements.play_ordering_transcript_format = false;
        }
        if (shortcut_key == U"play_ordering_transcript_format") {
            getData().menu_elements.play_ordering_board_format = false;
            getData().menu_elements.play_ordering_transcript_format = true;
        }
        // info area
        if (shortcut_key == U"show_opening_name") {
            getData().menu_elements.show_opening_name = !getData().menu_elements.show_opening_name;
        }
        if (shortcut_key == U"show_principal_variation") {
            getData().menu_elements.show_principal_variation = !getData().menu_elements.show_principal_variation;
        }
        // graph area
        if (shortcut_key == U"show_graph") {
            getData().menu_elements.show_graph = !getData().menu_elements.show_graph;
        }
        if (shortcut_key == U"show_graph_value") {
            getData().menu_elements.show_graph_value = true;
            getData().menu_elements.show_graph_sum_of_loss = false;
        }
        if (shortcut_key == U"show_graph_sum_of_loss") {
            getData().menu_elements.show_graph_value = false;
            getData().menu_elements.show_graph_sum_of_loss = true;
        }
        if (shortcut_key == U"show_endgame_error") {
            getData().menu_elements.show_endgame_error = !getData().menu_elements.show_endgame_error;
        }
        // others
        if (shortcut_key == U"show_ai_focus") {
            getData().menu_elements.show_ai_focus = !getData().menu_elements.show_ai_focus;
        }
        if (shortcut_key == U"show_laser_pointer") {
            getData().menu_elements.show_laser_pointer = !getData().menu_elements.show_laser_pointer;
        }
        if (shortcut_key == U"show_log") {
            getData().menu_elements.show_log = !getData().menu_elements.show_log;
        }
        if (shortcut_key == U"change_color_type") {
            getData().menu_elements.change_color_type = !getData().menu_elements.change_color_type;
        }
    }

    void menu_manipulate() {
        if (getData().menu_elements.stop_calculating || shortcut_key == U"stop_calculating") {
            stop_calculating();
            
            // ignore recalculate hint
            ai_status.hint_calculated = true;

            // stop pv calculation
            ai_status.pv_calculated = true;

            // stop local strategy
            ai_status.local_strategy_calculated = true;
            
            // If it's AI's turn, stop AI
            if (
                (getData().history_elem.player == BLACK && getData().menu_elements.ai_put_black) || 
                (getData().history_elem.player == WHITE && getData().menu_elements.ai_put_white)
            ) {
                need_start_game_button_calculation();
            }
            resume_calculating();
        }
        if (!ai_status.analyzing) {
            if ((getData().menu_elements.put_1_move_by_ai || shortcut_key == U"put_1_move_by_ai") && !ai_status.ai_thinking) {
                stop_calculating();
                resume_calculating();
                putting_1_move_by_ai = true;
                ai_move();
            }
            if (getData().menu_elements.backward) {
                stop_calculating();
                --getData().graph_resources.n_discs;
                getData().graph_resources.delta = -1;
                resume_calculating();
            }
            if (getData().menu_elements.forward) {
                stop_calculating();
                ++getData().graph_resources.n_discs;
                getData().graph_resources.delta = 1;
                resume_calculating();
            }
            if ((getData().menu_elements.undo || shortcut_key == U"undo") && getData().book_information.changing == BOOK_CHANGE_NO_CELL) {
                stop_calculating();
                resume_calculating();
                int n_discs_before = getData().history_elem.board.n_discs();
                while (getData().graph_resources.nodes[getData().graph_resources.branch].back().board.n_discs() >= n_discs_before && 
                    ((getData().graph_resources.branch == GRAPH_MODE_NORMAL && getData().graph_resources.nodes[GRAPH_MODE_NORMAL].size() > 1) || (getData().graph_resources.branch == GRAPH_MODE_INSPECT && getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size() > 0))) {
                    getData().graph_resources.nodes[getData().graph_resources.branch].pop_back();
                    History_elem n_history_elem = getData().history_elem;
                    if (getData().graph_resources.branch == GRAPH_MODE_INSPECT && getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size() == 0) {
                        getData().graph_resources.branch = GRAPH_MODE_NORMAL;
                        getData().graph_resources.nodes[GRAPH_MODE_NORMAL][getData().graph_resources.node_find(0, getData().history_elem.board.n_discs())].next_policy = -1;
                        n_history_elem = getData().graph_resources.nodes[0][getData().graph_resources.node_find(0, getData().history_elem.board.n_discs())];
                    } else {
                        getData().graph_resources.nodes[getData().graph_resources.branch].back().next_policy = -1;
                        n_history_elem = getData().graph_resources.nodes[getData().graph_resources.branch].back();
                    }
                    n_history_elem.next_policy = -1;
                    getData().history_elem = n_history_elem;
                    getData().graph_resources.n_discs = getData().history_elem.board.n_discs();
                    stop_calculating();
                    resume_calculating();
                }
                need_start_game_button_calculation();
            }
            if (getData().menu_elements.save_this_branch || shortcut_key == U"save_this_branch") {
                stop_calculating();
                if (getData().graph_resources.branch == GRAPH_MODE_INSPECT) {
                    std::vector<History_elem> new_branch;
                    int fork_start_idx = getData().graph_resources.node_find(0, getData().graph_resources.nodes[1].front().board.n_discs());
                    for (int i = 0; i < fork_start_idx; ++i) {
                        new_branch.emplace_back(getData().graph_resources.nodes[0][i]);
                    }
                    new_branch.back().next_policy = getData().graph_resources.nodes[1][0].policy;
                    for (History_elem elem: getData().graph_resources.nodes[1]) {
                        new_branch.emplace_back(elem);
                    }
                    new_branch.back().next_policy = -1;
                    getData().graph_resources.nodes[0].clear();
                    getData().graph_resources.nodes[1].clear();
                    for (History_elem elem: new_branch) {
                        getData().graph_resources.nodes[0].emplace_back(elem);
                    }
                    getData().graph_resources.branch = 0;
                } else if (getData().graph_resources.branch == GRAPH_MODE_NORMAL) {
                    int n_discs_before = getData().history_elem.board.n_discs();
                    while (getData().graph_resources.nodes[GRAPH_MODE_NORMAL].back().board.n_discs() > n_discs_before && getData().graph_resources.nodes[GRAPH_MODE_NORMAL].size() > 1) {
                        getData().graph_resources.nodes[GRAPH_MODE_NORMAL].pop_back();
                        History_elem n_history_elem = getData().history_elem;
                        getData().graph_resources.nodes[GRAPH_MODE_NORMAL].back().next_policy = -1;
                        n_history_elem = getData().graph_resources.nodes[GRAPH_MODE_NORMAL].back();
                        n_history_elem.next_policy = -1;
                        getData().history_elem = n_history_elem;
                        getData().graph_resources.n_discs = getData().history_elem.board.n_discs();
                    }
                }
                resume_calculating();
                need_start_game_button_calculation();
            }
            if (getData().menu_elements.generate_random_board || shortcut_key == U"generate_random_board") {
                int light_level = 1;
                int adjustment_level = 15;
                stop_calculating();
                getData().history_elem.reset();
                getData().graph_resources.init();
                getData().graph_resources.nodes[getData().graph_resources.branch].emplace_back(getData().history_elem);
                getData().game_information.init();
                pausing_in_pass = false;
                resume_calculating();
                ai_status.random_board_generator_calculating = true;
                ai_status.random_board_generator_future = std::async(std::launch::async, random_board_generator, getData().menu_elements.generate_random_board_score_range_min, getData().menu_elements.generate_random_board_score_range_max, getData().menu_elements.generate_random_board_moves, light_level, adjustment_level, &ai_status.random_board_generator_calculating);
            }
        }
        if (getData().menu_elements.convert_180 || shortcut_key == U"convert_180") {
            stop_calculating();
            getData().history_elem.board.board_rotate_180();
            if (0 <= getData().history_elem.policy && getData().history_elem.policy < HW2) {
                getData().history_elem.policy = HW2_M1 - getData().history_elem.policy;
            }
            if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy < HW2) {
                getData().history_elem.next_policy = HW2_M1 - getData().history_elem.next_policy;
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
                    getData().graph_resources.nodes[i][j].board.board_rotate_180();
                    if (0 <= getData().graph_resources.nodes[i][j].policy && getData().graph_resources.nodes[i][j].policy < HW2) {
                        getData().graph_resources.nodes[i][j].policy = HW2_M1 - getData().graph_resources.nodes[i][j].policy;
                    }
                    if (0 <= getData().graph_resources.nodes[i][j].next_policy && getData().graph_resources.nodes[i][j].next_policy < HW2) {
                        getData().graph_resources.nodes[i][j].next_policy = HW2_M1 - getData().graph_resources.nodes[i][j].next_policy;
                    }
                }
            }
            resume_calculating();
        }
        if (getData().menu_elements.convert_90_clock || shortcut_key == U"convert_90_clock") {
            stop_calculating();
            getData().history_elem.board.board_rotate_270();
            if (0 <= getData().history_elem.policy && getData().history_elem.policy < HW2) {
                int y0 = getData().history_elem.policy / HW;
                int x0 = getData().history_elem.policy % HW;
                getData().history_elem.policy = x0 * HW + (HW_M1 - y0);
            }
            if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy < HW2) {
                int y0 = getData().history_elem.next_policy / HW;
                int x0 = getData().history_elem.next_policy % HW;
                getData().history_elem.next_policy = x0 * HW + (HW_M1 - y0);
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
                    getData().graph_resources.nodes[i][j].board.board_rotate_270();
                    if (0 <= getData().graph_resources.nodes[i][j].policy && getData().graph_resources.nodes[i][j].policy < HW2) {
                        int y0 = getData().graph_resources.nodes[i][j].policy / HW;
                        int x0 = getData().graph_resources.nodes[i][j].policy % HW;
                        getData().graph_resources.nodes[i][j].policy = x0 * HW + (HW_M1 - y0);
                    }
                    if (0 <= getData().graph_resources.nodes[i][j].next_policy && getData().graph_resources.nodes[i][j].next_policy < HW2) {
                        int y0 = getData().graph_resources.nodes[i][j].next_policy / HW;
                        int x0 = getData().graph_resources.nodes[i][j].next_policy % HW;
                        getData().graph_resources.nodes[i][j].next_policy = x0 * HW + (HW_M1 - y0);
                    }
                }
            }
            resume_calculating();
        }
        if (getData().menu_elements.convert_90_anti_clock || shortcut_key == U"convert_90_anti_clock") {
            stop_calculating();
            getData().history_elem.board.board_rotate_90();
            if (0 <= getData().history_elem.policy && getData().history_elem.policy < HW2) {
                int y0 = getData().history_elem.policy / HW;
                int x0 = getData().history_elem.policy % HW;
                getData().history_elem.policy = (HW_M1 - x0) * HW + y0;
            }
            if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy < HW2) {
                int y0 = getData().history_elem.next_policy / HW;
                int x0 = getData().history_elem.next_policy % HW;
                getData().history_elem.next_policy = (HW_M1 - x0) * HW + y0;
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
                    getData().graph_resources.nodes[i][j].board.board_rotate_90();
                    if (0 <= getData().graph_resources.nodes[i][j].policy && getData().graph_resources.nodes[i][j].policy < HW2) {
                        int y0 = getData().graph_resources.nodes[i][j].policy / HW;
                        int x0 = getData().graph_resources.nodes[i][j].policy % HW;
                        getData().graph_resources.nodes[i][j].policy = (HW_M1 - x0) * HW + y0;
                    }
                    if (0 <= getData().graph_resources.nodes[i][j].next_policy && getData().graph_resources.nodes[i][j].next_policy < HW2) {
                        int y0 = getData().graph_resources.nodes[i][j].next_policy / HW;
                        int x0 = getData().graph_resources.nodes[i][j].next_policy % HW;
                        getData().graph_resources.nodes[i][j].next_policy = (HW_M1 - x0) * HW + y0;
                    }
                }
            }
            resume_calculating();
        }
        if (getData().menu_elements.convert_blackline || shortcut_key == U"convert_blackline") {
            stop_calculating();
            getData().history_elem.board.board_black_line_mirror();
            if (0 <= getData().history_elem.policy && getData().history_elem.policy < HW2) {
                int x = getData().history_elem.policy % HW;
                int y = getData().history_elem.policy / HW;
                getData().history_elem.policy = (HW_M1 - x) * HW + (HW_M1 - y);
            }
            if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy < HW2) {
                int x = getData().history_elem.next_policy % HW;
                int y = getData().history_elem.next_policy / HW;
                getData().history_elem.next_policy = (HW_M1 - x) * HW + (HW_M1 - y);
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
                    getData().graph_resources.nodes[i][j].board.board_black_line_mirror();
                    if (0 <= getData().graph_resources.nodes[i][j].policy && getData().graph_resources.nodes[i][j].policy < HW2) {
                        int x = getData().graph_resources.nodes[i][j].policy % HW;
                        int y = getData().graph_resources.nodes[i][j].policy / HW;
                        getData().graph_resources.nodes[i][j].policy = (HW_M1 - x) * HW + (HW_M1 - y);
                    }
                    if (0 <= getData().graph_resources.nodes[i][j].next_policy && getData().graph_resources.nodes[i][j].next_policy < HW2) {
                        int x = getData().graph_resources.nodes[i][j].next_policy % HW;
                        int y = getData().graph_resources.nodes[i][j].next_policy / HW;
                        getData().graph_resources.nodes[i][j].next_policy = (HW_M1 - x) * HW + (HW_M1 - y);
                    }
                }
            }
            resume_calculating();
        }
        if (getData().menu_elements.convert_whiteline || shortcut_key == U"convert_whiteline") {
            stop_calculating();
            getData().history_elem.board.board_white_line_mirror();
            if (0 <= getData().history_elem.policy && getData().history_elem.policy < HW2) {
                int x = getData().history_elem.policy % HW;
                int y = getData().history_elem.policy / HW;
                getData().history_elem.policy = x * HW + y;
            }
            if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy < HW2) {
                int x = getData().history_elem.next_policy % HW;
                int y = getData().history_elem.next_policy / HW;
                getData().history_elem.next_policy = x * HW + y;
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
                    getData().graph_resources.nodes[i][j].board.board_white_line_mirror();
                    if (0 <= getData().graph_resources.nodes[i][j].policy && getData().graph_resources.nodes[i][j].policy < HW2) {
                        int x = getData().graph_resources.nodes[i][j].policy % HW;
                        int y = getData().graph_resources.nodes[i][j].policy / HW;
                        getData().graph_resources.nodes[i][j].policy = x * HW + y;
                    }
                    if (0 <= getData().graph_resources.nodes[i][j].next_policy && getData().graph_resources.nodes[i][j].next_policy < HW2) {
                        int x = getData().graph_resources.nodes[i][j].next_policy % HW;
                        int y = getData().graph_resources.nodes[i][j].next_policy / HW;
                        getData().graph_resources.nodes[i][j].next_policy = x * HW + y;
                    }
                }
            }
            resume_calculating();
        }
        if (getData().menu_elements.convert_horizontal || shortcut_key == U"convert_horizontal") {
            stop_calculating();
            getData().history_elem.board.board_horizontal_mirror();
            if (0 <= getData().history_elem.policy && getData().history_elem.policy < HW2) {
                int x = getData().history_elem.policy % HW;
                int y = getData().history_elem.policy / HW;
                getData().history_elem.policy = y * HW + (HW_M1 - x);
            }
            if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy < HW2) {
                int x = getData().history_elem.next_policy % HW;
                int y = getData().history_elem.next_policy / HW;
                getData().history_elem.next_policy = y * HW + (HW_M1 - x);
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
                    getData().graph_resources.nodes[i][j].board.board_horizontal_mirror();
                    if (0 <= getData().graph_resources.nodes[i][j].policy && getData().graph_resources.nodes[i][j].policy < HW2) {
                        int x = getData().graph_resources.nodes[i][j].policy % HW;
                        int y = getData().graph_resources.nodes[i][j].policy / HW;
                        getData().graph_resources.nodes[i][j].policy = y * HW + (HW_M1 - x);
                    }
                    if (0 <= getData().graph_resources.nodes[i][j].next_policy && getData().graph_resources.nodes[i][j].next_policy < HW2) {
                        int x = getData().graph_resources.nodes[i][j].next_policy % HW;
                        int y = getData().graph_resources.nodes[i][j].next_policy / HW;
                        getData().graph_resources.nodes[i][j].next_policy = y * HW + (HW_M1 - x);
                    }
                }
            }
            resume_calculating();
        }
        if (getData().menu_elements.convert_vertical || shortcut_key == U"convert_vertical") {
            stop_calculating();
            getData().history_elem.board.board_vertical_mirror();
            if (0 <= getData().history_elem.policy && getData().history_elem.policy < HW2) {
                int x = getData().history_elem.policy % HW;
                int y = getData().history_elem.policy / HW;
                getData().history_elem.policy = (HW_M1 - y) * HW + x;
            }
            if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy < HW2) {
                int x = getData().history_elem.next_policy % HW;
                int y = getData().history_elem.next_policy / HW;
                getData().history_elem.next_policy = (HW_M1 - y) * HW + x;
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
                    getData().graph_resources.nodes[i][j].board.board_vertical_mirror();
                    if (0 <= getData().graph_resources.nodes[i][j].policy && getData().graph_resources.nodes[i][j].policy < HW2) {
                        int x = getData().graph_resources.nodes[i][j].policy % HW;
                        int y = getData().graph_resources.nodes[i][j].policy / HW;
                        getData().graph_resources.nodes[i][j].policy = (HW_M1 - y) * HW + x;
                    }
                    if (0 <= getData().graph_resources.nodes[i][j].next_policy && getData().graph_resources.nodes[i][j].next_policy < HW2) {
                        int x = getData().graph_resources.nodes[i][j].next_policy % HW;
                        int y = getData().graph_resources.nodes[i][j].next_policy / HW;
                        getData().graph_resources.nodes[i][j].next_policy = (HW_M1 - y) * HW + x;
                    }
                }
            }
            resume_calculating();
        }
        if (getData().menu_elements.cache_clear || shortcut_key == U"cache_clear") {
            stop_calculating();
            transposition_table.init();
            resume_calculating();
        }
    }

    void menu_in_out() {
        if (getData().menu_elements.input_from_clipboard || shortcut_key == U"input_from_clipboard") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            input_from_clipboard();
            return;
        }
        if (getData().menu_elements.input_text || shortcut_key == U"input_text") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Import_text", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.edit_board || shortcut_key == U"edit_board") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Edit_board", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.input_game || shortcut_key == U"input_game") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Import_game", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.input_bitboard || shortcut_key == U"input_bitboard") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Import_bitboard", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.copy_transcript || shortcut_key == U"output_transcript") {
            copy_transcript();
        }
        if (getData().menu_elements.copy_board || shortcut_key == U"output_board") {
            copy_board();
        }
        if (getData().menu_elements.screen_shot || shortcut_key == U"screen_shot") {
            taking_screen_shot_state = 1;
            getData().menu_elements.screen_shot = false; // because skip drawing menu in next frame
        }
        if (getData().menu_elements.change_screenshot_saving_dir || shortcut_key == U"change_screenshot_saving_dir") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Change_screenshot_saving_dir", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.board_image || shortcut_key == U"board_image") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Board_image", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.save_game || shortcut_key == U"save_game") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Export_game", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.output_bitboard_player_opponent || shortcut_key == U"output_bitboard_player_opponent") {
            copy_bitboard_player_opponent();
            return;
        }
        if (getData().menu_elements.output_bitboard_black_white || shortcut_key == U"output_bitboard_black_white") {
            copy_bitboard_black_white();
            return;
        }
    }

    void menu_book() {
        // book operation
        if (shortcut_key == U"change_book_by_right_click") {
            getData().menu_elements.change_book_by_right_click = !getData().menu_elements.change_book_by_right_click;
        }
        if (getData().menu_elements.book_start_deviate || shortcut_key == U"book_start_deviate") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Enhance_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.book_start_deviate_with_transcript || shortcut_key == U"book_start_deviate_with_transcript") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Deviate_book_transcript", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.book_start_store || shortcut_key == U"book_start_store") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Store_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.book_start_fix || shortcut_key == U"book_start_fix") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Fix_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.book_start_reducing || shortcut_key == U"book_start_reducing") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Reduce_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.book_start_recalculate_leaf || shortcut_key == U"book_start_recalculate_leaf") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Leaf_recalculate_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.book_start_recalculate_n_lines || shortcut_key == U"book_start_recalculate_n_lines") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"N_lines_recalculate_book", SCENE_FADE_TIME);
            return;
        }
        //if (getData().menu_elements.book_start_upgrade_better_leaves || shortcut_key == U"book_start_upgrade_better_leaves") {
        //    changing_scene = true;
        //    stop_calculating();
        //    resume_calculating();
        //    changeScene(U"Upgrade_better_leaves_book", SCENE_FADE_TIME);
        //    return;
        //}
        // file operation
        if (getData().menu_elements.import_book || shortcut_key == U"import_book") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Import_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.export_book || shortcut_key == U"export_book") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Export_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.book_merge || shortcut_key == U"book_merge") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Merge_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.book_reference || shortcut_key == U"book_reference") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Refer_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.show_book_info || shortcut_key == U"show_book_info") {
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Show_book_info", SCENE_FADE_TIME);
            return;
        }
    }

    void menu_help() {
        if (getData().menu_elements.usage || shortcut_key == U"open_usage") {
            shortcut_key = SHORTCUT_KEY_UNDEFINED;
            if (language.get("lang_name") == U"日本語") {
                System::LaunchBrowser(U"https://www.egaroucid.nyanyan.dev/ja/usage/");
            } else {
                System::LaunchBrowser(U"https://www.egaroucid.nyanyan.dev/en/usage/");
            }
        }
        if (getData().menu_elements.website || shortcut_key == U"open_website") {
            shortcut_key = SHORTCUT_KEY_UNDEFINED;
            if (language.get("lang_name") == U"日本語") {
                System::LaunchBrowser(U"https://www.egaroucid.nyanyan.dev/ja/");
            } else {
                System::LaunchBrowser(U"https://www.egaroucid.nyanyan.dev/en/");
            }
        }
        if (getData().menu_elements.bug_report || shortcut_key == U"bug_report") {
            shortcut_key = SHORTCUT_KEY_UNDEFINED;
            System::LaunchBrowser(U"https://docs.google.com/forms/d/e/1FAIpQLSd6ML1T1fc707luPEefBXuImMnlM9cQP8j-YHKiSyFoS-8rmQ/viewform?usp=sf_link");
        }
        if (getData().menu_elements.update_check || shortcut_key == U"update_check") {
            shortcut_key = SHORTCUT_KEY_UNDEFINED;
            changing_scene = true;
            stop_calculating();
            resume_calculating();
            changeScene(U"Update_check", SCENE_FADE_TIME);
        }
        if (shortcut_key == U"auto_update_check") {
            getData().menu_elements.auto_update_check = !getData().menu_elements.auto_update_check;
        }
        if (getData().menu_elements.license || shortcut_key == U"license") {
            shortcut_key = SHORTCUT_KEY_UNDEFINED;
            LicenseManager::ShowInBrowser();
        }
    }

    void menu_language() {
        for (int i = 0; i < (int)getData().resources.language_names.size(); ++i) {
            if (getData().menu_elements.languages[i]) {
                if (getData().settings.lang_name != getData().resources.language_names[i]) {
                    std::string lang_file = "resources/languages/" + getData().resources.language_names[i] + ".json";
                    if (!language.init(lang_file)) {
                        std::cerr << "language setting error" << std::endl;
                        lang_file = "resources/languages/" + getData().settings.lang_name + ".json";
                        language.init(lang_file);
                    }
                    if (!opening_init(getData().resources.language_names[i])) {
                        std::cerr << "opening setting error" << std::endl;
                        opening_init(DEFAULT_OPENING_LANG_NAME);
                    }
                    getData().settings.lang_name = getData().resources.language_names[i];
                    getData().fonts.init(getData().settings.lang_name);
                    getData().menu = create_menu(&getData().menu_elements, &getData().resources, getData().fonts.font, getData().resources.language_names[i]);
                    start_game_button.init(START_GAME_BUTTON_SX, START_GAME_BUTTON_SY, START_GAME_BUTTON_WIDTH, START_GAME_BUTTON_HEIGHT, START_GAME_BUTTON_RADIUS, language.get("play", "start_game"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
                    pass_button.init(PASS_BUTTON_SX, PASS_BUTTON_SY, PASS_BUTTON_WIDTH, PASS_BUTTON_HEIGHT, PASS_BUTTON_RADIUS, language.get("play", "pass"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
                    re_calculate_openings();
                }
            }
        }
    }

    void interact_graph() {
        getData().graph_resources.n_discs = graph.update_n_discs(getData().graph_resources.nodes[0], getData().graph_resources.nodes[1], getData().graph_resources.n_discs);
        if (shortcut_key_pressed != U"backward") {
            move_board_button_status.left_pushed = BUTTON_NOT_PUSHED;
        }
        if (shortcut_key_pressed != U"forward") {
            move_board_button_status.right_pushed = BUTTON_NOT_PUSHED;
        }

        if (MouseX1.down() || shortcut_key == U"backward" || (move_board_button_status.left_pushed != BUTTON_NOT_PUSHED && tim() - move_board_button_status.left_pushed >= BUTTON_LONG_PRESS_THRESHOLD)) {
            stop_calculating();
            --getData().graph_resources.n_discs;
            getData().graph_resources.delta = -1;
            resume_calculating();
            if (shortcut_key == U"backward") {
                move_board_button_status.left_pushed = tim();
            }
        }
        else if (MouseX2.down() || shortcut_key == U"forward" || (move_board_button_status.right_pushed != BUTTON_NOT_PUSHED && tim() - move_board_button_status.right_pushed >= BUTTON_LONG_PRESS_THRESHOLD)) {
            stop_calculating();
            ++getData().graph_resources.n_discs;
            getData().graph_resources.delta = 1;
            resume_calculating();
            if (shortcut_key == U"forward") {
                move_board_button_status.right_pushed = tim();
            }
        }
    }

    void update_n_discs() {
        int max_n_discs = getData().graph_resources.nodes[getData().graph_resources.branch].back().board.n_discs();
        getData().graph_resources.n_discs = std::min(getData().graph_resources.n_discs, max_n_discs);
        int min_n_discs = getData().graph_resources.nodes[GRAPH_MODE_NORMAL][0].board.n_discs();
        if (getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
            min_n_discs = std::min(min_n_discs, getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs());
        }
        getData().graph_resources.n_discs = std::max(getData().graph_resources.n_discs, min_n_discs);
        if (getData().graph_resources.branch == GRAPH_MODE_INSPECT && getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
            if (getData().graph_resources.n_discs < getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs()) {
                getData().graph_resources.branch = GRAPH_MODE_NORMAL;
                getData().graph_resources.nodes[1].clear();
            }
        }
        int node_idx = getData().graph_resources.node_find(getData().graph_resources.branch, getData().graph_resources.n_discs);
        if (node_idx == -1 && getData().graph_resources.branch == GRAPH_MODE_INSPECT) {
            getData().graph_resources.nodes[GRAPH_MODE_INSPECT].clear();
            int node_idx_0 = getData().graph_resources.node_find(GRAPH_MODE_NORMAL, getData().graph_resources.n_discs);
            if (node_idx_0 == -1) {
                std::cerr << "history vector element not found 0" << std::endl;
                return;
            }
            getData().graph_resources.nodes[GRAPH_MODE_INSPECT].emplace_back(getData().graph_resources.nodes[GRAPH_MODE_NORMAL][node_idx_0]);
            node_idx = getData().graph_resources.node_find(getData().graph_resources.branch, getData().graph_resources.n_discs);
        }
        while (node_idx == -1) {
            //std::cerr << "history vector element not found 1" << std::endl;
            getData().graph_resources.n_discs += getData().graph_resources.delta;
            node_idx = getData().graph_resources.node_find(GRAPH_MODE_NORMAL, getData().graph_resources.n_discs);
        }
        if (getData().history_elem.board != getData().graph_resources.nodes[getData().graph_resources.branch][node_idx].board) {
            stop_calculating();
            resume_calculating();
        }
        getData().history_elem = getData().graph_resources.nodes[getData().graph_resources.branch][node_idx];
    }

    void move_processing(int_fast8_t cell) {
        stop_calculating();
        int parent_idx = getData().graph_resources.node_find(getData().graph_resources.branch, getData().history_elem.board.n_discs());
        if (parent_idx != -1) {
            if (getData().graph_resources.nodes[getData().graph_resources.branch][parent_idx].next_policy == HW2_M1 - cell && parent_idx + 1 < (int)getData().graph_resources.nodes[getData().graph_resources.branch].size()) {
                ++getData().graph_resources.n_discs;
                return;
            }
            getData().graph_resources.nodes[getData().graph_resources.branch][parent_idx].next_policy = HW2_M1 - cell;
            while (getData().graph_resources.nodes[getData().graph_resources.branch].size() > parent_idx + 1) {
                getData().graph_resources.nodes[getData().graph_resources.branch].pop_back();
            }
        }
        Flip flip;
        calc_flip(&flip, &getData().history_elem.board, HW2_M1 - cell);
        getData().history_elem.board.move_board(&flip);
        getData().history_elem.policy = HW2_M1 - cell;
        getData().history_elem.next_policy = -1;
        getData().history_elem.v = GRAPH_IGNORE_VALUE;
        getData().history_elem.level = -1;
        getData().history_elem.player ^= 1;
        if (getData().history_elem.board.get_legal() == 0ULL) {
            getData().history_elem.board.pass();
            getData().history_elem.player ^= 1;
        }
        getData().graph_resources.nodes[getData().graph_resources.branch].emplace_back(getData().history_elem);
        getData().graph_resources.n_discs++;
        if (getData().history_elem.board.is_end()) {
            int sgn = getData().history_elem.player == BLACK ? 1 : -1;
            getData().graph_resources.nodes[getData().graph_resources.branch].back().v = sgn * getData().history_elem.board.score_player();
            getData().graph_resources.nodes[getData().graph_resources.branch].back().level = 100; // 100% search
        }
        resume_calculating();
    }

    void interact_move() {
        uint64_t legal = getData().history_elem.board.get_legal();
        for (int_fast8_t cell = 0; cell < HW2; ++cell) {
            if (1 & (legal >> (HW2_M1 - cell))) {
                int x = cell % HW;
                int y = cell / HW;
                Rect cell_rect(BOARD_SX + x * BOARD_CELL_SIZE, BOARD_SY + y * BOARD_CELL_SIZE, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
                if (cell_rect.mouseOver()) {
                    Cursor::RequestStyle(CursorStyle::Hand);
                }
                if (cell_rect.leftClicked()) {
                    stop_calculating();
                    if (getData().graph_resources.branch == GRAPH_MODE_NORMAL) {
                        int parent_idx = getData().graph_resources.node_find(GRAPH_MODE_NORMAL, getData().history_elem.board.n_discs());
                        if (parent_idx != -1) {
                            bool go_to_inspection_mode =
                                getData().history_elem.board.n_discs() != getData().graph_resources.nodes[GRAPH_MODE_NORMAL][getData().graph_resources.nodes[GRAPH_MODE_NORMAL].size() - 1].board.n_discs() &&
                                HW2_M1 - cell != getData().graph_resources.nodes[GRAPH_MODE_NORMAL][parent_idx].next_policy;
                            if (go_to_inspection_mode) {
                                getData().graph_resources.branch = GRAPH_MODE_INSPECT;
                            }
                        }
                    }
                    move_processing(cell);
                    resume_calculating();
                }
            }
        }
    }

    void ai_move() {
        if (!changing_scene) {
            uint64_t legal = getData().history_elem.board.get_legal();
            if (!ai_status.ai_thinking) {
                if (legal) {
                    stop_calculating();
                    resume_calculating();
                    bool specified_opening_moved = false;
                    if (getData().menu_elements.force_specified_openings) {
                        int selected_policy = getData().forced_openings.get_one(getData().history_elem.board);
                        std::cerr << "getting opening " << idx_to_coord(selected_policy) << std::endl;
                        if (selected_policy != MOVE_UNDEFINED) {
                            int player_bef = getData().history_elem.player;
                            int sgn = getData().history_elem.player == BLACK ? 1 : -1;
                            //getData().graph_resources.nodes[getData().graph_resources.branch].back().v = sgn * search_result.value;
                            //getData().graph_resources.nodes[getData().graph_resources.branch].back().level = calc_ai_type(search_result);
                            move_processing(HW2_M1 - selected_policy);
                            if (getData().history_elem.player == player_bef && (getData().menu_elements.ai_put_black ^ getData().menu_elements.ai_put_white) && getData().menu_elements.pause_when_pass && !getData().history_elem.board.is_end()) {
                                pausing_in_pass = true;
                            }
                            specified_opening_moved = true;
                        }
                    }
                    if (!specified_opening_moved) {
                        if (getData().menu_elements.accept_ai_loss) {
                            double loss_ratio = 0.01 * getData().menu_elements.loss_percentage;
                            if (myrandom() <= loss_ratio) {
                                ai_status.ai_future = std::async(std::launch::async, ai_loss, getData().history_elem.board, getData().menu_elements.level, getData().menu_elements.use_book, 0, true, true, getData().menu_elements.max_loss);
                            } else {
                                ai_status.ai_future = std::async(std::launch::async, ai, getData().history_elem.board, getData().menu_elements.level, getData().menu_elements.use_book, 0, true, true);
                            }
                        } else {
                            ai_status.ai_future = std::async(std::launch::async, ai, getData().history_elem.board, getData().menu_elements.level, getData().menu_elements.use_book, 0, true, true);
                        }
                        ai_status.ai_thinking = true;
                    }
                }
            }
            else if (ai_status.ai_future.valid()) {
                if (ai_status.ai_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    Search_result search_result = ai_status.ai_future.get();
                    if (1 & (legal >> search_result.policy)) {
                        int player_bef = getData().history_elem.player;
                        int sgn = getData().history_elem.player == BLACK ? 1 : -1;
                        getData().graph_resources.nodes[getData().graph_resources.branch].back().v = sgn * search_result.value;
                        getData().graph_resources.nodes[getData().graph_resources.branch].back().level = calc_ai_type(search_result);
                        move_processing(HW2_M1 - search_result.policy);
                        if (getData().history_elem.player == player_bef && (getData().menu_elements.ai_put_black ^ getData().menu_elements.ai_put_white) && getData().menu_elements.pause_when_pass && !getData().history_elem.board.is_end()) {
                            pausing_in_pass = true;
                        }
                    }
                    ai_status.ai_thinking = false;
                    putting_1_move_by_ai = false;
                }
            }
        }
    }

    void update_opening() {
        std::string new_opening = opening.get(getData().history_elem.board, getData().history_elem.player ^ 1);
        if (new_opening.size() && getData().history_elem.opening_name != new_opening) {
            getData().history_elem.opening_name = new_opening;
            int node_idx = getData().graph_resources.node_find(getData().graph_resources.branch, getData().graph_resources.n_discs);
            if (node_idx == -1) {
                std::cerr << "history vector element not found 2" << std::endl;
                return;
            }
            getData().graph_resources.nodes[getData().graph_resources.branch][node_idx].opening_name = new_opening;
        }
    }

    void draw_legal(uint64_t ignore) {
        uint64_t legal = getData().history_elem.board.get_legal();
        for (int cell = 0; cell < HW2; ++cell) {
            int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
            int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
            if (1 & (legal >> (HW2_M1 - cell))) {
                if ((1 & (ignore >> (HW2_M1 - cell))) == 0) {
                    Circle(x, y, LEGAL_SIZE).draw(getData().colors.cyan);
                }
            }
        }
    }

    void draw_last_move() {
        if (getData().history_elem.policy != -1) {
            int x = BOARD_SX + (HW_M1 - getData().history_elem.policy % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
            int y = BOARD_SY + (HW_M1 - getData().history_elem.policy / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
            Circle(x, y, LEGAL_SIZE).draw(getData().colors.red);
        }
    }

    void draw_next_move() {
        if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy <= HW2) {
            uint64_t legal = getData().history_elem.board.get_legal();
            if (1 & (legal >> getData().history_elem.next_policy)) {
                if (getData().menu_elements.show_next_move_change_view) {
                    int sx = BOARD_SX + (HW_M1 - getData().history_elem.next_policy % HW) * BOARD_CELL_SIZE;
                    int sy = BOARD_SY + (HW_M1 - getData().history_elem.next_policy / HW) * BOARD_CELL_SIZE;
                    Rect(sx, sy, BOARD_CELL_SIZE, BOARD_CELL_SIZE).drawFrame(8, 0, ColorF(getData().colors.purple, 0.7));
                } else {
                    int x = BOARD_SX + (HW_M1 - getData().history_elem.next_policy % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
                    int y = BOARD_SY + (HW_M1 - getData().history_elem.next_policy / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
                    if (getData().history_elem.player == WHITE) {
                        Circle(x, y, DISC_SIZE).draw(ColorF(getData().colors.white, 0.2));
                    } else {
                        Circle(x, y, DISC_SIZE).draw(ColorF(getData().colors.black, 0.2));
                    }
                }
            }
        }
    }

    void draw_play_ordering() {
        int history_elem_n_discs = getData().history_elem.board.n_discs();
        std::vector<int> put_order = get_put_order(getData().graph_resources, getData().history_elem);
        std::vector<int> put_player = get_put_player(getData().graph_resources.nodes[0][0].board, getData().graph_resources.nodes[0][0].player, put_order);
        for (int i = 0; i < put_order.size(); ++i) {
            int cell = put_order[i];
            int x = BOARD_SX + ((HW2_M1 - cell) % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
            int y = BOARD_SY + ((HW2_M1 - cell) / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
            bool is_black_disc;
            if (getData().menu_elements.play_ordering_transcript_format) {
                is_black_disc = put_player[i] == BLACK;
            } else {
                is_black_disc = getData().history_elem.player == BLACK && (getData().history_elem.board.player & (1ULL << cell)) != 0;
                is_black_disc |= getData().history_elem.player == WHITE && (getData().history_elem.board.opponent & (1ULL << cell)) != 0;
            }
            Color color = getData().colors.black;
            if (is_black_disc) {
                color = getData().colors.white;
            }
            getData().fonts.font_bold(i + 1).draw(24, Arg::center(x, y), color);
        }
    }

    uint64_t draw_hint(bool ignore_book_info) {
        uint64_t res = 0ULL;
        if (ai_status.hint_calculating || ai_status.hint_calculated) {
            bool simplified_hint_mode = !getData().menu_elements.show_hint_level && !getData().menu_elements.show_book_accuracy && !getData().menu_elements.use_umigame_value;
            std::vector<Hint_info> hint_infos;
            for (int cell = 0; cell < HW2; ++cell) {
                if (ai_status.hint_use[HW2_M1 - cell] && -HW2 <= ai_status.hint_values[HW2_M1 - cell] && ai_status.hint_values[HW2_M1 - cell] <= (double)HW2 + HINT_PRIORITY + 0.009) {
                    Hint_info hint_info;
                    hint_info.value = ai_status.hint_values[HW2_M1 - cell];
                    hint_info.cell = cell;
                    hint_info.type = ai_status.hint_types[HW2_M1 - cell];
                    hint_infos.emplace_back(hint_info);
                }
            }
            if (hint_infos.size()) {
                sort(hint_infos.begin(), hint_infos.end(), compare_hint_info);
                int sgn = getData().history_elem.player == BLACK ? 1 : -1;
                int node_idx = getData().graph_resources.node_find(getData().graph_resources.branch, getData().graph_resources.n_discs);
                if (node_idx != -1) {
                    int value_signed = sgn * (int)round(hint_infos[0].value);
                    if (getData().graph_resources.nodes[getData().graph_resources.branch][node_idx].level < hint_infos[0].type) {
                        getData().graph_resources.nodes[getData().graph_resources.branch][node_idx].v = value_signed;
                        getData().graph_resources.nodes[getData().graph_resources.branch][node_idx].level = hint_infos[0].type;
                    }
                }
            }
            int n_disc_hint = std::min((int)hint_infos.size(), getData().menu_elements.n_disc_hint);
            double best_score = -SCORE_INF;
            for (int i = 0; i < n_disc_hint; ++i) {
                best_score = std::max(best_score, hint_infos[i].value);
            }
            for (int i = 0; i < n_disc_hint; ++i) {
                //std::cerr << idx_to_coord(hint_infos[i].cell) << " " << hint_infos[i].value << std::endl;
                int sx = BOARD_SX + (hint_infos[i].cell % HW) * BOARD_CELL_SIZE;
                int sy = BOARD_SY + (hint_infos[i].cell / HW) * BOARD_CELL_SIZE;
                bool is_best_score = (hint_infos[i].value == best_score);
                int value = (int)round(hint_infos[i].value);
                Font font = get_hint_font(is_best_score);
                Color color = get_hint_color(value, hint_infos[i].type, is_best_score);
                if (simplified_hint_mode) {
                    // main value
                    font(value).draw(23, Arg::center(sx + BOARD_CELL_SIZE / 2, sy + BOARD_CELL_SIZE / 2), color);
                    // level info for simple mode (only 100%)
                    if (hint_infos[i].type == 100) {
                        const double check_width = BOARD_CELL_SIZE * 0.2;
                        getData().resources.check.scaled(check_width / (double)getData().resources.check.height()).draw(Arg::topRight(sx + BOARD_CELL_SIZE - 2, sy + 2));
                    }
                } else {
                    // main value
                    font(value).draw(18, sx + 3, sy, color);
                    // level info
                    if (getData().menu_elements.show_hint_level) {
                        if (hint_infos[i].type == AI_TYPE_BOOK) {
                            if (!ignore_book_info) {
                                getData().fonts.font_bold(U"book").draw(10, sx + 3, sy + 21, color);
                            }
                        } else if (hint_infos[i].type > HINT_MAX_LEVEL) {
                            getData().fonts.font_bold(Format(hint_infos[i].type) + U"%").draw(10, sx + 3, sy + 21, color);
                        } else {
                            RectF lv_rect = getData().fonts.font(U"Lv.").region(8, sx + 3, sy + 25);
                            getData().fonts.font_bold(U"Lv").draw(8, sx + 3, sy + 22.5, color);
                            getData().fonts.font_bold(Format(hint_infos[i].type)).draw(9.5, lv_rect.x + lv_rect.w, sy + 21, color);
                        }
                    }
                }
                res |= 1ULL << (HW2_M1 - hint_infos[i].cell);
            }
        }
        return res;
    }

    // draw hint with transposition table
    uint64_t draw_hint_tt(bool use_book, bool ignore_book_info) {
        uint64_t legal = getData().history_elem.board.get_legal();
        std::vector<Hint_info> hint_infos_tt;
        Board board = getData().history_elem.board.copy();
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
            Hint_info hint_info;
            hint_info.cell = HW2_M1 - cell;
            bool book_found = false;
            bool value_found = false;
            Flip flip;
            calc_flip(&flip, &board, cell);
            board.move_board(&flip);
                if (use_book && book.contain(board)) {
                    hint_info.type = AI_TYPE_BOOK;
                    hint_info.value = -book.get(board).value;
                    book_found = true;
                    value_found = true;
                }
                if (!book_found) {
                    int lower = -SCORE_MAX, upper = SCORE_MAX;
                    uint_fast8_t moves[2];
                    int depth;
                    uint_fast8_t mpc_level;
                    transposition_table.get_info(board, &lower, &upper, moves, &depth, &mpc_level);
                    if (lower == upper) {
                        hint_info.value = -lower;
                        if (depth == HW2 - board.n_discs()) {
                            hint_info.type = SELECTIVITY_PERCENTAGE[mpc_level];
                        } else {
                            hint_info.type = depth;
                        }
                        value_found = true;
                    }
                }
            board.undo_board(&flip);
            if (value_found) {
                hint_infos_tt.emplace_back(hint_info);
            }
        }
        uint64_t res = 0ULL;
        sort(hint_infos_tt.begin(), hint_infos_tt.end(), compare_hint_info);
        int n_disc_hint = hint_infos_tt.size(); //std::min((int)hint_infos_tt.size(), getData().menu_elements.n_disc_hint);
        int best_level = -INF;
        double best_score = -SCORE_INF;
        for (int i = 0; i < n_disc_hint; ++i) {
            if (best_level < hint_infos_tt[i].type) {
                best_level = hint_infos_tt[i].type;
                best_score = hint_infos_tt[i].value;
            } else if (best_level == hint_infos_tt[i].type && best_score <= hint_infos_tt[i].value) {
                best_score = hint_infos_tt[i].value;
            }
        }
        for (int i = 0; i < n_disc_hint; ++i) {
            //std::cerr << idx_to_coord(hint_infos_tt[i].cell) << " " << hint_infos_tt[i].value << std::endl;
            int sx = BOARD_SX + (hint_infos_tt[i].cell % HW) * BOARD_CELL_SIZE;
            int sy = BOARD_SY + (hint_infos_tt[i].cell / HW) * BOARD_CELL_SIZE;
            Color color = getData().colors.white;
            Font font = getData().fonts.font;
            if (hint_infos_tt[i].type == best_level && hint_infos_tt[i].value == best_score) {
                color = getData().colors.cyan;
                font = getData().fonts.font_heavy;
            }
            font((int)round(hint_infos_tt[i].value)).draw(18, sx + 3, sy, color);
            // std::cerr << idx_to_coord(HW2_M1 - hint_infos_tt[i].cell) << " " << hint_infos_tt[i].type << std::endl;
            if (hint_infos_tt[i].type == AI_TYPE_BOOK) {
                if (!ignore_book_info) {
                    getData().fonts.font_bold(U"book").draw(10, sx + 3, sy + 21, color);
                }
            } else if (hint_infos_tt[i].type > HINT_MAX_LEVEL) {
                getData().fonts.font_bold(Format(hint_infos_tt[i].type) + U"%").draw(10, sx + 3, sy + 21, color);
            } else {
                RectF lv_rect = getData().fonts.font(U"Lv.").region(8, sx + 3, sy + 25);
                getData().fonts.font_bold(U"Lv").draw(8, sx + 3, sy + 22.5, color);
                getData().fonts.font_bold(Format(hint_infos_tt[i].type)).draw(9.5, lv_rect.x + lv_rect.w, sy + 21, color);
            }
            res |= 1ULL << (HW2_M1 - hint_infos_tt[i].cell);
        }
        return res;
    }

    void draw_stable_discs() {
        uint64_t stable = calc_stability_bits(&getData().history_elem.board);
        for (int cell = 0; cell < HW2; ++cell) {
            int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
            int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
            if (1 & (stable >> (HW2_M1 - cell))) {
                Circle(x, y, STABLE_SIZE).draw(getData().colors.burlywood);
            }
        }
    }

    void draw_opening_on_cell() {
        uint64_t legal = getData().history_elem.board.get_legal();
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
            int x = HW_M1 - cell % HW;
            int y = HW_M1 - cell / HW;
            Rect cell_rect(BOARD_SX + x * BOARD_CELL_SIZE, BOARD_SY + y * BOARD_CELL_SIZE, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
            if (cell_rect.mouseOver()) {
                Flip flip;
                calc_flip(&flip, &getData().history_elem.board, cell);
                std::string openings = opening_many.get(getData().history_elem.board.move_copy(&flip), getData().history_elem.player);
                if (openings.size()) {
                    String opening_name = U" " + Unicode::FromUTF8(openings).replace(U"|", U" \n ") + U" ";
                    Vec2 pos = Cursor::Pos();
                    pos.x += 20;
                    RectF background_rect = getData().fonts.font_bold(opening_name).region(15, pos);
                    const int rect_y_max = BOARD_SY + BOARD_SIZE + BOARD_ROUND_FRAME_WIDTH + 5;
                    if (background_rect.y + background_rect.h > rect_y_max) {
                        double delta = background_rect.y + background_rect.h - rect_y_max;
                        background_rect.y -= delta;
                        pos.y -= delta;
                    }
                    background_rect.draw(getData().colors.white);
                    getData().fonts.font_bold(opening_name).draw(15, pos, getData().colors.black);
                }
            }
        }
    }

    bool same_level(int l1, int l2, int n_moves) {
        bool m1, m2;
        int d1, d2;
        uint_fast8_t ml1, ml2;
        get_level(l1, n_moves, &m1, &d1, &ml1);
        get_level(l2, n_moves, &m2, &d2, &ml2);
        return m1 == m2 && d1 == d2 && ml1 == ml2;
    }

    void hint_calculate() {
        ai_status.hint_calculating = true;
        ai_status.hint_calculated = false;
        for (int i = 0; i < HW2; ++i) {
            ai_status.hint_use[i] = false;
            ai_status.hint_values[i] = SCORE_UNDEFINED;
        }
        uint64_t legal = getData().history_elem.board.get_legal();
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
            ai_status.hint_use[cell] = true;
        }
        ai_status.n_hint_display = getData().menu_elements.n_disc_hint;
        std::cerr << "start hint calculation" << std::endl;
        ai_status.hint_future = std::async(std::launch::async, std::bind(ai_hint, getData().history_elem.board, getData().menu_elements.level, getData().menu_elements.use_book, 0, true, true, getData().menu_elements.n_disc_hint, ai_status.hint_values, ai_status.hint_types));
    }

    void try_hint_get() {
        if (ai_status.hint_future.valid()) {
            if (ai_status.hint_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                ai_status.hint_future.get();
                ai_status.hint_calculating = false;
                ai_status.hint_calculated = true;
                std::cerr << "finish hint calculation" << std::endl;
            }
        }
    }

    void pv_calculate() {
        ai_status.pv_calculating = true;
        ai_status.pv_calculated = false;
        std::cerr << "start pv calculation" << std::endl;
        ai_status.pv_future = std::async(std::launch::async, std::bind(get_principal_variation_str, getData().history_elem.board, getData().menu_elements.pv_length, getData().menu_elements.level, &principal_variation));
    }

    void try_pv_get() {
        if (ai_status.pv_future.valid()) {
            if (ai_status.pv_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                ai_status.pv_future.get();
                ai_status.pv_calculating = false;
                ai_status.pv_calculated = true;
                std::cerr << "finish pv calculation" << std::endl;
            }
        }
    }

    void local_strategy_calculate() {
        ai_status.local_strategy_calculating = true;
        ai_status.local_strategy_calculated = false;
        ai_status.local_strategy_done_level = 0;
        ai_status.local_strategy_future = std::async(std::launch::async, std::bind(calc_local_strategy_player, getData().history_elem.board, std::min(MAX_LOCAL_STRATEGY_LEVEL, getData().menu_elements.level), ai_status.local_strategy, getData().history_elem.player, &ai_status.local_strategy_calculating, &ai_status.local_strategy_done_level, false));
        std::cerr << "start local strategy calculation" << std::endl;
    }

    void local_strategy_policy_calculate() {
        ai_status.local_strategy_policy_calculating = true;
        ai_status.local_strategy_policy_calculated = false;
        ai_status.local_strategy_policy_done_level = 0;
        ai_status.local_strategy_policy_future = std::async(std::launch::async, std::bind(calc_local_strategy_policy, getData().history_elem.board, std::min(MAX_LOCAL_STRATEGY_LEVEL, getData().menu_elements.level), ai_status.local_strategy_policy, &ai_status.local_strategy_policy_calculating, &ai_status.local_strategy_policy_done_level, false));
        std::cerr << "start local strategy policy calculation" << std::endl;
    }

    void try_local_strategy_get() {
        if (ai_status.local_strategy_future.valid()) {
            if (ai_status.local_strategy_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                ai_status.local_strategy_future.get();
                ai_status.local_strategy_calculating = false;
                ai_status.local_strategy_calculated = true;
                std::cerr << "finish local strategy calculation" << std::endl;
            }
        }
    }

    void try_local_strategy_policy_get() {
        if (ai_status.local_strategy_policy_future.valid()) {
            if (ai_status.local_strategy_policy_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                ai_status.local_strategy_policy_future.get();
                ai_status.local_strategy_policy_calculating = false;
                ai_status.local_strategy_policy_calculated = true;
                std::cerr << "finish local strategy policy calculation" << std::endl;
            }
        }
    }

    void draw_local_strategy() {
        for (uint_fast8_t cell = 0; cell < HW2; ++cell) {
            int sx = BOARD_SX + ((HW2_M1 - cell) % HW) * BOARD_CELL_SIZE;
            int sy = BOARD_SY + ((HW2_M1 - cell) / HW) * BOARD_CELL_SIZE;
            //Color cell_color = HSV{160.0, 0.76 - 0.24 * ai_status.local_strategy[cell], 0.60 - 0.30 * ai_status.local_strategy[cell]};
            //Color cell_color = HSV{160.0 + 10.0 * ai_status.local_strategy[cell], 0.76 - 0.20 * ai_status.local_strategy[cell], 0.60 - 0.20 * ai_status.local_strategy[cell]};
            // normal green = HSV{160.0, 0.76, 0.60}
            Color cell_color;
            if (ai_status.local_strategy[cell] > 0) { // black
                cell_color = ColorF{ getData().colors.black_advantage, ai_status.local_strategy[cell] }; // orange
            } else { // white
                cell_color = ColorF{ getData().colors.white_advantage, -ai_status.local_strategy[cell] }; // blue
            }
            Rect{ sx, sy,  BOARD_CELL_SIZE, BOARD_CELL_SIZE}.draw(cell_color);
        }
    }

    void draw_local_strategy_policy() {
        uint64_t legal = getData().history_elem.board.get_legal();
        for (uint_fast8_t policy = first_bit(&legal); legal; policy = next_bit(&legal)) {
            int x = HW_M1 - policy % HW;
            int y = HW_M1 - policy / HW;
            Rect cell_rect(BOARD_SX + x * BOARD_CELL_SIZE, BOARD_SY + y * BOARD_CELL_SIZE, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
            if (cell_rect.mouseOver()) {
                for (uint_fast8_t cell = 0; cell < HW2; ++cell) {
                    int sx = BOARD_SX + ((HW2_M1 - cell) % HW) * BOARD_CELL_SIZE;
                    int sy = BOARD_SY + ((HW2_M1 - cell) / HW) * BOARD_CELL_SIZE;
                    int cx = sx + BOARD_CELL_SIZE / 2;
                    int cy = sy + BOARD_CELL_SIZE / 2;
                    Color frame_color;
                    // if (ai_status.local_strategy_policy[policy][cell] == LOCAL_STRATEGY_POLICY_CHANGED_GOOD_MOVE_DISC) {
                    //     frame_color = getData().colors.blue;
                    // } else if (ai_status.local_strategy_policy[policy][cell] == LOCAL_STRATEGY_POLICY_CHANGED_BAD_MOVE_DISC) {
                    //     frame_color = getData().colors.red;
                    // }
                    bool use_dotted_frame = false;
                    bool draw_square = false;
                    if (ai_status.local_strategy_policy[policy][cell] & LOCAL_STRATEGY_POLICY_CHANGED_GOOD_MOVE_FLIPPED) {
                        draw_square = true;
                        frame_color = getData().colors.blue;
                    } else if (ai_status.local_strategy_policy[policy][cell] & LOCAL_STRATEGY_POLICY_CHANGED_GOOD_MOVE_UNFLIPPED) {
                        draw_square = true;
                        frame_color = getData().colors.blue;
                        use_dotted_frame = true;
                        //frame_color = Palette::Skyblue;
                    } else if (ai_status.local_strategy_policy[policy][cell] & LOCAL_STRATEGY_POLICY_CHANGED_BAD_MOVE_FLIPPED) {
                        draw_square = true;
                        frame_color = getData().colors.red;
                    } else if (ai_status.local_strategy_policy[policy][cell] & LOCAL_STRATEGY_POLICY_CHANGED_BAD_MOVE_UNFLIPPED) {
                        draw_square = true;
                        frame_color = getData().colors.red;
                        use_dotted_frame = true;
                        //frame_color = Palette::Orange;
                    }
                    if (draw_square) {
                        if (use_dotted_frame) {
                            Line{ sx, sy, sx + BOARD_CELL_SIZE, sy }.draw(LineStyle::SquareDot, 6, frame_color);
                            Line{ sx, sy + BOARD_CELL_SIZE, sx + BOARD_CELL_SIZE, sy + BOARD_CELL_SIZE }.draw(LineStyle::SquareDot, 6, frame_color);
                            Line{ sx, sy, sx, sy + BOARD_CELL_SIZE }.draw(LineStyle::SquareDot, 6, frame_color);
                            Line{ sx + BOARD_CELL_SIZE, sy, sx + BOARD_CELL_SIZE, sy + BOARD_CELL_SIZE }.draw(LineStyle::SquareDot, 6, frame_color);
                        } else {
                            Rect{ sx, sy,  BOARD_CELL_SIZE, BOARD_CELL_SIZE}.drawFrame(3, 3, frame_color);
                        }
                    }
                    // disabled for Egaroucid 7.6.0
                    // Color player_legal_color = getData().colors.blue;
                    // Color opponent_legal_color = getData().colors.red;
                    // if (ai_status.local_strategy_policy[policy][cell] & LOCAL_STRATEGY_POLICY_CHANGED_PLAYER_CANPUT) {
                    //     Circle(sx + BOARD_CELL_SIZE - 10, sy + BOARD_CELL_SIZE - 25, LEGAL_SIZE).draw(player_legal_color);
                    // } else if (ai_status.local_strategy_policy[policy][cell] & LOCAL_STRATEGY_POLICY_CHANGED_PLAYER_CANNOTPUT) {
                    //     Circle(sx + BOARD_CELL_SIZE - 10, sy + BOARD_CELL_SIZE - 25, LEGAL_SIZE).drawFrame(2, 0, player_legal_color);
                    // }
                    // if (ai_status.local_strategy_policy[policy][cell] & LOCAL_STRATEGY_POLICY_CHANGED_OPPONENT_CANPUT) {
                    //     Circle(sx + BOARD_CELL_SIZE - 10, sy + BOARD_CELL_SIZE - 10, LEGAL_SIZE).draw(opponent_legal_color);
                    // } else if (ai_status.local_strategy_policy[policy][cell] & LOCAL_STRATEGY_POLICY_CHANGED_OPPONENT_CANNOTPUT) {
                    //     Circle(sx + BOARD_CELL_SIZE - 10, sy + BOARD_CELL_SIZE - 10, LEGAL_SIZE).drawFrame(2, 0, opponent_legal_color);
                    // }
                }
                break;
            }
        }
    }

    void init_analyze() {
        ai_status.analyze_task_stack.clear();
        int idx = 0;
        for (History_elem& node : getData().graph_resources.nodes[getData().graph_resources.branch]) {
            Analyze_info analyze_info;
            analyze_info.idx = idx++;
            analyze_info.sgn = node.player ? -1 : 1;
            analyze_info.board = node.board;
            ai_status.analyze_task_stack.emplace_back(std::make_pair(analyze_info, std::bind(ai, node.board, getData().menu_elements.level, getData().menu_elements.use_book, 0, true, true)));
        }
        std::cerr << "analyze " << ai_status.analyze_task_stack.size() << " tasks" << std::endl;
        ai_status.analyzing = true;
        analyze_do_task();
    }

    void analyze_do_task() {
        if (!changing_scene) {
            if (ai_status.analyze_task_stack.size() == 0) {
                ai_status.analyzing = false;
                getData().history_elem = getData().graph_resources.nodes[getData().graph_resources.branch].back();
                getData().graph_resources.n_discs = getData().graph_resources.nodes[getData().graph_resources.branch].back().board.n_discs();
                return;
            }
            std::pair<Analyze_info, std::function<Search_result()>> task = ai_status.analyze_task_stack.back();
            ai_status.analyze_task_stack.pop_back();
            ai_status.analyze_future[task.first.idx] = std::async(std::launch::async, task.second);
            ai_status.analyze_sgn[task.first.idx] = task.first.sgn;
            getData().history_elem.board = task.first.board;
            getData().history_elem.policy = -1;
            getData().history_elem.next_policy = -1;
            getData().history_elem.player = task.first.sgn == 1 ? 0 : 1;
            getData().graph_resources.n_discs = getData().history_elem.board.n_discs();
        }
    }

    void analyze_get_task() {
        bool task_finished = false;
        for (int i = 0; i < ANALYZE_SIZE; ++i) {
            if (ai_status.analyze_future[i].valid()) {
                if (ai_status.analyze_future[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    Search_result search_result = ai_status.analyze_future[i].get();
                    int value = ai_status.analyze_sgn[i] * search_result.value;
                    getData().graph_resources.nodes[getData().graph_resources.branch][i].v = value;
                    getData().graph_resources.nodes[getData().graph_resources.branch][i].level = calc_ai_type(search_result);
                    task_finished = true;
                }
            }
        }
        if (task_finished) {
            analyze_do_task();
        }
    }

    void copy_transcript() {
        std::string transcript = get_transcript(getData().graph_resources, getData().history_elem);
        std::cerr << transcript << std::endl;
        Clipboard::SetText(Unicode::Widen(transcript));
    }

    void copy_board() {
        std::string board_str;
        for (int i = 0; i < HW2; ++i) {
            uint64_t bit = 1ULL << (HW2_M1 - i);
            if (getData().history_elem.player == BLACK) {
                if (getData().history_elem.board.player & bit) {
                    board_str += "X";
                } else if (getData().history_elem.board.opponent & bit) {
                    board_str += "O";
                } else {
                    board_str += "-";
                }
            } else{
                if (getData().history_elem.board.player & bit) {
                    board_str += "O";
                } else if (getData().history_elem.board.opponent & bit) {
                    board_str += "X";
                } else {
                    board_str += "-";
                }
            }
        }
        board_str += " ";
        if (getData().history_elem.player == BLACK) {
            board_str += "X";
        } else{
            board_str += "O";
        }
        std::cerr << board_str << std::endl;
        Clipboard::SetText(Unicode::Widen(board_str));
    }

    void copy_bitboard_player_opponent() {
        std::ostringstream ss;
        ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << getData().history_elem.board.player;
        ss << "\t";
        ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << getData().history_elem.board.opponent;
        std::string res = ss.str();
        std::cerr << res << std::endl;
        Clipboard::SetText(Unicode::Widen(res));
    }

    void copy_bitboard_black_white() {
        std::ostringstream ss;
        if (getData().history_elem.player == BLACK) {
            ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << getData().history_elem.board.player;
            ss << "\t";
            ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << getData().history_elem.board.opponent;
        } else {
            ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << getData().history_elem.board.opponent;
            ss << "\t";
            ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << getData().history_elem.board.player;
        }
        std::string res = ss.str();
        std::cerr << res << std::endl;
        Clipboard::SetText(Unicode::Widen(res));
    }

    void need_start_game_button_calculation() {
        need_start_game_button =
            ((getData().history_elem.player == BLACK && getData().menu_elements.ai_put_black) ||
            (getData().history_elem.player == WHITE && getData().menu_elements.ai_put_white)) &&
            getData().history_elem.board.n_discs() == getData().graph_resources.nodes[getData().graph_resources.branch].back().board.n_discs() &&
            !getData().history_elem.board.is_end() && 
            getData().graph_resources.branch == GRAPH_MODE_NORMAL;
    }

    void calculate_umigame() {
        if (!changing_scene) {
            uint64_t legal = getData().history_elem.board.get_legal();
            if (!umigame_status.umigame_calculating) {
                Board board = getData().history_elem.board;
                int n_player = getData().history_elem.player ^ 1;
                Flip flip;
                for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                    calc_flip(&flip, &board, cell);
                    board.move_board(&flip);
                        if (board.get_legal() == 0ULL) {
                            board.pass();
                                umigame_status.umigame_future[cell] = std::async(std::launch::async, get_umigame, board, n_player ^ 1, getData().menu_elements.umigame_value_depth);
                            board.pass();
                        } else {
                            umigame_status.umigame_future[cell] = std::async(std::launch::async, get_umigame, board, n_player, getData().menu_elements.umigame_value_depth);
                        }
                    board.undo_board(&flip);
                }
                umigame_status.umigame_calculating = true;
                std::cerr << "start umigame calculation" << std::endl;
            }
            else {
                bool all_done = true;
                for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                    if (umigame_status.umigame_future[cell].valid()) {
                        if (umigame_status.umigame_future[cell].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                            umigame_status.umigame[cell] = umigame_status.umigame_future[cell].get();
                        } else {
                            all_done = false;
                        }
                    }
                }
                if (all_done) {
                    umigame_status.umigame_calculated = all_done;
                    std::cerr << "finish umigame calculation" << std::endl;
                }
            }
        }
    }

    void draw_umigame(uint64_t legal_ignore) {
        uint64_t legal = getData().history_elem.board.get_legal();
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
            if (1 & (legal_ignore >> cell)) {
                int sx = BOARD_SX + ((HW2_M1 - cell) % HW) * BOARD_CELL_SIZE;
                int sy = BOARD_SY + ((HW2_M1 - cell) / HW) * BOARD_CELL_SIZE;
                if (umigame_status.umigame[cell].b != UMIGAME_UNDEFINED) {
                    getData().fonts.font_bold(umigame_status.umigame[cell].b).draw(9.5, Arg::topRight(sx + BOARD_CELL_SIZE - 3, sy + 21), getData().colors.black);
                    getData().fonts.font_bold(umigame_status.umigame[cell].w).draw(9.5, Arg::topRight(sx + BOARD_CELL_SIZE - 3, sy + 33), getData().colors.white);
                }
            }
        }
    }

    void draw_book_n_lines(uint64_t legal_ignore) {
        uint64_t legal = getData().history_elem.board.get_legal();
        double best_score = -SCORE_INF;
        for (int cell = 0; cell < HW2; ++cell) {
            if (ai_status.hint_use[HW2_M1 - cell] && -HW2 <= ai_status.hint_values[HW2_M1 - cell] && ai_status.hint_values[HW2_M1 - cell] <= (double)HW2 + HINT_PRIORITY + 0.009) {
                best_score = std::max(best_score, ai_status.hint_values[HW2_M1 - cell]);
            }
        }
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
            if (1 & (legal_ignore >> cell)) {
                int sx = BOARD_SX + ((HW2_M1 - cell) % HW) * BOARD_CELL_SIZE;
                int sy = BOARD_SY + ((HW2_M1 - cell) / HW) * BOARD_CELL_SIZE;
                Board board = getData().history_elem.board;
                Flip flip;
                calc_flip(&flip, &board, cell);
                board.move_board(&flip);
                if (book.contain(board)) {
                    ai_status.hint_types[cell] = AI_TYPE_BOOK;
                    Book_elem book_elem = book.get(board);
                    ai_status.hint_values[cell] = -book_elem.value + HINT_PRIORITY; // priority to book
                    uint32_t n_lines = book_elem.n_lines;
                    String n_lines_str = Format(n_lines);
                    if (n_lines >= 1000000000) {
                        if (n_lines / 1000000000 < 10) {
                            uint32_t d1 = n_lines / 1000000000;
                            uint32_t d2 = (n_lines - d1 * 1000000000) / 100000000;
                            n_lines_str = Format(d1) + U"." + Format(d2) + U"G";
                        } else {
                            n_lines_str = Format(n_lines / 1000000000) + U"G";
                        }
                    } else if (n_lines >= 1000000) {
                        if (n_lines / 1000000 < 10) {
                            uint32_t d1 = n_lines / 1000000;
                            uint32_t d2 = (n_lines - d1 * 1000000) / 100000;
                            n_lines_str = Format(d1) + U"." + Format(d2) + U"M";
                        } else{
                            n_lines_str = Format(n_lines / 1000000) + U"M";
                        }
                    } else if (n_lines >= 1000) {
                        if (n_lines / 1000 < 10) {
                            uint32_t d1 = n_lines / 1000;
                            uint32_t d2 = (n_lines - d1 * 1000) / 100;
                            n_lines_str = Format(d1) + U"." + Format(d2) + U"K";
                        } else {
                            n_lines_str = Format(n_lines / 1000) + U"K";
                        }
                    }
                    Color color = get_hint_color(-book_elem.value, AI_TYPE_BOOK, round(-book_elem.value) == round(best_score));
                    getData().fonts.font_bold(n_lines_str).draw(9.5, sx + 3, sy + 21, color);
                }
            }
        }
    }

    void calculate_book_accuracy() {
        if (!changing_scene) {
            uint64_t legal = getData().history_elem.board.get_legal();
            if (!book_accuracy_status.book_accuracy_calculating) {
                Board board = getData().history_elem.board;
                Flip flip;
                for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                    calc_flip(&flip, &board, cell);
                    board.move_board(&flip);
                        if (board.get_legal() == 0ULL) {
                            board.pass();
                                book_accuracy_status.book_accuracy_future[cell] = std::async(std::launch::async, get_book_accuracy, board);
                            board.pass();
                        } else {
                            book_accuracy_status.book_accuracy_future[cell] = std::async(std::launch::async, get_book_accuracy, board);
                        }
                    board.undo_board(&flip);
                }
                book_accuracy_status.book_accuracy_calculating = true;
                std::cerr << "start book accuracy calculation" << std::endl;
            }
            else {
                bool all_done = true;
                for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                    if (book_accuracy_status.book_accuracy_future[cell].valid()) {
                        if (book_accuracy_status.book_accuracy_future[cell].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                            book_accuracy_status.book_accuracy[cell] = book_accuracy_status.book_accuracy_future[cell].get();
                        } else {
                            all_done = false;
                        }
                    }
                }
                if (all_done) {
                    book_accuracy_status.book_accuracy_calculated = true;
                    std::cerr << "finish book accuracy calculation" << std::endl;
                }
            }
        }
    }

    void draw_book_accuracy(uint64_t legal_ignore) {
        double best_score = -SCORE_INF;
        for (int cell = 0; cell < HW2; ++cell) {
            if (ai_status.hint_use[HW2_M1 - cell] && -HW2 <= ai_status.hint_values[HW2_M1 - cell] && ai_status.hint_values[HW2_M1 - cell] <= (double)HW2 + HINT_PRIORITY + 0.009) {
                best_score = std::max(best_score, ai_status.hint_values[HW2_M1 - cell]);
            }
        }
        uint64_t legal = getData().history_elem.board.get_legal();
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
            if (1 & (legal_ignore >> cell)) {
                int sx = BOARD_SX + ((HW2_M1 - cell) % HW) * BOARD_CELL_SIZE;
                int sy = BOARD_SY + ((HW2_M1 - cell) / HW) * BOARD_CELL_SIZE;
                if (book_accuracy_status.book_accuracy[cell] != BOOK_ACCURACY_LEVEL_UNDEFINED) {
                    //std::cerr << idx_to_coord(cell) << " " << book_accuracy_status.book_accuracy[cell] << std::endl;
                    std::string judge;
                    const std::string judge_list = "ABCDEF";
                    if (book_accuracy_status.book_accuracy[cell] > BOOK_ACCURACY_LEVEL_A) { // B-F
                        judge = judge_list[book_accuracy_status.book_accuracy[cell]];
                    } else { // AA-AF
                        judge = "A";
                        judge += judge_list[book_accuracy_status.book_accuracy[cell] + BOOK_ACCURACY_A_SHIFT];
                    }
                    Board board = getData().history_elem.board;
                    Flip flip;
                    calc_flip(&flip, &board, cell);
                    board.move_board(&flip);
                    Book_elem book_elem = book.get(board);
                    int book_level = book_elem.level;
                    String book_level_info = Format(book_level);
                    if (book_level == LEVEL_HUMAN) {
                        book_level_info = U"S";
                    }
                    Color color = get_hint_color(-book_elem.value, AI_TYPE_BOOK, round(-book_elem.value) == round(best_score));
                    getData().fonts.font_bold(Unicode::Widen(judge) + U" " + book_level_info).draw(9.5, sx + 3, sy + 33, color);
                }
            }
        }
    }

    void change_book_by_right_click() {
        if (getData().book_information.changing != BOOK_CHANGE_NO_CELL) {
            getData().fonts.font(language.get("book", "modifying_book_by_right_click") + U"  " + language.get("book", "changed_value") + U"(" + Unicode::Widen(idx_to_coord(getData().book_information.changing)) + U"): " + getData().book_information.val_str).draw(15, CHANGE_BOOK_INFO_SX, CHANGE_BOOK_INFO_SY, getData().colors.white);
            if (KeyEscape.down()) {
                getData().book_information.changing = BOOK_CHANGE_NO_CELL;
            } else if (Key0.down() || KeyNum0.down()) {
                if (getData().book_information.val_str != U"0" && getData().book_information.val_str != U"-") {
                    getData().book_information.val_str += U"0";
                }
            } else if (Key1.down() || KeyNum1.down()) {
                getData().book_information.val_str += U"1";
            } else if (Key2.down() || KeyNum2.down()) {
                getData().book_information.val_str += U"2";
            } else if (Key3.down() || KeyNum3.down()) {
                getData().book_information.val_str += U"3";
            } else if (Key4.down() || KeyNum4.down()) {
                getData().book_information.val_str += U"4";
            } else if (Key5.down() || KeyNum5.down()) {
                getData().book_information.val_str += U"5";
            } else if (Key6.down() || KeyNum6.down()) {
                getData().book_information.val_str += U"6";
            } else if (Key7.down() || KeyNum7.down()) {
                getData().book_information.val_str += U"7";
            } else if (Key8.down() || KeyNum8.down()) {
                getData().book_information.val_str += U"8";
            } else if (Key9.down() || KeyNum9.down()) {
                getData().book_information.val_str += U"9";
            } else if (KeyMinus.down()) {
                if (getData().book_information.val_str == U"" || getData().book_information.val_str == U"-") {
                    getData().book_information.val_str += U"-";
                }
            } else if (KeyBackspace.down()) {
                if (getData().book_information.val_str.size()) {
                    getData().book_information.val_str.pop_back();
                }
            }
        }
        uint64_t legal = getData().history_elem.board.get_legal();
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
            int x = HW_M1 - cell % HW;
            int y = HW_M1 - cell / HW;
            Rect cell_rect(BOARD_SX + x * BOARD_CELL_SIZE, BOARD_SY + y * BOARD_CELL_SIZE, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
            int state = 0;
            if (cell_rect.rightClicked()) {
                state = 1;
            } else if (KeyEnter.down()) {
                state = 2;
            }
            if (state) {
                if (getData().book_information.changing == cell) {
                    if (getData().book_information.val_str.size() == 0) {
                        getData().book_information.changing = BOOK_CHANGE_NO_CELL;
                    } else {
                        int changed_book_value = ParseOr<int>(getData().book_information.val_str, CHANGE_BOOK_ERR);
                        if (changed_book_value < -HW2 || HW2 < changed_book_value) {
                            getData().book_information.val_str.clear();
                            changed_book_value = CHANGE_BOOK_ERR;
                        }
                        if (changed_book_value != CHANGE_BOOK_ERR) {
                            std::cerr << "new value " << changed_book_value << std::endl;
                            Flip flip;
                            calc_flip(&flip, &getData().history_elem.board, getData().book_information.changing);
                            Board moved_board = getData().history_elem.board.move_copy(&flip);
                            if (moved_board.get_legal() == 0) {
                                if (moved_board.is_end()) { // game over
                                    book.change(moved_board, -changed_book_value, LEVEL_HUMAN);
                                } else { // just pass
                                    moved_board.pass();
                                    book.change(moved_board, changed_book_value, LEVEL_HUMAN);
                                }
                            } else {
                                book.change(moved_board, -changed_book_value, LEVEL_HUMAN);
                            }
                            reset_book_additional_information();
                            getData().book_information.changed = true;
                            getData().book_information.changing = BOOK_CHANGE_NO_CELL;
                            getData().book_information.val_str.clear();
                            stop_calculating();
                            resume_calculating();
                        } else {
                            Flip flip;
                            calc_flip(&flip, &getData().history_elem.board, getData().book_information.changing);
                            Board moved_board = getData().history_elem.board.move_copy(&flip);
                            book.delete_elem(moved_board);
                            if (moved_board.get_legal() == 0) {
                                moved_board.pass();
                                book.delete_elem(moved_board);
                            }
                            reset_book_additional_information();
                            getData().book_information.changed = true;
                            getData().book_information.changing = BOOK_CHANGE_NO_CELL;
                            getData().book_information.val_str.clear();
                            stop_calculating();
                            resume_calculating();
                        }
                    }
                } else if (state == 1) {
                    getData().book_information.val_str.clear();
                    getData().book_information.changing = cell;
                }
            }
        }
    }

    void re_calculate_openings() {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
                getData().graph_resources.nodes[i][j].opening_name.clear();
            }
        }
        getData().history_elem.opening_name.clear();
        std::string opening_name = "";
        for (int i = 0; i < (int)getData().graph_resources.nodes[0].size(); ++i) {
            std::string new_opening = opening.get(getData().graph_resources.nodes[0][i].board, getData().graph_resources.nodes[0][i].player ^ 1);
            if (new_opening.size()) {
                opening_name = new_opening;
            }
            getData().graph_resources.nodes[0][i].opening_name = opening_name;
        }
        if (getData().graph_resources.nodes[1].size()) {
            int fork_n_discs = getData().graph_resources.nodes[1][0].board.n_discs();
            int node_idx = getData().graph_resources.node_find(0, fork_n_discs);
            if (node_idx == -1) {
                std::cerr << "history vector element not found" << std::endl;
            } else {
                node_idx = std::max(0, node_idx - 1);
                opening_name = getData().graph_resources.nodes[0][node_idx].opening_name;
                for (int i = 0; i < (int)getData().graph_resources.nodes[1].size(); ++i) {
                    std::string new_opening = opening.get(getData().graph_resources.nodes[1][i].board, getData().graph_resources.nodes[1][i].player ^ 1);
                    if (new_opening.size()) {
                        opening_name = new_opening;
                    }
                    getData().graph_resources.nodes[1][i].opening_name = opening_name;
                }
            }
        }
        getData().history_elem.opening_name = opening.get(getData().history_elem.board, getData().history_elem.player ^ 1);
        if (getData().history_elem.opening_name.size() == 0) {
            int now_node_idx = getData().graph_resources.node_find(getData().graph_resources.branch, getData().history_elem.board.n_discs());
            if (now_node_idx == -1) {
                std::cerr << "history vector element not found" << std::endl;
            } else {
                now_node_idx = std::max(0, now_node_idx - 1);
                getData().history_elem.opening_name = getData().graph_resources.nodes[getData().graph_resources.branch][now_node_idx].opening_name;
            }
        }
    }

    int calc_ai_type(Search_result search_result) {
        if (search_result.depth == SEARCH_BOOK) {
            return AI_TYPE_BOOK;
        }
        if (search_result.is_end_search) {
            return search_result.probability;
        }
        return search_result.depth;
    }

    bool input_from_clipboard() {
        bool res = false;
        String s;
        if (Clipboard::GetText(s)) {
            std::string str = s.narrow();
            std::cerr << "import from clipboard " << str << std::endl;
            bool failed;
            Game_import_t imported_game = import_any_format_processing(str, &failed);
            if (!failed) {
                getData().graph_resources.init();
                getData().graph_resources.nodes[0] = imported_game.history;
                getData().graph_resources.n_discs = getData().graph_resources.nodes[0].back().board.n_discs();
                getData().game_information.init();
                getData().game_information.black_player_name = imported_game.black_player_name;
                getData().game_information.white_player_name = imported_game.white_player_name;
                std::string opening_name, n_opening_name;
                for (int i = 0; i < (int)getData().graph_resources.nodes[getData().graph_resources.branch].size(); ++i) {
                    n_opening_name.clear();
                    n_opening_name = opening.get(getData().graph_resources.nodes[getData().graph_resources.branch][i].board, getData().graph_resources.nodes[getData().graph_resources.branch][i].player ^ 1);
                    if (n_opening_name.size()) {
                        opening_name = n_opening_name;
                    }
                    getData().graph_resources.nodes[getData().graph_resources.branch][i].opening_name = opening_name;
                }
                getData().graph_resources.n_discs = getData().graph_resources.nodes[getData().graph_resources.branch].back().board.n_discs();
                getData().graph_resources.need_init = false;
                getData().history_elem = getData().graph_resources.nodes[getData().graph_resources.branch].back();
                res = true;
            }
        }
        init_main_scene();
        return res;
    }

    void check_random_board_generater() {
        if (ai_status.random_board_generator_future.valid() && ai_status.random_board_generator_calculating) {
            if (ai_status.random_board_generator_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                std::vector<int> moves = ai_status.random_board_generator_future.get();
                ai_status.random_board_generator_calculating = false;
                std::cerr << "finish random board generation" << std::endl;
                stop_calculating();
                getData().history_elem.reset();
                getData().graph_resources.init();
                getData().graph_resources.nodes[getData().graph_resources.branch].emplace_back(getData().history_elem);
                getData().game_information.init();
                pausing_in_pass = false;
                resume_calculating();
                for (int policy : moves) {
                    move_processing(HW2_M1 - policy);
                }
                need_start_game_button_calculation();
            }
        }
    }

    Color get_hint_color(int value, int hint_type, bool is_best_score) {
        Color color;
        if (getData().menu_elements.hint_colorize) {
            if (hint_type == 100 || hint_type == AI_TYPE_BOOK) { // 100% or book
                if (value > 0) {
                    color = getData().colors.cyan;
                } else if (value == 0) {
                    color = getData().colors.white;
                } else {
                    color = getData().colors.light_green;
                }
            } else { // midgame or endgame with MPC
                if (value >= 0) {
                    color = getData().colors.yellow;
                } else {
                    color = getData().colors.light_green;
                }
            }
        } else {
            if (is_best_score) {
                color = getData().colors.cyan;
            } else {
                color = getData().colors.white;
            }
        }
        return color;
    }

    Font get_hint_font(bool is_best_score) {
        Font font;
        if (is_best_score) {
            font = getData().fonts.font_heavy;
        } else {
            font = getData().fonts.font;
        }
        return font;
    }
};