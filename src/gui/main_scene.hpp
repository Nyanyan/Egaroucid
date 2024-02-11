/*
    Egaroucid Project

    @file main_scene.hpp
        Main scene for GUI
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"
#include "draw.hpp"
#include "screen_shot.hpp"

#define HINT_SINGLE_TASK_N_THREAD 4

bool compare_value_cell(std::pair<int, int>& a, std::pair<int, int>& b) {
    return a.first > b.first;
}

bool compare_hint_info(Hint_info& a, Hint_info& b) {
    return a.value > b.value;
}

Umigame_result get_umigame(Board board, int player) {
    return calculate_umigame(&board, player);
}

int get_book_accuracy(Board board){
    return calculate_book_accuracy(&board);
}

class Main_scene : public App::Scene {
private:
    Graph graph;
    //Level_display level_display;
    Move_board_button_status move_board_button_status;
    AI_status ai_status;
    Button start_game_button;
    Button pass_button;
    bool need_start_game_button;
    Umigame_status umigame_status;
    Book_accuracy_status book_accuracy_status;
    bool changing_scene;
    bool taking_screen_shot;
    bool pausing_in_pass;
    bool putting_1_move_by_ai;

public:
    Main_scene(const InitData& init) : IScene{ init } {
        std::cerr << "main scene loading" << std::endl;
        getData().menu = create_menu(&getData().menu_elements);
        graph.sx = GRAPH_SX;
        graph.sy = GRAPH_SY;
        graph.size_x = GRAPH_WIDTH;
        graph.size_y = GRAPH_HEIGHT;
        graph.resolution = GRAPH_RESOLUTION;
        if (getData().graph_resources.need_init) {
            getData().game_information.init();
            getData().graph_resources.init();
            getData().graph_resources.nodes[getData().graph_resources.branch].emplace_back(getData().history_elem);
        }
        start_game_button.init(START_GAME_BUTTON_SX, START_GAME_BUTTON_SY, START_GAME_BUTTON_WIDTH, START_GAME_BUTTON_HEIGHT, START_GAME_BUTTON_RADIUS, language.get("play", "start_game"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
        pass_button.init(PASS_BUTTON_SX, PASS_BUTTON_SY, PASS_BUTTON_WIDTH, PASS_BUTTON_HEIGHT, PASS_BUTTON_RADIUS, language.get("play", "pass"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
        need_start_game_button_calculation();
        changing_scene = false;
        taking_screen_shot = false;
        pausing_in_pass = false;
        putting_1_move_by_ai = false;
        std::cerr << "main scene loaded" << std::endl;
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
        if (getData().menu_elements.hash_level != global_hash_level) {
            int n_hash_level = getData().menu_elements.hash_level;
            if (n_hash_level != global_hash_level) {
                stop_calculating();
                if (!hash_resize(global_hash_level, n_hash_level, true)) {
                    std::cerr << "hash resize failed. use default level" << std::endl;
                    hash_resize(DEFAULT_HASH_LEVEL, DEFAULT_HASH_LEVEL, true);
                    getData().menu_elements.hash_level = DEFAULT_HASH_LEVEL;
                    global_hash_level = DEFAULT_HASH_LEVEL;
                }
                else {
                    global_hash_level = n_hash_level;
                }
                resume_calculating();
            }
        }

        // init
        getData().graph_resources.delta = 0;

        // opening
        update_opening();

        // screen shot
        if (taking_screen_shot) {
            take_screen_shot(getData().window_state.window_scale, getData().directories.document_dir);
            taking_screen_shot = false;
        }

        // menu
        menu_game();
        menu_manipulate();
        menu_in_out();
        menu_book();
        menu_help();
        menu_language();

        // analyze
        if (ai_status.analyzing) {
            analyze_get_task();
        }

        // move
        if (KeyB.down())
            getData().menu_elements.ai_put_black = !getData().menu_elements.ai_put_black;
        if (KeyW.down())
            getData().menu_elements.ai_put_white = !getData().menu_elements.ai_put_white;
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
        bool ignore_move = (getData().book_information.changing != BOOK_CHANGE_NO_CELL) || 
            ai_status.analyzing;
        if (need_start_game_button) {
            need_start_game_button_calculation();
            start_game_button.draw();
            if (start_game_button.clicked()) {
                need_start_game_button = false;
                stop_calculating();
                resume_calculating();
            }
        }
        if (pausing_in_pass){
            bool ai_to_move = ((getData().history_elem.player == BLACK && getData().menu_elements.ai_put_black) || (getData().history_elem.player == WHITE && getData().menu_elements.ai_put_white));
            if (!ai_to_move)
                pausing_in_pass = false;
            else{
                pass_button.draw();
                if (pass_button.clicked()){
                    pausing_in_pass = false;
                }
            }
        }
        if (!ignore_move) {
            if (ai_should_move) {
                ai_move();
            }
            else if (!getData().menu.active() && !need_start_game_button && !pausing_in_pass) {
                interact_move();
            }
        }

        bool graph_interact_ignore = ai_status.analyzing || ai_should_move;
        // transcript move
        if (!ignore_move && !graph_interact_ignore && !getData().menu.active()) {
            interact_graph();
        }
        update_n_discs();

        // book modifying by right-clicking
        if (!ai_should_move && !need_start_game_button && getData().menu_elements.change_book_by_right_click) {
            change_book_by_right_click();
        }

        // board drawing
        draw_board(getData().fonts, getData().colors, getData().history_elem);

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
        if (getData().menu_elements.show_play_ordering){
            draw_play_ordering();
        }

        // draw on cells
        // next move drawing
        if (getData().menu_elements.show_next_move) {
            draw_next_move();
        }

        uint64_t legal_ignore = 0ULL;

        // hint calculating & drawing
        bool hint_ignore = ai_should_move || ai_status.analyzing || need_start_game_button || pausing_in_pass;
        if (KeyV.down())
            getData().menu_elements.use_disc_hint = !getData().menu_elements.use_disc_hint;
        if (!hint_ignore) {
            if (getData().menu_elements.use_disc_hint) {
                if (!ai_status.hint_calculating && ai_status.hint_level < getData().menu_elements.level) {
                    hint_init_calculating();
                }
                hint_do_task();
                legal_ignore = draw_hint(getData().menu_elements.use_book && getData().menu_elements.show_book_accuracy && !hint_ignore);
            }
        }

        // legal drawing
        if (getData().menu_elements.show_legal && !pausing_in_pass) {
            draw_legal(legal_ignore);
        }

        // book information drawing
        if (getData().menu_elements.use_book){
            // book accuracy drawing
            if (getData().menu_elements.show_book_accuracy && !hint_ignore){
                draw_book_n_lines(legal_ignore);
                if (book_accuracy_status.book_accuracy_calculated){
                    draw_book_accuracy(legal_ignore);
                } else
                    calculate_book_accuracy();
            }

            // umigame calculating / drawing
            if (KeyU.down())
                getData().menu_elements.use_umigame_value = !getData().menu_elements.use_umigame_value;
            if (getData().menu_elements.use_umigame_value && !hint_ignore) {
                if (umigame_status.umigame_calculated) {
                    draw_umigame(legal_ignore);
                }
                else {
                    calculate_umigame();
                }
            }
        }

        // graph drawing
        graph.draw(getData().graph_resources.nodes[0], getData().graph_resources.nodes[1], getData().graph_resources.n_discs, getData().menu_elements.show_graph, getData().menu_elements.level, getData().fonts.font, getData().menu_elements.change_color_type);

        // level display drawing
        //level_display.draw(getData().menu_elements.level, getData().history_elem.board.n_discs());

        // info drawing
        draw_info(getData().colors, getData().history_elem, getData().fonts, getData().menu_elements, pausing_in_pass);

        // opening on cell drawing
        if (getData().menu_elements.show_opening_on_cell) {
            draw_opening_on_cell();
        }

        // menu drawing
        bool draw_menu_flag = !taking_screen_shot;
        if (draw_menu_flag) {
            getData().menu.draw();
        }

        // for screen shot
        if (taking_screen_shot) {
            ScreenCapture::RequestCurrentFrame();
        }
    }

    void draw() const override {

    }

private:
    void reset_hint() {
        ai_status.hint_level = HINT_NOT_CALCULATING;
        ai_status.hint_available = false;
        ai_status.hint_calculating = false;
        ai_status.hint_task_stack.clear();
    }

    void reset_ai() {
        ai_status.ai_thinking = false;
    }

    void reset_analyze() {
        ai_status.analyzing = false;
        ai_status.analyze_task_stack.clear();
    }

    void reset_book_additional_features(){
        umigame_status.umigame_calculated = false;
        umigame_status.umigame_calculating = false;
        book_accuracy_status.book_accuracy_calculated = false;
        book_accuracy_status.book_accuracy_calculating = false;
    }

    void stop_calculating() {
        std::cerr << "terminating calculation" << std::endl;
        global_searching = false;
        if (ai_status.ai_future.valid()) {
            ai_status.ai_future.get();
        }
        for (int i = 0; i < HW2; ++i) {
            if (ai_status.hint_future[i].valid()) {
                ai_status.hint_future[i].get();
            }
        }
        for (int i = 0; i < ANALYZE_SIZE; ++i) {
            if (ai_status.analyze_future[i].valid()) {
                ai_status.analyze_future[i].get();
            }
        }
        for (int i = 0; i < HW2; ++i) {
            if (umigame_status.umigame_future[i].valid()) {
                umigame_status.umigame_future[i].get();
            }
        }
        std::cerr << "calculation terminated" << std::endl;
        reset_ai();
        reset_hint();
        reset_analyze();
        reset_book_additional_features();
        std::cerr << "reset all calculations" << std::endl;
    }

    void resume_calculating() {
        global_searching = true;
    }

    void menu_game() {
        if (getData().menu_elements.start_game || KeyN.down()) {
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
        if (getData().menu_elements.start_game_human_black) {
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
        if (getData().menu_elements.start_game_human_white) {
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
        if (getData().menu_elements.start_selfplay) {
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
        if (getData().menu_elements.analyze && !ai_status.ai_thinking && !ai_status.analyzing) {
            stop_calculating();
            init_analyze();
            resume_calculating();
        }
    }

    void menu_in_out() {
        if (getData().menu_elements.input_transcript) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Import_transcript", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.input_board) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Import_board", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.edit_board) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Edit_board", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.input_game) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Import_game", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.copy_transcript) {
            copy_transcript();
        }
        if (getData().menu_elements.copy_board) {
            copy_board();
        }
        if (getData().menu_elements.save_game) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Export_game", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.screen_shot) {
            taking_screen_shot = true;
            getData().menu_elements.screen_shot = false; // because skip drawing menu in next frame
        }
        if (getData().menu_elements.board_image) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Board_image", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.output_bitboard_player_opponent) {
            copy_bitboard_player_opponent();
            return;
        }
        if (getData().menu_elements.output_bitboard_black_white) {
            copy_bitboard_black_white();
            return;
        }
    }

    void menu_manipulate() {
        if (getData().menu_elements.stop_calculating) {
            stop_calculating();
            ai_status.hint_level = HINT_INF_LEVEL;
            resume_calculating();
        }
        if (!ai_status.analyzing) {
            if ((getData().menu_elements.put_1_move_by_ai || KeyG.down()) && !ai_status.ai_thinking) {
                putting_1_move_by_ai = true;
                ai_move();
            }
            if (getData().menu_elements.backward) {
                --getData().graph_resources.n_discs;
                getData().graph_resources.delta = -1;
            }
            if (getData().menu_elements.forward) {
                ++getData().graph_resources.n_discs;
                getData().graph_resources.delta = 1;
            }
            if (getData().menu_elements.undo || (KeyBackspace.down() && getData().book_information.changing == BOOK_CHANGE_NO_CELL)) {
                int n_discs_before = getData().history_elem.board.n_discs();
                while (getData().graph_resources.nodes[getData().graph_resources.branch].back().board.n_discs() >= n_discs_before && 
                    ((getData().graph_resources.branch == GRAPH_MODE_NORMAL && getData().graph_resources.nodes[GRAPH_MODE_NORMAL].size() > 1) || (getData().graph_resources.branch == GRAPH_MODE_INSPECT && getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size() > 0))) {
                    getData().graph_resources.nodes[getData().graph_resources.branch].pop_back();
                    History_elem n_history_elem = getData().history_elem;
                    if (getData().graph_resources.branch == GRAPH_MODE_INSPECT && getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size() == 0){
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
                    reset_hint();
                }
                need_start_game_button_calculation();
            }
            if (getData().menu_elements.save_this_branch || KeyL.down()) {
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
                } else if (getData().graph_resources.branch == GRAPH_MODE_NORMAL){
                    int n_discs_before = getData().history_elem.board.n_discs();
                    while (getData().graph_resources.nodes[GRAPH_MODE_NORMAL].back().board.n_discs() > n_discs_before && getData().graph_resources.nodes[GRAPH_MODE_NORMAL].size() > 1) {
                        getData().graph_resources.nodes[GRAPH_MODE_NORMAL].pop_back();
                        History_elem n_history_elem = getData().history_elem;
                        getData().graph_resources.nodes[GRAPH_MODE_NORMAL].back().next_policy = -1;
                        n_history_elem = getData().graph_resources.nodes[GRAPH_MODE_NORMAL].back();
                        n_history_elem.next_policy = -1;
                        getData().history_elem = n_history_elem;
                        getData().graph_resources.n_discs = getData().history_elem.board.n_discs();
                        reset_hint();
                    }
                }
                need_start_game_button_calculation();
            }
            if (getData().menu_elements.generate_random_board || KeyR.down()){
                int max_n_moves = getData().menu_elements.generate_random_board_moves;
                int level = 2;
                std::random_device seed_gen;
                std::default_random_engine engine(seed_gen());
                std::normal_distribution<> dist(0.0, 4.0); // acceptable loss avg = 0.0, sd = 4.0 discs
                stop_calculating();
                getData().history_elem.reset();
                getData().graph_resources.init();
                getData().graph_resources.nodes[getData().graph_resources.branch].emplace_back(getData().history_elem);
                getData().game_information.init();
                pausing_in_pass = false;
                resume_calculating();
                for (int i = 0; i < max_n_moves; ++i){
                    if (getData().history_elem.board.get_legal() == 0)
                        break;
                    int acceptable_loss = std::abs(std::round(dist(engine)));
                    Search_result search_result = ai_accept_loss(getData().history_elem.board, level, acceptable_loss);
                    int policy = search_result.policy;
                    std::cerr << acceptable_loss << " " << idx_to_coord(policy) << " " << search_result.value << std::endl;
                    move_processing(HW2_M1 - policy);
                }
                need_start_game_button_calculation();
            }
        }
        if (getData().menu_elements.convert_180) {
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
        if (getData().menu_elements.convert_blackline) {
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
        if (getData().menu_elements.convert_whiteline) {
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
        if (getData().menu_elements.cache_clear) {
            stop_calculating();
            transposition_table.init();
            resume_calculating();
        }
    }

    void menu_book() {
        if (getData().menu_elements.book_start_deviate) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Enhance_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.book_start_fix) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Fix_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.book_start_reducing) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Reduce_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.book_start_recalculate_leaf) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Leaf_recalculate_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.book_merge) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Merge_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.book_reference) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Refer_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.import_book) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Import_book", SCENE_FADE_TIME);
            return;
        }
        if (getData().menu_elements.export_book) {
            stop_calculating();
            resume_calculating();
            changing_scene = true;
            changeScene(U"Export_book", SCENE_FADE_TIME);
            return;
        }
    }

    void menu_help() {
        if (getData().menu_elements.usage) {
            System::LaunchBrowser(U"https://www.egaroucid.nyanyan.dev/ja/usage/");
        }
        if (getData().menu_elements.website) {
            if (language.get("lang_name") == U"日本語") {
                System::LaunchBrowser(U"https://www.egaroucid.nyanyan.dev/ja/");
            }
            else {
                System::LaunchBrowser(U"https://www.egaroucid.nyanyan.dev/en/");
            }
        }
        if (getData().menu_elements.bug_report) {
            System::LaunchBrowser(U"https://docs.google.com/forms/d/e/1FAIpQLSd6ML1T1fc707luPEefBXuImMnlM9cQP8j-YHKiSyFoS-8rmQ/viewform?usp=sf_link");
        }
        if (getData().menu_elements.license_egaroucid) {
            //System::LaunchBrowser(U"LICENSE");
            System::LaunchBrowser(U"https://github.com/Nyanyan/Egaroucid/blob/main/LICENSE");
        }
        if (getData().menu_elements.license_siv3d) {
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
                    getData().menu = create_menu(&getData().menu_elements);
                    start_game_button.init(START_GAME_BUTTON_SX, START_GAME_BUTTON_SY, START_GAME_BUTTON_WIDTH, START_GAME_BUTTON_HEIGHT, START_GAME_BUTTON_RADIUS, language.get("play", "start_game"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
                    pass_button.init(PASS_BUTTON_SX, PASS_BUTTON_SY, PASS_BUTTON_WIDTH, PASS_BUTTON_HEIGHT, PASS_BUTTON_RADIUS, language.get("play", "pass"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
                    re_calculate_openings();
                }
            }
        }
    }

    void interact_graph() {
        getData().graph_resources.n_discs = graph.update_n_discs(getData().graph_resources.nodes[0], getData().graph_resources.nodes[1], getData().graph_resources.n_discs);
        if (!KeyLeft.pressed() && !KeyA.pressed()) {
            move_board_button_status.left_pushed = BUTTON_NOT_PUSHED;
        }
        if (!KeyRight.pressed() && !KeyD.pressed()) {
            move_board_button_status.right_pushed = BUTTON_NOT_PUSHED;
        }

        if (MouseX1.down() || KeyLeft.down() || KeyA.down() || (move_board_button_status.left_pushed != BUTTON_NOT_PUSHED && tim() - move_board_button_status.left_pushed >= BUTTON_LONG_PRESS_THRESHOLD)) {
            --getData().graph_resources.n_discs;
            getData().graph_resources.delta = -1;
            if (KeyLeft.down() || KeyA.down()) {
                move_board_button_status.left_pushed = tim();
            }
        }
        else if (MouseX2.down() || KeyRight.down() || KeyD.down() || (move_board_button_status.right_pushed != BUTTON_NOT_PUSHED && tim() - move_board_button_status.right_pushed >= BUTTON_LONG_PRESS_THRESHOLD)) {
            ++getData().graph_resources.n_discs;
            getData().graph_resources.delta = 1;
            if (KeyRight.down() || KeyD.down()) {
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
        if (getData().history_elem.board.is_end()){
            int sgn = getData().history_elem.player == 0 ? 1 : -1;
            getData().graph_resources.nodes[getData().graph_resources.branch].back().v = sgn * getData().history_elem.board.score_player();
            getData().graph_resources.nodes[getData().graph_resources.branch].back().level = N_LEVEL - 1;
        }
        reset_hint();
        reset_book_additional_features();
    }

    void interact_move() {
        uint64_t legal = getData().history_elem.board.get_legal();
        for (int_fast8_t cell = 0; cell < HW2; ++cell) {
            if (1 & (legal >> (HW2_M1 - cell))) {
                int x = cell % HW;
                int y = cell / HW;
                Rect cell_rect(BOARD_SX + x * BOARD_CELL_SIZE, BOARD_SY + y * BOARD_CELL_SIZE, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
                if (cell_rect.leftClicked()) {
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
                    stop_calculating();
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
                    ai_status.ai_future = std::async(std::launch::async, ai, getData().history_elem.board, getData().menu_elements.level, getData().menu_elements.use_book, getData().menu_elements.book_acc_level, true, true);
                    ai_status.ai_thinking = true;
                }
            }
            else if (ai_status.ai_future.valid()) {
                if (ai_status.ai_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    Search_result search_result = ai_status.ai_future.get();
                    if (1 & (legal >> search_result.policy)) {
                        int player_bef = getData().history_elem.player;
                        int sgn = getData().history_elem.player == 0 ? 1 : -1;
                        getData().graph_resources.nodes[getData().graph_resources.branch].back().v = sgn * search_result.value;
                        getData().graph_resources.nodes[getData().graph_resources.branch].back().level = getData().menu_elements.level;
                        move_processing(HW2_M1 - search_result.policy);
                        if (getData().history_elem.player == player_bef && (getData().menu_elements.ai_put_black ^ getData().menu_elements.ai_put_white) && getData().menu_elements.pause_when_pass && !getData().history_elem.board.is_end())
                            pausing_in_pass = true;
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

    Menu create_menu(Menu_elements* menu_elements) {
        Menu menu;
        menu_title title;
        menu_elem menu_e, side_menu, side_side_menu;



        title.init(language.get("play", "game"));

        menu_e.init_button(language.get("play", "new_game"), &menu_elements->start_game);
        title.push(menu_e);
        menu_e.init_button(language.get("play", "new_game_human_black"), &menu_elements->start_game_human_black);
        title.push(menu_e);
        menu_e.init_button(language.get("play", "new_game_human_white"), &menu_elements->start_game_human_white);
        title.push(menu_e);
        menu_e.init_button(language.get("play", "new_selfplay"), &menu_elements->start_selfplay);
        title.push(menu_e);
        menu_e.init_button(language.get("play", "analyze"), &menu_elements->analyze);
        title.push(menu_e);

        menu.push(title);




        title.init(language.get("settings", "settings"));

        menu_e.init_check(language.get("ai_settings", "use_book"), &menu_elements->use_book, menu_elements->use_book);
        title.push(menu_e);
        menu_e.init_bar(language.get("ai_settings", "book_accuracy_level"), &menu_elements->book_acc_level, menu_elements->book_acc_level, 0, BOOK_ACCURACY_LEVEL_INF);
        title.push(menu_e);
        menu_e.init_bar(language.get("ai_settings", "level"), &menu_elements->level, menu_elements->level, 1, 60);
        title.push(menu_e);
        menu_e.init_bar(language.get("settings", "thread", "thread"), &menu_elements->n_threads, menu_elements->n_threads, 1, 48);
        title.push(menu_e);
        menu_e.init_bar(language.get("settings", "hash_level"), &menu_elements->hash_level, menu_elements->hash_level, MIN_HASH_LEVEL, MAX_HASH_LEVEL);
        title.push(menu_e);

        menu_e.init_check(language.get("settings", "play", "ai_put_black"), &menu_elements->ai_put_black, menu_elements->ai_put_black);
        title.push(menu_e);
        menu_e.init_check(language.get("settings", "play", "ai_put_white"), &menu_elements->ai_put_white, menu_elements->ai_put_white);
        title.push(menu_e);

        menu_e.init_check(language.get("settings", "play", "pause_when_pass"), &menu_elements->pause_when_pass, menu_elements->pause_when_pass);
        title.push(menu_e);

        menu.push(title);




        title.init(language.get("display", "display"));

        menu_e.init_button(language.get("display", "cell", "display_on_cell"), &menu_elements->dummy);
        side_menu.init_check(language.get("display", "cell", "legal"), &menu_elements->show_legal, menu_elements->show_legal);
        menu_e.push(side_menu);
        side_menu.init_check(language.get("display", "cell", "disc_value"), &menu_elements->use_disc_hint, menu_elements->use_disc_hint);
        side_side_menu.init_bar(language.get("display", "cell", "disc_value_number"), &menu_elements->n_disc_hint, menu_elements->n_disc_hint, 1, SHOW_ALL_HINT);
        side_menu.push(side_side_menu);
        menu_e.push(side_menu);
        side_menu.init_check(language.get("display", "cell", "umigame_value"), &menu_elements->use_umigame_value, menu_elements->use_umigame_value);
        menu_e.push(side_menu);
        side_menu.init_check(language.get("display", "cell", "opening"), &menu_elements->show_opening_on_cell, menu_elements->show_opening_on_cell);
        menu_e.push(side_menu);
        side_menu.init_check(language.get("display", "cell", "next_move"), &menu_elements->show_next_move, menu_elements->show_next_move);
        side_side_menu.init_check(language.get("display", "cell", "next_move_change_view"), &menu_elements->show_next_move_change_view, menu_elements->show_next_move_change_view);
        side_menu.push(side_side_menu);
        menu_e.push(side_menu);
        side_menu.init_check(language.get("display", "cell", "show_book_accuracy"), &menu_elements->show_book_accuracy, menu_elements->show_book_accuracy);
        menu_e.push(side_menu);
        title.push(menu_e);

        menu_e.init_button(language.get("display", "disc", "display_on_disc"), &menu_elements->dummy);
        side_menu.init_check(language.get("display", "disc", "last_move"), &menu_elements->show_last_move, menu_elements->show_last_move);
        menu_e.push(side_menu);
        side_menu.init_check(language.get("display", "disc", "stable"), &menu_elements->show_stable_discs, menu_elements->show_stable_discs);
        menu_e.push(side_menu);
        side_menu.init_check(language.get("display", "disc", "play_ordering"), &menu_elements->show_play_ordering, menu_elements->show_play_ordering);
        menu_e.push(side_menu);
        title.push(menu_e);

        menu_e.init_check(language.get("display", "graph"), &menu_elements->show_graph, menu_elements->show_graph);
        title.push(menu_e);
        menu_e.init_check(language.get("display", "log"), &menu_elements->show_log, menu_elements->show_log);
        title.push(menu_e);
        menu_e.init_check(language.get("display", "change_color_type"), &menu_elements->change_color_type, menu_elements->change_color_type);
        title.push(menu_e);

        menu.push(title);




        title.init(language.get("operation", "operation"));

        menu_e.init_button(language.get("operation", "put_1_move_by_ai"), &menu_elements->put_1_move_by_ai);
        title.push(menu_e);
        menu_e.init_button(language.get("operation", "forward"), &menu_elements->forward);
        title.push(menu_e);
        menu_e.init_button(language.get("operation", "backward"), &menu_elements->backward);
        title.push(menu_e);
        menu_e.init_button(language.get("operation", "undo"), &menu_elements->undo);
        title.push(menu_e);
        menu_e.init_button(language.get("operation", "save_this_branch"), &menu_elements->save_this_branch);
        title.push(menu_e);
        menu_e.init_button(language.get("operation", "generate_random_board", "generate_random_board"), &menu_elements->dummy);
        side_menu.init_button(language.get("operation", "generate_random_board", "generate"), &menu_elements->generate_random_board);
        menu_e.push(side_menu);
        side_menu.init_bar(language.get("operation", "generate_random_board", "generate_n_moves"), &menu_elements->generate_random_board_moves, menu_elements->generate_random_board_moves, 1, 60);
        menu_e.push(side_menu);
        title.push(menu_e);

        menu_e.init_button(language.get("operation", "convert", "convert"), &menu_elements->dummy);
        side_menu.init_button(language.get("operation", "convert", "vertical"), &menu_elements->convert_180);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("operation", "convert", "black_line"), &menu_elements->convert_blackline);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("operation", "convert", "white_line"), &menu_elements->convert_whiteline);
        menu_e.push(side_menu);
        title.push(menu_e);

        menu_e.init_button(language.get("operation", "ai_operation", "ai_operation"), &menu_elements->dummy);
        side_menu.init_button(language.get("operation", "ai_operation", "stop_calculating"), &menu_elements->stop_calculating);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("operation", "ai_operation", "cache_clear"), &menu_elements->cache_clear);
        menu_e.push(side_menu);
        title.push(menu_e);

        menu.push(title);



        title.init(language.get("in_out", "in_out"));

        menu_e.init_button(language.get("in_out", "in"), &menu_elements->dummy);
        side_menu.init_button(language.get("in_out", "input_transcript"), &menu_elements->input_transcript);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("in_out", "input_board"), &menu_elements->input_board);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("in_out", "edit_board"), &menu_elements->edit_board);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("in_out", "input_game"), &menu_elements->input_game);
        menu_e.push(side_menu);
        title.push(menu_e);

        menu_e.init_button(language.get("in_out", "out"), &menu_elements->dummy);
        side_menu.init_button(language.get("in_out", "output_transcript"), &menu_elements->copy_transcript);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("in_out", "output_board"), &menu_elements->copy_board);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("in_out", "screen_shot"), &menu_elements->screen_shot);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("in_out", "board_image"), &menu_elements->board_image);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("in_out", "output_game"), &menu_elements->save_game);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("in_out", "output_bitboard"), &menu_elements->dummy);
        side_side_menu.init_button(language.get("in_out", "player_opponent"), &menu_elements->output_bitboard_player_opponent);
        side_menu.push(side_side_menu);
        side_side_menu.init_button(language.get("in_out", "black_white"), &menu_elements->output_bitboard_black_white);
        side_menu.push(side_side_menu);
        menu_e.push(side_menu);
        title.push(menu_e);

        menu.push(title);




        title.init(language.get("book", "book"));

        menu_e.init_button(language.get("book", "settings"), &menu_elements->dummy);
        side_menu.init_bar_check(language.get("book", "depth"), &menu_elements->book_learn_depth, menu_elements->book_learn_depth, 0, 60, &menu_elements->use_book_learn_depth, menu_elements->use_book_learn_depth);
        menu_e.push(side_menu);
        side_menu.init_bar_check(language.get("book", "error_per_move"), &menu_elements->book_learn_error_per_move, menu_elements->book_learn_error_per_move, 0, 24, &menu_elements->use_book_learn_error_per_move, menu_elements->use_book_learn_error_per_move);
        menu_e.push(side_menu);
        side_menu.init_bar_check(language.get("book", "error_sum"), &menu_elements->book_learn_error_sum, menu_elements->book_learn_error_sum, 0, 32, &menu_elements->use_book_learn_error_sum, menu_elements->use_book_learn_error_sum);
        menu_e.push(side_menu);
        title.push(menu_e);

        menu_e.init_button(language.get("book", "book_operation"), &menu_elements->dummy);
        side_menu.init_check(language.get("book", "right_click_to_modify"), &menu_elements->change_book_by_right_click, menu_elements->change_book_by_right_click);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("book", "book_deviate"), &menu_elements->book_start_deviate);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("book", "book_fix"), &menu_elements->book_start_fix);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("book", "book_reduce"), &menu_elements->book_start_reducing);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("book", "book_recalculate_leaf"), &menu_elements->book_start_recalculate_leaf);
        menu_e.push(side_menu);
        title.push(menu_e);

        menu_e.init_button(language.get("book", "file_operation"), &menu_elements->dummy);
        side_menu.init_button(language.get("book", "import_book"), &menu_elements->import_book);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("book", "export_book"), &menu_elements->export_book);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("book", "book_merge"), &menu_elements->book_merge);
        menu_e.push(side_menu);
        side_menu.init_button(language.get("book", "book_reference"), &menu_elements->book_reference);
        menu_e.push(side_menu);
        title.push(menu_e);

        menu.push(title);




        title.init(language.get("help", "help"));
        menu_e.init_button(language.get("help", "usage"), &menu_elements->usage);
        title.push(menu_e);
        menu_e.init_button(language.get("help", "website"), &menu_elements->website);
        title.push(menu_e);
        menu_e.init_button(language.get("help", "bug_report"), &menu_elements->bug_report);
        title.push(menu_e);
        menu_e.init_check(language.get("help", "auto_update_check"), &menu_elements->auto_update_check, menu_elements->auto_update_check);
        title.push(menu_e);
        menu_e.init_button(language.get("help", "license_egaroucid"), &menu_elements->license_egaroucid);
        title.push(menu_e);
        menu_e.init_button(language.get("help", "license_siv3d"), &menu_elements->license_siv3d);
        title.push(menu_e);
        menu.push(title);





        title.init(U"Language");
        for (int i = 0; i < (int)getData().resources.language_names.size(); ++i) {
            menu_e.init_radio(getData().resources.lang_img[i], &menu_elements->languages[i], menu_elements->languages[i]);
            title.push(menu_e);
        }
        menu.push(title);




        menu.init(0, 0, 12, getData().fonts.font, getData().resources.checkbox, getData().resources.unchecked);
        return menu;
    }

    void draw_legal(uint64_t ignore) {
        uint64_t legal = getData().history_elem.board.get_legal();
        for (int cell = 0; cell < HW2; ++cell) {
            int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
            int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
            if (1 & (legal >> (HW2_M1 - cell))) {
                if ((1 & (ignore >> (HW2_M1 - cell))) == 0)
                    Circle(x, y, LEGAL_SIZE).draw(getData().colors.cyan);
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
                if (getData().menu_elements.show_next_move_change_view){
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
        for (History_elem &elem: getData().graph_resources.nodes[getData().graph_resources.branch]){
            int cell = elem.policy;
            if (0 <= cell && cell < HW2 && ((getData().history_elem.board.player | getData().history_elem.board.opponent) & (1ULL << cell))){
                int x = BOARD_SX + ((HW2_M1 - cell) % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
                int y = BOARD_SY + ((HW2_M1 - cell) / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
                bool is_black_disc = getData().history_elem.player == BLACK && (getData().history_elem.board.player & (1ULL << cell)) != 0;
                is_black_disc |= getData().history_elem.player == WHITE && (getData().history_elem.board.opponent & (1ULL << cell)) != 0;
                Color color = getData().colors.black;
                if (is_black_disc)
                    color = getData().colors.white;
                getData().fonts.font_bold(elem.board.n_discs() - 4).draw(24, Arg::center(x, y), color);
            }
        }
    }

    uint64_t draw_hint(bool ignore_book_info) {
        uint64_t res = 0ULL;
        if (ai_status.hint_available) {
            std::vector<Hint_info> hint_infos;
            for (int cell = 0; cell < HW2; ++cell) {
                if (ai_status.hint_use_stable[cell]) {
                    Hint_info hint_info;
                    hint_info.value = ai_status.hint_values_stable[cell];
                    hint_info.cell = cell;
                    hint_info.type = ai_status.hint_types_stable[cell];
                    hint_infos.emplace_back(hint_info);
                }
            }
            if (hint_infos.size()) {
                sort(hint_infos.begin(), hint_infos.end(), compare_hint_info);
                int sgn = getData().history_elem.player == 0 ? 1 : -1;
                int node_idx = getData().graph_resources.node_find(getData().graph_resources.branch, getData().graph_resources.n_discs);
                if (node_idx != -1) {
                    int min_hint_type = 1000;
                    for (Hint_info &hint_info: hint_infos)
                        min_hint_type = std::min(min_hint_type, hint_info.type);
                    if (getData().graph_resources.nodes[getData().graph_resources.branch][node_idx].level < min_hint_type) {
                        getData().graph_resources.nodes[getData().graph_resources.branch][node_idx].v = sgn * (int)round(hint_infos[0].value);
                        getData().graph_resources.nodes[getData().graph_resources.branch][node_idx].level = min_hint_type;
                    }
                }
            }
            int n_disc_hint = std::min((int)hint_infos.size(), getData().menu_elements.n_disc_hint);
            for (int i = 0; i < n_disc_hint; ++i) {
                int sx = BOARD_SX + (hint_infos[i].cell % HW) * BOARD_CELL_SIZE;
                int sy = BOARD_SY + (hint_infos[i].cell / HW) * BOARD_CELL_SIZE;
                Color color = getData().colors.white;
                Font font = getData().fonts.font;
                if (hint_infos[i].value == hint_infos[0].value) {
                    color = getData().colors.cyan;
                    font = getData().fonts.font_heavy;
                }
                font((int)round(hint_infos[i].value)).draw(18, sx + 3, sy, color);
                if (hint_infos[i].type == HINT_TYPE_BOOK) {
                    if (!ignore_book_info)
                        getData().fonts.font(U"book").draw(10, sx + 3, sy + 19, color);
                }
                else if (hint_infos[i].type > HINT_MAX_LEVEL) {
                    getData().fonts.font(Format(hint_infos[i].type) + U"%").draw(10, sx + 3, sy + 19, color);
                }
                else {
                    getData().fonts.font(U"Lv." + Format(hint_infos[i].type)).draw(10, sx + 3, sy + 19, color);
                }
                res |= 1ULL << (HW2_M1 - hint_infos[i].cell);
            }
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

    void hint_init_calculating() {
        uint64_t legal = getData().history_elem.board.get_legal();
        if (ai_status.hint_level == HINT_NOT_CALCULATING) {
            stop_calculating();
            resume_calculating();
            for (int cell = 0; cell < HW2; ++cell) {
                ai_status.hint_values[cell] = HINT_INIT_VALUE;
                ai_status.hint_use[cell] = (bool)(1 & (legal >> (HW2_M1 - cell)));
                ai_status.hint_types[cell] = HINT_NOT_CALCULATING;
            }
            for (int cell = 0; cell < HW2; ++cell) {
                ai_status.hint_values_stable[cell] = ai_status.hint_values[cell];
                ai_status.hint_use_stable[cell] = ai_status.hint_use[cell];
                ai_status.hint_types_stable[cell] = ai_status.hint_types[cell];
            }
        }
        else {
            ai_status.hint_available = true;
        }
        int before_level = ai_status.hint_level;
        int n_moves = getData().history_elem.board.n_discs() - 4;
        ++ai_status.hint_level;
        if (ai_status.hint_level > 1) {
            while (same_level(before_level, ai_status.hint_level, n_moves) && ai_status.hint_level <= getData().menu_elements.level) {
                ++ai_status.hint_level;
            }
        }
        if (ai_status.hint_level <= getData().menu_elements.level) {
            std::vector<std::pair<int, int>> value_cells;
            for (int cell = 0; cell < HW2; ++cell) {
                if (ai_status.hint_use[cell]) {
                    value_cells.emplace_back(std::make_pair(ai_status.hint_values[cell], cell));
                }
            }
            sort(value_cells.begin(), value_cells.end(), compare_value_cell);
            int n_legal = pop_count_ull(legal);
            int hint_adoption_threshold = getData().menu_elements.n_disc_hint + std::max(1, n_legal * (getData().menu_elements.level - ai_status.hint_level) / getData().menu_elements.level);
            hint_adoption_threshold = std::min(hint_adoption_threshold, (int)value_cells.size());
            ai_status.hint_task_stack.clear();
            Board board;
            Flip flip;
            int next_task_size = 0;
            int idx = 0;
            for (std::pair<int, int>& value_cell : value_cells) {
                if (idx++ >= hint_adoption_threshold) {
                    break;
                }
                if (ai_status.hint_types[value_cell.second] != HINT_TYPE_BOOK) {
                    ++next_task_size;
                }
            }
            ai_status.hint_use_multi_thread = next_task_size < getData().menu_elements.n_threads;
            
            if (ai_status.hint_level <= 6) {
                ai_status.hint_use_multi_thread = false;
            }
            
            //ai_status.hint_use_multi_thread = true;
            idx = 0;
            for (std::pair<int, int>& value_cell : value_cells) {
                if (idx++ >= hint_adoption_threshold) {
                    break;
                }
                if (ai_status.hint_types[value_cell.second] != HINT_TYPE_BOOK) {
                    board = getData().history_elem.board;
                    calc_flip(&flip, &board, (uint_fast8_t)(HW2_M1 - value_cell.second));
                    board.move_board(&flip);
                    ai_status.hint_task_stack.emplace_back(std::make_pair(value_cell.second, std::bind(ai_hint, board, ai_status.hint_level, getData().menu_elements.use_book, ai_status.hint_use_multi_thread, false)));
                }
            }
            ai_status.hint_n_doing_tasks = 0;
            ai_status.hint_calculating = true;
            std::cerr << "hint search level " << ai_status.hint_level << " n_tasks " << ai_status.hint_task_stack.size() << " multi_threading " << ai_status.hint_use_multi_thread << std::endl;
        }
    }

    void hint_do_task() {
        if (!changing_scene) {
            if (ai_status.hint_n_doing_tasks > 0) {
                for (int cell = 0; cell < HW2; ++cell) {
                    if (ai_status.hint_future[cell].valid()) {
                        if (ai_status.hint_future[cell].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                            Search_result search_result = ai_status.hint_future[cell].get();
                            if (ai_status.hint_values[cell] == HINT_INIT_VALUE || search_result.is_end_search || search_result.depth == SEARCH_BOOK) {
                                ai_status.hint_values[cell] = -search_result.value;
                                if (search_result.depth == SEARCH_BOOK)
                                    ai_status.hint_values[cell] += 0.49999999; // to give priority for book values
                            }
                            else {
                                ai_status.hint_values[cell] -= 1.2 * search_result.value;
                                ai_status.hint_values[cell] /= 2.2;
                            }
                            if (search_result.depth == SEARCH_BOOK) {
                                ai_status.hint_types[cell] = HINT_TYPE_BOOK;
                            }
                            else if (search_result.is_end_search) {
                                ai_status.hint_types[cell] = search_result.probability;
                            }
                            else {
                                ai_status.hint_types[cell] = ai_status.hint_level;
                            }
                            --ai_status.hint_n_doing_tasks;
                        }
                    }
                }
            }
            if (ai_status.hint_task_stack.size() == 0 && ai_status.hint_n_doing_tasks == 0) {
                for (int cell = 0; cell < HW2; ++cell) {
                    ai_status.hint_use_stable[cell] = ai_status.hint_use[cell];
                    ai_status.hint_values_stable[cell] = ai_status.hint_values[cell];
                    ai_status.hint_types_stable[cell] = ai_status.hint_types[cell];
                }
                ai_status.hint_calculating = false;
            }
            else if (ai_status.hint_task_stack.size()) {
                int loop_time = std::min((int)ai_status.hint_task_stack.size(), getData().menu_elements.n_threads - ai_status.hint_n_doing_tasks);
                
                if (ai_status.hint_use_multi_thread) {
                    if (std::max(1, getData().menu_elements.n_threads / HINT_SINGLE_TASK_N_THREAD) - ai_status.hint_n_doing_tasks > 0) {
                        loop_time = std::min((int)ai_status.hint_task_stack.size(), std::max(1, getData().menu_elements.n_threads / HINT_SINGLE_TASK_N_THREAD) - ai_status.hint_n_doing_tasks);
                    }
                    else {
                        loop_time = 0;
                    }
                }
                else {
                    loop_time = std::min((int)ai_status.hint_task_stack.size(), getData().menu_elements.n_threads - ai_status.hint_n_doing_tasks);
                }
                
                if (loop_time > 0) {
                    for (int i = 0; i < loop_time; ++i) {
                        std::pair<int, std::function<Search_result()>> task = ai_status.hint_task_stack.back();
                        ai_status.hint_task_stack.pop_back();
                        ai_status.hint_future[task.first] = std::async(std::launch::async, task.second);
                    }
                    ai_status.hint_n_doing_tasks += loop_time;
                }
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
                    getData().graph_resources.nodes[getData().graph_resources.branch][i].level = getData().menu_elements.level;
                    task_finished = true;
                }
            }
        }
        if (task_finished) {
            analyze_do_task();
        }
    }

    void copy_transcript() {
        std::string transcript;
        int inspect_switch_n_discs = INF;
        if (getData().graph_resources.branch == 1) {
            if (getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
                inspect_switch_n_discs = getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs();
            }
            else {
                std::cerr << "no node found in inspect mode" << std::endl;
            }
        }
        std::cerr << inspect_switch_n_discs << std::endl;
        for (History_elem& history_elem : getData().graph_resources.nodes[GRAPH_MODE_NORMAL]) {
            if (history_elem.board.n_discs() + 1 >= inspect_switch_n_discs || history_elem.board.n_discs() >= getData().history_elem.board.n_discs()) {
                break;
            }
            if (history_elem.next_policy != -1) {
                transcript += idx_to_coord(history_elem.next_policy);
            }
        }
        if (inspect_switch_n_discs != INF) {
            if (getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].policy != -1) {
                transcript += idx_to_coord(getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].policy);
            }
            for (History_elem& history_elem : getData().graph_resources.nodes[GRAPH_MODE_INSPECT]) {
                if (history_elem.board.n_discs() >= getData().history_elem.board.n_discs()) {
                    break;
                }
                if (history_elem.next_policy != -1) {
                    transcript += idx_to_coord(history_elem.next_policy);
                }
            }
        }
        std::cerr << transcript << std::endl;
        Clipboard::SetText(Unicode::Widen(transcript));
    }

    void copy_board() {
        std::string board_str;
        for (int i = 0; i < HW2; ++i){
            if (getData().history_elem.player == BLACK){
                if (getData().history_elem.board.player & (1ULL << (HW2_M1 - i))){
                    board_str += "X";
                } else if (getData().history_elem.board.opponent & (1ULL << (HW2_M1 - i))){
                    board_str += "O";
                } else{
                    board_str += "-";
                }
            } else{
                if (getData().history_elem.board.player & (1ULL << (HW2_M1 - i))){
                    board_str += "O";
                } else if (getData().history_elem.board.opponent & (1ULL << (HW2_M1 - i))){
                    board_str += "X";
                } else{
                    board_str += "-";
                }
            }
        }
        board_str += " ";
        if (getData().history_elem.player == BLACK){
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
        ss << " ";
        ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << getData().history_elem.board.opponent;
        std::string res = ss.str();
        std::cerr << res << std::endl;
        Clipboard::SetText(Unicode::Widen(res));
    }

    void copy_bitboard_black_white() {
        std::ostringstream ss;
        if (getData().history_elem.player == BLACK){
            ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << getData().history_elem.board.player;
            ss << " ";
            ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << getData().history_elem.board.opponent;
        } else{
            ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << getData().history_elem.board.opponent;
            ss << " ";
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
            !getData().history_elem.board.is_end();
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
                        if (board.get_legal() == 0ULL){
                            board.pass();
                                umigame_status.umigame_future[cell] = std::async(std::launch::async, get_umigame, board, n_player ^ 1);
                            board.pass();
                        } else
                            umigame_status.umigame_future[cell] = std::async(std::launch::async, get_umigame, board, n_player);
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
                        }
                        else {
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
                    getData().fonts.font_heavy(umigame_status.umigame[cell].b).draw(13, Arg::bottomRight(sx + BOARD_CELL_SIZE - 3, sy + BOARD_CELL_SIZE - 17), getData().colors.black);
                    getData().fonts.font_heavy(umigame_status.umigame[cell].w).draw(13, Arg::bottomRight(sx + BOARD_CELL_SIZE - 3, sy + BOARD_CELL_SIZE - 1), getData().colors.white);
                }
            }
        }
    }

    void draw_book_n_lines(uint64_t legal_ignore) {
        uint64_t legal = getData().history_elem.board.get_legal();
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
            if (1 & (legal_ignore >> cell)) {
                int sx = BOARD_SX + ((HW2_M1 - cell) % HW) * BOARD_CELL_SIZE;
                int sy = BOARD_SY + ((HW2_M1 - cell) / HW) * BOARD_CELL_SIZE;
                Board board = getData().history_elem.board;
                Flip flip;
                calc_flip(&flip, &board, cell);
                board.move_board(&flip);
                if (book.contain(board)){
                    uint32_t n_lines = book.get(board).n_lines;
                    String n_lines_str = Format(n_lines);
                    if (n_lines >= 1000000000){
                        n_lines_str = Format(n_lines / 1000000000) + U"G";
                    } else if (n_lines >= 1000000){
                        n_lines_str = Format(n_lines / 1000000) + U"M";
                    } else if (n_lines >= 1000){
                        n_lines_str = Format(n_lines / 1000) + U"K";
                    }
                    getData().fonts.font_heavy(n_lines_str).draw(9, sx + 4, sy + 21, getData().colors.white);
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
                        if (board.get_legal() == 0ULL){
                            board.pass();
                                book_accuracy_status.book_accuracy_future[cell] = std::async(std::launch::async, get_book_accuracy, board);
                            board.pass();
                        } else
                            book_accuracy_status.book_accuracy_future[cell] = std::async(std::launch::async, get_book_accuracy, board);
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
                        }
                        else {
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
        uint64_t legal = getData().history_elem.board.get_legal();
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
            if (1 & (legal_ignore >> cell)) {
                int sx = BOARD_SX + ((HW2_M1 - cell) % HW) * BOARD_CELL_SIZE;
                int sy = BOARD_SY + ((HW2_M1 - cell) / HW) * BOARD_CELL_SIZE;
                if (book_accuracy_status.book_accuracy[cell] != BOOK_ACCURACY_LEVEL_UNDEFINED){
                    std::string judge = {(char)book_accuracy_status.book_accuracy[cell] + 'A'};
                    Board board = getData().history_elem.board;
                    Flip flip;
                    calc_flip(&flip, &board, cell);
                    board.move_board(&flip);
                    int book_level = book.get(board).level;
                    String book_level_info = Format(book_level);
                    if (book_level == LEVEL_HUMAN)
                        book_level_info = U"S";
                    getData().fonts.font_heavy(Unicode::Widen(judge) + U" " + book_level_info).draw(9, sx + 4, sy + 33, getData().colors.white);
                }
            }
        }
    }

    void change_book_by_right_click() {
        if (getData().book_information.changing != BOOK_CHANGE_NO_CELL) {
            getData().fonts.font(language.get("book", "changed_value") + U"(" + Unicode::Widen(idx_to_coord(getData().book_information.changing)) + U"): " + getData().book_information.val_str).draw(13, CHANGE_BOOK_INFO_SX, CHANGE_BOOK_INFO_SY, getData().colors.white);
            if (KeyEscape.down()) {
                getData().book_information.changing = BOOK_CHANGE_NO_CELL;
            }
            else if (Key0.down() || KeyNum0.down()) {
                if (getData().book_information.val_str != U"0" && getData().book_information.val_str != U"-") {
                    getData().book_information.val_str += U"0";
                }
            }
            else if (Key1.down() || KeyNum1.down()) {
                getData().book_information.val_str += U"1";
            }
            else if (Key2.down() || KeyNum2.down()) {
                getData().book_information.val_str += U"2";
            }
            else if (Key3.down() || KeyNum3.down()) {
                getData().book_information.val_str += U"3";
            }
            else if (Key4.down() || KeyNum4.down()) {
                getData().book_information.val_str += U"4";
            }
            else if (Key5.down() || KeyNum5.down()) {
                getData().book_information.val_str += U"5";
            }
            else if (Key6.down() || KeyNum6.down()) {
                getData().book_information.val_str += U"6";
            }
            else if (Key7.down() || KeyNum7.down()) {
                getData().book_information.val_str += U"7";
            }
            else if (Key8.down() || KeyNum8.down()) {
                getData().book_information.val_str += U"8";
            }
            else if (Key9.down() || KeyNum9.down()) {
                getData().book_information.val_str += U"9";
            }
            else if (KeyMinus.down()) {
                if (getData().book_information.val_str == U"" || getData().book_information.val_str == U"-") {
                    getData().book_information.val_str += U"-";
                }
            }
            else if (KeyBackspace.down()) {
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
            }
            else if (KeyEnter.down()) {
                state = 2;
            }
            if (state) {
                if (getData().book_information.changing == cell) {
                    if (getData().book_information.val_str.size() == 0) {
                        getData().book_information.changing = BOOK_CHANGE_NO_CELL;
                    }
                    else {
                        int changed_book_value = ParseOr<int>(getData().book_information.val_str, CHANGE_BOOK_ERR);
                        if (changed_book_value < -HW2 || HW2 < changed_book_value) {
                            getData().book_information.val_str.clear();
                            changed_book_value = CHANGE_BOOK_ERR;
                        }
                        if (changed_book_value != CHANGE_BOOK_ERR) {
                            std::cerr << "new value " << changed_book_value << std::endl;
                            Flip flip;
                            calc_flip(&flip, &getData().history_elem.board, getData().book_information.changing);
                            Board b = getData().history_elem.board.move_copy(&flip);
                            book.change(getData().history_elem.board.move_copy(&flip), -changed_book_value, LEVEL_HUMAN);
                            reset_book_additional_information();
                            getData().book_information.changed = true;
                            getData().book_information.changing = BOOK_CHANGE_NO_CELL;
                            getData().book_information.val_str.clear();
                            stop_calculating();
                            resume_calculating();
                        }
                        else {
                            Flip flip;
                            calc_flip(&flip, &getData().history_elem.board, getData().book_information.changing);
                            Board b = getData().history_elem.board.move_copy(&flip);
                            book.delete_elem(b);
                            reset_book_additional_information();
                            getData().book_information.changed = true;
                            getData().book_information.changing = BOOK_CHANGE_NO_CELL;
                            getData().book_information.val_str.clear();
                            stop_calculating();
                            resume_calculating();
                        }
                    }
                }
                else if (state == 1) {
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
            }
            else {
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
            }
            else {
                now_node_idx = std::max(0, now_node_idx - 1);
                getData().history_elem.opening_name = getData().graph_resources.nodes[getData().graph_resources.branch][now_node_idx].opening_name;
            }
        }
    }
};