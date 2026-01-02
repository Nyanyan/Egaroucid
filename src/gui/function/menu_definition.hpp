/*
    Egaroucid Project

    @file menu_definition.hpp
        Main scene for GUI
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once

#include "const/gui_common.hpp"
#include "menu.hpp"
#include "language.hpp"
#include "shortcut_key.hpp"
#include "./../../engine/setting.hpp"

String get_shortcut_key_info(String key) {
    String res = shortcut_keys.get_shortcut_key_str(key);
    if (res == U"") {
        return U"";
    }
    return U" (" + res + U")";
}

constexpr int AI_MAX_LOSS_INF = 129;
constexpr int AI_LOSS_PERCENTAGE_INF = 100;

Menu create_menu(Menu_elements* menu_elements, Resources *resources, Font font, std::string lang_name) {
    Menu menu;
    menu_title title;
    menu_elem menu_e, side_menu, side_side_menu;



    title.init(language.get("play", "game"));
        menu_e.init_button(language.get("play", "new_game") + get_shortcut_key_info(U"new_game"), &menu_elements->start_game);
        title.push(menu_e);
        menu_e.init_button(language.get("play", "new_game_human_black") + get_shortcut_key_info(U"new_game_human_black"), &menu_elements->start_game_human_black);
        title.push(menu_e);
        menu_e.init_button(language.get("play", "new_game_human_white") + get_shortcut_key_info(U"new_game_human_white"), &menu_elements->start_game_human_white);
        title.push(menu_e);
        menu_e.init_button(language.get("play", "new_selfplay") + get_shortcut_key_info(U"new_selfplay"), &menu_elements->start_selfplay);
        title.push(menu_e);
        menu_e.init_button(language.get("play", "analyze") + get_shortcut_key_info(U"analyze"), &menu_elements->analyze);
        title.push(menu_e);
        menu_e.init_button(language.get("play", "game_information") + get_shortcut_key_info(U"game_information"), &menu_elements->game_information);
        title.push(menu_e);
    menu.push(title);




    title.init(language.get("settings", "settings"));
        menu_e.init_check(language.get("ai_settings", "use_book") + get_shortcut_key_info(U"use_book"), &menu_elements->use_book, menu_elements->use_book);
        title.push(menu_e);
        //menu_e.init_bar(language.get("ai_settings", "book_accuracy_level"), &menu_elements->book_acc_level, menu_elements->book_acc_level, 0, BOOK_ACCURACY_LEVEL_INF);
        //title.push(menu_e);
        menu_e.init_check(language.get("ai_settings", "accept_ai_loss") + get_shortcut_key_info(U"accept_ai_loss"), &menu_elements->accept_ai_loss, menu_elements->accept_ai_loss);
            side_menu.init_bar(language.get("ai_settings", "max_loss"), &menu_elements->max_loss, menu_elements->max_loss, 1, AI_MAX_LOSS_INF);
            menu_e.push(side_menu);
            side_menu.init_bar(language.get("ai_settings", "loss_percentage"), &menu_elements->loss_percentage, menu_elements->loss_percentage, 1, AI_LOSS_PERCENTAGE_INF);
            menu_e.push(side_menu);
        title.push(menu_e);
        menu_e.init_bar(language.get("ai_settings", "level"), &menu_elements->level, menu_elements->level, 1, 60);
        title.push(menu_e);
        menu_e.init_bar(language.get("settings", "thread", "thread"), &menu_elements->n_threads, menu_elements->n_threads, 1, 48);
        title.push(menu_e);
#if USE_CHANGEABLE_HASH_LEVEL
        menu_e.init_bar(language.get("settings", "hash_level"), &menu_elements->hash_level, menu_elements->hash_level, MIN_HASH_LEVEL, MAX_HASH_LEVEL);
        title.push(menu_e);
#endif
        menu_e.init_check(language.get("settings", "play", "ai_put_black") + get_shortcut_key_info(U"ai_put_black"), &menu_elements->ai_put_black, menu_elements->ai_put_black);
        title.push(menu_e);
        menu_e.init_check(language.get("settings", "play", "ai_put_white") + get_shortcut_key_info(U"ai_put_white"), &menu_elements->ai_put_white, menu_elements->ai_put_white);
        title.push(menu_e);
        menu_e.init_check(language.get("settings", "play", "pause_when_pass") + get_shortcut_key_info(U"pause_when_pass"), &menu_elements->pause_when_pass, menu_elements->pause_when_pass);
        title.push(menu_e);
        menu_e.init_check(language.get("settings", "play", "force_specified_openings") + get_shortcut_key_info(U"force_specified_openings"), &menu_elements->force_specified_openings, menu_elements->force_specified_openings);
            side_menu.init_button(language.get("settings", "play", "opening_setting") + get_shortcut_key_info(U"opening_setting"), &menu_elements->opening_setting);
            menu_e.push(side_menu);
        title.push(menu_e);
        menu_e.init_button(language.get("settings", "shortcut_keys", "settings") + get_shortcut_key_info(U"shortcut_key_setting"), &menu_elements->shortcut_key_setting);
        title.push(menu_e);
    menu.push(title);




    title.init(language.get("display", "display"));
        menu_e.init_button(language.get("display", "cell", "display_on_cell"), &menu_elements->dummy);
            side_menu.init_check(language.get("display", "cell", "legal") + get_shortcut_key_info(U"show_legal"), &menu_elements->show_legal, menu_elements->show_legal);
            menu_e.push(side_menu);
            side_menu.init_check(language.get("display", "cell", "disc_value") + get_shortcut_key_info(U"show_disc_hint"), &menu_elements->use_disc_hint, menu_elements->use_disc_hint);
                side_side_menu.init_bar(language.get("display", "cell", "disc_value_number"), &menu_elements->n_disc_hint, menu_elements->n_disc_hint, 1, SHOW_ALL_HINT);
                side_menu.push(side_side_menu);
                side_side_menu.init_check(language.get("display", "cell", "show_hint_level") + get_shortcut_key_info(U"show_hint_level"), &menu_elements->show_hint_level, menu_elements->show_hint_level);
                side_menu.push(side_side_menu);
                side_side_menu.init_check(language.get("display", "cell", "show_value_when_ai_calculating") + get_shortcut_key_info(U"show_value_when_ai_calculating"), &menu_elements->show_value_when_ai_calculating, menu_elements->show_value_when_ai_calculating);
                side_menu.push(side_side_menu);
                side_side_menu.init_check(language.get("display", "cell", "show_book_accuracy") + get_shortcut_key_info(U"show_book_accuracy"), &menu_elements->show_book_accuracy, menu_elements->show_book_accuracy);
                side_menu.push(side_side_menu);
                side_side_menu.init_check(language.get("display", "cell", "hint_colorize") + get_shortcut_key_info(U"hint_colorize"), &menu_elements->hint_colorize, menu_elements->hint_colorize);
                side_menu.push(side_side_menu);
            menu_e.push(side_menu);
            side_menu.init_check(language.get("display", "cell", "umigame_value") + get_shortcut_key_info(U"show_umigame_value"), &menu_elements->use_umigame_value, menu_elements->use_umigame_value);
            side_side_menu.init_bar(language.get("display", "cell", "depth"), &menu_elements->umigame_value_depth, menu_elements->umigame_value_depth, 1, 60);
            side_menu.push(side_side_menu);
            menu_e.push(side_menu);
            side_menu.init_check(language.get("display", "cell", "opening") + get_shortcut_key_info(U"show_opening_on_cell"), &menu_elements->show_opening_on_cell, menu_elements->show_opening_on_cell);
            menu_e.push(side_menu);
            side_menu.init_check(language.get("display", "cell", "next_move") + get_shortcut_key_info(U"show_next_move"), &menu_elements->show_next_move, menu_elements->show_next_move);
            side_side_menu.init_check(language.get("display", "cell", "next_move_change_view"), &menu_elements->show_next_move_change_view, menu_elements->show_next_move_change_view);
            side_menu.push(side_side_menu);
            menu_e.push(side_menu);
        title.push(menu_e);
        menu_e.init_button(language.get("display", "disc", "display_on_disc"), &menu_elements->dummy);
            side_menu.init_check(language.get("display", "disc", "last_move") + get_shortcut_key_info(U"show_last_move"), &menu_elements->show_last_move, menu_elements->show_last_move);
            menu_e.push(side_menu);
            side_menu.init_check(language.get("display", "disc", "stable") + get_shortcut_key_info(U"show_stable_discs"), &menu_elements->show_stable_discs, menu_elements->show_stable_discs);
            menu_e.push(side_menu);
            side_menu.init_check(language.get("display", "disc", "play_ordering") + get_shortcut_key_info(U"show_play_ordering"), &menu_elements->show_play_ordering, menu_elements->show_play_ordering);
                side_side_menu.init_radio(language.get("display", "disc", "play_ordering_board_format") + get_shortcut_key_info(U"play_ordering_board_format"), &menu_elements->play_ordering_board_format, menu_elements->play_ordering_board_format);
                side_menu.push(side_side_menu);
                side_side_menu.init_radio(language.get("display", "disc", "play_ordering_transcript_format") + get_shortcut_key_info(U"play_ordering_transcript_format"), &menu_elements->play_ordering_transcript_format, menu_elements->play_ordering_transcript_format);
                side_menu.push(side_side_menu);
            menu_e.push(side_menu);
        title.push(menu_e);
        menu_e.init_button(language.get("display", "info", "display_on_info_area"), &menu_elements->dummy);
            side_menu.init_check(language.get("display", "info", "opening_name") + get_shortcut_key_info(U"show_opening_name"), &menu_elements->show_opening_name, menu_elements->show_opening_name);
            menu_e.push(side_menu);
            side_menu.init_bar_check(language.get("display", "info", "principal_variation") + get_shortcut_key_info(U"show_principal_variation"), &menu_elements->pv_length, menu_elements->pv_length, PV_LENGTH_SETTING_MIN, PV_LENGTH_SETTING_MAX, &menu_elements->show_principal_variation, menu_elements->show_principal_variation, U"-");
            menu_e.push(side_menu);
        title.push(menu_e);
        menu_e.init_button(language.get("display", "graph", "display_on_graph_area"), &menu_elements->dummy);
            side_menu.init_check(language.get("display", "graph", "graph") + get_shortcut_key_info(U"show_graph"), &menu_elements->show_graph, menu_elements->show_graph);
                side_side_menu.init_radio(language.get("display", "graph", "value") + get_shortcut_key_info(U"show_graph_value"), &menu_elements->show_graph_value, menu_elements->show_graph_value);
                side_menu.push(side_side_menu);
                side_side_menu.init_radio(language.get("display", "graph", "sum_of_loss") + get_shortcut_key_info(U"show_graph_sum_of_loss"), &menu_elements->show_graph_sum_of_loss, menu_elements->show_graph_sum_of_loss);
                side_menu.push(side_side_menu);
            menu_e.push(side_menu);
            side_menu.init_check(language.get("display", "graph", "endgame_error") + get_shortcut_key_info(U"show_endgame_error"), &menu_elements->show_endgame_error, menu_elements->show_endgame_error);
            menu_e.push(side_menu);
        title.push(menu_e);
        menu_e.init_check(language.get("display", "ai_focus") + get_shortcut_key_info(U"show_ai_focus"), &menu_elements->show_ai_focus, menu_elements->show_ai_focus);
        title.push(menu_e);
        menu_e.init_check(language.get("display", "laser_pointer") + get_shortcut_key_info(U"show_laser_pointer"), &menu_elements->show_laser_pointer, menu_elements->show_laser_pointer);
        title.push(menu_e);
        menu_e.init_check(language.get("display", "log") + get_shortcut_key_info(U"show_log"), &menu_elements->show_log, menu_elements->show_log);
        title.push(menu_e);
        menu_e.init_check(language.get("display", "change_color_type") + get_shortcut_key_info(U"change_color_type"), &menu_elements->change_color_type, menu_elements->change_color_type);
        title.push(menu_e);
    menu.push(title);




    title.init(language.get("operation", "operation"));
        menu_e.init_button(language.get("operation", "put_1_move_by_ai") + get_shortcut_key_info(U"put_1_move_by_ai"), &menu_elements->put_1_move_by_ai);
        title.push(menu_e);
        menu_e.init_button(language.get("operation", "forward") + get_shortcut_key_info(U"forward"), &menu_elements->forward);
        title.push(menu_e);
        menu_e.init_button(language.get("operation", "backward") + get_shortcut_key_info(U"backward"), &menu_elements->backward);
        title.push(menu_e);
        menu_e.init_button(language.get("operation", "undo") + get_shortcut_key_info(U"undo"), &menu_elements->undo);
        title.push(menu_e);
        menu_e.init_button(language.get("operation", "save_this_branch") + get_shortcut_key_info(U"save_this_branch"), &menu_elements->save_this_branch);
        title.push(menu_e);
        menu_e.init_button(language.get("operation", "generate_random_board", "generate_random_board"), &menu_elements->dummy);
            side_menu.init_button(language.get("operation", "generate_random_board", "generate") + get_shortcut_key_info(U"generate_random_board"), &menu_elements->generate_random_board);
            menu_e.push(side_menu);
            side_menu.init_bar(language.get("operation", "generate_random_board", "generate_n_moves"), &menu_elements->generate_random_board_moves, menu_elements->generate_random_board_moves, 1, 60);
            menu_e.push(side_menu);
            side_menu.init_2bars(language.get("operation", "generate_random_board", "score_range"), &menu_elements->generate_random_board_score_range_min, &menu_elements->generate_random_board_score_range_max, menu_elements->generate_random_board_score_range_min, menu_elements->generate_random_board_score_range_max, -64, 64, 2, resources->arrow_left);
            menu_e.push(side_menu);
        title.push(menu_e);
        menu_e.init_button(language.get("operation", "convert", "convert"), &menu_elements->dummy);
            side_menu.init_button(resources->rotate_180, language.get("operation", "convert", "rotate_180") + get_shortcut_key_info(U"convert_180"), &menu_elements->convert_180);
            menu_e.push(side_menu);
            side_menu.init_button(resources->rotate_cw, language.get("operation", "convert", "rotate_90_clock") + get_shortcut_key_info(U"convert_90_clock"), &menu_elements->convert_90_clock);
            menu_e.push(side_menu);
            side_menu.init_button(resources->rotate_ccw, language.get("operation", "convert", "rotate_90_anti_clock") + get_shortcut_key_info(U"convert_90_anti_clock"), &menu_elements->convert_90_anti_clock);
            menu_e.push(side_menu);
            side_menu.init_button(resources->mirror_black_line, language.get("operation", "convert", "black_line") + get_shortcut_key_info(U"convert_blackline"), &menu_elements->convert_blackline);
            menu_e.push(side_menu);
            side_menu.init_button(resources->mirror_white_line, language.get("operation", "convert", "white_line") + get_shortcut_key_info(U"convert_whiteline"), &menu_elements->convert_whiteline);
            menu_e.push(side_menu);
            side_menu.init_button(resources->flip_horizontal, language.get("operation", "convert", "horizontal") + get_shortcut_key_info(U"convert_horizontal"), &menu_elements->convert_horizontal);
            menu_e.push(side_menu);
            side_menu.init_button(resources->flip_vertical, language.get("operation", "convert", "vertical") + get_shortcut_key_info(U"convert_vertical"), &menu_elements->convert_vertical);
            menu_e.push(side_menu);
        title.push(menu_e);
        menu_e.init_button(language.get("operation", "ai_operation", "ai_operation"), &menu_elements->dummy);
            side_menu.init_button(language.get("operation", "ai_operation", "stop_calculating") + get_shortcut_key_info(U"stop_calculating"), &menu_elements->stop_calculating);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("operation", "ai_operation", "cache_clear") + get_shortcut_key_info(U"cache_clear"), &menu_elements->cache_clear);
            menu_e.push(side_menu);
        title.push(menu_e);
    menu.push(title);



    title.init(language.get("in_out", "in_out"));
        menu_e.init_button(language.get("in_out", "in"), &menu_elements->dummy);
            side_menu.init_button(language.get("in_out", "input_from_clipboard") + get_shortcut_key_info(U"input_from_clipboard"), &menu_elements->input_from_clipboard);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("in_out", "input_text") + get_shortcut_key_info(U"input_text"), &menu_elements->input_text);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("in_out", "edit_board") + get_shortcut_key_info(U"edit_board"), &menu_elements->edit_board);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("in_out", "input_game") + get_shortcut_key_info(U"input_game"), &menu_elements->input_game);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("in_out", "input_bitboard") + get_shortcut_key_info(U"input_bitboard"), &menu_elements->input_bitboard);
            menu_e.push(side_menu);
        title.push(menu_e);
        menu_e.init_button(language.get("in_out", "out"), &menu_elements->dummy);
            side_menu.init_button(language.get("in_out", "output_transcript") + get_shortcut_key_info(U"output_transcript"), &menu_elements->copy_transcript);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("in_out", "output_board") + get_shortcut_key_info(U"output_board"), &menu_elements->copy_board);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("in_out", "screen_shot") + get_shortcut_key_info(U"screen_shot"), &menu_elements->screen_shot);
            side_side_menu.init_button(language.get("in_out", "change_screenshot_saving_dir"), &menu_elements->change_screenshot_saving_dir);
            side_menu.push(side_side_menu);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("in_out", "board_image") + get_shortcut_key_info(U"board_image"), &menu_elements->board_image);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("in_out", "output_game") + get_shortcut_key_info(U"save_game"), &menu_elements->save_game);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("in_out", "output_bitboard"), &menu_elements->dummy);
            side_side_menu.init_button(language.get("in_out", "player_opponent") + get_shortcut_key_info(U"output_bitboard_player_opponent"), &menu_elements->output_bitboard_player_opponent);
            side_menu.push(side_side_menu);
            side_side_menu.init_button(language.get("in_out", "black_white") + get_shortcut_key_info(U"output_bitboard_black_white"), &menu_elements->output_bitboard_black_white);
            side_menu.push(side_side_menu);
            menu_e.push(side_menu);
        title.push(menu_e);
    menu.push(title);




    title.init(language.get("book", "book"));
        menu_e.init_button(language.get("book", "settings"), &menu_elements->dummy);
            side_menu.init_bar_check(language.get("book", "depth"), &menu_elements->book_learn_depth, menu_elements->book_learn_depth, 0, 60, &menu_elements->use_book_learn_depth, menu_elements->use_book_learn_depth, U"Inf");
            menu_e.push(side_menu);
            side_menu.init_bar_check(language.get("book", "error_per_move"), &menu_elements->book_learn_error_per_move, menu_elements->book_learn_error_per_move, 0, 24, &menu_elements->use_book_learn_error_per_move, menu_elements->use_book_learn_error_per_move, U"Inf");
            menu_e.push(side_menu);
            side_menu.init_bar_check(language.get("book", "error_sum"), &menu_elements->book_learn_error_sum, menu_elements->book_learn_error_sum, 0, 24, &menu_elements->use_book_learn_error_sum, menu_elements->use_book_learn_error_sum, U"Inf");
            menu_e.push(side_menu);
            side_menu.init_bar_check(language.get("book", "error_leaf"), &menu_elements->book_learn_error_leaf, menu_elements->book_learn_error_leaf, 0, 24, &menu_elements->use_book_learn_error_leaf, menu_elements->use_book_learn_error_leaf, U"Inf");
            menu_e.push(side_menu);
        title.push(menu_e);
        menu_e.init_button(language.get("book", "book_operation"), &menu_elements->dummy);
            side_menu.init_check(language.get("book", "right_click_to_modify") + get_shortcut_key_info(U"change_book_by_right_click"), &menu_elements->change_book_by_right_click, menu_elements->change_book_by_right_click);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("book", "book_deviate") + get_shortcut_key_info(U"book_start_deviate"), &menu_elements->book_start_deviate);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("book", "book_deviate_with_transcript") + get_shortcut_key_info(U"book_start_deviate_with_transcript"), &menu_elements->book_start_deviate_with_transcript);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("book", "book_store") + get_shortcut_key_info(U"book_start_store"), &menu_elements->book_start_store);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("book", "book_fix") + get_shortcut_key_info(U"book_start_fix"), &menu_elements->book_start_fix);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("book", "book_reduce") + get_shortcut_key_info(U"book_start_reducing"), &menu_elements->book_start_reducing);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("book", "book_recalculate_leaf") + get_shortcut_key_info(U"book_start_recalculate_leaf"), &menu_elements->book_start_recalculate_leaf);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("book", "book_recalculate_n_lines") + get_shortcut_key_info(U"book_start_recalculate_n_lines"), &menu_elements->book_start_recalculate_n_lines);
            menu_e.push(side_menu);
            //side_menu.init_button(language.get("book", "book_upgrade_better_leaves") + get_shortcut_key_info(U"book_start_upgrade_better_leaves"), &menu_elements->book_start_upgrade_better_leaves);
            //menu_e.push(side_menu);
        title.push(menu_e);
        menu_e.init_button(language.get("book", "file_operation"), &menu_elements->dummy);
            side_menu.init_button(language.get("book", "import_book") + get_shortcut_key_info(U"import_book"), &menu_elements->import_book);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("book", "export_book") + get_shortcut_key_info(U"export_book"), &menu_elements->export_book);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("book", "book_merge") + get_shortcut_key_info(U"book_merge"), &menu_elements->book_merge);
            menu_e.push(side_menu);
            side_menu.init_button(language.get("book", "book_reference") + get_shortcut_key_info(U"book_reference"), &menu_elements->book_reference);
            menu_e.push(side_menu);
        title.push(menu_e);
        menu_e.init_button(language.get("book", "show_book_info") + get_shortcut_key_info(U"show_book_info"), &menu_elements->show_book_info);
        title.push(menu_e);
    menu.push(title);




    title.init(language.get("help", "help"));
        menu_e.init_button(language.get("help", "usage") + get_shortcut_key_info(U"open_usage"), &menu_elements->usage);
        title.push(menu_e);
        menu_e.init_button(language.get("help", "website") + get_shortcut_key_info(U"open_website"), &menu_elements->website);
        title.push(menu_e);
        menu_e.init_button(language.get("help", "bug_report") + get_shortcut_key_info(U"bug_report"), &menu_elements->bug_report);
        title.push(menu_e);
        menu_e.init_button(language.get("help", "update_check") + get_shortcut_key_info(U"update_check"), &menu_elements->update_check);
        title.push(menu_e);
        menu_e.init_check(language.get("help", "auto_update_check") + get_shortcut_key_info(U"auto_update_check"), &menu_elements->auto_update_check, menu_elements->auto_update_check);
        title.push(menu_e);
        menu_e.init_button(language.get("help", "license") + get_shortcut_key_info(U"license"), &menu_elements->license);
        title.push(menu_e);
    menu.push(title);





    title.init(U"Language");
    for (int i = 0; i < (int)resources->language_names.size(); ++i) {
        menu_e.init_radio(resources->lang_img[i], &menu_elements->languages[i], menu_elements->languages[i]);
        title.push(menu_e);
    }
    menu.push(title);




    menu.init(0, 0, 12, font, resources->checkbox, resources->unchecked, lang_name);
    return menu;
}