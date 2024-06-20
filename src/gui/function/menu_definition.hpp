/*
    Egaroucid Project

    @file menu_definition.hpp
        Main scene for GUI
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once

#include "const/gui_common.hpp"
#include "menu.hpp"
#include "language.hpp"
#include "./../../engine/setting.hpp"

Menu create_menu(Menu_elements* menu_elements, Resources *resources, Font font) {
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
    #if USE_CHANGEABLE_HASH_LEVEL
        menu_e.init_bar(language.get("settings", "hash_level"), &menu_elements->hash_level, menu_elements->hash_level, MIN_HASH_LEVEL, MAX_HASH_LEVEL);
        title.push(menu_e);
    #endif

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
    side_side_menu.init_bar(language.get("display", "cell", "depth"), &menu_elements->umigame_value_depth, menu_elements->umigame_value_depth, 1, 60);
    side_menu.push(side_side_menu);
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

    menu_e.init_button(language.get("display", "info", "display_on_info_area"), &menu_elements->dummy);
    side_menu.init_check(language.get("display", "info", "opening_name"), &menu_elements->show_opening_name, menu_elements->show_opening_name);
    menu_e.push(side_menu);
    side_menu.init_check(language.get("display", "info", "principal_variation"), &menu_elements->show_principal_variation, menu_elements->show_principal_variation);
    menu_e.push(side_menu);
    title.push(menu_e);

    menu_e.init_check(language.get("display", "graph", "graph"), &menu_elements->show_graph, menu_elements->show_graph);
    side_menu.init_radio(language.get("display", "graph", "value"), &menu_elements->show_graph_value, menu_elements->show_graph_value);
    menu_e.push(side_menu);
    side_menu.init_radio(language.get("display", "graph", "sum_of_loss"), &menu_elements->show_graph_sum_of_loss, menu_elements->show_graph_sum_of_loss);
    menu_e.push(side_menu);
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
    side_menu.init_button(language.get("in_out", "input_bitboard"), &menu_elements->input_bitboard);
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
    side_menu.init_check(language.get("book", "right_click_to_modify"), &menu_elements->change_book_by_right_click, menu_elements->change_book_by_right_click);
    menu_e.push(side_menu);
    side_menu.init_button(language.get("book", "book_deviate"), &menu_elements->book_start_deviate);
    menu_e.push(side_menu);
    side_menu.init_button(language.get("book", "book_deviate_with_transcript"), &menu_elements->book_start_deviate_with_transcript);
    menu_e.push(side_menu);
    side_menu.init_button(language.get("book", "book_fix"), &menu_elements->book_start_fix);
    menu_e.push(side_menu);
    side_menu.init_button(language.get("book", "book_reduce"), &menu_elements->book_start_reducing);
    menu_e.push(side_menu);
    side_menu.init_button(language.get("book", "book_recalculate_leaf"), &menu_elements->book_start_recalculate_leaf);
    menu_e.push(side_menu);
    side_menu.init_button(language.get("book", "book_recalculate_n_lines"), &menu_elements->book_start_recalculate_n_lines);
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

    menu_e.init_button(language.get("book", "show_book_info"), &menu_elements->show_book_info);
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
    for (int i = 0; i < (int)resources->language_names.size(); ++i) {
        menu_e.init_radio(resources->lang_img[i], &menu_elements->languages[i], menu_elements->languages[i]);
        title.push(menu_e);
    }
    menu.push(title);




    menu.init(0, 0, 12, font, resources->checkbox, resources->unchecked);
    return menu;
}
