/*
    Egaroucid Project

    @file close.hpp
        Closing scene
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"

void save_settings(Menu_elements menu_elements, Settings settings, Directories directories, User_settings user_settings) {
    JSON setting_json;
    setting_json[U"n_threads"] = menu_elements.n_threads;
    setting_json[U"auto_update_check"] = menu_elements.auto_update_check;
    setting_json[U"lang_name"] = Unicode::Widen(settings.lang_name);
    setting_json[U"book_file"] = Unicode::Widen(settings.book_file);
    setting_json[U"use_book"] = menu_elements.use_book;
    setting_json[U"level"] = menu_elements.level;
    setting_json[U"ai_put_black"] = menu_elements.ai_put_black;
    setting_json[U"ai_put_white"] = menu_elements.ai_put_white;
    setting_json[U"use_disc_hint"] = menu_elements.use_disc_hint;
    setting_json[U"use_umigame_value"] = menu_elements.use_umigame_value;
    setting_json[U"n_disc_hint"] = menu_elements.n_disc_hint;
    setting_json[U"show_legal"] = menu_elements.show_legal;
    setting_json[U"show_graph"] = menu_elements.show_graph;
    setting_json[U"show_opening_on_cell"] = menu_elements.show_opening_on_cell;
    setting_json[U"show_log"] = menu_elements.show_log;
    setting_json[U"book_learn_depth"] = menu_elements.book_learn_depth;
    setting_json[U"book_learn_error_per_move"] = menu_elements.book_learn_error_per_move;
    setting_json[U"show_stable_discs"] = menu_elements.show_stable_discs;
    setting_json[U"change_book_by_right_click"] = menu_elements.change_book_by_right_click;
    setting_json[U"show_last_move"] = menu_elements.show_last_move;
    setting_json[U"show_next_move"] = menu_elements.show_next_move;
#if USE_CHANGEABLE_HASH_LEVEL
    setting_json[U"hash_level"] = menu_elements.hash_level;
#endif
    //setting_json[U"book_acc_level"] = menu_elements.book_acc_level;
    setting_json[U"pause_when_pass"] = menu_elements.pause_when_pass;
    setting_json[U"book_learn_error_sum"] = menu_elements.book_learn_error_sum;
    setting_json[U"show_next_move_change_view"] = menu_elements.show_next_move_change_view;
    setting_json[U"change_color_type"] = menu_elements.change_color_type;
    setting_json[U"show_play_ordering"] = menu_elements.show_play_ordering;
    setting_json[U"generate_random_board_moves"] = menu_elements.generate_random_board_moves;
    setting_json[U"show_book_accuracy"] = menu_elements.show_book_accuracy;
    setting_json[U"use_book_learn_depth"] = menu_elements.use_book_learn_depth;
    setting_json[U"use_book_learn_error_per_move"] = menu_elements.use_book_learn_error_per_move;
    setting_json[U"use_book_learn_error_sum"] = menu_elements.use_book_learn_error_sum;
    setting_json[U"umigame_value_depth"] = menu_elements.umigame_value_depth;
    setting_json[U"show_graph_value"] = menu_elements.show_graph_value;
    setting_json[U"show_graph_sum_of_loss"] = menu_elements.show_graph_sum_of_loss;
    setting_json[U"book_learn_error_leaf"] = menu_elements.book_learn_error_leaf;
    setting_json[U"use_book_learn_error_leaf"] = menu_elements.use_book_learn_error_leaf;
    setting_json[U"show_opening_name"] = menu_elements.show_opening_name;
    setting_json[U"show_principal_variation"] = menu_elements.show_principal_variation;
    setting_json[U"show_laser_pointer"] = menu_elements.show_laser_pointer;
    setting_json[U"show_ai_focus"] = menu_elements.show_ai_focus;
    setting_json[U"pv_length"] = menu_elements.pv_length;
    setting_json[U"screenshot_saving_dir"] = Unicode::Widen(user_settings.screenshot_saving_dir);
    setting_json[U"accept_ai_loss"] = menu_elements.accept_ai_loss;
    setting_json[U"max_loss"] = menu_elements.max_loss;
    setting_json[U"loss_percentage"] = menu_elements.loss_percentage;
    setting_json[U"force_specified_openings"] = menu_elements.force_specified_openings;
    setting_json[U"show_value_when_ai_calculating"] = menu_elements.show_value_when_ai_calculating;
    setting_json[U"generate_random_board_score_range_min"] = menu_elements.generate_random_board_score_range_min;
    setting_json[U"generate_random_board_score_range_max"] = menu_elements.generate_random_board_score_range_max;
    setting_json[U"show_hint_level"] = menu_elements.show_hint_level;
    setting_json[U"show_endgame_error"] = menu_elements.show_endgame_error;
    setting_json[U"hint_colorize"] = menu_elements.hint_colorize;
    setting_json[U"play_ordering_board_format"] = menu_elements.play_ordering_board_format;
    setting_json[U"play_ordering_transcript_format"] = menu_elements.play_ordering_transcript_format;
    setting_json.save(U"{}setting.json"_fmt(Unicode::Widen(directories.appdata_dir)));
}

void close_app(Menu_elements menu_elements, Settings settings, Directories directories, User_settings user_settings, Book_information book_information, Forced_openings forced_openings, Window_state window_state) {
    if (!window_state.loading) {
        save_settings(menu_elements, settings, directories, user_settings);
        String shortcut_key_file = U"{}shortcut_key.json"_fmt(Unicode::Widen(directories.appdata_dir));
        shortcut_keys.save_settings(shortcut_key_file);
        // Note: forced_openings are now saved via the opening_setting scene to the folder structure
        // No need to save here as the folder structure is already saved when OK is clicked
    }
    if (book_information.changed) {
        book.save_egbk3(settings.book_file, settings.book_file + ".bak");
    }
}

class Close : public App::Scene {
private:
    std::future<void> close_future;

public:
    Close(const InitData& init) : IScene{ init } {
        close_future = std::async(std::launch::async, close_app, getData().menu_elements, getData().settings, getData().directories, getData().user_settings, getData().book_information, getData().forced_openings, getData().window_state);
    }

    void update() override {
        Scene::SetBackground(getData().colors.green);
        const int icon_width = (LEFT_RIGHT - LEFT_LEFT);
        getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(LEFT_LEFT, Y_CENTER - icon_width / 2);
        getData().resources.logo.scaled((double)icon_width * 0.8 / getData().resources.logo.width()).draw(RIGHT_LEFT, Y_CENTER - 40);
        if (close_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            close_future.get();
            System::Exit();
        }
        getData().fonts.font(language.get("closing")).draw(50, RIGHT_LEFT, Y_CENTER + 40, getData().colors.white);
    }

    void draw() const override {

    }
};