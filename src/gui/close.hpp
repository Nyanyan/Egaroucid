/*
    Egaroucid Project

    @file close.hpp
        Closing scene
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"

void save_settings(Menu_elements menu_elements, Settings settings, Directories directories) {
    TextWriter writer(U"{}setting.txt"_fmt(Unicode::Widen(directories.appdata_dir)));
    if (writer) {
        writer.writeln(menu_elements.n_threads);
        writer.writeln((int)menu_elements.auto_update_check);
        writer.writeln(Unicode::Widen(settings.lang_name));
        writer.writeln(Unicode::Widen(settings.book_file));
        writer.writeln((int)menu_elements.use_book);
        writer.writeln(menu_elements.level);
        writer.writeln((int)menu_elements.ai_put_black);
        writer.writeln((int)menu_elements.ai_put_white);
        writer.writeln((int)menu_elements.use_disc_hint);
        writer.writeln((int)menu_elements.use_umigame_value);
        writer.writeln(menu_elements.n_disc_hint);
        writer.writeln((int)menu_elements.show_legal);
        writer.writeln((int)menu_elements.show_graph);
        writer.writeln((int)menu_elements.show_opening_on_cell);
        writer.writeln((int)menu_elements.show_log);
        writer.writeln(menu_elements.book_learn_depth);
        writer.writeln(menu_elements.book_learn_error_per_move);
        writer.writeln((int)menu_elements.show_stable_discs);
        writer.writeln((int)menu_elements.change_book_by_right_click);
        writer.writeln((int)menu_elements.show_last_move);
        writer.writeln((int)menu_elements.show_next_move);
        writer.writeln(menu_elements.hash_level);
		writer.writeln(menu_elements.book_acc_level);
        writer.writeln((int)menu_elements.pause_when_pass);
        writer.writeln(menu_elements.book_learn_error_sum);
    }
}

void close_app(Menu_elements menu_elements, Settings settings, Directories directories, Book_information book_information, Window_state window_state) {
	if (!window_state.loading) {
		save_settings(menu_elements, settings, directories);
	}
    if (book_information.changed) {
        book.save_bin(settings.book_file, settings.book_file + ".bak");
    }
}

class Close : public App::Scene {
private:
    std::future<void> close_future;

public:
    Close(const InitData& init) : IScene{ init } {
        close_future = std::async(std::launch::async, close_app, getData().menu_elements, getData().settings, getData().directories, getData().book_information, getData().window_state);
    }

    void update() override {
        Scene::SetBackground(getData().colors.green);
        const int icon_width = (LEFT_RIGHT - LEFT_LEFT);
        getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(LEFT_LEFT, Y_CENTER - icon_width / 2);
        getData().resources.logo.scaled((double)icon_width * 0.8 / getData().resources.logo.width()).draw(RIGHT_LEFT, Y_CENTER - 40);
        if (close_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            close_future.get();
            //thread_pool.terminate();
            System::Exit();
        }
        getData().fonts.font(language.get("closing")).draw(50, RIGHT_LEFT, Y_CENTER + 40, getData().colors.white);
    }

    void draw() const override {

    }
};