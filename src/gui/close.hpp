/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include "./ai.hpp"
#include "function/language.hpp"
#include "function/menu.hpp"
#include "function/graph.hpp"
#include "function/opening.hpp"
#include "function/button.hpp"
#include "function/radio_button.hpp"
#include "gui_common.hpp"

void save_settings(Menu_elements menu_elements, Settings settings, Directories directories) {
	TextWriter writer(U"{}Egaroucid/setting.txt"_fmt(Unicode::Widen(directories.appdata_dir)));
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
		writer.writeln(menu_elements.book_learn_error);
		writer.writeln((int)menu_elements.show_stable_discs);
		writer.writeln((int)menu_elements.change_book_by_right_click);
		writer.writeln((int)menu_elements.ignore_book);
	}
}

void close_app(Menu_elements menu_elements, Settings settings, Directories directories, Book_information book_information) {
	save_settings(menu_elements, settings, directories);
	if (book_information.changed) {
		book.save_bin(settings.book_file, settings.book_file + ".bak");
	}
}

class Close : public App::Scene {
private:
	future<void> close_future;

public:
	Close(const InitData& init) : IScene{ init } {
		close_future = async(launch::async, close_app, getData().menu_elements, getData().settings, getData().directories, getData().book_information);
	}

	void update() override {
		Scene::SetBackground(getData().colors.green);
		const int icon_width = (LEFT_RIGHT - LEFT_LEFT);
		getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(LEFT_LEFT, Y_CENTER - icon_width / 2);
		getData().resources.logo.scaled((double)icon_width * 0.8 / getData().resources.logo.width()).draw(RIGHT_LEFT, Y_CENTER - 40);
		if (close_future.wait_for(chrono::seconds(0)) == future_status::ready) {
			close_future.get();
			//thread_pool.terminate();
			System::Exit();
		}
		getData().fonts.font50(language.get("closing")).draw(RIGHT_LEFT, Y_CENTER + 40, getData().colors.white);
	}

	void draw() const override {

	}
};
