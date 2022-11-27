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

void delete_book() {
	book.delete_all();
}

bool import_book(string file) {
	cerr << "book import" << endl;
	bool result = true;
	vector<string> lst;
	auto offset = string::size_type(0);
	while (1) {
		auto pos = file.find(".", offset);
		if (pos == string::npos) {
			lst.push_back(file.substr(offset));
			break;
		}
		lst.push_back(file.substr(offset, pos - offset));
		offset = pos + 1;
	}
	if (lst[lst.size() - 1] == "egbk") {
		cerr << "importing Egaroucid book" << endl;
		result = !book.import_file_bin(file);
	}
	else if (lst[lst.size() - 1] == "dat") {
		cerr << "importing Edax book" << endl;
		result = !book.import_edax_book(file);
	}
	else {
		cerr << "this is not a book" << endl;
	}
	return result;
}

bool import_book_egaroucid(string file) {
	cerr << "book import" << endl;
	bool result = true;
	vector<string> lst;
	auto offset = string::size_type(0);
	while (1) {
		auto pos = file.find(".", offset);
		if (pos == string::npos) {
			lst.push_back(file.substr(offset));
			break;
		}
		lst.push_back(file.substr(offset, pos - offset));
		offset = pos + 1;
	}
	if (lst[lst.size() - 1] == "egbk") {
		cerr << "importing Egaroucid book" << endl;
		result = !book.import_file_bin(file);
	}
	else {
		cerr << "this is not an Egaroucid book" << endl;
	}
	return result;
}

class Import_book : public App::Scene {
private:
	future<bool> import_book_future;
	Button back_button;
	bool importing;
	bool imported;
	bool failed;

public:
	Import_book(const InitData& init) : IScene{ init } {
		back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
		importing = false;
		imported = false;
		failed = false;
	}

	void update() override {
		if (System::GetUserActions() & UserAction::CloseButtonClicked) {
			changeScene(U"Close", SCENE_FADE_TIME);
		}
		Scene::SetBackground(getData().colors.green);
		const int icon_width = (LEFT_RIGHT - LEFT_LEFT) / 2;
		getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
		getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
		int sy = 20 + icon_width + 50;
		if (!importing) {
			getData().fonts.font(language.get("book", "merge_explanation")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
			back_button.draw();
			if (back_button.clicked() || KeyEscape.pressed()) {
				changeScene(U"Main_scene", SCENE_FADE_TIME);
			}
			if (DragDrop::HasNewFilePaths()) {
				import_book_future = async(launch::async, import_book, DragDrop::GetDroppedFilePaths()[0].path.narrow());
				importing = true;
			}
		}
		else if (!imported) {
			getData().fonts.font(language.get("book", "loading")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
			if (import_book_future.wait_for(chrono::seconds(0)) == future_status::ready) {
				failed = import_book_future.get();
				imported = true;
			}
		}
		else {
			if (failed) {
				getData().fonts.font(language.get("book", "import_failed")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
				back_button.draw();
				if (back_button.clicked() || KeyEscape.pressed()) {
					changeScene(U"Main_scene", SCENE_FADE_TIME);
				}
			}
			else {
				getData().book_information.changed = true;
				changeScene(U"Main_scene", SCENE_FADE_TIME);
			}
		}
	}

	void draw() const override {

	}
};

class Refer_book : public App::Scene {
private:
	Button single_back_button;
	Button back_button;
	Button default_button;
	Button go_button;
	string book_file;
	future<void> delete_book_future;
	future<bool> import_book_future;
	bool book_deleting;
	bool book_importing;
	bool failed;
	bool done;

public:
	Refer_book(const InitData& init) : IScene{ init } {
		single_back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
		back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
		default_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("book", "use_default"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
		go_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("book", "import"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
		book_file = getData().settings.book_file;
		book_deleting = false;
		book_importing = false;
		failed = false;
		done = false;
	}

	void update() override {
		if (System::GetUserActions() & UserAction::CloseButtonClicked) {
			changeScene(U"Close", SCENE_FADE_TIME);
		}
		Scene::SetBackground(getData().colors.green);
		const int icon_width = (LEFT_RIGHT - LEFT_LEFT) / 2;
		getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
		getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
		int sy = 20 + icon_width + 50;
		if (!book_deleting && !book_importing && !failed && !done) {
			getData().fonts.font(language.get("book", "input_book_path")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
			Rect text_area{ X_CENTER - 300, sy + 40, 600, 70 };
			text_area.draw(getData().colors.light_cyan).drawFrame(2, getData().colors.black);
			String book_file_str = Unicode::Widen(book_file);
			TextInput::UpdateText(book_file_str);
			const String editingText = TextInput::GetEditingText();
			bool return_pressed = false;
			if (KeyControl.pressed() && KeyV.down()) {
				String clip_text;
				Clipboard::GetText(clip_text);
				book_file_str += clip_text;
			}
			if (book_file_str.size()) {
				if (book_file_str[book_file_str.size() - 1] == '\n') {
					book_file_str.replace(U"\n", U"");
					return_pressed = true;
				}
			}
			book_file = book_file_str.narrow();
			getData().fonts.font(book_file_str + U'|' + editingText).draw(15, text_area.stretched(-4), getData().colors.black);
			back_button.draw();
			if (back_button.clicked() || KeyEscape.pressed()) {
				changeScene(U"Main_scene", SCENE_FADE_TIME);
			}
			default_button.draw();
			if (default_button.clicked()) {
				book_file = getData().directories.document_dir + "Egaroucid/book.egbk";
			}
			go_button.draw();
			if (go_button.clicked() || return_pressed) {
				getData().book_information.changed = true;
				getData().settings.book_file = book_file;
				cerr << "book reference changed to " << book_file << endl;
				delete_book_future = async(launch::async, delete_book);
				book_deleting = true;
			}
		}
		else if (book_deleting || book_importing) {
			getData().fonts.font(language.get("book", "loading")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
			if (book_deleting) {
				if (delete_book_future.wait_for(chrono::seconds(0)) == future_status::ready) {
					delete_book_future.get();
					book_deleting = false;
					import_book_future = async(launch::async, import_book_egaroucid, getData().settings.book_file);
					book_importing = true;
				}
			}
			else if (book_importing) {
				if (import_book_future.wait_for(chrono::seconds(0)) == future_status::ready) {
					failed = import_book_future.get();
					book_importing = false;
					done = true;
				}
			}
		}
		else if (done) {
			if (failed) {
				getData().fonts.font(language.get("book", "import_failed")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
				single_back_button.draw();
				if (single_back_button.clicked() || KeyEscape.pressed()) {
					changeScene(U"Main_scene", SCENE_FADE_TIME);
				}
			}
			else {
				changeScene(U"Main_scene", SCENE_FADE_TIME);
			}
		}
	}

	void draw() const override {

	}
};

class Learn_book : public App::Scene {
private:
	Button stop_button;
	Button back_button;
	History_elem history_elem;
	bool book_learning;
	bool done;
	future<void> book_learn_future;
	Board root_board;

public:
	Learn_book(const InitData& init) : IScene{ init } {
		stop_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("book", "stop_learn"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
		back_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
		root_board = getData().history_elem.board;
		history_elem = getData().history_elem;
		history_elem.policy = -1;
		book_learning = true;
		done = false;
		book_learn_future = async(launch::async, learn_book, root_board, getData().menu_elements.level, getData().menu_elements.book_learn_depth, getData().menu_elements.book_learn_error, &history_elem.board, &history_elem.player, getData().settings.book_file, getData().settings.book_file + ".bak", getData().menu_elements.ignore_book,  &book_learning);
	}

	void update() override {
		//if (System::GetUserActions() & UserAction::CloseButtonClicked) {
		//	changeScene(U"Close", SCENE_FADE_TIME);
		//}
		Scene::SetBackground(getData().colors.green);
		draw_board(getData().fonts, getData().colors, history_elem);
		draw_info(getData().colors, history_elem, getData().fonts, getData().menu_elements);
		getData().fonts.font(language.get("book", "book_learn")).draw(25, 480, 190, getData().colors.white);
		getData().fonts.font(language.get("book", "depth") + U": " + Format(getData().menu_elements.book_learn_depth)).draw(15, 480, 300, getData().colors.white);
		getData().fonts.font(language.get("book", "accept") + U": " + Format(getData().menu_elements.book_learn_error)).draw(15, 480, 320, getData().colors.white);
		if (book_learning) {
			getData().fonts.font(language.get("book", "learning")).draw(20, 480, 230, getData().colors.white);
			stop_button.draw();
			if (stop_button.clicked()) {
				global_searching = false;
				book_learning = false;
			}
		}
		else if (!done) {
			getData().fonts.font(language.get("book", "stopping")).draw(20, 480, 230, getData().colors.white);
			if (book_learn_future.wait_for(chrono::seconds(0)) == future_status::ready) {
				book_learn_future.get();
				done = true;
				global_searching = true;
			}
		}
		else {
			getData().fonts.font(language.get("book", "complete")).draw(20, 480, 230, getData().colors.white);
			back_button.draw();
			if (back_button.clicked()) {
				getData().graph_resources.need_init = false;
				changeScene(U"Main_scene", SCENE_FADE_TIME);
			}
		}
	}

	void draw() const override {

	}
};
