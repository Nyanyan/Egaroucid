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

int init_ai(const Settings* settings, const Directories* directories) {
	thread_pool.resize(settings->n_threads - 1);
	cerr << "there are " << thread_pool.size() << " additional threads" << endl;
	bit_init();
	if (!board_init()) {
		return ERR_HASH_FILE_NOT_IMPORTED;
	}
	stability_init();
	if (!evaluate_init(directories->eval_file)) {
		return ERR_EVAL_FILE_NOT_IMPORTED;
	}
	parent_transpose_table.first_init();
	child_transpose_table.first_init();
	if (!book_init(settings->book_file)) {
		return ERR_BOOK_FILE_NOT_IMPORTED;
	}
	return ERR_OK;
}

int check_update(const Directories* directories, String *new_version) {
	const String version_url = U"https://www.egaroucid.nyanyan.dev/version.txt";
	const FilePath version_save_path = U"{}Egaroucid/version.txt"_fmt(Unicode::Widen(directories->appdata_dir));
	if (SimpleHTTP::Save(version_url, version_save_path).isOK()) {
		TextReader reader(version_save_path);
		if (reader) {
			reader.readLine(*new_version);
			if (EGAROUCID_VERSION != *new_version) {
				return UPDATE_CHECK_UPDATE_FOUND;
			}
		}
	}
	return UPDATE_CHECK_ALREADY_UPDATED;
}



int load_app(Directories* directories, Resources* resources, Settings* settings, bool* update_found, String *new_version) {
	if (settings->auto_update_check) {
		if (check_update(directories, new_version) == UPDATE_CHECK_UPDATE_FOUND) {
			*update_found = true;
		}
	}
	return init_ai(settings, directories);
}



class Load : public App::Scene {
private:
	bool load_failed;
	bool book_failed;
	String tips;
	bool update_found;
	future<int> load_future;
	Button skip_button;
	Button update_button;
	Button book_ignore_button;
	String new_version;

public:
	Load(const InitData& init) : IScene{ init } {
		skip_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("help", "skip"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
		update_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("help", "download"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
		book_ignore_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("loading", "launch"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
		load_failed = false;
		book_failed = false;
		tips = language.get_random("tips", "tips");
		update_found = false;
		load_future = async(launch::async, load_app, &getData().directories, &getData().resources, &getData().settings, &update_found, &new_version);
	}

	void update() override {
		Scene::SetBackground(getData().colors.green);
		if (update_found) {
			const int icon_width = (LEFT_RIGHT - LEFT_LEFT) / 2;
			getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
			getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
			int sy = 20 + icon_width + 50;
			getData().fonts.font(language.get("help", "new_version_available")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
			sy += 35;
			getData().fonts.font(language.get("help", "download?")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
			skip_button.draw();
			update_button.draw();
			if (skip_button.clicked() || KeyEscape.pressed()) {
				update_found = false;
			}
			if (update_button.clicked() || KeyEnter.pressed()) {
				System::LaunchBrowser(U"https://github.com/Nyanyan/Egaroucid/releases/tag/v{}"_fmt(new_version));
			}
		}
		else {
			const int icon_width = (LEFT_RIGHT - LEFT_LEFT);
			getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(LEFT_LEFT, Y_CENTER - icon_width / 2);
			getData().resources.logo.scaled((double)icon_width * 0.8 / getData().resources.logo.width()).draw(RIGHT_LEFT, Y_CENTER - 40);
			if (load_future.valid()) {
				if (load_future.wait_for(chrono::seconds(0)) == future_status::ready) {
					int load_code = load_future.get();
					if (load_code == ERR_OK) {
						cerr << "loaded" << endl;
						getData().menu_elements.init(&getData().settings, &getData().resources);
						changeScene(U"Main_scene", SCENE_FADE_TIME);
					}
					else {
						load_failed = true;
						if (load_code == ERR_BOOK_FILE_NOT_IMPORTED) {
							book_failed = true;
						}
					}
				}
			}
			if (load_failed) {
				if (book_failed) {
					getData().fonts.font(language.get("loading", "book_failed")).draw(20, RIGHT_LEFT, Y_CENTER + 50, getData().colors.white);
					book_ignore_button.draw();
					if (book_ignore_button.clicked()) {
						cerr << "loaded" << endl;
						getData().menu_elements.init(&getData().settings, &getData().resources);
						changeScene(U"Main_scene", SCENE_FADE_TIME);
					}
				}
				else {
					getData().fonts.font(language.get("loading", "load_failed")).draw(20, RIGHT_LEFT, Y_CENTER + 50, getData().colors.white);
					if (System::GetUserActions() & UserAction::CloseButtonClicked) {
						System::Exit();
					}
				}

			}
			else {
				getData().fonts.font(language.get("loading", "loading")).draw(50, RIGHT_LEFT, Y_CENTER + 40, getData().colors.white);
				getData().fonts.font(language.get("tips", "do_you_know")).draw(20, RIGHT_LEFT, Y_CENTER + 110, getData().colors.white);
				getData().fonts.font(tips).draw(15, RIGHT_LEFT, Y_CENTER + 140, getData().colors.white);
			}
		}
	}

	void draw() const override {

	}
};
