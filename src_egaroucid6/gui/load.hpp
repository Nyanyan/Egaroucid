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
	board_init();
	stability_init();
	if (!evaluate_init(directories->eval_file)) {
		return ERR_EVAL_FILE_NOT_IMPORTED;
	}
	if (!book_init(settings->book_file)) {
		return ERR_BOOK_FILE_NOT_IMPORTED;
	}
	parent_transpose_table.first_init();
	child_transpose_table.first_init();
	return ERR_OK;
}

int check_update(const Directories* directories) {
	const String version_url = U"https://www.egaroucid-app.nyanyan.dev/version.txt";
	const FilePath version_save_path = U"{}Egaroucid/version.txt"_fmt(Unicode::Widen(directories->appdata_dir));
	if (SimpleHTTP::Save(version_url, version_save_path).isOK()) {
		TextReader reader(version_save_path);
		if (reader) {
			String new_version;
			reader.readLine(new_version);
			if (EGAROUCID_VERSION != new_version) {
				return UPDATE_CHECK_UPDATE_FOUND;
			}
		}
	}
	return UPDATE_CHECK_ALREADY_UPDATED;
}



int load_app(Directories* directories, Resources* resources, Settings* settings, bool* update_found) {
	if (settings->auto_update_check) {
		if (check_update(directories) == UPDATE_CHECK_UPDATE_FOUND) {
			*update_found = true;
		}
	}
	return init_ai(settings, directories);
}



class Load : public App::Scene {
private:
	bool load_failed;
	String tips;
	bool update_found;
	future<int> load_future;

public:
	Load(const InitData& init) : IScene{ init } {
		load_failed = false;
		tips = language.get_random("tips", "tips");
		update_found = false;
		load_future = async(launch::async, load_app, &getData().directories, &getData().resources, &getData().settings, &update_found);
	}

	void update() override {
		Scene::SetBackground(getData().colors.green);
		const int icon_width = (LEFT_RIGHT - LEFT_LEFT);
		getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(LEFT_LEFT, Y_CENTER - icon_width / 2);
		getData().resources.logo.scaled((double)icon_width * 0.8 / getData().resources.logo.width()).draw(RIGHT_LEFT, Y_CENTER - 40);
		if (load_future.wait_for(chrono::seconds(0)) == future_status::ready) {
			int load_code = load_future.get();
			if (load_code == ERR_OK) {
				cerr << "loaded" << endl;
				getData().menu_elements.init(&getData().settings, &getData().resources);
				changeScene(U"Main_scene", SCENE_FADE_TIME);
			}
			else {
				load_failed = true;
			}
		}
		if (load_failed) {
			getData().fonts.font50(language.get("loading", "load_failed")).draw(RIGHT_LEFT, Y_CENTER + 30, getData().colors.white);
		}
		else {
			getData().fonts.font50(language.get("loading", "loading")).draw(RIGHT_LEFT, Y_CENTER + 40, getData().colors.white);
			getData().fonts.font20(language.get("tips", "do_you_know")).draw(RIGHT_LEFT, Y_CENTER + 110, getData().colors.white);
			getData().fonts.font15(tips).draw(RIGHT_LEFT, Y_CENTER + 140, getData().colors.white);
		}
	}

	void draw() const override {

	}
};
