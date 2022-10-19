/*
    Egaroucid for Web Project

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

void init_default_settings(const Directories* directories, const Resources* resources, Settings* settings) {
	cerr << "use default settings" << endl;
	settings->n_threads = min(32, (int)thread::hardware_concurrency());
	settings->auto_update_check = 1;
	settings->lang_name = "japanese";
	settings->book_file = directories->document_dir + "Egaroucid/book.egbk";
	settings->use_book = true;
	settings->level = 13;
	settings->ai_put_black = false;
	settings->ai_put_white = false;
	settings->use_disc_hint = true;
	settings->use_umigame_value = false;
	settings->n_disc_hint = SHOW_ALL_HINT;
	settings->show_legal = true;
	settings->show_graph = true;
	settings->show_opening_on_cell = true;
	settings->show_log = true;
	settings->book_learn_depth = 40;
	settings->book_learn_error = 3;
	settings->show_stable_discs = false;
	settings->change_book_by_right_click = false;
	settings->ignore_book = false;
	settings->show_last_move = true;
	settings->show_next_move = true;
}

int init_resources(Resources* resources, Settings* settings) {
	// language names
	ifstream ifs_lang("resources/languages/languages.txt");
	if (ifs_lang.fail()) {
		return ERR_LANG_LIST_NOT_LOADED;
	}
	string lang_line;
	while (getline(ifs_lang, lang_line)) {
		while (lang_line.back() == '\n' || lang_line.back() == '\r') {
			lang_line.pop_back();
		}
		resources->language_names.emplace_back(lang_line);
	}
	if (resources->language_names.size() == 0) {
		return ERR_LANG_LIST_NOT_LOADED;
	}

	// language json
	if (!language_name.init()) {
		return ERR_LANG_JSON_NOT_LOADED;
	}

	// language
	string lang_file = "resources/languages/" + settings->lang_name + ".json";
	if (!language.init(lang_file)) {
		return ERR_LANG_NOT_LOADED;
	}

	// textures
	Texture icon(U"resources/img/icon.png", TextureDesc::Mipped);
	Texture logo(U"resources/img/logo.png", TextureDesc::Mipped);
	Texture checkbox(U"resources/img/checked.png", TextureDesc::Mipped);
	Texture unchecked(U"resources/img/unchecked.png", TextureDesc::Mipped);
	if (icon.isEmpty() || logo.isEmpty() || checkbox.isEmpty() || unchecked.isEmpty()) {
		return ERR_TEXTURE_NOT_LOADED;
	}
	resources->icon = icon;
	resources->logo = logo;
	resources->checkbox = checkbox;
	resources->unchecked = unchecked;

	// opening
	if (!opening_init(settings->lang_name)) {
		return ERR_OPENING_NOT_LOADED;
	}

	return ERR_OK;

}

int silent_load(Directories* directories, Resources* resources, Settings* settings) {
	init_directories(directories);
	init_default_settings(directories, resources, settings);
	return init_resources(resources, settings);
}

class Silent_load : public App::Scene {
private:
	future<int> silent_load_future;
	bool silent_load_failed;

public:
	Silent_load(const InitData& init) : IScene{ init } {
		silent_load_future = async(launch::async, silent_load, &getData().directories, &getData().resources, &getData().settings);
		silent_load_failed = false;
		cerr << "start silent loading" << endl;
	}

	void update() override {
		if (silent_load_future.wait_for(chrono::seconds(0)) == future_status::ready) {
			int load_code = silent_load_future.get();
			if (load_code == ERR_OK) {
				cerr << "silent loaded" << endl;
				changeScene(U"Load", SCENE_FADE_TIME);
			}
			else {
				silent_load_failed = true;
			}
		}
		if (silent_load_failed) {
			getData().fonts.font(U"BASIC DATA NOT LOADED. PLEASE RE-INSTALL.").draw(30, LEFT_LEFT, Y_CENTER + 50, getData().colors.white);
		}
	}

	void draw() const override {
		//Scene::SetBackground(getData().colors.green);
		Scene::SetBackground(getData().colors.black);
	}
};
