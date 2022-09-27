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

void init_directories(Directories* directories) {
	// system directory
	directories->document_dir = FileSystem::GetFolderPath(SpecialFolder::Documents).narrow();
	directories->appdata_dir = FileSystem::GetFolderPath(SpecialFolder::LocalAppData).narrow();
	cerr << "document_dir " << directories->document_dir << endl;
	cerr << "appdata_dir " << directories->appdata_dir << endl;

	// file directories
	directories->eval_file = "resources/eval.egev";
}

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
	settings->book_learn_error = 6;
}

int init_settings_import_int(TextReader* reader, int* res) {
	String line;
	if (reader->readLine(line)) {
		try {
			*res = Parse<int32>(line);
			return ERR_OK;
		}
		catch (const ParseError& e) {
			return ERR_IMPORT_SETTINGS;
		}
	}
	else {
		return ERR_IMPORT_SETTINGS;
	}
}

int init_settings_import_bool(TextReader* reader, bool* res) {
	String line;
	if (reader->readLine(line)) {
		try {
			int int_res = Parse<int32>(line);
			if (int_res != 0 && int_res != 1) {
				return ERR_IMPORT_SETTINGS;
			}
			*res = (bool)int_res;
			return ERR_OK;
		}
		catch (const ParseError& e) {
			return ERR_IMPORT_SETTINGS;
		}
	}
	else {
		return ERR_IMPORT_SETTINGS;
	}
}

int init_settings_import_str(TextReader* reader, string* res) {
	String line;
	if (reader->readLine(line)) {
		*res = line.narrow();
		return ERR_OK;
	}
	else {
		return ERR_IMPORT_SETTINGS;
	}
}

void init_settings(const Directories* directories, const Resources* resources, Settings* settings) {
	TextReader reader(U"{}Egaroucid/setting.txt"_fmt(Unicode::Widen(directories->appdata_dir)));
	if (!reader) {
		goto use_default_settings;
	}
	else {
		if (init_settings_import_int(&reader, &settings->n_threads) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_bool(&reader, &settings->auto_update_check) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_str(&reader, &settings->lang_name) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_str(&reader, &settings->book_file) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_bool(&reader, &settings->use_book) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_int(&reader, &settings->level) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_bool(&reader, &settings->ai_put_black) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_bool(&reader, &settings->ai_put_white) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_bool(&reader, &settings->use_disc_hint) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_bool(&reader, &settings->use_umigame_value) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_int(&reader, &settings->n_disc_hint) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_bool(&reader, &settings->show_legal) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_bool(&reader, &settings->show_graph) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_bool(&reader, &settings->show_opening_on_cell) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_bool(&reader, &settings->show_log) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_int(&reader, &settings->book_learn_depth) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_int(&reader, &settings->book_learn_error) != ERR_OK) {
			goto use_default_settings;
		}
	}
use_default_settings:
	init_default_settings(directories, resources, settings);
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
	if (icon.isEmpty() || logo.isEmpty() || checkbox.isEmpty()) {
		return ERR_TEXTURE_NOT_LOADED;
	}
	resources->icon = icon;
	resources->logo = logo;
	resources->checkbox = checkbox;

	// opening
	if (!opening_init()) {
		return ERR_OPENING_NOT_LOADED;
	}

	return ERR_OK;

}

int silent_load(Directories* directories, Resources* resources, Settings* settings) {
	init_directories(directories);
	init_settings(directories, resources, settings);
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
			getData().fonts.font30(U"BASIC DATA NOT LOADED. PLEASE RE-INSTALL.").draw(LEFT_LEFT, Y_CENTER + getData().fonts.font50.fontSize(), getData().colors.white);
		}
	}

	void draw() const override {
		//Scene::SetBackground(getData().colors.green);
		Scene::SetBackground(getData().colors.black);
	}
};
