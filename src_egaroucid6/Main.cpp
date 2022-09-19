#include <iostream>
#include "ai.hpp"
#include <Siv3D.hpp> // OpenSiv3D v0.6.3

using namespace std;

// version definition
#define EGAROUCID_VERSION U"6.0.0"

// error definition
#define ERR_OK 0
#define ERR_LANG_LIST_NOT_LOADED 1
#define ERR_TEXTURE_NOT_LOADED 2
#define ERR_EVAL_FILE_NOT_IMPORTED 1
#define ERR_BOOK_FILE_NOT_IMPORTED 2
#define ERR_IMPORT_SETTINGS 1

// constant definition
#define AI_MODE_HUMAN_AI 0
#define AI_MODE_AI_HUMAN 1
#define AI_MODE_AI_AI 2
#define AI_MODE_HUMAN_HUMAN 3
#define SHOW_ALL_HINT 100
#define UPDATE_CHECK_ALREADY_UPDATED 0
#define UPDATE_CHECK_UPDATE_FOUND 1

struct Colors {
	Color green;
};

struct Directories {
	string document_dir;
	string appdata_dir;
	string eval_file;
};

struct Resources {
	vector<string> language_names;
	Texture icon;
	Texture logo;
	Texture checkbox;
};

struct Settings {
	int n_threads;
	int use_auto_update_check;
	string lang_name;
	string book_file;
	int use_book;
	int ai_level;
	int hint_level;
	int use_ai_mode;
	int use_hint_all;
	int use_normal_hint;
	int use_umigame_value;
	int show_hint_num;
};

void init_colors(Colors* colors) {
	colors->green = Color(36, 153, 114, 100);
}

void init_directories(Directories* directories) {
	// system directory
	directories->document_dir = FileSystem::GetFolderPath(SpecialFolder::Documents).narrow();
	directories->appdata_dir = FileSystem::GetFolderPath(SpecialFolder::LocalAppData).narrow();

	// file directories
	directories->eval_file = "resources/eval.egev";
}

int init_resources(Resources* resources) {
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
	return ERR_OK;

}

void init_default_settings(const Directories* directories, const Resources *resources, Settings* settings) {
	settings->n_threads = min(32, (int)thread::hardware_concurrency());
	settings->use_auto_update_check = 1;
	settings->lang_name = resources->language_names[0];
	settings->book_file = directories->document_dir + "Egaroucid/book.egbk";
	settings->use_book = 1;
	settings->ai_level = 13;
	settings->hint_level = 13;
	settings->use_ai_mode = AI_MODE_HUMAN_HUMAN;
	settings->use_hint_all = 1;
	settings->use_normal_hint = 1;
	settings->use_umigame_value = 0;
	settings->show_hint_num = SHOW_ALL_HINT;
}

int init_settings_import_int(TextReader* reader, int *res) {
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

int init_settings_import_str(TextReader* reader, string *res) {
	String line;
	if (reader->readLine(line)) {
		*res = line.narrow();
		return ERR_OK;
	}
	else {
		return ERR_IMPORT_SETTINGS;
	}
}

void init_settings(const Directories* directories, const Resources *resources, Settings* settings) {
	TextReader reader(U"{}Egaroucid/setting.txt"_fmt(Unicode::Widen(directories->appdata_dir)));
	if (!reader) {
		goto use_default_settings;
	}
	else {
		if (init_settings_import_int(&reader, &settings->n_threads) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_int(&reader, &settings->use_auto_update_check) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_str(&reader, &settings->lang_name) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_str(&reader, &settings->book_file) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_int(&reader, &settings->use_book) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_int(&reader, &settings->ai_level) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_int(&reader, &settings->hint_level) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_int(&reader, &settings->use_ai_mode) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_int(&reader, &settings->use_hint_all) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_int(&reader, &settings->use_normal_hint) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_int(&reader, &settings->use_umigame_value) != ERR_OK) {
			goto use_default_settings;
		}
		if (init_settings_import_int(&reader, &settings->show_hint_num) != ERR_OK) {
			goto use_default_settings;
		}
	}
use_default_settings:
	init_default_settings(directories, resources, settings);
}

int init_ai(const Settings *settings, const Directories *directories) {
	thread_pool.resize(settings->n_threads);
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
	const FilePath version_save_path = U"{}Egaroucid/version.txt"_fmt(directories->appdata_dir);
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

void info_update_found() {

}

void error_resources(int err_code) {

}

void error_ai(int err_code) {

}

void Main() {
	Size window_size = Size(1000, 720);
	Window::Resize(window_size);
	Window::SetStyle(WindowStyle::Sizable);
	Scene::SetResizeMode(ResizeMode::Keep);
	Window::SetTitle(U"Egaroucid {}"_fmt(EGAROUCID_VERSION));
	System::SetTerminationTriggers(UserAction::NoAction);
	Console.open();
	stringstream logger_stream;
	//cerr.rdbuf(logger_stream.rdbuf());
	string logger;
	String logger_String;

	Colors colors;
	Directories directories;
	Resources resources;
	Settings settings;
	init_colors(&colors);
	init_directories(&directories);
	if (int resources_init_code = init_resources(&resources) != ERR_OK) {
		error_resources(resources_init_code);
		exit(1);
	}
	init_settings(&directories, &resources, &settings);
	if (settings.use_auto_update_check) {
		if (check_update(&directories) == UPDATE_CHECK_UPDATE_FOUND) {
			info_update_found();
		}
	}
	if (int ai_init_code = init_ai(&settings, &directories) != ERR_OK) {
		error_ai(ai_init_code);
		exit(1);
	}

	Scene::SetBackground(colors.green);
}
