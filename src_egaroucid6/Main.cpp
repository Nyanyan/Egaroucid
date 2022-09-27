#include <iostream>
#include <future>
#include "ai.hpp"
#include "gui/language.hpp"
#include "gui/menu.hpp"
#include "gui/gui_common.hpp"
#include "gui/graph.hpp"
#include "gui/opening.hpp"
#include "gui/button.hpp"
#include "gui/radio_button.hpp"
#include <Siv3D.hpp> // OpenSiv3D v0.6.3

using namespace std;

// version definition
#define EGAROUCID_VERSION U"6.0.0"

// scene definition
#define SCENE_FADE_TIME 200

// coordinate definition
#define WINDOW_SIZE_X 800
#define WINDOW_SIZE_Y 500
#define PADDING 20
constexpr int LEFT_LEFT = PADDING;
constexpr int LEFT_RIGHT = WINDOW_SIZE_X / 2 - PADDING;
constexpr int LEFT_CENTER = (LEFT_LEFT + LEFT_RIGHT) / 2;
constexpr int RIGHT_LEFT = WINDOW_SIZE_X / 2 + PADDING;
constexpr int RIGHT_RIGHT = WINDOW_SIZE_X - PADDING;
constexpr int RIGHT_CENTER = (RIGHT_LEFT + RIGHT_RIGHT) / 2;
constexpr int X_CENTER = WINDOW_SIZE_X / 2;
constexpr int Y_CENTER = WINDOW_SIZE_Y / 2;

// error definition
#define ERR_OK 0
#define ERR_LANG_LIST_NOT_LOADED 1
#define ERR_LANG_JSON_NOT_LOADED 2
#define ERR_LANG_NOT_LOADED 3
#define ERR_TEXTURE_NOT_LOADED 4
#define ERR_OPENING_NOT_LOADED 5
#define ERR_EVAL_FILE_NOT_IMPORTED 1
#define ERR_BOOK_FILE_NOT_IMPORTED 2
#define ERR_IMPORT_SETTINGS 1

// constant definition
#define AI_MODE_HUMAN_AI 0
#define AI_MODE_AI_HUMAN 1
#define AI_MODE_AI_AI 2
#define AI_MODE_HUMAN_HUMAN 3
#define SHOW_ALL_HINT 35
#define UPDATE_CHECK_ALREADY_UPDATED 0
#define UPDATE_CHECK_UPDATE_FOUND 1

// board drawing constants
#define BOARD_SIZE 400
#define BOARD_COORD_SIZE 20
#define DISC_SIZE 20
#define LEGAL_SIZE 7
#define BOARD_CELL_FRAME_WIDTH 2
#define BOARD_DOT_SIZE 5
#define BOARD_ROUND_FRAME_WIDTH 10
#define BOARD_ROUND_DIAMETER 20
#define BOARD_SY 60
constexpr int BOARD_SX = LEFT_LEFT + BOARD_COORD_SIZE;
constexpr int BOARD_CELL_SIZE = BOARD_SIZE / HW;

// graph drawing constants
#define GRAPH_RESOLUTION 8
constexpr int GRAPH_SX = BOARD_SX + BOARD_SIZE + 50;
constexpr int GRAPH_SY = Y_CENTER + 30;
constexpr int GRAPH_WIDTH = WINDOW_SIZE_X - GRAPH_SX - 20;
constexpr int GRAPH_HEIGHT = WINDOW_SIZE_Y - GRAPH_SY - 40;

// info drawing constants
#define INFO_SY 35
#define INFO_DISC_RADIUS 12
constexpr int INFO_SX = BOARD_SX + BOARD_SIZE + 25;

// graph mode constants
#define GRAPH_MODE_NORMAL 0
#define GRAPH_MODE_INSPECT 1

// button press constants
#define BUTTON_NOT_PUSHED 0
#define BUTTON_LONG_PRESS_THRESHOLD 500

// hint constants
#define HINT_NOT_CALCULATING -1
#define HINT_INIT_VALUE -INF
#define HINT_TYPE_BOOK 1000
#define HINT_INF_LEVEL 100
#define HINT_MAX_LEVEL 60

// analyze constants
#define ANALYZE_SIZE 62

// back button constants
#define BACK_BUTTON_WIDTH 200
#define BACK_BUTTON_HEIGHT 50
#define BACK_BUTTON_SY 420
#define BACK_BUTTON_RADIUS 20
constexpr int BACK_BUTTON_SX = X_CENTER - BACK_BUTTON_WIDTH / 2;

// go/back button constants
#define GO_BACK_BUTTON_WIDTH 200
#define GO_BACK_BUTTON_HEIGHT 50
#define GO_BACK_BUTTON_SY 420
#define GO_BACK_BUTTON_RADIUS 20
constexpr int GO_BACK_BUTTON_GO_SX = X_CENTER + 10;
constexpr int GO_BACK_BUTTON_BACK_SX = X_CENTER - GO_BACK_BUTTON_WIDTH - 10;

// 3 buttons constants
#define BUTTON3_WIDTH 200
#define BUTTON3_HEIGHT 50
#define BUTTON3_SY 420
#define BUTTON3_RADIUS 20
constexpr int BUTTON3_1_SX = X_CENTER - BUTTON3_WIDTH * 3 / 2 - 10;
constexpr int BUTTON3_2_SX = X_CENTER - BUTTON3_WIDTH / 2;
constexpr int BUTTON3_3_SX = X_CENTER + BUTTON3_WIDTH / 2 + 10;

#define BUTTON2_VERTICAL_WIDTH 200
#define BUTTON2_VERTICAL_HEIGHT 50
#define BUTTON2_VERTICAL_1_SY 350
#define BUTTON2_VERTICAL_2_SY 420
#define BUTTON2_VERTICAL_SX 520
#define BUTTON2_VERTICAL_RADIUS 20

struct Colors {
	Color green{ Color(36, 153, 114, 100) };
	Color black{ Palette::Black };
	Color white{ Palette::White };
	Color dark_gray{ Color(51, 51, 51) };
	Color cyan{ Palette::Cyan };
	Color red{ Palette::Red };
	Color light_cyan{ Palette::Lightcyan };
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
	bool auto_update_check;
	string lang_name;
	string book_file;
	bool use_book;
	int level;
	bool ai_put_black;
	bool ai_put_white;
	bool use_disc_hint;
	bool use_umigame_value;
	int n_disc_hint;
	bool show_legal;
	bool show_graph;
	bool show_opening_on_cell;
	bool show_log;
	int book_learn_depth;
	int book_learn_error;
};

struct Fonts {
	Font font50{ 50 };
	Font font40{ 40 };
	Font font30{ 30 };
	Font font25{ 25 };
	Font font20{ 20 };
	Font font15{ 15 };
	Font font13{ 13 };
	Font font12{ 12 };
	Font font10{ 10 };
	Font font15_bold{ 15, Typeface::Bold };
	Font font13_heavy{ 13, Typeface::Heavy };
	Font font9_bold{ 9, Typeface::Bold };
};

struct Menu_elements {
	bool dummy;

	// 対局
	bool start_game;
	bool analyze;

	// 設定
	// AIの設定
	bool use_book;
	int level;
	int n_threads;
	// 着手
	bool ai_put_black;
	bool ai_put_white;

	// 表示
	bool use_disc_hint;
	int n_disc_hint;
	bool use_umigame_value;
	bool show_legal;
	bool show_graph;
	bool show_opening_on_cell;
	bool show_log;

	// 定石
	bool book_start_learn;
	int book_learn_depth;
	int book_learn_error;
	bool book_import;
	bool book_reference;

	// 入出力
	// 入力
	bool input_transcript;
	bool input_board;
	bool edit_board;
	bool input_game;
	// 出力
	bool copy_transcript;
	bool save_game;

	// 操作
	bool stop_calculating;
	bool forward;
	bool backward;
	// 変換
	bool convert_180;
	bool convert_blackline;
	bool convert_whiteline;

	// ヘルプ
	bool usage;
	bool bug_report;
	bool auto_update_check;
	bool license;

	// language
	bool languages[200];

	void init(Settings *settings, Resources *resources) {
		dummy = false;

		start_game = false;
		analyze = false;

		use_book = settings->use_book;
		level = settings->level;
		n_threads = settings->n_threads;
		ai_put_black = settings->ai_put_black;
		ai_put_white = settings->ai_put_white;

		use_disc_hint = settings->use_disc_hint;
		n_disc_hint = settings->n_disc_hint;
		use_umigame_value = settings->use_umigame_value;
		show_legal = settings->show_legal;
		show_graph = settings->show_graph;
		show_opening_on_cell = settings->show_opening_on_cell;
		show_log = settings->show_log;

		book_start_learn = false;
		book_learn_depth = settings->book_learn_depth;
		book_learn_error = settings->book_learn_error;
		book_import = false;
		book_reference = false;

		input_transcript = false;
		input_board = false;
		edit_board = false;
		input_game = false;
		copy_transcript = false;
		save_game = false;

		stop_calculating = false;
		forward = false;
		backward = false;
		convert_180 = false;
		convert_blackline = false;
		convert_whiteline = false;

		usage = false;
		bug_report = false;
		auto_update_check = settings->auto_update_check;
		license = false;

		bool lang_found = false;
		for (int i = 0; i < resources->language_names.size(); ++i) {
			if (resources->language_names[i] == settings->lang_name) {
				lang_found = true;
				languages[i] = true;
			}
			else {
				languages[i] = false;
			}
		}
		if (!lang_found) {
			settings->lang_name = resources->language_names[0];
			languages[0] = true;
		}
	}
};

struct Graph_resources {
	vector<History_elem> nodes[2];
	int n_discs;
	int delta;
	int put_mode;
	bool need_init;

	Graph_resources() {
		init();
	}

	void init() {
		nodes[0].clear();
		nodes[1].clear();
		n_discs = 4;
		delta = 0;
		put_mode = 0;
		need_init = true;
	}

	int node_find(int mode, int n_discs) {
		for (int i = 0; i < (int)nodes[mode].size(); ++i) {
			if (nodes[mode][i].board.n_discs() == n_discs) {
				return i;
			}
		}
		return -1;
	}
};

struct Common_resources {
	Colors colors;
	Directories directories;
	Resources resources;
	Settings settings;
	Fonts fonts;
	Menu_elements menu_elements;
	Menu menu;
	History_elem history_elem;
	Graph_resources graph_resources;
};

struct Hint_info {
	double value;
	int cell;
	int type;
};

struct Move_board_button_status {
	uint64_t left_pushed{ BUTTON_NOT_PUSHED };
	uint64_t right_pushed{ BUTTON_NOT_PUSHED };
};

struct Analyze_info {
	int idx;
	int sgn;
	Board board;
};

struct AI_status {
	bool ai_thinking{ false };
	future<Search_result> ai_future;

	bool hint_calculating{ false };
	int hint_level{ HINT_NOT_CALCULATING };
	future<Search_result> hint_future[HW2];
	vector<pair<int, function<Search_result()>>> hint_task_stack;
	bool hint_use[HW2];
	double hint_values[HW2];
	int hint_types[HW2];
	bool hint_available{ false };
	bool hint_use_stable[HW2];
	double hint_values_stable[HW2];
	int hint_types_stable[HW2];

	bool analyzing{ false };
	future<Search_result> analyze_future[ANALYZE_SIZE];
	int analyze_sgn[ANALYZE_SIZE];
	vector<pair<Analyze_info, function<Search_result()>>> analyze_task_stack;
};

struct Game_abstract {
	String black_player;
	String white_player;
	int black_score;
	int white_score;
	String memo;
	String date;
	String transcript;
};

using App = SceneManager<String, Common_resources>;

void init_directories(Directories* directories) {
	// system directory
	directories->document_dir = FileSystem::GetFolderPath(SpecialFolder::Documents).narrow();
	directories->appdata_dir = FileSystem::GetFolderPath(SpecialFolder::LocalAppData).narrow();
	cerr << "document_dir " << directories->document_dir << endl;
	cerr << "appdata_dir " << directories->appdata_dir << endl;

	// file directories
	directories->eval_file = "resources/eval.egev";
}

int init_resources(Resources* resources, Settings *settings) {
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

void init_default_settings(const Directories* directories, const Resources *resources, Settings* settings) {
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

int init_ai(const Settings *settings, const Directories *directories) {
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

int silent_load(Directories* directories, Resources* resources, Settings *settings) {
	init_directories(directories);
	init_settings(directories, resources, settings);
	return init_resources(resources, settings);
}

int load_app(Directories *directories, Resources *resources, Settings *settings, bool *update_found) {
	if (settings->auto_update_check) {
		if (check_update(directories) == UPDATE_CHECK_UPDATE_FOUND) {
			*update_found = true;
		}
	}
	return init_ai(settings, directories);
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

bool compare_value_cell(pair<int, int>& a, pair<int, int>& b) {
	return a.first > b.first;
}

bool compare_hint_info(Hint_info& a, Hint_info& b) {
	return a.value > b.value;
}

void draw_board(Fonts fonts, Colors colors, History_elem history_elem) {
	String coord_x = U"abcdefgh";
	for (int i = 0; i < HW; ++i) {
		fonts.font15_bold(i + 1).draw(Arg::center(BOARD_SX - BOARD_COORD_SIZE, BOARD_SY + BOARD_CELL_SIZE * i + BOARD_CELL_SIZE / 2), colors.dark_gray);
		fonts.font15_bold(coord_x[i]).draw(Arg::center(BOARD_SX + BOARD_CELL_SIZE * i + BOARD_CELL_SIZE / 2, BOARD_SY - BOARD_COORD_SIZE - 2), colors.dark_gray);
	}
	for (int i = 0; i < HW_M1; ++i) {
		Line(BOARD_SX + BOARD_CELL_SIZE * (i + 1), BOARD_SY, BOARD_SX + BOARD_CELL_SIZE * (i + 1), BOARD_SY + BOARD_CELL_SIZE * HW).draw(BOARD_CELL_FRAME_WIDTH, colors.dark_gray);
		Line(BOARD_SX, BOARD_SY + BOARD_CELL_SIZE * (i + 1), BOARD_SX + BOARD_CELL_SIZE * HW, BOARD_SY + BOARD_CELL_SIZE * (i + 1)).draw(BOARD_CELL_FRAME_WIDTH, colors.dark_gray);
	}
	Circle(BOARD_SX + 2 * BOARD_CELL_SIZE, BOARD_SY + 2 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(colors.dark_gray);
	Circle(BOARD_SX + 2 * BOARD_CELL_SIZE, BOARD_SY + 6 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(colors.dark_gray);
	Circle(BOARD_SX + 6 * BOARD_CELL_SIZE, BOARD_SY + 2 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(colors.dark_gray);
	Circle(BOARD_SX + 6 * BOARD_CELL_SIZE, BOARD_SY + 6 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(colors.dark_gray);
	RoundRect(BOARD_SX, BOARD_SY, BOARD_CELL_SIZE * HW, BOARD_CELL_SIZE * HW, BOARD_ROUND_DIAMETER).drawFrame(0, BOARD_ROUND_FRAME_WIDTH, colors.white);
	Flip flip;
	int board_arr[HW2];
	history_elem.board.translate_to_arr(board_arr, history_elem.player);
	for (int cell = 0; cell < HW2; ++cell) {
		int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
		int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
		if (board_arr[cell] == BLACK) {
			Circle(x, y, DISC_SIZE).draw(colors.black);
		}
		else if (board_arr[cell] == WHITE) {
			Circle(x, y, DISC_SIZE).draw(colors.white);
		}
	}
	if (history_elem.policy != -1) {
		int x = BOARD_SX + (HW_M1 - history_elem.policy % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
		int y = BOARD_SY + (HW_M1 - history_elem.policy / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
		Circle(x, y, LEGAL_SIZE).draw(colors.red);
	}
}

class Main_scene : public App::Scene {
private:
	Graph graph;
	Move_board_button_status move_board_button_status;
	AI_status ai_status;

public:
	Main_scene(const InitData& init) : IScene{ init } {
		cerr << "main scene loading" << endl;
		getData().menu = create_menu(&getData().menu_elements);
		graph.sx = GRAPH_SX;
		graph.sy = GRAPH_SY;
		graph.size_x = GRAPH_WIDTH;
		graph.size_y = GRAPH_HEIGHT;
		graph.resolution = GRAPH_RESOLUTION;
		graph.font = getData().fonts.font15;
		graph.font_size = 15;
		if (getData().graph_resources.need_init) {
			getData().graph_resources.init();
			getData().graph_resources.nodes[getData().graph_resources.put_mode].emplace_back(getData().history_elem);
		}
		cerr << "main scene loaded" << endl;
	}

	void update() override {
		Scene::SetBackground(getData().colors.green);

		// init
		getData().graph_resources.delta = 0;

		// opening
		update_opening();

		// menu
		menu_game();
		menu_manipulate();
		menu_in_out();
		menu_book();

		// analyze
		if (ai_status.analyzing) {
			analyze_get_task();
		}

		bool graph_interact_ignore = ai_status.analyzing;
		// transcript move
		if (!graph_interact_ignore && !getData().menu.active()) {
			interact_graph();
			update_n_discs();
		}

		bool move_ignore = ai_status.analyzing;
		// move
		bool ai_should_move =
			getData().graph_resources.put_mode == GRAPH_MODE_NORMAL &&
			((getData().history_elem.player == BLACK && getData().menu_elements.ai_put_black) || (getData().history_elem.player == WHITE && getData().menu_elements.ai_put_white)) &&
			getData().history_elem.board.n_discs() == getData().graph_resources.nodes[GRAPH_MODE_NORMAL][getData().graph_resources.nodes[GRAPH_MODE_NORMAL].size() - 1].board.n_discs();
		if (!move_ignore) {
			if (ai_should_move) {
				ai_move();
			}
			else if (!getData().menu.active()) {
				interact_move();
			}
		}

		// board drawing
		draw_board(getData().fonts, getData().colors, getData().history_elem);

		bool hint_ignore = ai_should_move || ai_status.analyzing;

		// hint / legalcalculating & drawing
		if (!hint_ignore) {
			if (getData().menu_elements.use_disc_hint) {
				if (!ai_status.hint_calculating && ai_status.hint_level < getData().menu_elements.level) {
					hint_init_calculating();
					cerr << "hint search level " << ai_status.hint_level << endl;
				}
				hint_do_task();
				uint64_t legal_ignore = draw_hint();
				if (getData().menu_elements.show_legal) {
					draw_legal(legal_ignore);
				}
			}
			else if (getData().menu_elements.show_legal) {
				draw_legal(0);
			}
		}

		// graph drawing
		if (getData().menu_elements.show_graph) {
			graph.draw(getData().graph_resources.nodes[0], getData().graph_resources.nodes[1], getData().graph_resources.n_discs);
		}

		// info drawing
		draw_info();

		// opening on cell drawing
		if (getData().menu_elements.show_opening_on_cell) {
			draw_opening_on_cell();
		}

		// menu drawing
		getData().menu.draw();
	}

	void draw() const override {

	}

private:
	void reset_hint() {
		ai_status.hint_level = HINT_NOT_CALCULATING;
		ai_status.hint_available = false;
		ai_status.hint_calculating = false;
	}

	void reset_ai() {
		ai_status.ai_thinking = false;
	}

	void reset_analyze() {
		ai_status.analyzing = false;
		ai_status.analyze_task_stack.clear();
	}

	void stop_calculating() {
		cerr << "terminating calculation" << endl;
		global_searching = false;
		if (ai_status.ai_future.valid()) {
			ai_status.ai_future.get();
		}
		for (int i = 0; i < HW2; ++i) {
			if (ai_status.hint_future[i].valid()) {
				ai_status.hint_future[i].get();
			}
		}
		for (int i = 0; i < ANALYZE_SIZE; ++i) {
			if (ai_status.analyze_future[i].valid()) {
				ai_status.analyze_future[i].get();
			}
		}
		global_searching = true;
		cerr << "calculation terminated" << endl;
		reset_ai();
		reset_hint();
		reset_analyze();
		cerr << "reset all calculations" << endl;
	}

	void menu_game() {
		if (getData().menu_elements.start_game) {
			stop_calculating();
			getData().history_elem.reset();
			getData().graph_resources.init();
			getData().graph_resources.nodes[getData().graph_resources.put_mode].emplace_back(getData().history_elem);
		}
		if (getData().menu_elements.analyze && !ai_status.ai_thinking && !ai_status.analyzing) {
			stop_calculating();
			init_analyze();
		}
	}

	void menu_in_out() {
		if (getData().menu_elements.input_transcript) {
			changeScene(U"Import_transcript", SCENE_FADE_TIME);
		}
		if (getData().menu_elements.input_board) {
			changeScene(U"Import_board", SCENE_FADE_TIME);
		}
		if (getData().menu_elements.edit_board) {
			changeScene(U"Edit_board", SCENE_FADE_TIME);
		}
		if (getData().menu_elements.input_game) {
			changeScene(U"Import_game", SCENE_FADE_TIME);
		}
	}

	void menu_manipulate() {
		if (getData().menu_elements.stop_calculating) {
			stop_calculating();
			ai_status.hint_level = HINT_INF_LEVEL;
		}
		if (!ai_status.analyzing) {
			if (getData().menu_elements.backward) {
				--getData().graph_resources.n_discs;
				getData().graph_resources.delta = -1;
			}
			if (getData().menu_elements.forward) {
				++getData().graph_resources.n_discs;
				getData().graph_resources.delta = 1;
			}
		}
		if (getData().menu_elements.convert_180) {
			stop_calculating();
			getData().history_elem.board.board_rotate_180();
			if (0 <= getData().history_elem.policy && getData().history_elem.policy < HW2) {
				getData().history_elem.policy = HW2_M1 - getData().history_elem.policy;
			}
			if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy < HW2) {
				getData().history_elem.next_policy = HW2_M1 - getData().history_elem.next_policy;
			}
			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
					getData().graph_resources.nodes[i][j].board.board_rotate_180();
					if (0 <= getData().graph_resources.nodes[i][j].policy && getData().graph_resources.nodes[i][j].policy < HW2) {
						getData().graph_resources.nodes[i][j].policy = HW2_M1 - getData().graph_resources.nodes[i][j].policy;
					}
					if (0 <= getData().graph_resources.nodes[i][j].next_policy && getData().graph_resources.nodes[i][j].next_policy < HW2) {
						getData().graph_resources.nodes[i][j].next_policy = HW2_M1 - getData().graph_resources.nodes[i][j].next_policy;
					}
				}
			}
			reset_hint();
		}
		if (getData().menu_elements.convert_blackline) {
			stop_calculating();
			getData().history_elem.board.board_black_line_mirror();
			if (0 <= getData().history_elem.policy && getData().history_elem.policy < HW2) {
				int x = getData().history_elem.policy % HW;
				int y = getData().history_elem.policy / HW;
				getData().history_elem.policy = (HW_M1 - x) * HW + (HW_M1 - y);
			}
			if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy < HW2) {
				int x = getData().history_elem.next_policy % HW;
				int y = getData().history_elem.next_policy / HW;
				getData().history_elem.next_policy = (HW_M1 - x) * HW + (HW_M1 - y);
			}
			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
					getData().graph_resources.nodes[i][j].board.board_black_line_mirror();
					if (0 <= getData().graph_resources.nodes[i][j].policy && getData().graph_resources.nodes[i][j].policy < HW2) {
						int x = getData().graph_resources.nodes[i][j].policy % HW;
						int y = getData().graph_resources.nodes[i][j].policy / HW;
						getData().graph_resources.nodes[i][j].policy = (HW_M1 - x) * HW + (HW_M1 - y);
					}
					if (0 <= getData().graph_resources.nodes[i][j].next_policy && getData().graph_resources.nodes[i][j].next_policy < HW2) {
						int x = getData().graph_resources.nodes[i][j].next_policy % HW;
						int y = getData().graph_resources.nodes[i][j].next_policy / HW;
						getData().graph_resources.nodes[i][j].next_policy = (HW_M1 - x) * HW + (HW_M1 - y);
					}
				}
			}
			reset_hint();
		}
		if (getData().menu_elements.convert_whiteline) {
			stop_calculating();
			getData().history_elem.board.board_white_line_mirror();
			if (0 <= getData().history_elem.policy && getData().history_elem.policy < HW2) {
				int x = getData().history_elem.policy % HW;
				int y = getData().history_elem.policy / HW;
				getData().history_elem.policy = x * HW + y;
			}
			if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy < HW2) {
				int x = getData().history_elem.next_policy % HW;
				int y = getData().history_elem.next_policy / HW;
				getData().history_elem.next_policy = x * HW + y;
			}
			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
					getData().graph_resources.nodes[i][j].board.board_white_line_mirror();
					if (0 <= getData().graph_resources.nodes[i][j].policy && getData().graph_resources.nodes[i][j].policy < HW2) {
						int x = getData().graph_resources.nodes[i][j].policy % HW;
						int y = getData().graph_resources.nodes[i][j].policy / HW;
						getData().graph_resources.nodes[i][j].policy = x * HW + y;
					}
					if (0 <= getData().graph_resources.nodes[i][j].next_policy && getData().graph_resources.nodes[i][j].next_policy < HW2) {
						int x = getData().graph_resources.nodes[i][j].next_policy % HW;
						int y = getData().graph_resources.nodes[i][j].next_policy / HW;
						getData().graph_resources.nodes[i][j].next_policy = x * HW + y;
					}
				}
			}
			reset_hint();
		}
	}

	void menu_book() {
		if (getData().menu_elements.book_import) {
			changeScene(U"Import_book", SCENE_FADE_TIME);
		}
		if (getData().menu_elements.book_reference) {
			changeScene(U"Refer_book", SCENE_FADE_TIME);
		}
	}

	void interact_graph() {
		getData().graph_resources.n_discs = graph.update_n_discs(getData().graph_resources.nodes[0], getData().graph_resources.nodes[1], getData().graph_resources.n_discs);
		if (!KeyLeft.pressed() && !KeyA.pressed()) {
			move_board_button_status.left_pushed = BUTTON_NOT_PUSHED;
		}
		if (!KeyRight.pressed() && !KeyD.pressed()) {
			move_board_button_status.right_pushed = BUTTON_NOT_PUSHED;
		}

		if (MouseX1.down() || KeyLeft.down() || KeyA.down() || (move_board_button_status.left_pushed != BUTTON_NOT_PUSHED && tim() - move_board_button_status.left_pushed >= BUTTON_LONG_PRESS_THRESHOLD)) {
			--getData().graph_resources.n_discs;
			getData().graph_resources.delta = -1;
			if (KeyLeft.down() || KeyA.down()) {
				move_board_button_status.left_pushed = tim();
			}
		}
		else if (MouseX2.down() || KeyRight.down() || KeyD.down() || (move_board_button_status.right_pushed != BUTTON_NOT_PUSHED && tim() - move_board_button_status.right_pushed >= BUTTON_LONG_PRESS_THRESHOLD)) {
			++getData().graph_resources.n_discs;
			getData().graph_resources.delta = 1;
			if (KeyRight.down() || KeyD.down()) {
				move_board_button_status.right_pushed = tim();
			}
		}
	}

	void update_n_discs() {
		int max_n_discs = getData().graph_resources.nodes[getData().graph_resources.put_mode].back().board.n_discs();
		getData().graph_resources.n_discs = min(getData().graph_resources.n_discs, max_n_discs);
		int min_n_discs = getData().graph_resources.nodes[GRAPH_MODE_NORMAL][0].board.n_discs();
		if (getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
			min_n_discs = min(min_n_discs, getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs());
		}
		getData().graph_resources.n_discs = max(getData().graph_resources.n_discs, min_n_discs);
		if (getData().graph_resources.put_mode == GRAPH_MODE_INSPECT && getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
			if (getData().graph_resources.n_discs < getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs()) {
				getData().graph_resources.put_mode = GRAPH_MODE_NORMAL;
				getData().graph_resources.nodes[1].clear();
			}
		}
		int node_idx = getData().graph_resources.node_find(getData().graph_resources.put_mode, getData().graph_resources.n_discs);
		if (node_idx == -1 && getData().graph_resources.put_mode == GRAPH_MODE_INSPECT) {
			getData().graph_resources.nodes[GRAPH_MODE_INSPECT].clear();
			int node_idx_0 = getData().graph_resources.node_find(GRAPH_MODE_NORMAL, getData().graph_resources.n_discs);
			if (node_idx_0 == -1) {
				cerr << "history vector element not found 0" << endl;
				return;
			}
			getData().graph_resources.nodes[GRAPH_MODE_INSPECT].emplace_back(getData().graph_resources.nodes[GRAPH_MODE_NORMAL][node_idx_0]);
			node_idx = getData().graph_resources.node_find(getData().graph_resources.put_mode, getData().graph_resources.n_discs);
		}
		while (node_idx == -1) {
			//cerr << "history vector element not found 1" << endl;
			getData().graph_resources.n_discs += getData().graph_resources.delta;
			node_idx = getData().graph_resources.node_find(GRAPH_MODE_NORMAL, getData().graph_resources.n_discs);
		}
		if (getData().history_elem.board != getData().graph_resources.nodes[getData().graph_resources.put_mode][node_idx].board) {
			stop_calculating();
			reset_hint();
		}
		getData().history_elem = getData().graph_resources.nodes[getData().graph_resources.put_mode][node_idx];
	}

	void move_processing(int_fast8_t cell) {
		int parent_idx = getData().graph_resources.node_find(getData().graph_resources.put_mode, getData().history_elem.board.n_discs());
		if (parent_idx != -1) {
			if (getData().graph_resources.nodes[getData().graph_resources.put_mode][parent_idx].next_policy == HW2_M1 - cell && parent_idx + 1 < (int)getData().graph_resources.nodes[getData().graph_resources.put_mode].size()) {
				++getData().graph_resources.n_discs;
				return;
			}
			getData().graph_resources.nodes[getData().graph_resources.put_mode][parent_idx].next_policy = HW2_M1 - cell;
			while (getData().graph_resources.nodes[getData().graph_resources.put_mode].size() > parent_idx + 1) {
				getData().graph_resources.nodes[getData().graph_resources.put_mode].pop_back();
			}
		}
		Flip flip;
		calc_flip(&flip, &getData().history_elem.board, HW2_M1 - cell);
		getData().history_elem.board.move_board(&flip);
		getData().history_elem.policy = HW2_M1 - cell;
		getData().history_elem.next_policy = -1;
		getData().history_elem.v = GRAPH_IGNORE_VALUE;
		getData().history_elem.level = -1;
		getData().history_elem.player ^= 1;
		if (getData().history_elem.board.get_legal() == 0ULL) {
			getData().history_elem.board.pass();
			getData().history_elem.player ^= 1;
		}
		getData().graph_resources.nodes[getData().graph_resources.put_mode].emplace_back(getData().history_elem);
		getData().graph_resources.n_discs++;
		reset_hint();
	}

	void interact_move() {
		uint64_t legal = getData().history_elem.board.get_legal();
		for (int_fast8_t cell = 0; cell < HW2; ++cell) {
			if (1 & (legal >> (HW2_M1 - cell))) {
				int x = cell % HW;
				int y = cell / HW;
				Rect cell_rect(BOARD_SX + x * BOARD_CELL_SIZE, BOARD_SY + y * BOARD_CELL_SIZE, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
				if (cell_rect.leftClicked()) {
					if (getData().graph_resources.put_mode == GRAPH_MODE_NORMAL) {
						int parent_idx = getData().graph_resources.node_find(GRAPH_MODE_NORMAL, getData().history_elem.board.n_discs());
						if (parent_idx != -1) {
							bool go_to_inspection_mode =
								getData().history_elem.board.n_discs() != getData().graph_resources.nodes[GRAPH_MODE_NORMAL][getData().graph_resources.nodes[GRAPH_MODE_NORMAL].size() - 1].board.n_discs() &&
								HW2_M1 - cell != getData().graph_resources.nodes[GRAPH_MODE_NORMAL][parent_idx].next_policy;
							if (go_to_inspection_mode) {
								getData().graph_resources.put_mode = GRAPH_MODE_INSPECT;
							}
						}
					}
					stop_calculating();
					move_processing(cell);
				}
			}
		}
	}

	void ai_move() {
		uint64_t legal = getData().history_elem.board.get_legal();
		if (!ai_status.ai_thinking) {
			if (legal) {
				ai_status.ai_future = async(launch::async, ai, getData().history_elem.board, getData().menu_elements.level, getData().menu_elements.use_book, true);
				ai_status.ai_thinking = true;
			}
		}
		else if (ai_status.ai_future.valid()) {
			if (ai_status.ai_future.wait_for(chrono::seconds(0)) == future_status::ready) {
				Search_result search_result = ai_status.ai_future.get();
				if (1 & (legal >> search_result.policy)) {
					int sgn = getData().history_elem.player == 0 ? 1 : -1;
					move_processing(HW2_M1 - search_result.policy);
					getData().graph_resources.nodes[getData().graph_resources.put_mode].back().v = sgn * search_result.value;
					getData().graph_resources.nodes[getData().graph_resources.put_mode].back().level = getData().menu_elements.level;
				}
				ai_status.ai_thinking = false;
			}
		}
	}

	void update_opening() {
		string new_opening = opening.get(getData().history_elem.board, getData().history_elem.player ^ 1);
		if (new_opening.size() && getData().history_elem.opening_name != new_opening) {
			getData().history_elem.opening_name = new_opening;
			int node_idx = getData().graph_resources.node_find(getData().graph_resources.put_mode, getData().graph_resources.n_discs);
			if (node_idx == -1) {
				cerr << "history vector element not found 2" << endl;
				return;
			}
			getData().graph_resources.nodes[getData().graph_resources.put_mode][node_idx].opening_name = new_opening;
		}
	}

	Menu create_menu(Menu_elements* menu_elements) {
		Menu menu;
		menu_title title;
		menu_elem menu_e, side_menu, side_side_menu;
		Font menu_font = getData().fonts.font12;



		title.init(language.get("play", "game"));

		menu_e.init_button(language.get("play", "new_game"), &menu_elements->start_game);
		title.push(menu_e);
		menu_e.init_button(language.get("play", "analyze"), &menu_elements->analyze);
		title.push(menu_e);

		menu.push(title);




		title.init(language.get("settings", "settings"));

		menu_e.init_check(language.get("ai_settings", "use_book"), &menu_elements->use_book, menu_elements->use_book);
		title.push(menu_e);
		menu_e.init_bar(language.get("ai_settings", "level"), &menu_elements->level, menu_elements->level, 0, 60);
		title.push(menu_e);
		menu_e.init_bar(language.get("settings", "thread", "thread"), &menu_elements->n_threads, menu_elements->n_threads, 1, 32);
		title.push(menu_e);

		menu_e.init_check(language.get("settings", "play", "ai_put_black"), &menu_elements->ai_put_black, menu_elements->ai_put_black);
		title.push(menu_e);
		menu_e.init_check(language.get("settings", "play", "ai_put_white"), &menu_elements->ai_put_white, menu_elements->ai_put_white);
		title.push(menu_e);

		menu.push(title);




		title.init(language.get("display", "display"));

		menu_e.init_button(language.get("display", "hint", "hint"), &menu_elements->dummy);
		side_menu.init_check(language.get("display", "hint", "disc_value"), &menu_elements->use_disc_hint, menu_elements->use_disc_hint);
		menu_e.push(side_menu);
		side_menu.init_bar(language.get("display", "hint", "disc_value_number"), &menu_elements->n_disc_hint, menu_elements->n_disc_hint, 1, SHOW_ALL_HINT);
		menu_e.push(side_menu);
		side_menu.init_check(language.get("display", "hint", "umigame_value"), &menu_elements->use_umigame_value, menu_elements->use_umigame_value);
		menu_e.push(side_menu);
		title.push(menu_e);

		menu_e.init_check(language.get("display", "legal"), &menu_elements->show_legal, menu_elements->show_legal);
		title.push(menu_e);
		menu_e.init_check(language.get("display", "graph"), &menu_elements->show_graph, menu_elements->show_graph);
		title.push(menu_e);
		menu_e.init_check(language.get("display", "opening_on_cell"), &menu_elements->show_opening_on_cell, menu_elements->show_opening_on_cell);
		title.push(menu_e);
		menu_e.init_check(language.get("display", "log"), &menu_elements->show_log, menu_elements->show_log);
		title.push(menu_e);

		menu.push(title);




		title.init(language.get("operation", "operation"));

		menu_e.init_button(language.get("operation", "stop_calculating"), &menu_elements->stop_calculating);
		title.push(menu_e);

		menu_e.init_button(language.get("operation", "forward"), &menu_elements->forward);
		title.push(menu_e);
		menu_e.init_button(language.get("operation", "backward"), &menu_elements->backward);
		title.push(menu_e);

		menu_e.init_button(language.get("operation", "convert", "convert"), &menu_elements->dummy);
		side_menu.init_button(language.get("operation", "convert", "vertical"), &menu_elements->convert_180);
		menu_e.push(side_menu);
		side_menu.init_button(language.get("operation", "convert", "black_line"), &menu_elements->convert_blackline);
		menu_e.push(side_menu);
		side_menu.init_button(language.get("operation", "convert", "white_line"), &menu_elements->convert_whiteline);
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
		title.push(menu_e);

		menu_e.init_button(language.get("in_out", "out"), &menu_elements->dummy);
		side_menu.init_button(language.get("in_out", "output_transcript"), &menu_elements->copy_transcript);
		menu_e.push(side_menu);
		side_menu.init_button(language.get("in_out", "output_game"), &menu_elements->save_game);
		menu_e.push(side_menu);
		title.push(menu_e);

		menu.push(title);




		title.init(language.get("book", "book"));

		menu_e.init_button(language.get("book", "import"), &menu_elements->book_import);
		title.push(menu_e);
		menu_e.init_button(language.get("book", "book_reference"), &menu_elements->book_reference);
		title.push(menu_e);
		menu_e.init_button(language.get("book", "settings"), &menu_elements->dummy);
		side_menu.init_bar(language.get("book", "depth"), &menu_elements->book_learn_depth, menu_elements->book_learn_depth, 0, 60);
		menu_e.push(side_menu);
		side_menu.init_bar(language.get("book", "accept"), &menu_elements->book_learn_error, menu_elements->book_learn_error, 0, 64);
		menu_e.push(side_menu);
		title.push(menu_e);
		menu_e.init_button(language.get("book", "start_learn"), &menu_elements->book_start_learn);
		title.push(menu_e);

		menu.push(title);




		title.init(language.get("help", "help"));
		menu_e.init_button(language.get("help", "how_to_use"), &menu_elements->usage);
		title.push(menu_e);
		menu_e.init_button(language.get("help", "bug_report"), &menu_elements->bug_report);
		title.push(menu_e);
		menu_e.init_check(language.get("help", "auto_update_check"), &menu_elements->auto_update_check, menu_elements->auto_update_check);
		title.push(menu_e);
		menu_e.init_button(language.get("help", "license"), &menu_elements->license);
		title.push(menu_e);
		menu.push(title);





		title.init(U"Language");
		for (int i = 0; i < (int)getData().resources.language_names.size(); ++i) {
			menu_e.init_radio(language_name.get(getData().resources.language_names[i]), &menu_elements->languages[i], menu_elements->languages[i]);
			title.push(menu_e);
		}
		menu.push(title);




		menu.init(0, 0, menu_font, getData().resources.checkbox);
		return menu;
	}

	void draw_legal(uint64_t ignore) {
		Flip flip;
		uint64_t legal = getData().history_elem.board.get_legal();
		for (int cell = 0; cell < HW2; ++cell) {
			int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
			int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
			if (1 & (legal >> (HW2_M1 - cell))) {
				if (HW2_M1 - cell == getData().history_elem.next_policy) {
					if (getData().history_elem.player == WHITE) {
						Circle(x, y, DISC_SIZE).draw(ColorF(getData().colors.white, 0.2));
					}
					else {
						Circle(x, y, DISC_SIZE).draw(ColorF(getData().colors.black, 0.2));
					}
				}
				if ((1 & (ignore >> (HW2_M1 - cell))) == 0)
					Circle(x, y, LEGAL_SIZE).draw(getData().colors.cyan);
			}
		}
	}

	void draw_info() {
		if (getData().history_elem.board.get_legal()) {
			getData().fonts.font20(Format(getData().history_elem.board.n_discs() - 3) + language.get("info", "moves")).draw(INFO_SX, INFO_SY);
			if (getData().history_elem.player == BLACK) {
				getData().fonts.font20(language.get("info", "black")).draw(INFO_SX + 100, INFO_SY);
			}
			else {
				getData().fonts.font20(language.get("info", "white")).draw(INFO_SX + 100, INFO_SY);
			}
		}
		else {
			getData().fonts.font20(language.get("info", "game_end")).draw(INFO_SX, INFO_SY);
		}
		getData().fonts.font15(language.get("info", "opening_name") + U": " + Unicode::FromUTF8(getData().history_elem.opening_name)).draw(INFO_SX, INFO_SY + 30);
		Circle(INFO_SX + INFO_DISC_RADIUS, INFO_SY + 75, INFO_DISC_RADIUS).draw(getData().colors.black);
		Circle(INFO_SX + INFO_DISC_RADIUS, INFO_SY + 110, INFO_DISC_RADIUS).draw(getData().colors.white);
		if (getData().history_elem.player == BLACK) {
			getData().fonts.font20(getData().history_elem.board.count_player()).draw(Arg::leftCenter(INFO_SX + 40, INFO_SY + 75));
			getData().fonts.font20(getData().history_elem.board.count_opponent()).draw(Arg::leftCenter(INFO_SX + 40, INFO_SY + 110));
		}
		else {
			getData().fonts.font20(getData().history_elem.board.count_opponent()).draw(Arg::leftCenter(INFO_SX + 40, INFO_SY + 75));
			getData().fonts.font20(getData().history_elem.board.count_player()).draw(Arg::leftCenter(INFO_SX + 40, INFO_SY + 110));
		}
		getData().fonts.font15(language.get("common", "level") + Format(getData().menu_elements.level)).draw(INFO_SX, INFO_SY + 135);
		int mid_depth, end_depth;
		get_level_depth(getData().menu_elements.level, &mid_depth, &end_depth);
		getData().fonts.font15(language.get("info", "lookahead_0") + Format(mid_depth) + language.get("info", "lookahead_1")).draw(INFO_SX, INFO_SY + 160);
		getData().fonts.font15(language.get("info", "complete_0") + Format(end_depth) + language.get("info", "complete_1")).draw(INFO_SX, INFO_SY + 185);
	}

	uint64_t draw_hint() {
		uint64_t res = 0ULL;
		if (ai_status.hint_available) {
			vector<Hint_info> hint_infos;
			for (int cell = 0; cell < HW2; ++cell) {
				if (ai_status.hint_use_stable[cell]) {
					Hint_info hint_info;
					hint_info.value = ai_status.hint_values[cell];
					hint_info.cell = cell;
					hint_info.type = ai_status.hint_types_stable[cell];
					hint_infos.emplace_back(hint_info);
				}
			}
			sort(hint_infos.begin(), hint_infos.end(), compare_hint_info);
			if (hint_infos.size()) {
				int sgn = getData().history_elem.player == 0 ? 1 : -1;
				int node_idx = getData().graph_resources.node_find(getData().graph_resources.put_mode, getData().graph_resources.n_discs);
				if (node_idx != -1) {
					if (getData().graph_resources.nodes[getData().graph_resources.put_mode][node_idx].level < hint_infos[0].type) {
						getData().graph_resources.nodes[getData().graph_resources.put_mode][node_idx].v = sgn * (int)round(hint_infos[0].value);
						getData().graph_resources.nodes[getData().graph_resources.put_mode][node_idx].level = hint_infos[0].type;
					}
				}
			}
			int n_disc_hint = min((int)hint_infos.size(), getData().menu_elements.n_disc_hint);
			for (int i = 0; i < n_disc_hint; ++i) {
				int sx = BOARD_SX + (hint_infos[i].cell % HW) * BOARD_CELL_SIZE;
				int sy = BOARD_SY + (hint_infos[i].cell / HW) * BOARD_CELL_SIZE;
				Color color = getData().colors.white;
				if (hint_infos[i].value == hint_infos[0].value) {
					color = getData().colors.cyan;
				}
				getData().fonts.font15_bold((int)round(hint_infos[i].value)).draw(sx + 2, sy, color);
				if (hint_infos[i].type == HINT_TYPE_BOOK) {
					getData().fonts.font10(U"book").draw(sx + 2, sy + 16, color);
				}
				else if (hint_infos[i].type > HINT_MAX_LEVEL) {
					getData().fonts.font10(Format(hint_infos[i].type) + U"%").draw(sx + 2, sy + 16, color);
				}
				else {
					getData().fonts.font10(U"Lv." + Format(hint_infos[i].type)).draw(sx + 2, sy + 16, color);
				}
				res |= 1ULL << (HW2_M1 - hint_infos[i].cell);
			}
		}
		return res;
	}

	void draw_opening_on_cell() {
		uint64_t legal = getData().history_elem.board.get_legal();
		for (int cell = 0; cell < HW2; ++cell) {
			int x = HW_M1 - cell % HW;
			int y = HW_M1 - cell / HW;
			Rect cell_rect(BOARD_SX + x * BOARD_CELL_SIZE, BOARD_SY + y * BOARD_CELL_SIZE, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
			if ((1 & (legal >> cell)) && cell_rect.mouseOver()) {
				Flip flip;
				calc_flip(&flip, &getData().history_elem.board, cell);
				string openings = opening_many.get(getData().history_elem.board.move_copy(&flip), getData().history_elem.player);
				if (openings.size()) {
					String opening_name = U" " + Unicode::FromUTF8(openings).replace(U" ", U" \n ");
					Vec2 pos = Cursor::Pos();
					pos.x += 20;
					RectF background_rect = getData().fonts.font15_bold(opening_name).region(pos);
					background_rect.draw(getData().colors.white);
					getData().fonts.font15_bold(opening_name).draw(pos, getData().colors.black);
				}
			}
		}
	}

	void hint_init_calculating() {
		uint64_t legal = getData().history_elem.board.get_legal();
		if (ai_status.hint_level == HINT_NOT_CALCULATING) {
			for (int cell = 0; cell < HW2; ++cell) {
				ai_status.hint_values[cell] = HINT_INIT_VALUE;
				ai_status.hint_use[cell] = (bool)(1 & (legal >> (HW2_M1 - cell)));
			}
		}
		else {
			ai_status.hint_available = true;
		}
		++ai_status.hint_level;
		vector<pair<int, int>> value_cells;
		for (int cell = 0; cell < HW2; ++cell) {
			if (ai_status.hint_use[cell]) {
				value_cells.emplace_back(make_pair(ai_status.hint_values[cell], cell));
			}
		}
		sort(value_cells.begin(), value_cells.end(), compare_value_cell);
		int n_legal = pop_count_ull(legal);
		int hint_adoption_threshold = getData().menu_elements.n_disc_hint + max(1, n_legal * (getData().menu_elements.level - ai_status.hint_level) / getData().menu_elements.level);
		hint_adoption_threshold = min(hint_adoption_threshold, (int)value_cells.size());
		ai_status.hint_task_stack.clear();
		int idx = 0;
		Board board;
		Flip flip;
		for (pair<int, int>& value_cell : value_cells) {
			if (idx++ >= hint_adoption_threshold) {
				break;
			}
			board = getData().history_elem.board;
			calc_flip(&flip, &board, (uint_fast8_t)(HW2_M1 - value_cell.second));
			board.move_board(&flip);
			ai_status.hint_task_stack.emplace_back(make_pair(value_cell.second, bind(ai_hint, board, ai_status.hint_level, getData().menu_elements.use_book, false)));
		}
		ai_status.hint_calculating = true;
	}

	void hint_do_task() {
		bool has_remaining_task = false;
		for (int cell = 0; cell < HW2; ++cell) {
			if (ai_status.hint_future[cell].valid()) {
				if (ai_status.hint_future[cell].wait_for(chrono::seconds(0)) == future_status::ready) {
					Search_result search_result = ai_status.hint_future[cell].get();
					if (ai_status.hint_values[cell] == HINT_INIT_VALUE || search_result.is_end_search || search_result.depth == SEARCH_BOOK) {
						ai_status.hint_values[cell] = -search_result.value;
					}
					else {
						ai_status.hint_values[cell] += -search_result.value;
						ai_status.hint_values[cell] /= 2.0;
					}
					if (search_result.depth == SEARCH_BOOK) {
						ai_status.hint_types[cell] = HINT_TYPE_BOOK;
					}
					else if (search_result.is_end_search) {
						ai_status.hint_types[cell] = search_result.probability;
					}
					else {
						ai_status.hint_types[cell] = ai_status.hint_level;
					}
				}
				else {
					has_remaining_task = true;
				}
			}
		}
		if (!has_remaining_task) {
			int loop_time = 1;
			if (ai_status.hint_level <= 10) {
				loop_time = max(1, getData().menu_elements.n_threads - 1);
			}
			bool task_pushed = false;
			for (int i = 0; i < loop_time; ++i) {
				if (ai_status.hint_task_stack.size()) {
					pair<int, function<Search_result()>> task = ai_status.hint_task_stack.back();
					ai_status.hint_task_stack.pop_back();
					ai_status.hint_future[task.first] = async(launch::async, task.second);
					task_pushed = true;
				}
			}
			if (!task_pushed) {
				for (int cell = 0; cell < HW2; ++cell) {
					ai_status.hint_use_stable[cell] = ai_status.hint_use[cell];
					ai_status.hint_values_stable[cell] = ai_status.hint_values[cell];
					ai_status.hint_types_stable[cell] = ai_status.hint_types[cell];
				}
				ai_status.hint_calculating = false;
			}
		}
	}

	void init_analyze() {
		ai_status.analyze_task_stack.clear();
		int idx = 0;
		for (History_elem& node : getData().graph_resources.nodes[getData().graph_resources.put_mode]) {
			Analyze_info analyze_info;
			analyze_info.idx = idx++;
			analyze_info.sgn = node.player ? -1 : 1;
			analyze_info.board = node.board;
			ai_status.analyze_task_stack.emplace_back(make_pair(analyze_info, bind(ai, node.board, getData().menu_elements.level, getData().menu_elements.use_book, true)));
		}
		cerr << "analyze " << ai_status.analyze_task_stack.size() << " tasks" << endl;
		ai_status.analyzing = true;
		analyze_do_task();
	}

	void analyze_do_task() {
		pair<Analyze_info, function<Search_result()>> task = ai_status.analyze_task_stack.back();
		ai_status.analyze_task_stack.pop_back();
		ai_status.analyze_future[task.first.idx] = async(launch::async, task.second);
		ai_status.analyze_sgn[task.first.idx] = task.first.sgn;
		getData().history_elem.board = task.first.board;
		getData().history_elem.policy = -1;
		getData().history_elem.next_policy = -1;
		getData().history_elem.player = task.first.sgn == 1 ? 0 : 1;
		getData().graph_resources.n_discs = getData().history_elem.board.n_discs();
	}

	void analyze_get_task() {
		if (ai_status.analyze_task_stack.size() == 0) {
			ai_status.analyzing = false;
			getData().history_elem = getData().graph_resources.nodes[getData().graph_resources.put_mode].back();
			getData().graph_resources.n_discs = getData().graph_resources.nodes[getData().graph_resources.put_mode].back().board.n_discs();
			return;
		}
		bool task_finished = false;
		for (int i = 0; i < ANALYZE_SIZE; ++i) {
			if (ai_status.analyze_future[i].valid()) {
				if (ai_status.analyze_future[i].wait_for(chrono::seconds(0)) == future_status::ready) {
					Search_result search_result = ai_status.analyze_future[i].get();
					int value = ai_status.analyze_sgn[i] * search_result.value;
					cerr << i << " " << value << endl;
					getData().graph_resources.nodes[getData().graph_resources.put_mode][i].v = value;
					getData().graph_resources.nodes[getData().graph_resources.put_mode][i].level = getData().menu_elements.level;
					task_finished = true;
				}
			}
		}
		if (task_finished) {
			analyze_do_task();
		}
	}
};

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
		back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		importing = false;
		imported = false;
		failed = false;
	}

	void update() override {
		Scene::SetBackground(getData().colors.green);
		const int icon_width = (LEFT_RIGHT - LEFT_LEFT) / 2;
		getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
		getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
		int sy = 20 + icon_width + 50;
		if (!importing) {
			getData().fonts.font25(language.get("book", "import_explanation")).draw(Arg::topCenter(X_CENTER, sy), getData().colors.white);
			back_button.draw();
			if (back_button.clicked() || KeyEscape.pressed()) {
				changeScene(U"Main_scene", SCENE_FADE_TIME);
			}
			if (DragDrop::HasNewFilePaths()) {
				for (const auto& dropped : DragDrop::GetDroppedFilePaths()) {
					import_book_future = async(launch::async, import_book, dropped.path.narrow());
					importing = true;
				}
			}
		}
		else if (!imported) {
			getData().fonts.font25(language.get("book", "loading")).draw(Arg::topCenter(X_CENTER, sy), getData().colors.white);
			if (import_book_future.wait_for(chrono::seconds(0)) == future_status::ready) {
				failed = import_book_future.get();
				imported = true;
			}
		} else {
			if (failed) {
				getData().fonts.font25(language.get("book", "import_failed")).draw(Arg::topCenter(X_CENTER, sy), getData().colors.white);
				back_button.draw();
				if (back_button.clicked() || KeyEscape.pressed()) {
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
		single_back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		default_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("book", "use_default"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		go_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("book", "import"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		book_file = getData().settings.book_file;
		book_deleting = false;
		book_importing = false;
		failed = false;
		done = false;
	}

	void update() override {
		Scene::SetBackground(getData().colors.green);
		const int icon_width = (LEFT_RIGHT - LEFT_LEFT) / 2;
		getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
		getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
		int sy = 20 + icon_width + 50;
		if (!book_deleting && !book_importing && !failed && !done) {
			getData().fonts.font25(language.get("book", "input_book_path")).draw(Arg::topCenter(X_CENTER, sy), getData().colors.white);
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
			getData().fonts.font15(book_file_str + U'|' + editingText).draw(text_area.stretched(-4), getData().colors.black);
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
				getData().settings.book_file = book_file;
				cerr << "book reference changed to " << book_file << endl;
				delete_book_future = async(launch::async, delete_book);
				book_deleting = true;
			}
		}
		else if (book_deleting || book_importing) {
			getData().fonts.font25(language.get("book", "loading")).draw(Arg::topCenter(X_CENTER, sy), getData().colors.white);
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
				getData().fonts.font25(language.get("book", "import_failed")).draw(Arg::topCenter(X_CENTER, sy), getData().colors.white);
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

vector<History_elem> import_transcript_processing(string transcript, bool *failed) {
	Board h_bd;
	h_bd.reset();
	vector<History_elem> n_history;
	String transcript_str = Unicode::Widen(transcript).replace(U"\r", U"").replace(U"\n", U"").replace(U" ", U"");
	if (transcript_str.size() % 2 != 0 && transcript_str.size() >= 120) {
		*failed = true;
	}
	else {
		int y, x;
		uint64_t legal;
		Flip flip;
		h_bd.reset();
		History_elem history_elem;
		int player = BLACK;
		history_elem.set(h_bd, player, GRAPH_IGNORE_VALUE, -1, -1, -1, "");
		n_history.emplace_back(history_elem);
		for (int i = 0; i < (int)transcript_str.size(); i += 2) {
			x = (int)transcript_str[i] - (int)'a';
			if (x < 0 || HW <= x) {
				x = (int)transcript_str[i] - (int)'A';
				if (x < 0 || HW <= x) {
					*failed = true;
					break;
				}
			}
			y = (int)transcript_str[i + 1] - (int)'1';
			if (y < 0 || HW <= y) {
				*failed = true;
				break;
			}
			y = HW_M1 - y;
			x = HW_M1 - x;
			legal = h_bd.get_legal();
			if (1 & (legal >> (y * HW + x))) {
				calc_flip(&flip, &h_bd, y * HW + x);
				h_bd.move_board(&flip);
				player ^= 1;
				if (h_bd.get_legal() == 0ULL) {
					h_bd.pass();
					player ^= 1;
					if (h_bd.get_legal() == 0ULL) {
						h_bd.pass();
						player ^= 1;
						if (i != transcript_str.size() - 2) {
							*failed = true;
							break;
						}
					}
				}
			}
			else {
				*failed = true;
				break;
			}
			n_history.back().next_policy = y * HW + x;
			history_elem.set(h_bd, player, GRAPH_IGNORE_VALUE, -1, y * HW + x, -1, "");
			n_history.emplace_back(history_elem);
		}
	}
	return n_history;
}

class Import_transcript : public App::Scene {
private:
	Button single_back_button;
	Button back_button;
	Button import_button;
	bool done;
	bool failed;
	string transcript;
	vector<History_elem> n_history;

public:
	Import_transcript(const InitData& init) : IScene{ init } {
		single_back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		import_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "import"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		done = false;
		failed = false;
		transcript.clear();
	}

	void update() override {
		Scene::SetBackground(getData().colors.green);
		const int icon_width = (LEFT_RIGHT - LEFT_LEFT) / 2;
		getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
		getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
		int sy = 20 + icon_width + 50;
		if (!done) {
			getData().fonts.font25(language.get("in_out", "input_transcript")).draw(Arg::topCenter(X_CENTER, sy), getData().colors.white);
			Rect text_area{ X_CENTER - 300, sy + 40, 600, 70 };
			text_area.draw(getData().colors.light_cyan).drawFrame(2, getData().colors.black);
			String str = Unicode::Widen(transcript);
			TextInput::UpdateText(str);
			const String editingText = TextInput::GetEditingText();
			bool return_pressed = false;
			if (KeyControl.pressed() && KeyV.down()) {
				String clip_text;
				Clipboard::GetText(clip_text);
				str += clip_text;
			}
			if (str.size()) {
				if (str[str.size() - 1] == '\n') {
					str.replace(U"\n", U"");
					return_pressed = true;
				}
			}
			transcript = str.narrow();
			getData().fonts.font15(str + U'|' + editingText).draw(text_area.stretched(-4), getData().colors.black);
			back_button.draw();
			import_button.draw();
			if (back_button.clicked() || KeyEscape.pressed()) {
				changeScene(U"Main_scene", SCENE_FADE_TIME);
			}
			if (import_button.clicked() || KeyEnter.pressed()) {
				n_history = import_transcript_processing(transcript, &failed);
				done = true;
			}
		}
		else {
			if (!failed) {
				getData().graph_resources.init();
				getData().graph_resources.nodes[0] = n_history;
				getData().graph_resources.n_discs = getData().graph_resources.nodes[0].back().board.n_discs();
				getData().graph_resources.need_init = false;
				changeScene(U"Main_scene", SCENE_FADE_TIME);
			}
			else {
				getData().fonts.font25(language.get("in_out", "import_failed")).draw(Arg::topCenter(X_CENTER, sy), getData().colors.white);
				single_back_button.draw();
				if (single_back_button.clicked() || KeyEscape.pressed()) {
					changeScene(U"Main_scene", SCENE_FADE_TIME);
				}
			}
		}
	}

	void draw() const override {

	}
};

class Import_board : public App::Scene {
private:
	Button single_back_button;
	Button back_button;
	Button import_button;
	bool done;
	bool failed;
	Board board;
	int player;
	string board_str;

public:
	Import_board(const InitData& init) : IScene{ init } {
		single_back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		import_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "import"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		done = false;
		failed = false;
		board_str.clear();
	}

	void update() override {
		Scene::SetBackground(getData().colors.green);
		const int icon_width = (LEFT_RIGHT - LEFT_LEFT) / 2;
		getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
		getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
		int sy = 20 + icon_width + 50;
		if (!done) {
			getData().fonts.font25(language.get("in_out", "input_board")).draw(Arg::topCenter(X_CENTER, sy), getData().colors.white);
			Rect text_area{ X_CENTER - 300, sy + 40, 600, 70 };
			text_area.draw(getData().colors.light_cyan).drawFrame(2, getData().colors.black);
			String str = Unicode::Widen(board_str);
			TextInput::UpdateText(str);
			const String editingText = TextInput::GetEditingText();
			bool return_pressed = false;
			if (KeyControl.pressed() && KeyV.down()) {
				String clip_text;
				Clipboard::GetText(clip_text);
				str += clip_text;
			}
			if (str.size()) {
				if (str[str.size() - 1] == '\n') {
					str.replace(U"\n", U"");
					return_pressed = true;
				}
			}
			board_str = str.narrow();
			getData().fonts.font15(str + U'|' + editingText).draw(text_area.stretched(-4), getData().colors.black);
			back_button.draw();
			import_button.draw();
			if (back_button.clicked() || KeyEscape.pressed()) {
				changeScene(U"Main_scene", SCENE_FADE_TIME);
			}
			if (import_button.clicked() || KeyEnter.pressed()) {
				failed = import_board_processing();
				done = true;
			}
		}
		else {
			if (!failed) {
				getData().graph_resources.init();
				History_elem history_elem;
				history_elem.reset();
				getData().graph_resources.nodes[0].emplace_back(history_elem);
				history_elem.player = player;
				history_elem.board = board;
				getData().graph_resources.nodes[0].emplace_back(history_elem);
				getData().graph_resources.n_discs = board.n_discs();
				getData().graph_resources.need_init = false;
				changeScene(U"Main_scene", SCENE_FADE_TIME);
			}
			else {
				getData().fonts.font25(language.get("in_out", "import_failed")).draw(Arg::topCenter(X_CENTER, sy), getData().colors.white);
				single_back_button.draw();
				if (single_back_button.clicked() || KeyEscape.pressed()) {
					changeScene(U"Main_scene", SCENE_FADE_TIME);
				}
			}
		}
	}

	void draw() const override {

	}

private:
	bool import_board_processing() {
		String board_str_str = Unicode::Widen(board_str).replace(U"\r", U"").replace(U"\n", U"").replace(U" ", U"");
		bool failed_res = false;
		int bd_arr[HW2];
		Board bd;
		if (board_str_str.size() != HW2 + 1) {
			failed_res = true;
		}
		else {
			for (int i = 0; i < HW2; ++i) {
				if (board_str_str[i] == '0' || board_str_str[i] == 'B' || board_str_str[i] == 'b' || board_str_str[i] == 'X' || board_str_str[i] == 'x' || board_str_str[i] == '*')
					bd_arr[i] = BLACK;
				else if (board_str_str[i] == '1' || board_str_str[i] == 'W' || board_str_str[i] == 'w' || board_str_str[i] == 'O' || board_str_str[i] == 'o')
					bd_arr[i] = WHITE;
				else if (board_str_str[i] == '.' || board_str_str[i] == '-')
					bd_arr[i] = VACANT;
				else {
					failed_res = true;
					break;
				}
			}
			if (board_str_str[HW2] == '0' || board_str_str[HW2] == 'B' || board_str_str[HW2] == 'b' || board_str_str[HW2] == 'X' || board_str_str[HW2] == 'x' || board_str_str[HW2] == '*')
				player = 0;
			else if (board_str_str[HW2] == '1' || board_str_str[HW2] == 'W' || board_str_str[HW2] == 'w' || board_str_str[HW2] == 'O' || board_str_str[HW2] == 'o')
				player = 1;
			else
				failed_res = true;
		}
		if (!failed_res) {
			board.translate_from_arr(bd_arr, player);
		}
		return failed_res;
	}
};

class Edit_board : public App::Scene {
private:
	Button back_button;
	Button set_button;
	Radio_button player_radio;
	Radio_button disc_radio;
	bool done;
	bool failed;
	History_elem history_elem;

public:
	Edit_board(const InitData& init) : IScene{ init } {
		back_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_1_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		set_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("in_out", "import"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		done = false;
		failed = false;
		history_elem = getData().history_elem;
		Radio_button_element radio_button_elem;
		player_radio.init();
		radio_button_elem.init(480, 120, getData().fonts.font15, 20, language.get("common", "black"), true);
		player_radio.push(radio_button_elem);
		radio_button_elem.init(480, 140, getData().fonts.font15, 20, language.get("common", "white"), false);
		player_radio.push(radio_button_elem);
		disc_radio.init();
		radio_button_elem.init(480, 210, getData().fonts.font15, 20, language.get("edit_board", "black"), true);
		disc_radio.push(radio_button_elem);
		radio_button_elem.init(480, 230, getData().fonts.font15, 20, language.get("edit_board", "white"), false);
		disc_radio.push(radio_button_elem);
		radio_button_elem.init(480, 250, getData().fonts.font15, 20, language.get("edit_board", "empty"), false);
		disc_radio.push(radio_button_elem);

	}

	void update() override {
		int board_arr[HW2];
		history_elem.board.translate_to_arr(board_arr, BLACK);
		for (int cell = 0; cell < HW2; ++cell) {
			int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
			int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
			if (board_arr[cell] == BLACK) {
				Circle(x, y, DISC_SIZE).draw(Palette::Black);
			}
			else if (board_arr[cell] == WHITE) {
				Circle(x, y, DISC_SIZE).draw(Palette::White);
			}
		}
		for (int cell = 0; cell < HW2; ++cell) {
			int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE;
			int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE;
			Rect cell_region(x, y, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
			if (cell_region.leftPressed()) {
				board_arr[cell] = disc_radio.checked;
			}
		}
		history_elem.board.translate_from_arr(board_arr, BLACK);
		if (KeyB.pressed()) {
			disc_radio.checked = BLACK;
		}
		else if (KeyW.pressed()) {
			disc_radio.checked = WHITE;
		}
		else if (KeyE.pressed()) {
			disc_radio.checked = VACANT;
		}
		Scene::SetBackground(getData().colors.green);
		getData().fonts.font25(language.get("in_out", "edit_board")).draw(480, 20, getData().colors.white);
		getData().fonts.font20(language.get("in_out", "player")).draw(480, 80, getData().colors.white);
		getData().fonts.font20(language.get("in_out", "color")).draw(480, 170, getData().colors.white);
		draw_board(getData().fonts, getData().colors, history_elem);
		player_radio.draw();
		disc_radio.draw();
		back_button.draw();
		set_button.draw();
		if (back_button.clicked() || KeyEscape.pressed()) {
			changeScene(U"Main_scene", SCENE_FADE_TIME);
		}
		if (set_button.clicked() || KeyEnter.pressed()) {
			if (player_radio.checked != BLACK) {
				history_elem.board.pass();
			}
			history_elem.player = player_radio.checked;
			history_elem.v = GRAPH_IGNORE_VALUE;
			history_elem.level = -1;
			getData().history_elem = history_elem;
			int n_discs = history_elem.board.n_discs();
			int insert_place = (int)getData().graph_resources.nodes[getData().graph_resources.put_mode].size();
			int replace_place = -1;
			for (int i = 0; i < (int)getData().graph_resources.nodes[getData().graph_resources.put_mode].size(); ++i) {
				int node_n_discs = getData().graph_resources.nodes[getData().graph_resources.put_mode][i].board.n_discs();
				if (node_n_discs == n_discs) {
					replace_place = i;
					insert_place = -1;
					break;
				}
				else if (node_n_discs > n_discs) {
					insert_place = i;
					break;
				}
			}
			if (replace_place != -1) {
				cerr << "replace" << endl;
				getData().graph_resources.nodes[getData().graph_resources.put_mode][replace_place] = history_elem;
			}
			else {
				cerr << "insert" << endl;
				getData().graph_resources.nodes[getData().graph_resources.put_mode].insert(getData().graph_resources.nodes[getData().graph_resources.put_mode].begin() + insert_place, history_elem);
			}
			getData().graph_resources.need_init = false;
			getData().graph_resources.n_discs = n_discs;
			changeScene(U"Main_scene", SCENE_FADE_TIME);
		}
	}

	void draw() const override {

	}
};

class Import_game : public App::Scene {
private:
	vector<Game_abstract> game_abstracts;
	vector<Button> buttons;
	Button back_button;
	int strt_idx;
	int n_games;

public:
	Import_game(const InitData& init) : IScene{ init } {
		strt_idx = 0;
		back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		n_games = 0;
	}

	void update() override {
		getData().fonts.font25(language.get("in_out", "input_game")).draw(Arg::topCenter(X_CENTER, 10), getData().colors.white);

		back_button.draw();
		if (back_button.clicked() || KeyEscape.pressed()) {
			changeScene(U"Main_scene", SCENE_FADE_TIME);
		}
		for (int i = 0; i < n_games; ++i) {
			if (buttons[i].clicked()) {
				string transcript = game_abstracts[i].transcript.narrow();
			}
		}
	}

	void draw() const override {

	}
};

void Main() {
	Size window_size = Size(WINDOW_SIZE_X, WINDOW_SIZE_Y);
	Window::Resize(window_size);
	Window::SetStyle(WindowStyle::Sizable);
	Scene::SetResizeMode(ResizeMode::Keep);
	Window::SetTitle(U"Egaroucid {}"_fmt(EGAROUCID_VERSION));
	System::SetTerminationTriggers(UserAction::NoAction);
	Console.open();
	
	App scene_manager;
	scene_manager.add <Silent_load> (U"Silent_load");
	scene_manager.add <Load>(U"Load");
	scene_manager.add <Main_scene>(U"Main_scene");
	scene_manager.add <Import_book>(U"Import_book");
	scene_manager.add <Refer_book>(U"Refer_book");
	scene_manager.add <Import_transcript>(U"Import_transcript");
	scene_manager.add <Import_board>(U"Import_board");
	scene_manager.add <Edit_board>(U"Edit_board");
	scene_manager.add <Import_game>(U"Import_game");
	scene_manager.setFadeColor(Palette::Black);
	scene_manager.init(U"Silent_load");

	while (System::Update()) {
		scene_manager.update();
	}
}
