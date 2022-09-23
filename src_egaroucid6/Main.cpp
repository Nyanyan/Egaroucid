#include <iostream>
#include <future>
#include "ai.hpp"
#include "gui/language.hpp"
#include "gui/menu.hpp"
#include "gui/gui_common.hpp"
#include "gui/graph.hpp"
#include "gui/opening.hpp"
#include <Siv3D.hpp> // OpenSiv3D v0.6.3

using namespace std;

// version definition
#define EGAROUCID_VERSION U"6.0.0"

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
constexpr int GRAPH_SX = BOARD_SX + BOARD_SIZE + 40;
constexpr int GRAPH_SY = Y_CENTER + 30;
constexpr int GRAPH_WIDTH = WINDOW_SIZE_X - GRAPH_SX - 20;
constexpr int GRAPH_HEIGHT = WINDOW_SIZE_Y - GRAPH_SY - 20;

// info drawing constants
#define INFO_SY 35
#define INFO_DISC_RADIUS 12
constexpr int INFO_SX = BOARD_SX + BOARD_SIZE + 25;

// button press constants
#define BUTTON_NOT_PUSHED 0
#define BUTTON_LONG_PRESS_THRESHOLD 500


struct Colors {
	Color green{ Color(36, 153, 114, 100) };
	Color black{ Palette::Black };
	Color white{ Palette::White };
	Color dark_gray{ Color(51, 51, 51) };
	Color cyan{ Palette::Cyan };
	Color red{ Palette::Red };
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

struct Common_resources {
	Colors colors;
	Directories directories;
	Resources resources;
	Settings settings;
	Fonts fonts;
	Menu_elements menu_elements;
	Menu menu;
	History_elem history_elem;
};

struct Graph_resources {
	vector<History_elem> nodes[2];
	int n_discs;
	int put_mode;

	void init() {
		nodes[0].clear();
		nodes[1].clear();
		n_discs = 4;
		put_mode = 0;
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

struct Move_board_button_status {
	uint64_t left_pushed{ BUTTON_NOT_PUSHED };
	uint64_t right_pushed{ BUTTON_NOT_PUSHED };
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
				changeScene(U"Load", 0.0);
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
		Scene::SetBackground(getData().colors.green);
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
		getData().resources.icon.scaled((double)(LEFT_RIGHT - LEFT_LEFT) / getData().resources.icon.width()).draw(LEFT_LEFT, Y_CENTER - (LEFT_RIGHT - LEFT_LEFT) / 2);
		getData().resources.logo.scaled((double)(LEFT_RIGHT - LEFT_LEFT) * 0.8 / getData().resources.logo.width()).draw(RIGHT_LEFT, Y_CENTER - 40);
		if (load_future.wait_for(chrono::seconds(0)) == future_status::ready) {
			int load_code = load_future.get();
			if (load_code == ERR_OK) {
				cerr << "loaded" << endl;
				changeScene(U"Main_scene", 0.5);
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
			getData().fonts.font20(tips).draw(RIGHT_LEFT, Y_CENTER + 140, getData().colors.white);
		}
	}

	void draw() const override {

	}
};

class Main_scene : public App::Scene {
private:
	Graph graph;
	Graph_resources graph_resources;
	Move_board_button_status move_board_button_status;
public:
	Main_scene(const InitData& init) : IScene{ init } {
		cerr << "main scene loading" << endl;
		getData().menu_elements.init(&getData().settings, &getData().resources);
		getData().menu = create_menu(&getData().menu_elements);
		graph.sx = GRAPH_SX;
		graph.sy = GRAPH_SY;
		graph.size_x = GRAPH_WIDTH;
		graph.size_y = GRAPH_HEIGHT;
		graph.resolution = GRAPH_RESOLUTION;
		graph.font = getData().fonts.font15;
		graph.font_size = 15;
		graph_resources.init();
		graph_resources.nodes[graph_resources.put_mode].emplace_back(getData().history_elem);
		cerr << "main scene loaded" << endl;
	}

	void update() override {
		Scene::SetBackground(getData().colors.green);

		// opening
		update_opening();

		// transcript move
		menu_manipulate();
		if (!getData().menu.active()) {
			interact_graph();
		}
		update_n_discs();

		// move
		if (!getData().menu.active()) {
			interact_move();
		}
		ai_move();

		draw_board();
		if (getData().menu_elements.show_graph) {
			graph.draw(graph_resources.nodes[0], graph_resources.nodes[1], graph_resources.n_discs);
		}
		draw_info();
		if (getData().menu_elements.show_opening_on_cell) {
			draw_opening_on_cell();
		}
		
		getData().menu.draw();
	}

	void draw() const override {

	}

private:
	void menu_manipulate() {
		if (getData().menu_elements.backward) {
			--graph_resources.n_discs;
		}
		if (getData().menu_elements.forward) {
			++graph_resources.n_discs;
		}
	}

	void interact_graph() {
		graph_resources.n_discs = graph.update_n_discs(graph_resources.nodes[0], graph_resources.nodes[1], graph_resources.n_discs);
		if (!KeyLeft.pressed() && !KeyA.pressed()) {
			move_board_button_status.left_pushed = BUTTON_NOT_PUSHED;
		}
		if (!KeyRight.pressed() && !KeyD.pressed()) {
			move_board_button_status.right_pushed = BUTTON_NOT_PUSHED;
		}

		if (MouseX1.down() || KeyLeft.down() || KeyA.down() || (move_board_button_status.left_pushed != BUTTON_NOT_PUSHED && tim() - move_board_button_status.left_pushed >= BUTTON_LONG_PRESS_THRESHOLD)) {
			--graph_resources.n_discs;
			if (KeyLeft.down() || KeyA.down()) {
				move_board_button_status.left_pushed = tim();
			}
		}
		else if (MouseX2.down() || KeyRight.down() || KeyD.down() || (move_board_button_status.right_pushed != BUTTON_NOT_PUSHED && tim() - move_board_button_status.right_pushed >= BUTTON_LONG_PRESS_THRESHOLD)) {
			++graph_resources.n_discs;
			if (KeyRight.down() || KeyD.down()) {
				move_board_button_status.right_pushed = tim();
			}
		}
	}

	void update_n_discs() {
		int max_n_discs = graph_resources.nodes[0][graph_resources.nodes[0].size() - 1].board.n_discs();
		if (graph_resources.nodes[1].size()) {
			max_n_discs = max(max_n_discs, graph_resources.nodes[1][graph_resources.nodes[1].size() - 1].board.n_discs());
		}
		graph_resources.n_discs = min(graph_resources.n_discs, max_n_discs);
		int min_n_discs = graph_resources.nodes[0][0].board.n_discs();
		if (graph_resources.nodes[1].size()) {
			min_n_discs = min(min_n_discs, graph_resources.nodes[1][0].board.n_discs());
		}
		graph_resources.n_discs = max(graph_resources.n_discs, min_n_discs);

		if (graph_resources.put_mode == 0 && graph_resources.n_discs != graph_resources.nodes[0][graph_resources.nodes[0].size() - 1].board.n_discs()) {
			graph_resources.put_mode = 1;
		}
		else if (graph_resources.put_mode == 1 && graph_resources.n_discs < graph_resources.nodes[1][0].board.n_discs()) {
			graph_resources.put_mode = 0;
			graph_resources.nodes[1].clear();
		}
		else if (graph_resources.put_mode == 1 && graph_resources.nodes[1].size() == 1 && getData().history_elem.board.n_discs() == graph_resources.nodes[0][graph_resources.nodes[0].size() - 1].board.n_discs()) {
			graph_resources.put_mode = 0;
			graph_resources.nodes[1].clear();
		}
		int node_idx = graph_resources.node_find(graph_resources.put_mode, graph_resources.n_discs);
		if (node_idx == -1 && graph_resources.put_mode == 1) {
			graph_resources.nodes[1].clear();
			int node_idx_0 = graph_resources.node_find(0, graph_resources.n_discs);
			if (node_idx_0 == -1) {
				cerr << "history vector element not found" << endl;
				return;
			}
			graph_resources.nodes[1].emplace_back(graph_resources.nodes[0][node_idx_0]);
			node_idx = graph_resources.node_find(graph_resources.put_mode, graph_resources.n_discs);
		}
		if (node_idx == -1 && graph_resources.put_mode == 0) {
			cerr << "history vector element not found" << endl;
			return;
		}
		getData().history_elem = graph_resources.nodes[graph_resources.put_mode][node_idx];
	}

	void move_processing(int_fast8_t cell) {
		int parent_idx = graph_resources.node_find(graph_resources.put_mode, getData().history_elem.board.n_discs());
		if (parent_idx == -1) {
			cerr << "history vector element not found" << endl;
			return;
		}
		graph_resources.nodes[graph_resources.put_mode][parent_idx].next_policy = HW2_M1 - cell;
		if (parent_idx + 1 < (int)graph_resources.nodes[graph_resources.put_mode].size()) {
			for (int i = parent_idx + 1; i < (int)graph_resources.nodes[graph_resources.put_mode].size(); ++i) {
				graph_resources.nodes[graph_resources.put_mode].pop_back();
			}
		}
		Flip flip;
		calc_flip(&flip, &getData().history_elem.board, HW2_M1 - cell);
		getData().history_elem.board.move_board(&flip);
		getData().history_elem.policy = HW2_M1 - cell;
		getData().history_elem.next_policy = -1;
		getData().history_elem.v = GRAPH_IGNORE_VALUE;
		getData().history_elem.player ^= 1;
		if (getData().history_elem.board.get_legal() == 0ULL) {
			getData().history_elem.board.pass();
			getData().history_elem.player ^= 1;
		}
		graph_resources.nodes[graph_resources.put_mode].emplace_back(getData().history_elem);
		graph_resources.n_discs++;
	}

	void interact_move() {
		if ((getData().history_elem.player == BLACK && getData().menu_elements.ai_put_black) || (getData().history_elem.player == WHITE && getData().menu_elements.ai_put_white)) {
			return;
		}
		uint64_t legal = getData().history_elem.board.get_legal();
		for (int_fast8_t cell = 0; cell < HW2; ++cell) {
			if (1 & (legal >> (HW2_M1 - cell))) {
				int x = cell % HW;
				int y = cell / HW;
				Rect cell_rect(BOARD_SX + x * BOARD_CELL_SIZE, BOARD_SY + y * BOARD_CELL_SIZE, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
				if (cell_rect.leftClicked()) {
					move_processing(cell);
				}
			}
		}
	}

	void ai_move() {
		if (graph_resources.put_mode == 0) {
			if ((getData().history_elem.player == BLACK && getData().menu_elements.ai_put_black) || (getData().history_elem.player == WHITE && getData().menu_elements.ai_put_white)) {
				uint64_t legal = getData().history_elem.board.get_legal();
				if (legal) {
					Search_result search_result = ai(getData().history_elem.board, getData().menu_elements.level, getData().menu_elements.use_book, true);
					if (1 & (legal >> search_result.policy)) {
						move_processing(HW2_M1 - search_result.policy);
					}
				}
			}
		}
	}

	void update_opening() {
		string new_opening = opening.get(getData().history_elem.board, getData().history_elem.player ^ 1);
		if (new_opening.size()) {
			getData().history_elem.opening_name = new_opening;
			int node_idx = graph_resources.node_find(graph_resources.put_mode, graph_resources.n_discs);
			if (node_idx == -1) {
				cerr << "history vector element not found" << endl;
				return;
			}
			graph_resources.nodes[graph_resources.put_mode][node_idx].opening_name = new_opening;
		}
	}

	Menu create_menu(Menu_elements* menu_elements) {
		Menu menu;
		menu_title title;
		menu_elem menu_e, side_menu, side_side_menu;
		Font menu_font = getData().fonts.font15;



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

		menu_e.init_check(language.get("display", "graph"), &menu_elements->show_graph, menu_elements->show_graph);
		title.push(menu_e);
		menu_e.init_check(language.get("display", "opening_on_cell"), &menu_elements->show_opening_on_cell, menu_elements->show_opening_on_cell);
		title.push(menu_e);
		menu_e.init_check(language.get("display", "log"), &menu_elements->show_log, menu_elements->show_log);
		title.push(menu_e);

		menu.push(title);




		title.init(language.get("book", "book"));

		menu_e.init_button(language.get("book", "start_learn"), &menu_elements->book_start_learn);
		title.push(menu_e);
		menu_e.init_button(language.get("book", "settings"), &menu_elements->dummy);
		side_menu.init_bar(language.get("book", "depth"), &menu_elements->book_learn_depth, menu_elements->book_learn_depth, 0, 60);
		menu_e.push(side_menu);
		side_menu.init_bar(language.get("book", "accept"), &menu_elements->book_learn_error, menu_elements->book_learn_error, 0, 64);
		menu_e.push(side_menu);
		title.push(menu_e);
		menu_e.init_button(language.get("book", "import"), &menu_elements->book_import);
		title.push(menu_e);
		menu_e.init_button(language.get("book", "book_reference"), &menu_elements->book_reference);
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

	void draw_board() {
		String coord_x = U"abcdefgh";
		for (int i = 0; i < HW; ++i) {
			getData().fonts.font15_bold(i + 1).draw(Arg::center(BOARD_SX - BOARD_COORD_SIZE, BOARD_SY + BOARD_CELL_SIZE * i + BOARD_CELL_SIZE / 2), getData().colors.dark_gray);
			getData().fonts.font15_bold(coord_x[i]).draw(Arg::center(BOARD_SX + BOARD_CELL_SIZE * i + BOARD_CELL_SIZE / 2, BOARD_SY - BOARD_COORD_SIZE - 2), getData().colors.dark_gray);
		}
		for (int i = 0; i < HW_M1; ++i) {
			Line(BOARD_SX + BOARD_CELL_SIZE * (i + 1), BOARD_SY, BOARD_SX + BOARD_CELL_SIZE * (i + 1), BOARD_SY + BOARD_CELL_SIZE * HW).draw(BOARD_CELL_FRAME_WIDTH, getData().colors.dark_gray);
			Line(BOARD_SX, BOARD_SY + BOARD_CELL_SIZE * (i + 1), BOARD_SX + BOARD_CELL_SIZE * HW, BOARD_SY + BOARD_CELL_SIZE * (i + 1)).draw(BOARD_CELL_FRAME_WIDTH, getData().colors.dark_gray);
		}
		Circle(BOARD_SX + 2 * BOARD_CELL_SIZE, BOARD_SY + 2 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(getData().colors.dark_gray);
		Circle(BOARD_SX + 2 * BOARD_CELL_SIZE, BOARD_SY + 6 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(getData().colors.dark_gray);
		Circle(BOARD_SX + 6 * BOARD_CELL_SIZE, BOARD_SY + 2 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(getData().colors.dark_gray);
		Circle(BOARD_SX + 6 * BOARD_CELL_SIZE, BOARD_SY + 6 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(getData().colors.dark_gray);
		RoundRect(BOARD_SX, BOARD_SY, BOARD_CELL_SIZE * HW, BOARD_CELL_SIZE * HW, BOARD_ROUND_DIAMETER).drawFrame(0, BOARD_ROUND_FRAME_WIDTH, getData().colors.white);
		Flip flip;
		uint64_t legal = getData().history_elem.board.get_legal();
		int board_arr[HW2];
		getData().history_elem.board.translate_to_arr(board_arr, getData().history_elem.player);
		for (int cell = 0; cell < HW2; ++cell) {
			int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
			int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
			if (board_arr[cell] == BLACK) {
				Circle(x, y, DISC_SIZE).draw(getData().colors.black);
			}
			else if (board_arr[cell] == WHITE) {
				Circle(x, y, DISC_SIZE).draw(getData().colors.white);
			}
			if (1 & (legal >> (HW2_M1 - cell))) {
				if (HW2_M1 - cell == getData().history_elem.next_policy) {
					if (getData().history_elem.player == WHITE) {
						Circle(x, y, DISC_SIZE).draw(ColorF(getData().colors.white, 0.2));
					}
					else {
						Circle(x, y, DISC_SIZE).draw(ColorF(getData().colors.black, 0.2));
					}
				}
				if (getData().menu_elements.show_legal && !getData().menu_elements.use_disc_hint) {
					Circle(x, y, LEGAL_SIZE).draw(getData().colors.cyan);
				}
			}
		}
		if (getData().history_elem.policy != -1) {
			int x = BOARD_SX + (HW_M1 - getData().history_elem.policy % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
			int y = BOARD_SY + (HW_M1 - getData().history_elem.policy / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
			Circle(x, y, LEGAL_SIZE).draw(getData().colors.red);
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
	scene_manager.setFadeColor(Color(36, 153, 114, 100));
	scene_manager.init(U"Silent_load");

	while (System::Update()) {
		scene_manager.update();
	}
}
