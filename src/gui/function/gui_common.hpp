/*
    Egaroucid Project

    @file gui_common.hpp
        Common things about GUI
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include "./../../engine/engine_all.hpp"
#include "menu.hpp"
#include "version.hpp"
#include "url.hpp"

// graph definition
#define GRAPH_IGNORE_VALUE INF

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
#define ERR_HASH_NOT_RESIZED 3
#define ERR_IMPORT_SETTINGS 1

// constant definition
#define SHOW_ALL_HINT 35
#define UPDATE_CHECK_ALREADY_UPDATED 0
#define UPDATE_CHECK_UPDATE_FOUND 1

// board drawing constants
#define BOARD_SIZE 400
#define BOARD_COORD_SIZE 20
#define DISC_SIZE 20
#define LEGAL_SIZE 7
#define STABLE_SIZE 4
#define BOARD_CELL_FRAME_WIDTH 2
#define BOARD_DOT_SIZE 5
#define BOARD_ROUND_FRAME_WIDTH 10
#define BOARD_ROUND_DIAMETER 20
#define BOARD_SY 60
constexpr int BOARD_SX = LEFT_LEFT + BOARD_COORD_SIZE;
constexpr int BOARD_CELL_SIZE = BOARD_SIZE / HW;

// main start game button constants
#define START_GAME_BUTTON_SX 700
#define START_GAME_BUTTON_SY 45
#define START_GAME_BUTTON_WIDTH 80
#define START_GAME_BUTTON_HEIGHT 30
#define START_GAME_BUTTON_RADIUS 10

// graph drawing constants
#define GRAPH_RESOLUTION 8
constexpr int GRAPH_SX = BOARD_SX + BOARD_SIZE + 60;
constexpr int GRAPH_SY = Y_CENTER + 20;
constexpr int GRAPH_WIDTH = WINDOW_SIZE_X - GRAPH_SX - 35;
constexpr int GRAPH_HEIGHT = WINDOW_SIZE_Y - GRAPH_SY - 60;

// level drawing constants
#define LEVEL_DEPTH_DY -15
#define LEVEL_INFO_DY -60
#define LEVEL_INFO_WIDTH 40
#define LEVEL_INFO_HEIGHT 20
#define LEVEL_PROB_WIDTH 90

// level graph roundrect constants
#define GRAPH_RECT_DY -80
#define GRAPH_RECT_DX -40
#define GRAPH_RECT_RADIUS 20
#define GRAPH_RECT_THICKNESS 5
constexpr int GRAPH_RECT_WIDTH = WINDOW_SIZE_X - (GRAPH_SX + GRAPH_RECT_DX) - 10;
constexpr int GRAPH_RECT_HEIGHT = GRAPH_HEIGHT - GRAPH_RECT_DY + 27;
/*
#define LEVEL_INFO_WIDTH 40
#define LEVEL_HEIGHT 15
#define LEVEL_INFO_HEIGHT 20
constexpr int LEVEL_SX = BOARD_SX + BOARD_SIZE + 50;
constexpr int LEVEL_SY = Y_CENTER - 32;
constexpr int LEVEL_WIDTH = WINDOW_SIZE_X - LEVEL_SX - 30;
constexpr int LEVEL_INFO_SX = LEVEL_SX + LEVEL_WIDTH - LEVEL_INFO_WIDTH * 5;
constexpr int LEVEL_INFO_SY = LEVEL_SY - 55;
constexpr int LEVEL_INFO_RECT_SY = LEVEL_SY - 45;
constexpr int LEVEL_DEPTH_SY = LEVEL_SY + LEVEL_HEIGHT + 8;
*/

// info drawing constants
#define INFO_SY 35
#define INFO_DISC_RADIUS 12
#define INFO_SX 460
#define INFO_RECT_RADIUS 20
#define INFO_RECT_THICKNESS 5
constexpr int INFO_WIDTH = WINDOW_SIZE_X - 10 - INFO_SX;
constexpr int INFO_HEIGHT = 190 - INFO_SY - 12;

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

// export game constants
#define EXPORT_GAME_PLAYER_WIDTH 300
#define EXPORT_GAME_PLAYER_HEIGHT 30
#define EXPORT_GAME_MEMO_WIDTH 600
#define EXPORT_GAME_MEMO_HEIGHT 250
#define EXPORT_GAME_RADIUS 15

// import game constants
#define IMPORT_GAME_N_GAMES_ON_WINDOW 7
#define IMPORT_GAME_SX 30
#define IMPORT_GAME_SY 65
#define IMPORT_GAME_HEIGHT 45
#define IMPORT_GAME_PLAYER_WIDTH 220
#define IMPORT_GAME_PLAYER_HEIGHT 25
#define IMPORT_GAME_SCORE_WIDTH 60
#define IMPORT_GAME_WINNER_BLACK 0
#define IMPORT_GAME_WINNER_WHITE 1
#define IMPORT_GAME_WINNER_DRAW 2
#define IMPORT_GAME_BUTTON_SX 660
#define IMPORT_GAME_BUTTON_WIDTH 100
#define IMPORT_GAME_BUTTON_HEIGHT 25
#define IMPORT_GAME_BUTTON_RADIUS 7
#define IMPORT_GAME_DATE_WIDTH 120
constexpr int IMPORT_GAME_BUTTON_SY = (IMPORT_GAME_HEIGHT - IMPORT_GAME_BUTTON_HEIGHT) / 2;
constexpr int IMPORT_GAME_WIDTH = WINDOW_SIZE_X - IMPORT_GAME_SX * 2;

// game saving constants
#define GAME_DATE U"date"
#define GAME_BLACK_PLAYER U"black_player"
#define GAME_WHITE_PLAYER U"white_player"
#define GAME_BLACK_DISCS U"black_discs"
#define GAME_WHITE_DISCS U"white_discs"
#define GAME_MEMO U"memo"
#define GAME_BOARD_PLAYER U"board_player"
#define GAME_BOARD_OPPONENT U"board_opponent"
#define GAME_PLAYER U"player"
#define GAME_VALUE U"value"
#define GAME_LEVEL U"level"
#define GAME_POLICY U"policy"
#define GAME_NEXT_POLICY U"next_policy"
#define GAME_DISCS_UNDEFINED -1
#define GAME_MEMO_SUMMARY_SIZE 40

// book modification
#define BOOK_CHANGE_NO_CELL 64
#define CHANGE_BOOK_ERR -1000
#define CHANGE_BOOK_INFO_SX 660
#define CHANGE_BOOK_INFO_SY 40

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

// 2 button vertical constants
#define BUTTON2_VERTICAL_WIDTH 200
#define BUTTON2_VERTICAL_HEIGHT 50
#define BUTTON2_VERTICAL_1_SY 350
#define BUTTON2_VERTICAL_2_SY 420
#define BUTTON2_VERTICAL_SX 520
#define BUTTON2_VERTICAL_RADIUS 20

// font constant
#define FONT_DEFAULT_SIZE 50

struct History_elem {
    Board board;
    int player;
    int v;
    int level;
    int policy;
    int next_policy;
    std::string opening_name;

    History_elem() {
        reset();
    }

    void reset() {
        board.reset();
        player = 0;
        v = GRAPH_IGNORE_VALUE;
        policy = -1;
        next_policy = -1;
        level = -1;
        opening_name.clear();
    }

    void set(Board b, int p, int vv, int l, int pl, int npl, std::string o) {
        board = b;
        player = p;
        v = vv;
        level = l;
        policy = pl;
        next_policy = npl;
        opening_name = o;
    }
};

struct Colors {
    Color green{ Color(36, 153, 114) };
    Color black{ Palette::Black };
    Color white{ Palette::White };
    Color dark_gray{ Color(51, 51, 51) };
    Color cyan{ Color(70, 250, 255) };
    Color red{ Palette::Red };
    Color light_cyan{ Palette::Lightcyan };
    Color chocolate{ Color(210, 105, 30) };
    Color darkred{ Color(178, 34, 34) };
    Color darkblue{ Color(34, 34, 178) };
    Color burlywood{ Color(222, 184, 135) };
};

struct Directories {
    std::string document_dir;
    std::string appdata_dir;
    std::string eval_file;
};

struct Resources {
    std::vector<std::string> language_names;
    Texture icon;
    Texture logo;
    Texture checkbox;
    Texture unchecked;
};

struct Settings {
    int n_threads;
    bool auto_update_check;
    std::string lang_name;
    std::string book_file;
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
    bool show_stable_discs;
    bool change_book_by_right_click;
    bool show_last_move;
    bool show_next_move;
    int hash_level;
};

struct Fonts {
    Font font{ FontMethod::MSDF, FONT_DEFAULT_SIZE };
    Font font_bold{ FontMethod::MSDF, FONT_DEFAULT_SIZE, Typeface::Bold };
    Font font_heavy{ FontMethod::MSDF, FONT_DEFAULT_SIZE, Typeface::Heavy };
};

struct Menu_elements {
    bool dummy;

    // game
    bool start_game;
    bool analyze;

    // settings
    // AI settings
    bool use_book;
    int level;
    int n_threads;
    int hash_level;
    // player
    bool ai_put_black;
    bool ai_put_white;

    // display
    bool use_disc_hint;
    int n_disc_hint;
    bool use_umigame_value;
    bool show_legal;
    bool show_graph;
    bool show_opening_on_cell;
    bool show_stable_discs;
    bool show_log;
    bool show_last_move;
    bool show_next_move;

    // book
    bool book_start_widen;
	bool book_start_deepen;
    int book_learn_depth;
    int book_learn_error;
    bool book_import;
    bool book_reference;
    bool change_book_by_right_click;

    // input / output
    // input
    bool input_transcript;
    bool input_board;
    bool edit_board;
    bool input_game;
    // output
    bool copy_transcript;
    bool save_game;
    bool screen_shot;

    // manipulation
    bool stop_calculating;
    bool forward;
    bool backward;
    // conversion
    bool convert_180;
    bool convert_blackline;
    bool convert_whiteline;
	bool cache_clear;

    // help
    bool website;
    bool bug_report;
    bool auto_update_check;
    bool license_egaroucid;
    bool license_siv3d;

    // language
    bool languages[200];

    void init(Settings* settings, Resources* resources) {
        dummy = false;

        start_game = false;
        analyze = false;

        use_book = settings->use_book;
        level = settings->level;
        n_threads = settings->n_threads;
        hash_level = settings->hash_level;
        ai_put_black = settings->ai_put_black;
        ai_put_white = settings->ai_put_white;

        use_disc_hint = settings->use_disc_hint;
        n_disc_hint = settings->n_disc_hint;
        use_umigame_value = settings->use_umigame_value;
        show_legal = settings->show_legal;
        show_graph = settings->show_graph;
        show_opening_on_cell = settings->show_opening_on_cell;
        show_stable_discs = settings->show_stable_discs;
        show_log = settings->show_log;
        show_last_move = settings->show_last_move;
        show_next_move = settings->show_next_move;

        book_start_widen = false;
		book_start_deepen = false;
        book_learn_depth = settings->book_learn_depth;
        book_learn_error = settings->book_learn_error;
        book_import = false;
        book_reference = false;
        change_book_by_right_click = settings->change_book_by_right_click;

        input_transcript = false;
        input_board = false;
        edit_board = false;
        input_game = false;
        copy_transcript = false;
        save_game = false;
        screen_shot = false;

        stop_calculating = false;
        forward = false;
        backward = false;
        convert_180 = false;
        convert_blackline = false;
        convert_whiteline = false;
		cache_clear = false;

        website = false;
        bug_report = false;
        auto_update_check = settings->auto_update_check;
        license_egaroucid = false;
        license_siv3d = false;

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
    std::vector<History_elem> nodes[2];
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

struct Game_information {
    String black_player_name;
    String white_player_name;
    String memo;

    void init() {
        black_player_name.clear();
        white_player_name.clear();
        memo.clear();
    }
};

struct Book_information {
    bool changed{ false };
    uint_fast8_t changing{ BOOK_CHANGE_NO_CELL };
    String val_str;
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
    Game_information game_information;
    Book_information book_information;
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
    std::future<Search_result> ai_future;

    bool hint_calculating{ false };
    int hint_level{ HINT_NOT_CALCULATING };
    std::future<Search_result> hint_future[HW2];
    std::vector<std::pair<int, std::function<Search_result()>>> hint_task_stack;
    bool hint_use[HW2];
    double hint_values[HW2];
    int hint_types[HW2];
    bool hint_available{ false };
    bool hint_use_stable[HW2];
    double hint_values_stable[HW2];
    int hint_types_stable[HW2];
    bool hint_use_multi_thread;
    int hint_n_doing_tasks;

    bool analyzing{ false };
    std::future<Search_result> analyze_future[ANALYZE_SIZE];
    int analyze_sgn[ANALYZE_SIZE];
    std::vector<std::pair<Analyze_info, std::function<Search_result()>>> analyze_task_stack;

    bool book_learning{ false };
};

struct Game_abstract {
    String black_player;
    String white_player;
    int black_score;
    int white_score;
    String memo;
    String date;
};

struct Umigame_status {
    bool umigame_calculating{ false };
    bool umigame_calculated{ false };
    std::future<Umigame_result> umigame_future[HW2];
    Umigame_result umigame[HW2];
};

using App = SceneManager<String, Common_resources>;