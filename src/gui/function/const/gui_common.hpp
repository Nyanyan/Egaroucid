/*
    Egaroucid Project

    @file gui_common.hpp
        Common things about GUI
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <Siv3D.hpp>
#include "./../../../engine/engine_all.hpp"
#include "./../menu.hpp"
#include "info.hpp"
#include "url.hpp"

// graph definition
constexpr int GRAPH_IGNORE_VALUE = INF;

// scene definition
constexpr int SCENE_FADE_TIME = 100;

// coordinate definition
constexpr int WINDOW_SIZE_X = 800;
constexpr int WINDOW_SIZE_Y = 500;
constexpr int WINDOW_SIZE_X_MIN = 8;
constexpr int WINDOW_SIZE_Y_MIN = 5;
constexpr int PADDING = 20;
constexpr int LEFT_LEFT = PADDING;
constexpr int LEFT_RIGHT = WINDOW_SIZE_X / 2 - PADDING;
constexpr int LEFT_CENTER = (LEFT_LEFT + LEFT_RIGHT) / 2;
constexpr int RIGHT_LEFT = WINDOW_SIZE_X / 2 + PADDING;
constexpr int RIGHT_RIGHT = WINDOW_SIZE_X - PADDING;
constexpr int RIGHT_CENTER = (RIGHT_LEFT + RIGHT_RIGHT) / 2;
constexpr int X_CENTER = WINDOW_SIZE_X / 2;
constexpr int Y_CENTER = WINDOW_SIZE_Y / 2;

// icon width
constexpr int SCENE_ICON_WIDTH = 120;

// error definition
constexpr int ERR_OK = 0;
constexpr int ERR_IMPORT_SETTINGS = 1;
// silent load
constexpr int ERR_SILENT_LOAD_TERMINATED = 100;
constexpr int ERR_SILENT_LOAD_LANG_LIST_NOT_LOADED = 101;
constexpr int ERR_SILENT_LOAD_LANG_JSON_NOT_LOADED = 102;
constexpr int ERR_SILENT_LOAD_LANG_NOT_LOADED = 103;
constexpr int ERR_SILENT_LOAD_TEXTURE_NOT_LOADED = 104;
// load (resources)
constexpr int ERR_LOAD_TERMINATED = 200;
constexpr int ERR_LOAD_TEXTURE_NOT_LOADED = 201;
constexpr int ERR_LOAD_OPENING_NOT_LOADED = 202;
constexpr int ERR_LOAD_LICENSE_FILE_NOT_LOADED = 203;
// load (ai)
constexpr int ERR_LOAD_EVAL_FILE_NOT_IMPORTED = 301;
constexpr int ERR_LOAD_BOOK_FILE_NOT_IMPORTED = 302;
constexpr int ERR_LOAD_HASH_NOT_RESIZED = 303;

// constant definition
constexpr int UPDATE_CHECK_NONE = -1;
constexpr int UPDATE_CHECK_ALREADY_UPDATED = 0;
constexpr int UPDATE_CHECK_UPDATE_FOUND = 1;
constexpr int UPDATE_CHECK_FAILED = 2;
constexpr int SHOW_ALL_HINT = 35;

// board drawing constants
constexpr int BOARD_SIZE = 400;
constexpr int BOARD_COORD_SIZE = 20;
constexpr int DISC_SIZE = 20;
constexpr int LEGAL_SIZE = 7;
constexpr int STABLE_SIZE = 4;
constexpr int BOARD_CELL_FRAME_WIDTH = 2;
constexpr int BOARD_DOT_SIZE = 5;
constexpr int BOARD_ROUND_FRAME_WIDTH = 10;
constexpr int BOARD_ROUND_DIAMETER = 23;
constexpr int BOARD_SY = 60;
constexpr int BOARD_DISC_FRAME_WIDTH = 2;
constexpr int BOARD_SX = LEFT_LEFT + BOARD_COORD_SIZE;
constexpr int BOARD_CELL_SIZE = BOARD_SIZE / HW;

// main scene start game button constants
constexpr int START_GAME_BUTTON_SX = 700;
constexpr int START_GAME_BUTTON_SY = 11;
constexpr int START_GAME_BUTTON_WIDTH = 90;
constexpr int START_GAME_BUTTON_HEIGHT = 30;
constexpr int START_GAME_BUTTON_RADIUS = 10;

// main scene pass button constants
constexpr int PASS_BUTTON_SX = 700;
constexpr int PASS_BUTTON_SY = 11;
constexpr int PASS_BUTTON_WIDTH = 90;
constexpr int PASS_BUTTON_HEIGHT = 30;
constexpr int PASS_BUTTON_RADIUS = 10;

// graph drawing constants
constexpr int GRAPH_RESOLUTION = 4;
constexpr int GRAPH_SX = BOARD_SX + BOARD_SIZE + 65;
constexpr int GRAPH_SY = Y_CENTER + 20;
constexpr int GRAPH_WIDTH = WINDOW_SIZE_X - GRAPH_SX - 35;
constexpr int GRAPH_HEIGHT = WINDOW_SIZE_Y - GRAPH_SY - 60;

// level drawing constants
constexpr int LEVEL_DEPTH_DY = -15;
constexpr int LEVEL_INFO_DX = -14;
constexpr int LEVEL_INFO_DY = -56;
constexpr int LEVEL_INFO_WIDTH = 37;
constexpr int LEVEL_INFO_HEIGHT = 20;
constexpr int LEVEL_PROB_WIDTH = 80;

// level graph roundrect constants
constexpr int GRAPH_RECT_DY = -70;
constexpr int GRAPH_RECT_DX = -45;
constexpr int GRAPH_RECT_RADIUS = 20;
constexpr int GRAPH_RECT_THICKNESS = 5;
constexpr int GRAPH_RECT_WIDTH = WINDOW_SIZE_X - (GRAPH_SX + GRAPH_RECT_DX) - 10;
constexpr int GRAPH_RECT_HEIGHT = GRAPH_HEIGHT - GRAPH_RECT_DY + 27;
constexpr int N_GRPAPH_COLOR_TYPES = 2;

// info drawing constants
constexpr int INFO_SY = 53;
constexpr int INFO_DISC_RADIUS = 12;
constexpr int INFO_SX = 460;
constexpr int INFO_RECT_RADIUS = 20;
constexpr int INFO_RECT_THICKNESS = 5;
constexpr int INFO_WIDTH = WINDOW_SIZE_X - 10 - INFO_SX;
constexpr int INFO_HEIGHT = 200 - INFO_SY - 12;
constexpr int AI_FOCUS_INFO_COLOR_RECT_WIDTH = 90 + INFO_DISC_RADIUS + 3;

// graph mode constants
constexpr int GRAPH_MODE_NORMAL = 0;
constexpr int GRAPH_MODE_INSPECT = 1;

// button press constants
constexpr int BUTTON_NOT_PUSHED = 0;
constexpr int BUTTON_LONG_PRESS_THRESHOLD = 500;

// hint constants
constexpr int HINT_NOT_CALCULATING = -1;
constexpr int HINT_INIT_VALUE = -INF;
constexpr int HINT_INF_LEVEL = 100;
constexpr int HINT_MAX_LEVEL = 60;

// analyze constants
constexpr int ANALYZE_SIZE = 62;

// export game constants
constexpr int EXPORT_GAME_PLAYER_WIDTH = 300;
constexpr int EXPORT_GAME_PLAYER_HEIGHT = 30;
constexpr int EXPORT_GAME_MEMO_WIDTH = 600;
constexpr int EXPORT_GAME_MEMO_HEIGHT = 240;
constexpr int EXPORT_GAME_RADIUS = 15;
constexpr int EXPORT_GAME_FOLDER_AREA_HEIGHT = 45;
constexpr int EXPORT_GAME_N_GAMES_ON_WINDOW = 5;
constexpr int EXPORT_GAME_CREATE_FOLDER_Y_CENTER = 370;
constexpr int EXPORT_GAME_DATE_WIDTH = EXPORT_GAME_PLAYER_WIDTH;
constexpr int EXPORT_GAME_DATE_HEIGHT = EXPORT_GAME_PLAYER_HEIGHT;


// import game constants
constexpr int IMPORT_GAME_N_GAMES_ON_WINDOW = 7;
constexpr int IMPORT_GAME_LEFT_MARGIN = 10;
constexpr int IMPORT_GAME_SX = 30 - IMPORT_GAME_LEFT_MARGIN;
constexpr int IMPORT_GAME_SY = 70;
constexpr int IMPORT_GAME_HEIGHT = 45;
constexpr int IMPORT_GAME_PLAYER_WIDTH = 220;
constexpr int IMPORT_GAME_PLAYER_HEIGHT = 24;
constexpr int IMPORT_GAME_SCORE_WIDTH = 60;
constexpr int IMPORT_GAME_WINNER_BLACK = 0;
constexpr int IMPORT_GAME_WINNER_WHITE = 1;
constexpr int IMPORT_GAME_WINNER_DRAW = 2;
constexpr int IMPORT_GAME_BUTTON_SX = 660;
constexpr int IMPORT_GAME_BUTTON_WIDTH = 100;
constexpr int IMPORT_GAME_BUTTON_HEIGHT = 25;
constexpr int IMPORT_GAME_BUTTON_RADIUS = 9;
constexpr int IMPORT_GAME_DATE_WIDTH = 120;
constexpr int IMPORT_GAME_BUTTON_SY = (IMPORT_GAME_HEIGHT - IMPORT_GAME_BUTTON_HEIGHT) / 2;
constexpr int IMPORT_GAME_WIDTH = WINDOW_SIZE_X - (IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN) * 2 + IMPORT_GAME_LEFT_MARGIN;

// opening setting
constexpr int OPENING_SETTING_N_GAMES_ON_WINDOW = 7;
constexpr int OPENING_SETTING_LEFT_MARGIN = 10;
constexpr int OPENING_SETTING_SX = 30 - OPENING_SETTING_LEFT_MARGIN;
constexpr int OPENING_SETTING_SY = 85;
constexpr int OPENING_SETTING_HEIGHT = 40;  // Reduced from 45 to make room for fixed header
constexpr int OPENING_SETTING_HEADER_HEIGHT = 30;  // Small height for fixed header display
constexpr int OPENING_SETTING_WIDTH = WINDOW_SIZE_X - (OPENING_SETTING_SX + OPENING_SETTING_LEFT_MARGIN) * 2 + OPENING_SETTING_LEFT_MARGIN;

// folder creation / rename shared layout
constexpr int NEW_FOLDER_PANEL_SY = OPENING_SETTING_SY + 8;
constexpr int NEW_FOLDER_PANEL_HEIGHT = OPENING_SETTING_HEIGHT;
constexpr int NEW_FOLDER_LABEL_INNER_MARGIN = 8;
constexpr int NEW_FOLDER_TEXTBOX_OFFSET_X = 150;
constexpr int NEW_FOLDER_TEXTBOX_OFFSET_Y = -17;
constexpr int NEW_FOLDER_TEXTBOX_WIDTH = 400;
constexpr int NEW_FOLDER_TEXTBOX_HEIGHT = 30;

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
constexpr int GAME_DISCS_UNDEFINED = -1;
constexpr int GAME_MEMO_SUMMARY_SIZE = 40;

// book modification
constexpr int BOOK_CHANGE_NO_CELL = 64;
constexpr int CHANGE_BOOK_ERR = -1000;
constexpr int CHANGE_BOOK_INFO_SX = 460;
constexpr int CHANGE_BOOK_INFO_SY = 13;

// back button constants
constexpr int BACK_BUTTON_WIDTH = 200;
constexpr int BACK_BUTTON_HEIGHT = 50;
constexpr int BACK_BUTTON_SY = 420;
constexpr int BACK_BUTTON_RADIUS = 20;
constexpr int BACK_BUTTON_SX = X_CENTER - BACK_BUTTON_WIDTH / 2;

// go/back button constants
constexpr int GO_BACK_BUTTON_WIDTH = 200;
constexpr int GO_BACK_BUTTON_HEIGHT = 50;
constexpr int GO_BACK_BUTTON_SY = 420;
constexpr int GO_BACK_BUTTON_RADIUS = 20;
constexpr int GO_BACK_BUTTON_GO_SX = X_CENTER + 10;
constexpr int GO_BACK_BUTTON_BACK_SX = X_CENTER - GO_BACK_BUTTON_WIDTH - 10;

// 2 buttons constants
constexpr int BUTTON2_WIDTH = 200;
constexpr int BUTTON2_HEIGHT = 50;
constexpr int BUTTON2_SY = 420;
constexpr int BUTTON2_RADIUS = 20;
constexpr int BUTTON2_1_SX = X_CENTER - BUTTON2_WIDTH - 10;
constexpr int BUTTON2_2_SX = X_CENTER + 10;

// 3 buttons constants
constexpr int BUTTON3_WIDTH = 200;
constexpr int BUTTON3_HEIGHT = 50;
constexpr int BUTTON3_SY = 420;
constexpr int BUTTON3_RADIUS = 20;
constexpr int BUTTON3_1_SX = X_CENTER - BUTTON3_WIDTH * 3 / 2 - 10;
constexpr int BUTTON3_2_SX = X_CENTER - BUTTON3_WIDTH / 2;
constexpr int BUTTON3_3_SX = X_CENTER + BUTTON3_WIDTH / 2 + 10;

// 2 buttons vertical constants
constexpr int BUTTON2_VERTICAL_WIDTH = 200;
constexpr int BUTTON2_VERTICAL_HEIGHT = 50;
constexpr int BUTTON2_VERTICAL_1_SY = 350;
constexpr int BUTTON2_VERTICAL_2_SY = 420;
constexpr int BUTTON2_VERTICAL_SX = 520;
constexpr int BUTTON2_VERTICAL_RADIUS = 20;

// 3 buttons vertical constants
constexpr int BUTTON3_VERTICAL_WIDTH = 200;
constexpr int BUTTON3_VERTICAL_HEIGHT = 50;
constexpr int BUTTON3_VERTICAL_1_SY = 280;
constexpr int BUTTON3_VERTICAL_2_SY = 350;
constexpr int BUTTON3_VERTICAL_3_SY = 420;
constexpr int BUTTON3_VERTICAL_SX = 520;
constexpr int BUTTON3_VERTICAL_RADIUS = 20;

// font constant
constexpr int FONT_DEFAULT_SIZE = 48;

// default language
#define DEFAULT_LANGUAGE "english"
#define DEFAULT_OPENING_LANG_NAME "english"

// textbox constant
constexpr int TEXTBOX_MAX_CHARS = 10000;





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
        player = BLACK;
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

    bool operator==(const History_elem& other) const {
        return board == other.board && player == other.player && player == other.player;
    }
};

struct Colors {
    Color green{ Color(36, 153, 114) };
    Color dark_green{ Color(42, 114, 83) };
    Color light_green{ Color(190, 255, 215) };
    Color black{ Palette::Black };
    Color white{ Palette::White };
    Color dark_gray{ Color(51, 51, 51) };
    Color cyan{ Color(110, 255, 255) };
    Color purple{ Color(142, 68, 173) };
    Color red{ Palette::Red };
    Color blue{ Palette::Blue };
    Color light_cyan{ Palette::Lightcyan };
    Color chocolate{ Color(210, 105, 30) };
    Color darkred{ Color(178, 34, 34) };
    Color darkblue{ Color(34, 34, 178) };
    Color burlywood{ Color(222, 184, 135) };
    Color black_advantage{ Color(241, 196, 15) };
    Color white_advantage{ Color(94, 192, 255) };
    Color yellow{ Palette::Yellow };
};

struct Directories {
    std::string document_dir;
    std::string appdata_dir;
    std::string eval_file;
    std::string eval_mo_end_file;
};

struct Resources {
    std::vector<std::string> language_names;
    Texture icon;
    Texture logo;
    Texture checkbox;
    Texture unchecked;
    Texture laser_pointer;
    Texture cross;
    Texture flip_horizontal;
    Texture flip_vertical;
    Texture rotate_cw;
    Texture rotate_ccw;
    Texture rotate_180;
    Texture mirror_white_line;
    Texture mirror_black_line;
    Texture check;
    Texture folder;
    Texture arrow_left;
    Texture pencil;
    std::vector<Texture> lang_img;
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
    int umigame_value_depth;
    int n_disc_hint;
    bool show_legal;
    bool show_graph;
    bool show_opening_on_cell;
    bool show_laser_pointer;
    bool show_log;
    int book_learn_depth;
    int book_learn_error_per_move;
    int book_learn_error_sum;
    int book_learn_error_leaf;
    bool use_book_learn_depth;
    bool use_book_learn_error_per_move;
    bool use_book_learn_error_sum;
    bool use_book_learn_error_leaf;
    bool show_stable_discs;
    bool change_book_by_right_click;
    bool show_last_move;
    bool show_next_move;
#if USE_CHANGEABLE_HASH_LEVEL
    int hash_level;
#endif
    //int book_acc_level;
    bool accept_ai_loss;
    int max_loss;
    int loss_percentage;
    bool pause_when_pass;
    bool force_specified_openings;
    bool show_next_move_change_view;
    bool change_color_type;
    bool show_play_ordering;
    int generate_random_board_moves;
    bool show_book_accuracy;
    bool show_graph_value;
    bool show_graph_sum_of_loss;
    bool show_opening_name;
    bool show_principal_variation;
    bool show_ai_focus;
    int pv_length;
    std::string screenshot_saving_dir;
    bool show_value_when_ai_calculating;
    int generate_random_board_score_range_min;
    int generate_random_board_score_range_max;
    bool show_hint_level;
    bool show_endgame_error;
    bool hint_colorize;
    bool play_ordering_board_format;
    bool play_ordering_transcript_format;
};

struct Fonts {
    Font font;
    Font font_bold;
    Font font_heavy;

    // japanese / english
    Font font_default{ FontMethod::MSDF, FONT_DEFAULT_SIZE };
    Font font_bold_default{ FontMethod::MSDF, FONT_DEFAULT_SIZE, Typeface::Bold };
    Font font_heavy_default{ FontMethod::MSDF, FONT_DEFAULT_SIZE, Typeface::Heavy };

    // simplified chinese
    Font font_SC{ FontMethod::MSDF, FONT_DEFAULT_SIZE, Typeface::CJK_Regular_SC };
    Font font_bold_SC{ FontMethod::MSDF, FONT_DEFAULT_SIZE, Typeface::CJK_Regular_SC, FontStyle::Bold };
    Font font_heavy_SC{ FontMethod::MSDF, FONT_DEFAULT_SIZE, Typeface::CJK_Regular_SC, FontStyle::BoldItalic };

    // traditional chinese (taiwan)
    Font font_TC_Taiwan{ FontMethod::MSDF, FONT_DEFAULT_SIZE, Typeface::CJK_Regular_TC };
    Font font_bold_TC_Taiwan{ FontMethod::MSDF, FONT_DEFAULT_SIZE, Typeface::CJK_Regular_TC, FontStyle::Bold };
    Font font_heavy_TC_Taiwan{ FontMethod::MSDF, FONT_DEFAULT_SIZE, Typeface::CJK_Regular_TC, FontStyle::BoldItalic };

    void init(std::string lang) {
        std::cerr << "font init " << lang << std::endl;
        if (lang == "simplified_chinese") {
            font = font_SC;
            font_bold = font_bold_SC;
            font_heavy = font_heavy_SC;
        } else if (lang == "traditional_chinese_taiwan") {
            font = font_TC_Taiwan;
            font_bold = font_bold_TC_Taiwan;
            font_heavy = font_heavy_TC_Taiwan;
        } else { // japanese / english
            font = font_default;
            font_bold = font_bold_default;
            font_heavy = font_heavy_default;
        }
        add_fallback();
    }

    void add_fallback() {

        // japanese / english
        font.addFallback(font_default);
        font_bold.addFallback(font_bold_default);
        font_heavy.addFallback(font_heavy_default);

        // chinese
        font.addFallback(font_SC);
        font_bold.addFallback(font_bold_SC);
        font_heavy.addFallback(font_heavy_SC);
    }
};

struct Menu_elements {
    bool dummy;

    // game
    bool start_game;
    bool start_game_human_black;
    bool start_game_human_white;
    bool start_selfplay;
    bool analyze;
    bool game_information;

    // settings
    // AI settings
    bool use_book;
    //int book_acc_level;
    bool accept_ai_loss;
    int max_loss;
    int loss_percentage;
    int level;
    int n_threads;
#if USE_CHANGEABLE_HASH_LEVEL
    int hash_level;
#endif
    // player
    bool ai_put_black;
    bool ai_put_white;
    bool pause_when_pass;
    bool force_specified_openings;
    bool opening_setting;
    bool shortcut_key_setting;

    // display
    bool use_disc_hint;
    int n_disc_hint;
    bool show_hint_level;
    bool use_umigame_value;
    int umigame_value_depth;
    bool show_legal;
    bool show_graph;
    bool show_opening_on_cell;
    bool show_stable_discs;
    bool show_play_ordering;
    bool play_ordering_board_format;
    bool play_ordering_transcript_format;
    bool show_laser_pointer;
    bool show_log;
    bool show_last_move;
    bool show_next_move;
    bool show_next_move_change_view;
    bool change_color_type;
    bool show_book_accuracy;
    bool show_graph_value;
    bool show_graph_sum_of_loss;
    bool show_opening_name;
    bool show_principal_variation;
    bool show_ai_focus;
    int pv_length;
    bool show_value_when_ai_calculating;
    bool show_endgame_error;
    bool hint_colorize;

    // book
    bool book_start_deviate;
    bool book_start_deviate_with_transcript;
    bool book_start_store;
    bool book_start_fix;
    bool book_start_fix_edax;
    int book_learn_depth;
    int book_learn_error_per_move;
    int book_learn_error_sum;
    int book_learn_error_leaf;
    bool use_book_learn_depth;
    bool use_book_learn_error_per_move;
    bool use_book_learn_error_sum;
    bool use_book_learn_error_leaf;
    bool book_merge;
    bool book_reference;
    bool change_book_by_right_click;
    bool import_book;
    bool export_book;
    bool book_start_reducing;
    bool book_start_recalculate_leaf;
    bool show_book_info;
    bool book_start_recalculate_n_lines;
    bool book_start_upgrade_better_leaves;

    // input / output
    // input
    bool input_from_clipboard;
    bool input_text;
    bool edit_board;
    bool input_game;
    // output
    bool copy_transcript;
    bool copy_board;
    bool input_bitboard;
    bool save_game;
    bool screen_shot;
    bool change_screenshot_saving_dir;
    bool board_image;
    bool output_bitboard_player_opponent;
    bool output_bitboard_black_white;

    // manipulation
    bool stop_calculating;
    bool put_1_move_by_ai;
    bool forward;
    bool backward;
    bool undo;
    bool save_this_branch;
    bool generate_random_board;
    int generate_random_board_moves;
    int generate_random_board_score_range_min;
    int generate_random_board_score_range_max;
    // conversion
    bool convert_180;
    bool convert_90_clock;
    bool convert_90_anti_clock;
    bool convert_blackline;
    bool convert_whiteline;
    bool convert_horizontal;
    bool convert_vertical;
    bool cache_clear;

    // help
    bool usage;
    bool website;
    bool bug_report;
    bool update_check;
    bool auto_update_check;
    bool license;

    // language
    bool languages[200];

    void init(Settings* settings, Resources* resources) {
        dummy = false;

        start_game = false;
        start_game_human_black = false;
        start_game_human_white = false;
        start_selfplay = false;
        analyze = false;
        game_information = false;

        use_book = settings->use_book;
        //book_acc_level = settings->book_acc_level;
        accept_ai_loss = settings->accept_ai_loss;
        max_loss = settings->max_loss;
        loss_percentage = settings->loss_percentage;
        level = settings->level;
        n_threads = settings->n_threads;
#if USE_CHANGEABLE_HASH_LEVEL
        hash_level = settings->hash_level;
#endif
        ai_put_black = settings->ai_put_black;
        ai_put_white = settings->ai_put_white;
        pause_when_pass = settings->pause_when_pass;
        force_specified_openings = settings->force_specified_openings;
        opening_setting = false;
        shortcut_key_setting = false;

        use_disc_hint = settings->use_disc_hint;
        n_disc_hint = settings->n_disc_hint;
        show_hint_level = settings->show_hint_level;
        use_umigame_value = settings->use_umigame_value;
        umigame_value_depth = settings->umigame_value_depth;
        show_legal = settings->show_legal;
        show_graph = settings->show_graph;
        show_opening_on_cell = settings->show_opening_on_cell;
        show_stable_discs = settings->show_stable_discs;
        show_play_ordering = settings->show_play_ordering;
        play_ordering_board_format = settings->play_ordering_board_format;
        play_ordering_transcript_format = settings->play_ordering_transcript_format;
        show_laser_pointer = settings->show_laser_pointer;
        show_log = settings->show_log;
        show_last_move = settings->show_last_move;
        show_next_move = settings->show_next_move;
        show_next_move_change_view = settings->show_next_move_change_view;
        change_color_type = settings->change_color_type;
        show_book_accuracy = settings->show_book_accuracy;
        show_graph_value = settings->show_graph_value;
        show_graph_sum_of_loss = settings->show_graph_sum_of_loss;
        show_opening_name = settings->show_opening_name;
        show_principal_variation = settings->show_principal_variation;
        show_ai_focus = settings->show_ai_focus;
        pv_length = settings->pv_length;
        show_value_when_ai_calculating = settings->show_value_when_ai_calculating;
        show_endgame_error = settings->show_endgame_error;
        hint_colorize = settings->hint_colorize;

        book_start_deviate = false;
        book_start_deviate_with_transcript = false;
        book_start_store = false;
        book_start_fix = false;
        book_start_fix_edax = false;
        book_learn_depth = settings->book_learn_depth;
        book_learn_error_per_move = settings->book_learn_error_per_move;
        book_learn_error_sum = settings->book_learn_error_sum;
        book_learn_error_leaf = settings->book_learn_error_leaf;
        use_book_learn_depth = settings->use_book_learn_depth;
        use_book_learn_error_per_move = settings->use_book_learn_error_per_move;
        use_book_learn_error_sum = settings->use_book_learn_error_sum;
        use_book_learn_error_leaf = settings->use_book_learn_error_leaf;
        book_merge = false;
        book_reference = false;
        change_book_by_right_click = settings->change_book_by_right_click;
        book_start_reducing = false;
        book_start_recalculate_leaf = false;
        import_book = false;
        export_book = false;
        show_book_info = false;
        book_start_recalculate_n_lines = false;
        book_start_upgrade_better_leaves = false;

        input_from_clipboard = false;
        input_text = false;
        edit_board = false;
        input_game = false;
        copy_transcript = false;
        copy_board = false;
        input_bitboard = false;
        save_game = false;
        screen_shot = false;
        change_screenshot_saving_dir = false;
        board_image = false;
        output_bitboard_player_opponent = false;
        output_bitboard_black_white = false;

        stop_calculating = false;
        put_1_move_by_ai = false;
        forward = false;
        backward = false;
        undo = false;
        save_this_branch = false;
        generate_random_board = false;
        generate_random_board_moves = settings->generate_random_board_moves;
        generate_random_board_score_range_min = settings->generate_random_board_score_range_min;
        generate_random_board_score_range_max = settings->generate_random_board_score_range_max;
        convert_180 = false;
        convert_90_clock = false;
        convert_90_anti_clock = false;
        convert_blackline = false;
        convert_whiteline = false;
        convert_horizontal = false;
        convert_vertical = false;
        cache_clear = false;

        usage = false;
        website = false;
        bug_report = false;
        update_check = false;
        auto_update_check = settings->auto_update_check;
        license = false;

        bool lang_found = false;
        for (int i = 0; i < resources->language_names.size(); ++i) {
            if (resources->language_names[i] == settings->lang_name) {
                lang_found = true;
                languages[i] = true;
            } else {
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
    int branch;
    bool need_init;

    Graph_resources() {
        init();
    }

    void init() {
        nodes[0].clear();
        nodes[1].clear();
        n_discs = 4;
        delta = 0;
        branch = 0;
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
    String date;  // YYYY-MM-DD format
    bool is_game_loaded;  // true if a specific game is loaded from Import_game

    void init() {
        black_player_name.clear();
        white_player_name.clear();
        memo.clear();
        date.clear();
        is_game_loaded = false;
    }
};

struct Game_editor_info {
    String return_scene;
    bool is_editing_mode;
    String game_date;
    std::string subfolder;
    bool game_info_updated;
    int export_mode; // 0 = main line, 1 = until this board

    void init() {
        return_scene.clear();
        is_editing_mode = false;
        game_date.clear();
        subfolder.clear();
        game_info_updated = false;
        export_mode = 0;
    }
};

struct Save_location_picker_info {
    std::vector<History_elem> pending_history;
    std::string selected_subfolder;

    void init() {
        pending_history.clear();
        selected_subfolder.clear();
    }
};

struct Book_information {
    bool changed{ false };
    uint_fast8_t changing{ BOOK_CHANGE_NO_CELL };
    String val_str;
};

struct User_settings {
    std::string screenshot_saving_dir;
};

struct Window_state {
    double window_scale;
    bool loading;
    Window_state() {
        window_scale = 1.0;
        loading = true;
    }
};

struct Forced_openings {
    std::vector<std::pair<std::string, double>> openings;
    std::unordered_map<Board, std::vector<std::pair<int, double>>, Book_hash> selected_moves;
    std::unordered_set<Board, Book_hash> forced_boards;

    // Forced_openings() {
    //     openings = {
    //         {"f5d6c3d3c4f4f6", 1}, // stephenson
    //         {"f5d6c3d3c4f4e3", 1}, // brightwell
    //         {"f5d6c3d3c4f4e6", 1}, // leader's tiger
    //     };
    // }

    void init() {
        // Clear existing data before rebuilding
        selected_moves.clear();
        forced_boards.clear();
        
        Board board;
        Flip flip;
        for (const std::pair<std::string, double> opening : openings) {
            board.reset();
            int symmetry_idx = 0;
            Board unique_board = representative_board(board, &symmetry_idx);
            forced_boards.emplace(unique_board);
            std::string opening_str = opening.first;
            // std::cerr << opening_str << std::endl;
            double weight = opening.second;
            for (int i = 0; i < opening_str.size() - 1 && board.check_pass(); i += 2) {
                int policy = get_coord_from_chars(opening_str[i], opening_str[i + 1]);
                // std::cerr << idx_to_coord(policy) << std::endl;
                // board.print();
                int symmetry_idx1 = 0;
                Board unique_board1 = representative_board(board, &symmetry_idx1);
                selected_moves[unique_board1].emplace_back(std::make_pair(convert_coord_from_representative_board(policy, symmetry_idx1), weight));
                calc_flip(&flip, &board, policy);
                board.move_board(&flip);
                Board unique_board2 = representative_board(board);
                forced_boards.emplace(unique_board2);
            }
        }
        std::cerr << "forced openings initialized " << selected_moves.size() << std::endl;
    }

    void load(std::string file) {
        std::ifstream ifs(file);
        if (!ifs) {
            std::cerr << "forced openings file " << file << " not found" << std::endl;
            return;
        }
        openings.clear();
        std::string line;
        while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            std::string transcript, weight_str;
            iss >> transcript >> weight_str;
            double weight;
            try {
                weight = stoi(weight_str);
                if (is_valid_transcript(transcript)) {
                    openings.emplace_back(std::make_pair(transcript, weight));
                }
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument: " << e.what() << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range: " << e.what() << std::endl;
            }
        }
        init();
    }

    void save(std::string file) {
        std::ofstream ofs(file);
        for (const std::pair<std::string, double> opening : openings) {
            ofs << opening.first << " " << std::round(opening.second) << std::endl;
        }
        ofs.close();
    }

    int get_one(Board board) {
        int symmetry_idx = 0;
        Board unique_board = representative_board(board, &symmetry_idx);
        int selected_policy = MOVE_UNDEFINED;
        if (selected_moves.find(unique_board) != selected_moves.end()) {
            double sum_weight = 0.0;
            for (std::pair<int, double> &elem: selected_moves[unique_board]) {
                sum_weight += elem.second;
            }
            double rnd = myrandom() * sum_weight;
            double s = 0.0;
            for (std::pair<int, double> &elem: selected_moves[unique_board]) {
                s += elem.second;
                if (s >= rnd) {
                    selected_policy = convert_coord_to_representative_board(elem.first, symmetry_idx);
                    break;
                }
            }
        }
        return selected_policy;
    }

    void add(std::string str, double weight) {
        openings.emplace_back(std::make_pair(str, weight));
        init();
    }

    bool is_forced(Board board) {
        Board unique_board = representative_board(board);
        return forced_boards.find(unique_board) != forced_boards.end();
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
    Game_information game_information;
    Game_editor_info game_editor_info;
    Save_location_picker_info save_location_picker_info;
    Book_information book_information;
    User_settings user_settings;
    Forced_openings forced_openings;
    Window_state window_state;
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
    bool hint_calculated{ false };
    std::future<void> hint_future;
    bool hint_use[HW2];
    double hint_values[HW2];
    int hint_types[HW2];
    int n_hint_display;

    bool analyzing{ false };
    std::future<Search_result> analyze_future[ANALYZE_SIZE];
    int analyze_sgn[ANALYZE_SIZE];
    std::vector<std::pair<Analyze_info, std::function<Search_result()>>> analyze_task_stack;

    bool pv_calculating{ false };
    bool pv_calculated{ false };
    std::future<void> pv_future;

    bool book_learning{ false };

    bool local_strategy_calculating{ false };
    bool local_strategy_calculated{ false };
    std::future<void> local_strategy_future;
    double local_strategy[HW2];
    int local_strategy_done_level{ 0 };

    bool local_strategy_policy_calculating{ false };
    bool local_strategy_policy_calculated{ false };
    std::future<void> local_strategy_policy_future;
    int local_strategy_policy[HW2][HW2]; // [policy][cell]
    int local_strategy_policy_done_level{ 0 };

    bool random_board_generator_calculating{ false };
    std::future<std::vector<int>> random_board_generator_future;
};

struct Game_abstract {
    String black_player;
    String white_player;
    int black_score;
    int white_score;
    String memo;
    String filename_date;  // YYYY_MM_DD_hh_mm_ss format (used for file identification)
    String game_date;      // YYYY-MM-DD format (user-specified game date)
};

struct Umigame_status {
    bool umigame_calculating{ false };
    bool umigame_calculated{ false };
    std::future<Umigame_result> umigame_future[HW2];
    Umigame_result umigame[HW2];
};

struct Book_accuracy_status {
    bool book_accuracy_calculating{ false };
    bool book_accuracy_calculated{ false };
    std::future<int> book_accuracy_future[HW2];
    int book_accuracy[HW2];
};

using App = SceneManager<String, Common_resources>;