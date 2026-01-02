/*
    Egaroucid Project

    @file silent_load.hpp
        Load before GUI wake up
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"

std::string get_default_language() {
    std::string default_language = System::DefaultLanguage().narrow();
    std::string res = "english";
    if (default_language == "ja-JP") { // japanese
        res = "japanese";
    }
    if (default_language == "zh-CN" || default_language == "zh-cmn-Hans") { // chinese
        res = "chinese";
    }
    return res;
}

void init_default_settings(const Directories* directories, const Resources* resources, Settings* settings) {
    //std::cerr << "use default settings" << std::endl;
    settings->n_threads = std::min(32, (int)std::thread::hardware_concurrency());
    settings->auto_update_check = 1;
    settings->lang_name = get_default_language();
    settings->book_file = directories->document_dir + "book" + BOOK_EXTENSION;
    settings->use_book = true;
    settings->level = DEFAULT_LEVEL;
    settings->ai_put_black = false;
    settings->ai_put_white = false;
    settings->use_disc_hint = true;
    settings->use_umigame_value = false;
    settings->n_disc_hint = SHOW_ALL_HINT;
    settings->show_legal = true;
    settings->show_graph = true;
    settings->show_opening_on_cell = true;
    settings->show_log = true;
    settings->book_learn_depth = 30;
    settings->book_learn_error_per_move = 2;
    settings->book_learn_error_sum = 2;
    settings->show_stable_discs = false;
    settings->change_book_by_right_click = false;
    settings->show_last_move = true;
    settings->show_next_move = true;
#if USE_CHANGEABLE_HASH_LEVEL
    settings->hash_level = DEFAULT_HASH_LEVEL;
#endif
    //settings->book_acc_level = 0;
    settings->accept_ai_loss = false;
    settings->max_loss = 2;
    settings->loss_percentage = 30;
    settings->pause_when_pass = true;
    settings->force_specified_openings = false;
    settings->show_next_move_change_view = false;
    settings->change_color_type = false;
    settings->show_play_ordering = false;
    settings->generate_random_board_moves = 20;
    settings->show_book_accuracy = false;
    settings->use_book_learn_depth = true;
    settings->use_book_learn_error_per_move = true;
    settings->use_book_learn_error_sum = true;
    settings->umigame_value_depth = 60;
    settings->show_graph_value = true;
    settings->show_graph_sum_of_loss = false;
    settings->book_learn_error_leaf = 2;
    settings->use_book_learn_error_leaf = true;
    settings->show_opening_name = true;
    settings->show_principal_variation = true;
    settings->show_laser_pointer = false;
    settings->show_ai_focus = false;
    settings->pv_length = 7;
    settings->screenshot_saving_dir = directories->document_dir + "screenshots/";
    settings->show_value_when_ai_calculating = false;
    // settings->generate_random_board_score_range = 64;
    settings->generate_random_board_score_range_min = -64;
    settings->generate_random_board_score_range_max = 64;
    settings->show_hint_level = true;
    settings->show_endgame_error = false;
    settings->hint_colorize = true;
    settings->play_ordering_board_format = true;
    settings->play_ordering_transcript_format = false;
}

int init_settings_import_int(JSON &json, String key, int* res) {
    if (json[key].getType() != JSONValueType::Number) {
        return ERR_IMPORT_SETTINGS;
    }
    *res = (int)json[key].get<double>();
    return ERR_OK;
}

int init_settings_import_bool(JSON &json, String key, bool* res) {
    if (json[key].getType() != JSONValueType::Bool) {
        return ERR_IMPORT_SETTINGS;
    }
    *res = json[key].get<bool>();
    return ERR_OK;
}

int init_settings_import_str(JSON &json, String key, std::string* res) {
    if (json[key].getType() != JSONValueType::String) {
        return ERR_IMPORT_SETTINGS;
    }
    *res = json[key].getString().narrow();
    return ERR_OK;
}

int init_settings_import_int(TextReader* reader, int* res) {
    String line;
    int pre_res = *res;
    if (reader->readLine(line)) {
        try {
            *res = Parse<int32>(line);
            return ERR_OK;
        }
        catch (const ParseError& e) {
            *res = pre_res;
            return ERR_IMPORT_SETTINGS;
        }
    } else {
        *res = pre_res;
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
    } else {
        return ERR_IMPORT_SETTINGS;
    }
}

int init_settings_import_str(TextReader* reader, std::string* res) {
    String line;
    if (reader->readLine(line)) {
        *res = line.narrow();
        return ERR_OK;
    } else {
        return ERR_IMPORT_SETTINGS;
    }
}

void import_text_settings(const Directories* directories, const Resources* resources, Settings* settings) {
    TextReader reader(U"{}setting.txt"_fmt(Unicode::Widen(directories->appdata_dir)));
    if (!reader) {
        std::cerr << "err-1" << std::endl;
        return;
    } else {
        if (init_settings_import_int(&reader, &settings->n_threads) != ERR_OK) {
            std::cerr << "err0" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->auto_update_check) != ERR_OK) {
            std::cerr << "err1" << std::endl;
            return;
        }
        if (init_settings_import_str(&reader, &settings->lang_name) != ERR_OK) {
            std::cerr << "err2" << std::endl;
            return;
        }
        if (init_settings_import_str(&reader, &settings->book_file) != ERR_OK) {
            std::cerr << "err3" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->use_book) != ERR_OK) {
            std::cerr << "err4" << std::endl;
            return;
        }
        if (init_settings_import_int(&reader, &settings->level) != ERR_OK) {
            std::cerr << "err5" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->ai_put_black) != ERR_OK) {
            std::cerr << "err6" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->ai_put_white) != ERR_OK) {
            std::cerr << "err7" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->use_disc_hint) != ERR_OK) {
            std::cerr << "err8" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->use_umigame_value) != ERR_OK) {
            std::cerr << "err9" << std::endl;
            return;
        }
        if (init_settings_import_int(&reader, &settings->n_disc_hint) != ERR_OK) {
            std::cerr << "err10" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->show_legal) != ERR_OK) {
            std::cerr << "err11" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->show_graph) != ERR_OK) {
            std::cerr << "err12" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->show_opening_on_cell) != ERR_OK) {
            std::cerr << "err13" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->show_log) != ERR_OK) {
            std::cerr << "err14" << std::endl;
            return;
        }
        if (init_settings_import_int(&reader, &settings->book_learn_depth) != ERR_OK) {
            std::cerr << "err15" << std::endl;
            return;
        }
        if (init_settings_import_int(&reader, &settings->book_learn_error_per_move) != ERR_OK) {
            std::cerr << "err16" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->show_stable_discs) != ERR_OK) {
            std::cerr << "err17" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->change_book_by_right_click) != ERR_OK) {
            std::cerr << "err18" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->show_last_move) != ERR_OK) {
            std::cerr << "err20" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->show_next_move) != ERR_OK) {
            std::cerr << "err21" << std::endl;
            return;
        }
#if USE_CHANGEABLE_HASH_LEVEL
        if (init_settings_import_int(&reader, &settings->hash_level) != ERR_OK) {
            std::cerr << "err22" << std::endl;
            return;
        } else {
            settings->hash_level = std::max(settings->hash_level, DEFAULT_HASH_LEVEL);
        }
#endif
        // if (init_settings_import_int(&reader, &settings->book_acc_level) != ERR_OK) {
        //     std::cerr << "err23" << std::endl;
        //     return;
        // }
        if (init_settings_import_bool(&reader, &settings->pause_when_pass) != ERR_OK) {
            std::cerr << "err24" << std::endl;
            return;
        }
        if (init_settings_import_int(&reader, &settings->book_learn_error_sum) != ERR_OK) {
            std::cerr << "err25" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->show_next_move_change_view) != ERR_OK) {
            std::cerr << "err26" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->change_color_type) != ERR_OK) {
            std::cerr << "err27" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->show_play_ordering) != ERR_OK) {
            std::cerr << "err28" << std::endl;
            return;
        }
        if (init_settings_import_int(&reader, &settings->generate_random_board_moves) != ERR_OK) {
            std::cerr << "err29" << std::endl;
            return;
        }
    }
}

void init_settings(const Directories* directories, const Resources* resources, Settings* settings) {
    init_default_settings(directories, resources, settings);
    JSON setting_json = JSON::Load(U"{}setting.json"_fmt(Unicode::Widen(directories->appdata_dir)));
    if (setting_json.size() == 0) {
        std::cerr << "json not found, try legacy txt settings" << std::endl;
        import_text_settings(directories, resources, settings);
        return;
    }
    if (init_settings_import_int(setting_json, U"n_threads", &settings->n_threads) != ERR_OK) {
        std::cerr << "err0" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"auto_update_check", &settings->auto_update_check) != ERR_OK) {
        std::cerr << "err1" << std::endl;
    }
    if (init_settings_import_str(setting_json, U"lang_name", &settings->lang_name) != ERR_OK) {
        std::cerr << "err2" << std::endl;
    }
    if (init_settings_import_str(setting_json, U"book_file", &settings->book_file) != ERR_OK) {
        std::cerr << "err3" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"use_book", &settings->use_book) != ERR_OK) {
        std::cerr << "err4" << std::endl;
    }
    if (init_settings_import_int(setting_json, U"level", &settings->level) != ERR_OK) {
        std::cerr << "err5" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"ai_put_black", &settings->ai_put_black) != ERR_OK) {
        std::cerr << "err6" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"ai_put_white", &settings->ai_put_white) != ERR_OK) {
        std::cerr << "err7" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"use_disc_hint", &settings->use_disc_hint) != ERR_OK) {
        std::cerr << "err8" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"use_umigame_value", &settings->use_umigame_value) != ERR_OK) {
        std::cerr << "err9" << std::endl;
    }
    if (init_settings_import_int(setting_json, U"n_disc_hint", &settings->n_disc_hint) != ERR_OK) {
        std::cerr << "err10" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_legal", &settings->show_legal) != ERR_OK) {
        std::cerr << "err11" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_graph", &settings->show_graph) != ERR_OK) {
        std::cerr << "err12" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_opening_on_cell", &settings->show_opening_on_cell) != ERR_OK) {
        std::cerr << "err13" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_log", &settings->show_log) != ERR_OK) {
        std::cerr << "err14" << std::endl;
    }
    if (init_settings_import_int(setting_json, U"book_learn_depth", &settings->book_learn_depth) != ERR_OK) {
        std::cerr << "err15" << std::endl;
    }
    if (init_settings_import_int(setting_json, U"book_learn_error_per_move", &settings->book_learn_error_per_move) != ERR_OK) {
        std::cerr << "err16" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_stable_discs", &settings->show_stable_discs) != ERR_OK) {
        std::cerr << "err17" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"change_book_by_right_click", &settings->change_book_by_right_click) != ERR_OK) {
        std::cerr << "err18" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_last_move", &settings->show_last_move) != ERR_OK) {
        std::cerr << "err20" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_next_move", &settings->show_next_move) != ERR_OK) {
        std::cerr << "err21" << std::endl;
    }
#if USE_CHANGEABLE_HASH_LEVEL
    if (init_settings_import_int(setting_json, U"hash_level", &settings->hash_level) != ERR_OK) {
        std::cerr << "err22" << std::endl;
    } else {
        settings->hash_level = std::max(settings->hash_level, DEFAULT_HASH_LEVEL);
    }
#endif
    // if (init_settings_import_int(setting_json, U"book_acc_level", &settings->book_acc_level) != ERR_OK) {
    //     std::cerr << "err23" << std::endl;
    // }
    if (init_settings_import_bool(setting_json, U"pause_when_pass", &settings->pause_when_pass) != ERR_OK) {
        std::cerr << "err24" << std::endl;
    }
    if (init_settings_import_int(setting_json, U"book_learn_error_sum", &settings->book_learn_error_sum) != ERR_OK) {
        std::cerr << "err25" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_next_move_change_view", &settings->show_next_move_change_view) != ERR_OK) {
        std::cerr << "err26" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"change_color_type", &settings->change_color_type) != ERR_OK) {
        std::cerr << "err27" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_play_ordering", &settings->show_play_ordering) != ERR_OK) {
        std::cerr << "err28" << std::endl;
    }
    if (init_settings_import_int(setting_json, U"generate_random_board_moves", &settings->generate_random_board_moves) != ERR_OK) {
        std::cerr << "err29" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_book_accuracy", &settings->show_book_accuracy) != ERR_OK) {
        std::cerr << "err30" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"use_book_learn_depth", &settings->use_book_learn_depth) != ERR_OK) {
        std::cerr << "err31" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"use_book_learn_error_per_move", &settings->use_book_learn_error_per_move) != ERR_OK) {
        std::cerr << "err32" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"use_book_learn_error_sum", &settings->use_book_learn_error_sum) != ERR_OK) {
        std::cerr << "err33" << std::endl;
    }
    if (init_settings_import_int(setting_json, U"umigame_value_depth", &settings->umigame_value_depth) != ERR_OK) {
        std::cerr << "err34" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_graph_value", &settings->show_graph_value) != ERR_OK) {
        std::cerr << "err35" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_graph_sum_of_loss", &settings->show_graph_sum_of_loss) != ERR_OK) {
        std::cerr << "err36" << std::endl;
    }
    if (init_settings_import_int(setting_json, U"book_learn_error_leaf", &settings->book_learn_error_leaf) != ERR_OK) {
        std::cerr << "err37" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"use_book_learn_error_leaf", &settings->use_book_learn_error_leaf) != ERR_OK) {
        std::cerr << "err38" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_opening_name", &settings->show_opening_name) != ERR_OK) {
        std::cerr << "err39" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_principal_variation", &settings->show_principal_variation) != ERR_OK) {
        std::cerr << "err40" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_laser_pointer", &settings->show_laser_pointer) != ERR_OK) {
        std::cerr << "err41" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_ai_focus", &settings->show_ai_focus) != ERR_OK) {
        std::cerr << "err42" << std::endl;
    }
    if (init_settings_import_int(setting_json, U"pv_length", &settings->pv_length) != ERR_OK) {
        std::cerr << "err43" << std::endl;
    }
    if (init_settings_import_str(setting_json, U"screenshot_saving_dir", &settings->screenshot_saving_dir) != ERR_OK) {
        std::cerr << "err44" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"accept_ai_loss", &settings->accept_ai_loss) != ERR_OK) {
        std::cerr << "err45" << std::endl;
    }
    if (init_settings_import_int(setting_json, U"max_loss", &settings->max_loss) != ERR_OK) {
        std::cerr << "err46" << std::endl;
    }
    if (init_settings_import_int(setting_json, U"loss_percentage", &settings->loss_percentage) != ERR_OK) {
        std::cerr << "err47" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"force_specified_openings", &settings->force_specified_openings) != ERR_OK) {
        std::cerr << "err48" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_value_when_ai_calculating", &settings->show_value_when_ai_calculating) != ERR_OK) {
        std::cerr << "err49" << std::endl;
    }
    // not used
    // if (init_settings_import_int(setting_json, U"generate_random_board_score_range", &settings->generate_random_board_score_range) != ERR_OK) {
    //     std::cerr << "err50" << std::endl;
    // }
    if (init_settings_import_bool(setting_json, U"show_hint_level", &settings->show_hint_level) != ERR_OK) {
        std::cerr << "err51" << std::endl;
    }
    int n_random_board_range_failed = 0;
    if (init_settings_import_int(setting_json, U"generate_random_board_score_range_min", &settings->generate_random_board_score_range_min) != ERR_OK) {
        std::cerr << "err52" << std::endl;
        ++n_random_board_range_failed;
    }
    if (init_settings_import_int(setting_json, U"generate_random_board_score_range_max", &settings->generate_random_board_score_range_max) != ERR_OK) {
        std::cerr << "err53" << std::endl;
        ++n_random_board_range_failed;
    }
    int generate_random_board_score_range;
    if (n_random_board_range_failed == 2 && init_settings_import_int(setting_json, U"generate_random_board_score_range", &generate_random_board_score_range) == ERR_OK) {
        settings->generate_random_board_score_range_min = -generate_random_board_score_range;
        settings->generate_random_board_score_range_max = generate_random_board_score_range;
        std::cerr << "use abs generate_random_board_score_range" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"show_endgame_error", &settings->show_endgame_error) != ERR_OK) {
        std::cerr << "err54" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"hint_colorize", &settings->hint_colorize) != ERR_OK) {
        std::cerr << "err55" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"play_ordering_board_format", &settings->play_ordering_board_format) != ERR_OK) {
        std::cerr << "err56" << std::endl;
    }
    if (init_settings_import_bool(setting_json, U"play_ordering_transcript_format", &settings->play_ordering_transcript_format) != ERR_OK) {
        std::cerr << "err57" << std::endl;
    }
}

void init_directories(Directories* directories) {
    // system directory
#if GUI_PORTABLE_MODE
    directories->document_dir = "./document/";
    directories->appdata_dir = "./appdata/";
#else
    directories->document_dir = FileSystem::GetFolderPath(SpecialFolder::Documents).narrow() + "Egaroucid/";
    directories->appdata_dir = FileSystem::GetFolderPath(SpecialFolder::LocalAppData).narrow() + "Egaroucid/";
#endif
    std::cerr << "document_dir " << directories->document_dir << " appdata_dir " << directories->appdata_dir << std::endl;

    // file directories
    directories->eval_file = EXE_DIRECTORY_PATH + "resources/eval.egev2";
    directories->eval_mo_end_file = EXE_DIRECTORY_PATH + "resources/eval_move_ordering_end.egev";
}

int init_resources_silent_load(Resources* resources, Settings* settings, Fonts *fonts, bool *stop_loading) {
    // language json
    std::cerr << "loading language list" << std::endl;
    if (!language_name.init(resources->language_names)) {
        return ERR_SILENT_LOAD_LANG_JSON_NOT_LOADED;
    }

    if (*stop_loading) {
        return ERR_SILENT_LOAD_TERMINATED;
    }

    // language
    std::cerr << "loading language pack" << std::endl;
    std::string lang_file = EXE_DIRECTORY_PATH + "resources/languages/" + settings->lang_name + ".json";
    if (!language.init(lang_file)) {
        std::cerr << "language file not found. use alternative language" << std::endl;
        settings->lang_name = DEFAULT_LANGUAGE;
        lang_file = EXE_DIRECTORY_PATH + "resources/languages/" + settings->lang_name + ".json";
        if (!language.init(lang_file)) {
            return ERR_SILENT_LOAD_LANG_NOT_LOADED;
        }
    }

    if (*stop_loading) {
        return ERR_SILENT_LOAD_TERMINATED;
    }

    // font
    std::cerr << "loading font" << std::endl;
    fonts->init(settings->lang_name);

    if (*stop_loading) {
        return ERR_SILENT_LOAD_TERMINATED;
    }

    // textures
    std::cerr << "loading textures (1)" << std::endl;
    Texture icon(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/icon.png"), TextureDesc::Mipped);
    Texture logo(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/logo.png"), TextureDesc::Mipped);
    if (icon.isEmpty() || logo.isEmpty()) {
        return ERR_SILENT_LOAD_TEXTURE_NOT_LOADED;
    }
    resources->icon = icon;
    resources->logo = logo;

    if (*stop_loading) {
        return ERR_SILENT_LOAD_TERMINATED;
    }

    return ERR_OK;

}

void init_user_settings(Settings* settings, User_settings *user_settings) {
    user_settings->screenshot_saving_dir = settings->screenshot_saving_dir;
}

int silent_load(Directories* directories, Resources* resources, Settings* settings, User_settings *user_settings, Fonts *fonts, bool *stop_loading) {
    init_directories(directories);
    init_settings(directories, resources, settings);
    init_user_settings(settings, user_settings);
    return init_resources_silent_load(resources, settings, fonts, stop_loading);
}

class Silent_load : public App::Scene {
private:
    bool loading;
    bool loaded;
    int load_code;
    bool stop_loading;
    std::future<int> silent_load_future;
    Font err_font{ FontMethod::MSDF, FONT_DEFAULT_SIZE };
public:
    Silent_load(const InitData& init) : IScene{ init } {
        stop_loading = false;
        silent_load_future = std::async(std::launch::async, silent_load, &getData().directories, &getData().resources, &getData().settings, &getData().user_settings, &getData().fonts, &stop_loading);
        loading = true;
        loaded = false;
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            stop_loading = true;
            if (silent_load_future.valid()) {
                silent_load_future.get();
            }
            System::Exit();
        }
        if (loading) {
            if (silent_load_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                load_code = silent_load_future.get();
                loaded = load_code == ERR_OK;
                loading = false;
            }
        } else {
            if (loaded) {
                std::cerr << "silent loaded" << std::endl;
                // changeScene(U"Load", SCENE_FADE_TIME);
                changeScene(U"Load", 0);
            } else {
                String err_str = U"BASIC DATA NOT LOADED. PLEASE RE-INSTALL.\nERROR CODE: " + Format(load_code);
                err_font(err_str).draw(20, Arg::leftCenter(LEFT_LEFT, Y_CENTER), Palette::White);
            }
        }
    }

    void draw() const override {
        Scene::SetBackground(Color(36, 153, 114));
    }
};