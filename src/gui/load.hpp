/*
    Egaroucid Project

    @file load.hpp
        Loading scene
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include <windows.h>
#include <shlwapi.h>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"

std::string get_default_language(){
    std::string default_language = System::DefaultLanguage().narrow();
    std::string res = "english";
    if (default_language == "ja-JP") // japanese
        res = "japanese";
    if (default_language == "zh-CN" || default_language == "zh-cmn-Hans") // chinese
        res = "chinese";
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
    settings->book_acc_level = 0;
    settings->pause_when_pass = false;
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
}

int init_settings_import_int(JSON &json, String key, int* res) {
    if (json[key].getType() != JSONValueType::Number)
        return ERR_IMPORT_SETTINGS;
    *res = (int)json[key].get<double>();
    return ERR_OK;
}

int init_settings_import_bool(JSON &json, String key, bool* res) {
    if (json[key].getType() != JSONValueType::Bool)
        return ERR_IMPORT_SETTINGS;
    *res = json[key].get<bool>();
    return ERR_OK;
}

int init_settings_import_str(JSON &json, String key, std::string* res) {
    if (json[key].getType() != JSONValueType::String)
        return ERR_IMPORT_SETTINGS;
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
    }
    else {
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
    }
    else {
        return ERR_IMPORT_SETTINGS;
    }
}

int init_settings_import_str(TextReader* reader, std::string* res) {
    String line;
    if (reader->readLine(line)) {
        *res = line.narrow();
        return ERR_OK;
    }
    else {
        return ERR_IMPORT_SETTINGS;
    }
}

void import_text_settings(const Directories* directories, const Resources* resources, Settings* settings){
    TextReader reader(U"{}setting.txt"_fmt(Unicode::Widen(directories->appdata_dir)));
    if (!reader) {
        std::cerr << "err-1" << std::endl;
        return;
    }
    else {
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
            } else
                settings->hash_level = std::max(settings->hash_level, DEFAULT_HASH_LEVEL);
        #endif
        if (init_settings_import_int(&reader, &settings->book_acc_level) != ERR_OK) {
            std::cerr << "err23" << std::endl;
            return;
        }
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
    if (setting_json.size() == 0){
        std::cerr << "json not found, try legacy txt settings" << std::endl;
        import_text_settings(directories, resources, settings);
        return;
    }
    if (init_settings_import_int(setting_json, U"n_threads", &settings->n_threads) != ERR_OK){
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
        } else
            settings->hash_level = std::max(settings->hash_level, DEFAULT_HASH_LEVEL);
    #endif
    if (init_settings_import_int(setting_json, U"book_acc_level", &settings->book_acc_level) != ERR_OK) {
        std::cerr << "err23" << std::endl;
    }
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
}

void init_shortcut_keys(const Directories* directories){
    String file = U"{}shortcut_key.json"_fmt(Unicode::Widen(directories->appdata_dir));
    shortcut_keys.init(file);
}

int init_ai(Settings* settings, const Directories* directories, bool *stop_loading) {
    thread_pool.resize(settings->n_threads - 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cerr << "there are " << thread_pool.size() << " additional threads" << std::endl;
    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    endsearch_init();
    #if USE_MPC_PRE_CALCULATION
        mpc_init();
    #endif
    MEMORYSTATUSEX msex = { sizeof(MEMORYSTATUSEX) };
    GlobalMemoryStatusEx( &msex );
    double free_mb = (double)msex.ullAvailPhys / 1024 / 1024;
    double size_mb = (double)sizeof(Hash_node) / 1024 / 1024 * hash_sizes[MAX_HASH_LEVEL];
    std::cerr << "memory " << free_mb << " " << size_mb << std::endl;
    while (free_mb <= size_mb && MAX_HASH_LEVEL > 26){
        --MAX_HASH_LEVEL;
        size_mb = (double)sizeof(Hash_node) / 1024 / 1024 * hash_sizes[MAX_HASH_LEVEL];
    }
    settings->hash_level = std::min(settings->hash_level, MAX_HASH_LEVEL);
    std::cerr << "max hash level " << MAX_HASH_LEVEL << std::endl;
    #if USE_CHANGEABLE_HASH_LEVEL
        if (!hash_resize(DEFAULT_HASH_LEVEL, settings->hash_level, true)) {
            std::cerr << "hash resize failed. use default setting" << std::endl;
            settings->hash_level = DEFAULT_HASH_LEVEL;
        }
    #else
        hash_tt_init(true);
    #endif
    stability_init();
    if (!evaluate_init(directories->eval_file, directories->eval_mo_end_file, true)) {
        return ERR_EVAL_FILE_NOT_IMPORTED;
    }
    if (!book_init(settings->book_file, true, stop_loading)) {
        return ERR_BOOK_FILE_NOT_IMPORTED;
    }
    std::string ext = get_extension(settings->book_file);
    if (ext == "egbk"){
        settings->book_file += "2"; // force book version 3
        book.save_egbk3(settings->book_file, settings->book_file + ".bak");
    } else if (ext == "egbk2"){
        settings->book_file[settings->book_file.size() - 1] = '3'; // force book version 3
        book.save_egbk3(settings->book_file, settings->book_file + ".bak");
    }
    return ERR_OK;
}

int check_update(const Directories* directories, String *new_version) {
    const FilePath version_save_path = U"{}version.txt"_fmt(Unicode::Widen(directories->appdata_dir));
    AsyncHTTPTask task = SimpleHTTP::SaveAsync(VERSION_URL, version_save_path);
    uint64_t strt = tim();
    while (tim() - strt < 1000){ // timeout 1000 ms
        if (task.isReady()){
            if (task.getResponse().isOK()){
                TextReader reader(version_save_path);
                if (reader) {
                    reader.readLine(*new_version);
                    if (EGAROUCID_NUM_VERSION != *new_version) { // new version found
                        return UPDATE_CHECK_UPDATE_FOUND;
                    }
                }
            }
        }
    }
    if (task.getStatus() == HTTPAsyncStatus::Downloading){ // cancel task
        task.cancel();
    }
    return UPDATE_CHECK_ALREADY_UPDATED;
}

int load_app(Directories* directories, Resources* resources, Settings* settings, bool* update_found, String *new_version, bool *stop_loading) {
    init_settings(directories, resources, settings);
    init_shortcut_keys(directories);
    if (settings->auto_update_check) {
        if (check_update(directories, new_version) == UPDATE_CHECK_UPDATE_FOUND) {
            *update_found = true;
        }
    }
    return init_ai(settings, directories, stop_loading);
}



class Load : public App::Scene {
private:
    bool load_failed;
    bool book_failed;
    String tips;
    bool update_found;
    std::future<int> load_future;
    Button skip_button;
    Button update_button;
    Button book_ignore_button;
    String new_version;
    bool stop_loading;

public:
    Load(const InitData& init) : IScene{ init } {
        skip_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("help", "skip"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        update_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("help", "download"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        book_ignore_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("loading", "launch"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        load_failed = false;
        book_failed = false;
        tips = language.get_random("tips", "tips");
        update_found = false;
        stop_loading = false;
        load_future = std::async(std::launch::async, load_app, &getData().directories, &getData().resources, &getData().settings, &update_found, &new_version, &stop_loading);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            stop_loading = true;
            load_future.get();
            changeScene(U"Close", SCENE_FADE_TIME);
            return;
        }
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
            if (skip_button.clicked() || KeyEscape.down()) {
                update_found = false;
            }
            if (update_button.clicked() || KeyEnter.down()) {
                if (language.get("lang_name") == U"日本語") {
                    System::LaunchBrowser(U"https://www.egaroucid.nyanyan.dev/ja/download/");
                }
                else {
                    System::LaunchBrowser(U"https://www.egaroucid.nyanyan.dev/en/download/");
                }
                changeScene(U"Close", SCENE_FADE_TIME);
                return;
            }
        }
        else {
            const int icon_width = (LEFT_RIGHT - LEFT_LEFT);
            getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(LEFT_LEFT, Y_CENTER - icon_width / 2);
            getData().resources.logo.scaled((double)icon_width * 0.8 / getData().resources.logo.width()).draw(RIGHT_LEFT, Y_CENTER - 40);
            if (load_future.valid()) {
                if (load_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    int load_code = load_future.get();
                    if (load_code == ERR_OK) {
                        std::cerr << "loaded" << std::endl;
                        getData().menu_elements.init(&getData().settings, &getData().resources);
                        getData().window_state.loading = false;
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
                        std::cerr << "loaded" << std::endl;
                        getData().menu_elements.init(&getData().settings, &getData().resources);
                        getData().window_state.loading = false;
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