/*
    Egaroucid Project

    @file silent_load.hpp
        Load before GUI wake up
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"

void init_directories(Directories* directories) {
    // system directory
    #if GUI_PORTABLE_MODE
        directories->document_dir = "./document/";
        directories->appdata_dir = "./appdata/";
    #else
        directories->document_dir = FileSystem::GetFolderPath(SpecialFolder::Documents).narrow() + "Egaroucid/";
        directories->appdata_dir = FileSystem::GetFolderPath(SpecialFolder::LocalAppData).narrow() + "Egaroucid/";
    #endif
    std::cerr << "document_dir " << directories->document_dir << std::endl;
    std::cerr << "appdata_dir " << directories->appdata_dir << std::endl;

    // file directories
    directories->eval_file = "resources/eval.egev";
}

std::string get_default_language(){
    std::string default_language = System::DefaultLanguage().narrow();
    std::string res = "english";
    if (default_language == "ja-JP")
        res = "japanese";
    return res;
}

void init_default_settings(const Directories* directories, const Resources* resources, Settings* settings) {
    std::cerr << "use default settings" << std::endl;
    settings->n_threads = std::min(32, (int)std::thread::hardware_concurrency());
    settings->auto_update_check = 1;
    settings->lang_name = get_default_language();
    settings->book_file = directories->document_dir + "book.egbk2";
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
    settings->book_learn_error = 2;
    settings->show_stable_discs = false;
    settings->change_book_by_right_click = false;
    settings->show_last_move = true;
    settings->show_next_move = true;
    settings->hash_level = DEFAULT_HASH_LEVEL;
    settings->book_acc_level = 0;
    settings->pause_when_pass = false;
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

void init_settings(const Directories* directories, const Resources* resources, Settings* settings) {
    init_default_settings(directories, resources, settings);
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
        if (init_settings_import_int(&reader, &settings->book_learn_error) != ERR_OK) {
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
        if (init_settings_import_int(&reader, &settings->hash_level) != ERR_OK) {
            std::cerr << "err22" << std::endl;
            return;
        } else
            settings->hash_level = std::max(settings->hash_level, DEFAULT_HASH_LEVEL);
        if (init_settings_import_int(&reader, &settings->book_acc_level) != ERR_OK) {
            std::cerr << "err23" << std::endl;
            return;
        }
        if (init_settings_import_bool(&reader, &settings->pause_when_pass) != ERR_OK) {
            std::cerr << "err24" << std::endl;
            return;
        }
    }
}

int init_resources(Resources* resources, Settings* settings) {
    // language names
    std::ifstream ifs_lang("resources/languages/languages.txt");
    if (ifs_lang.fail()) {
        return ERR_LANG_LIST_NOT_LOADED;
    }
    std::string lang_line;
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
    std::string lang_file = "resources/languages/" + settings->lang_name + ".json";
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
    init_settings(directories, resources, settings);
    return init_resources(resources, settings);
}

class Silent_load : public App::Scene {
private:
    std::future<int> silent_load_future;
    bool silent_load_failed;

public:
    Silent_load(const InitData& init) : IScene{ init } {
        silent_load_future = std::async(std::launch::async, silent_load, &getData().directories, &getData().resources, &getData().settings);
        silent_load_failed = false;
        std::cerr << "start silent loading" << std::endl;
    }

    void update() override {
        if (silent_load_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            int load_code = silent_load_future.get();
            if (load_code == ERR_OK) {
                std::cerr << "silent loaded" << std::endl;
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