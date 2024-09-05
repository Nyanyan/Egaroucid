/*
    Egaroucid Project

    @file silent_load.hpp
        Load before GUI wake up
    @date 2021-2024
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
    std::cerr << "document_dir " << directories->document_dir << " appdata_dir " << directories->appdata_dir << std::endl;

    // file directories
    directories->eval_file = "resources/eval.egev2";
    directories->eval_mo_end_file = "resources/eval_move_ordering_end.egev";
}

int init_resources(Resources* resources, Settings* settings, Fonts *fonts, bool *stop_loading) {
    // language json
    if (!language_name.init(resources->language_names)) {
        return ERR_LANG_JSON_NOT_LOADED;
    }

    if (*stop_loading){
        return ERR_TERMINATED;
    }

    // language
    std::string lang_file = "resources/languages/" + settings->lang_name + ".json";
    if (!language.init(lang_file)) {
        std::cerr << "language file not found. use alternative language" << std::endl;
        settings->lang_name = DEFAULT_LANGUAGE;
        lang_file = "resources/languages/" + settings->lang_name + ".json";
        if (!language.init(lang_file))
            return ERR_LANG_NOT_LOADED;
    }

    if (*stop_loading){
        return ERR_TERMINATED;
    }

    fonts->init(settings->lang_name);

    if (*stop_loading){
        return ERR_TERMINATED;
    }

    // textures
    Texture icon(U"resources/img/icon.png", TextureDesc::Mipped);
    Texture logo(U"resources/img/logo.png", TextureDesc::Mipped);
    Texture checkbox(U"resources/img/checked.png", TextureDesc::Mipped);
    Texture unchecked(U"resources/img/unchecked.png", TextureDesc::Mipped);
    Texture laser_pointer(U"resources/img/laser_pointer.png", TextureDesc::Mipped);

    if (*stop_loading){
        return ERR_TERMINATED;
    }

    std::vector<Texture> lang_img;
    for (int i = 0; i < (int)resources->language_names.size() && !(*stop_loading); ++i) {
        Texture limg(U"resources/languages/" +  Unicode::Widen(resources->language_names[i]) + U".png", TextureDesc::Mipped);
        if (limg.isEmpty()) {
            return ERR_TEXTURE_NOT_LOADED;
        }
        lang_img.emplace_back(limg);
    }

    if (*stop_loading){
        return ERR_TERMINATED;
    }

    if (icon.isEmpty() || logo.isEmpty() || checkbox.isEmpty() || unchecked.isEmpty()) {
        return ERR_TEXTURE_NOT_LOADED;
    }
    resources->icon = icon;
    resources->logo = logo;
    resources->checkbox = checkbox;
    resources->unchecked = unchecked;
    resources->laser_pointer = laser_pointer;
    resources->lang_img = lang_img;

    // opening
    if (!opening_init(settings->lang_name)) {
        std::cerr << "opening file not found. use alternative opening file" << std::endl;
        if (!opening_init(DEFAULT_OPENING_LANG_NAME))
            return ERR_OPENING_NOT_LOADED;
    }

    // license
    TextReader reader{U"LICENSE"};
    if (not reader) {
        return ERR_LICENSE_FILE_NOT_LOADED;
    }
    String copyright = Unicode::Widen("(C) " + (std::string)EGAROUCID_DATE + " " + (std::string)EGAROUCID_AUTHOR);
    String license = reader.readAll();
    LicenseManager::AddLicense({
        .title = U"Egaroucid",
        .copyright = copyright,
        .text = license
    });

    return ERR_OK;

}

int silent_load(Directories* directories, Resources* resources, Settings* settings, Fonts *fonts, bool *stop_loading) {
    init_directories(directories);
    return init_resources(resources, settings, fonts, stop_loading);
}

class Silent_load : public App::Scene {
private:
    bool loading;
    bool loaded;
    bool stop_loading;
    std::future<int> silent_load_future;
public:
    Silent_load(const InitData& init) : IScene{ init } {
        stop_loading = false;
        silent_load_future = std::async(std::launch::async, silent_load, &getData().directories, &getData().resources, &getData().settings, &getData().fonts, &stop_loading);
        loading = true;
        loaded = false;
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            stop_loading = true;
            silent_load_future.get();
            System::Exit();
        }
        if (loading){
            if (silent_load_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                int load_code = silent_load_future.get();
                loaded = load_code == ERR_OK;
                loading = false;
            }
        } else{
            if (loaded){
                std::cerr << "silent loaded" << std::endl;
                changeScene(U"Load", SCENE_FADE_TIME);
            } else{
                getData().fonts.font(U"BASIC DATA NOT LOADED. PLEASE RE-INSTALL.").draw(30, LEFT_LEFT, Y_CENTER + 50, getData().colors.white);
            }
        }
    }

    void draw() const override {
        Scene::SetBackground(Color(36, 153, 114));
    }
};