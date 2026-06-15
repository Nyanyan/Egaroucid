/*
    Egaroucid Project

    @file data_migration_scene.hpp
        Data migration scenes
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <future>
#include <iostream>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"

inline String data_migration_input_path(const TextAreaEditState& text_area) {
    return data_migration_slash_path(text_area.text.replaced(U"\r", U"").replaced(U"\n", U"").trimmed());
}

inline bool data_migration_return_pressed(const TextAreaEditState& text_area) {
    return text_area.text.size() && text_area.text[text_area.text.size() - 1] == U'\n';
}

inline String data_migration_message_from_result(const Data_migration_result& result) {
    if (result.succeeded) {
        return language.get("data_migration", "complete");
    }
    switch (result.error) {
        case Data_migration_error::invalid_destination:
            return language.get("data_migration", "directory_not_found");
        case Data_migration_error::invalid_source:
            return language.get("data_migration", "invalid_backup");
        case Data_migration_error::unsupported_zip:
            return language.get("data_migration", "zip_not_supported");
        case Data_migration_error::unsafe_source:
            return language.get("data_migration", "unsafe_source");
        case Data_migration_error::copy_failed:
            return language.get("data_migration", "failed");
        case Data_migration_error::none:
        default:
            return language.get("data_migration", "failed");
    }
}

inline bool data_migration_backup_folder_valid(const String& path) {
    if (!FileSystem::IsDirectory(path)) {
        return false;
    }
    return FileSystem::IsDirectory(data_migration_join_path(path, U"appdata")) &&
        FileSystem::IsDirectory(data_migration_join_path(path, U"document"));
}

inline void reload_egaroucid_settings_after_data_import(Common_resources* data) {
    init_settings(&data->directories, &data->resources, &data->settings);
    init_user_settings(&data->settings, &data->user_settings);

    std::string lang_file = EXE_DIRECTORY_PATH + "resources/languages/" + data->settings.lang_name + ".json";
    if (!language.init(lang_file)) {
        data->settings.lang_name = DEFAULT_LANGUAGE;
        lang_file = EXE_DIRECTORY_PATH + "resources/languages/" + data->settings.lang_name + ".json";
        language.init(lang_file);
    }
    data->fonts.init(data->settings.lang_name);
    if (!opening_init(data->settings.lang_name)) {
        opening_init(DEFAULT_OPENING_LANG_NAME);
    }

    shortcut_keys.init(U"{}shortcut_key.json"_fmt(Unicode::Widen(data->directories.appdata_dir)), &data->directories);
    shortcut_buttons.init(U"{}shortcut_button.json"_fmt(Unicode::Widen(data->directories.appdata_dir)));
    mouse_additional_buttons.init(U"{}mouse_additional_button.json"_fmt(Unicode::Widen(data->directories.appdata_dir)));

    data->menu_elements.init(&data->settings, &data->resources);
    data->menu = create_menu(&data->menu_elements, &data->resources, data->fonts.font, data->settings.lang_name);
    data->book_information.changed = false;
    data->graph_resources.need_init = false;
}

class Export_settings_data : public App::Scene {
private:
    Button back_button;
    Button default_button;
    Button export_button;
    TextAreaEditState text_area;
    std::future<Data_migration_result> export_future;
    Data_migration_result result;
    bool exporting{ false };
    bool finished{ false };

    void init_buttons() {
        back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        default_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "use_default"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        export_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("data_migration", "export"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
    }

public:
    Export_settings_data(const InitData& init) : IScene{ init } {
        set_scene_ime_enabled(true);
        init_buttons();
        text_area.text = Unicode::Widen(getData().directories.document_dir);
        text_area.cursorPos = text_area.text.size();
        text_area.rebuildGlyphs();
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
            return;
        }

        Scene::SetBackground(getData().colors.green);
        const int icon_width = SCENE_ICON_WIDTH;
        getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
        getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
        const int sy = 20 + icon_width + 40;

        if (exporting) {
            getData().fonts.font(language.get("data_migration", "exporting")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            if (export_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                result = export_future.get();
                exporting = false;
                finished = true;
            }
        } else if (finished) {
            getData().fonts.font(data_migration_message_from_result(result)).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            if (!result.path.isEmpty()) {
                getData().fonts.font(result.path).draw(14, Arg::topCenter(X_CENTER, sy + 45), getData().colors.white);
            }
            back_button.draw();
            if (back_button.clicked() || gui_textarea_ime::escape_pressed_for_scene_change()) {
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        } else {
            getData().fonts.font(language.get("data_migration", "export_title")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            getData().fonts.font(language.get("data_migration", "export_description")).draw(14, Arg::topCenter(X_CENTER, sy + 38), getData().colors.white);

            text_area.active = true;
            text_area_with_ime_candidate_window(text_area, Vec2{ X_CENTER - 300, sy + 65 }, SizeF{ 600, 100 }, TEXTBOX_MAX_CHARS);
            if (DragDrop::HasNewFilePaths()) {
                text_area.text = DragDrop::GetDroppedFilePaths()[0].path;
                text_area.cursorPos = 0;
                text_area.scrollY = 0.0;
                text_area.textChanged = true;
            }

            const String destination_dir = data_migration_input_path(text_area);
            const bool valid_dir = FileSystem::IsDirectory(destination_dir);
            if (valid_dir) {
                export_button.enable();
            } else {
                export_button.disable();
                getData().fonts.font(language.get("data_migration", "directory_not_found")).draw(15, Arg::topCenter(X_CENTER, sy + 180), getData().colors.white);
            }

            back_button.draw();
            if (back_button.clicked() || gui_textarea_ime::escape_pressed_for_scene_change()) {
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
            default_button.draw();
            if (default_button.clicked()) {
                text_area.text = Unicode::Widen(getData().directories.document_dir);
                text_area.cursorPos = text_area.text.size();
                text_area.scrollY = 0.0;
                text_area.rebuildGlyphs();
            }
            export_button.draw();
            if (valid_dir && (export_button.clicked() || data_migration_return_pressed(text_area))) {
                export_future = std::async(std::launch::async, export_egaroucid_settings_data, getData().directories, destination_dir);
                exporting = true;
            }
        }
    }

    void draw() const override {
    }
};

class Import_settings_data : public App::Scene {
private:
    Button back_button;
    Button import_button;
    TextAreaEditState text_area;
    std::future<Data_migration_result> import_future;
    Data_migration_result result;
    bool importing{ false };
    bool finished{ false };

    void init_buttons() {
        back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        import_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("data_migration", "import"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
    }

public:
    Import_settings_data(const InitData& init) : IScene{ init } {
        set_scene_ime_enabled(true);
        init_buttons();
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
            return;
        }

        Scene::SetBackground(getData().colors.green);
        const int icon_width = SCENE_ICON_WIDTH;
        getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
        getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
        const int sy = 20 + icon_width + 40;

        if (importing) {
            getData().fonts.font(language.get("data_migration", "importing")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            if (import_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                result = import_future.get();
                if (result.succeeded) {
                    reload_egaroucid_settings_after_data_import(&getData());
                    init_buttons();
                }
                importing = false;
                finished = true;
            }
        } else if (finished) {
            getData().fonts.font(data_migration_message_from_result(result)).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            if (result.succeeded) {
                getData().fonts.font(language.get("data_migration", "restart_recommended")).draw(15, Arg::topCenter(X_CENTER, sy + 45), getData().colors.white);
            }
            back_button.draw();
            if (back_button.clicked() || gui_textarea_ime::escape_pressed_for_scene_change()) {
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        } else {
            getData().fonts.font(language.get("data_migration", "import_title")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            getData().fonts.font(language.get("data_migration", "import_description")).draw(14, Arg::topCenter(X_CENTER, sy + 38), getData().colors.white);
            getData().fonts.font(language.get("data_migration", "import_warning")).draw(13, Arg::topCenter(X_CENTER, sy + 60), getData().colors.white);

            text_area.active = true;
            text_area_with_ime_candidate_window(text_area, Vec2{ X_CENTER - 300, sy + 85 }, SizeF{ 600, 100 }, TEXTBOX_MAX_CHARS);
            bool path_dragged = false;
            if (DragDrop::HasNewFilePaths()) {
                text_area.text = DragDrop::GetDroppedFilePaths()[0].path;
                text_area.cursorPos = 0;
                text_area.scrollY = 0.0;
                text_area.textChanged = true;
                path_dragged = true;
            }

            const String backup_root = data_migration_input_path(text_area);
            const bool zip_selected = backup_root.lowercased().ends_with(U".zip");
            const bool valid_backup = data_migration_backup_folder_valid(backup_root);
            if (valid_backup) {
                import_button.enable();
            } else {
                import_button.disable();
                const String message = zip_selected ? language.get("data_migration", "zip_not_supported") : language.get("data_migration", "invalid_backup");
                getData().fonts.font(message).draw(15, Arg::topCenter(X_CENTER, sy + 200), getData().colors.white);
            }

            back_button.draw();
            if (back_button.clicked() || gui_textarea_ime::escape_pressed_for_scene_change()) {
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
            import_button.draw();
            if (valid_backup && (import_button.clicked() || data_migration_return_pressed(text_area) || path_dragged)) {
                import_future = std::async(std::launch::async, import_egaroucid_settings_data, getData().directories, backup_root);
                importing = true;
            }
        }
    }

    void draw() const override {
    }
};
