﻿/*
    Egaroucid Project

    @file book_scene.hpp
        Book expand / deepen
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"
#include "draw.hpp"

constexpr int BOOK_DEPTH_INF = 80;
constexpr int BOOK_ERROR_INF = 100000;

void reset_book_additional_information() {
    umigame.delete_all();
    book_accuracy.delete_all();
}

void delete_book() {
    book.delete_all();
    reset_book_additional_information();
}

bool import_book(std::string file) {
    std::cerr << "book import" << std::endl;
    bool result = book.import_book_extension_determination(file);
    reset_book_additional_information();
    return result;
}

bool import_book_with_level(std::string file, int level) {
    std::cerr << "book import with level " << level << std::endl;
    bool result = book.import_book_extension_determination(file, level);
    reset_book_additional_information();
    return result;
}

inline void draw_book_settings(Font &font, Color &color, int depth, int error_per_move, int error_sum, int error_leaf) {
    String depth_str = Format(depth);
    if (depth == BOOK_DEPTH_INF) {
        depth_str = language.get("book", "unlimited");
    }
    font(language.get("book", "depth") + U": " + depth_str).draw(15, 480, 260, color);
    String error_per_move_str = Format(error_per_move);
    if (error_per_move == BOOK_ERROR_INF) {
        error_per_move_str = language.get("book", "unlimited");
    }
    font(language.get("book", "error_per_move") + U": " + error_per_move_str).draw(15, 480, 280, color);
    String error_sum_str = Format(error_sum);
    if (error_sum == BOOK_ERROR_INF) {
        error_sum_str = language.get("book", "unlimited");
    }
    font(language.get("book", "error_sum") + U": " + error_sum_str).draw(15, 480, 300, color);
    String error_leaf_str = Format(error_leaf);
    if (error_leaf == BOOK_ERROR_INF) {
        error_leaf_str = language.get("book", "unlimited");
    }
    font(language.get("book", "error_leaf") + U": " + error_leaf_str).draw(15, 480, 320, color);
}

inline void draw_book_settings(Font &font, Color &color, int depth, int error_per_move, int error_sum) {
    String depth_str = Format(depth);
    if (depth == BOOK_DEPTH_INF) {
        depth_str = language.get("book", "unlimited");
    }
    font(language.get("book", "depth") + U": " + depth_str).draw(15, 480, 270, color);
    String error_per_move_str = Format(error_per_move);
    if (error_per_move == BOOK_ERROR_INF) {
        error_per_move_str = language.get("book", "unlimited");
    }
    font(language.get("book", "error_per_move") + U": " + error_per_move_str).draw(15, 480, 290, color);
    String error_sum_str = Format(error_sum);
    if (error_sum == BOOK_ERROR_INF) {
        error_sum_str = language.get("book", "unlimited");
    }
    font(language.get("book", "error_sum") + U": " + error_sum_str).draw(15, 480, 310, color);
}

class Import_book : public App::Scene {
private:
    std::future<bool> import_book_future;
    std::future<void> delete_book_future;
    Button single_back_button;
    Button back_button;
    Button go_button;
    Slidebar level_bar;
    std::string book_file;
    bool book_deleting;
    bool book_importing;
    bool failed;
    bool done;
    int level;
    bool need_level;
    TextAreaEditState text_area;

public:
    Import_book(const InitData& init) : IScene{ init } {
        single_back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        go_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("book", "import"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        book_deleting = false;
        book_importing = false;
        failed = false;
        done = false;
        need_level = false;
        level = getData().menu_elements.level;
        level_bar.init(X_CENTER - 220, 355, 440, 20, language.get("ai_settings", "level"), 20, getData().colors.white, getData().fonts.font, 1, 60, &level);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        const int icon_width = SCENE_ICON_WIDTH;
        getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
        getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
        int sy = 20 + icon_width + 40;
        if (!book_deleting && !book_importing && !failed && !done) {
            getData().fonts.font(language.get("book", "import_book")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            getData().fonts.font(language.get("book", "input_book_path")).draw(14, Arg::topCenter(X_CENTER, sy + 38), getData().colors.white);
            text_area.active = true;
            SimpleGUI::TextArea(text_area, Vec2{X_CENTER - 300, sy + 60}, SizeF{600, 100}, TEXTBOX_MAX_CHARS);
            bool return_pressed = false;
            if (text_area.text.size()) {
                if (text_area.text[text_area.text.size() - 1] == '\n') {
                    return_pressed = true;
                }
            }
            bool file_dragged = false;
            if (DragDrop::HasNewFilePaths()) {
                text_area.text = DragDrop::GetDroppedFilePaths()[0].path;
                text_area.cursorPos = 0;
                text_area.scrollY = 0.0;
                text_area.textChanged = true;
                file_dragged = true;
            }
            book_file = text_area.text.replaced(U"\r", U"").replaced(U"\n", U"").narrow();
            std::string ext = get_extension(book_file);
            bool formatted_file = false;
            if (ext == BOOK_EXTENSION_NODOT || ext == "egbk2" || ext == "egbk" || ext == "dat") {
                go_button.enable();
                formatted_file = true;
            } else {
                go_button.disable();
                getData().fonts.font(language.get("book", "wrong_extension") + U" " + language.get("book", "legal_extension3")).draw(15, Arg::topCenter(X_CENTER, sy + 170), getData().colors.white);
            }
            bool need_level_setting = ext == "egbk";
            if (need_level_setting) {
                level_bar.draw();
            }
            if (level_bar.is_changeable()) {
                back_button.disable_notransparent();
            } else {
                back_button.enable();
            }
            back_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                reset_book_additional_information();
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
            if (level_bar.is_changeable()) {
                go_button.disable_notransparent();
            } else {
                go_button.enable();
            }
            go_button.draw();
            if (formatted_file && (go_button.clicked() || return_pressed || (file_dragged && !need_level_setting))) {
                getData().book_information.changed = true;
                need_level = need_level_setting;
                delete_book_future = std::async(std::launch::async, delete_book);
                book_deleting = true;
            }
        } else if (book_deleting || book_importing) {
            getData().fonts.font(language.get("book", "loading")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            if (book_deleting) {
                if (delete_book_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    delete_book_future.get();
                    book_deleting = false;
                    if (need_level) {
                        import_book_future = std::async(std::launch::async, import_book_with_level, book_file, level);
                    } else {
                        import_book_future = std::async(std::launch::async, import_book, book_file);
                    }
                    book_importing = true;
                }
            } else if (book_importing) {
                if (import_book_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    failed = !import_book_future.get();
                    book_importing = false;
                    done = true;
                }
            }
        } else if (done) {
            if (failed) {
                getData().fonts.font(language.get("book", "import_failed")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
                single_back_button.draw();
                if (single_back_button.clicked() || KeyEscape.pressed()) {
                    reset_book_additional_information();
                    getData().graph_resources.need_init = false;
                    changeScene(U"Main_scene", SCENE_FADE_TIME);
                }
            } else {
                reset_book_additional_information();
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        }
    }

    void draw() const override {

    }
};

class Export_book: public App::Scene {
private:
    Button back_button;
    Button go_with_level_button;
    Button go_button;
    Slidebar level_bar;
    std::string book_file;
    int level;
    std::future<void> save_book_edax_future;
    bool book_exporting;
    bool done;
    TextAreaEditState text_area;

public:
    Export_book(const InitData& init) : IScene{ init } {
        back_button.init(BUTTON3_1_SX, GO_BACK_BUTTON_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        go_with_level_button.init(BUTTON3_2_SX, GO_BACK_BUTTON_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("book", "export_with_specified_level"), 18, getData().fonts.font, getData().colors.white, getData().colors.black);
        go_button.init(BUTTON3_3_SX, GO_BACK_BUTTON_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("book", "export"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        text_area.text = Unicode::Widen(getData().directories.document_dir + "book_copy.egbk3");
        text_area.cursorPos = text_area.text.size();
        text_area.rebuildGlyphs();
        book_exporting = false;
        done = false;
        level = getData().menu_elements.level;
        level_bar.init(X_CENTER - 220, 355, 440, 20, language.get("ai_settings", "level"), 20, getData().colors.white, getData().fonts.font, 1, 60, &level);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        const int icon_width = SCENE_ICON_WIDTH;
        getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
        getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
        int sy = 20 + icon_width + 40;
        if (!book_exporting) {
            getData().fonts.font(language.get("book", "export_book")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            text_area.active = true;
            SimpleGUI::TextArea(text_area, Vec2{X_CENTER - 300, sy + 40}, SizeF{600, 100}, TEXTBOX_MAX_CHARS);
            bool return_pressed = false;
            if (text_area.text.size()) {
                if (text_area.text[text_area.text.size() - 1] == '\n') {
                    return_pressed = true;
                }
            }
            book_file = text_area.text.replaced(U"\r", U"").replaced(U"\n", U"").narrow();
            std::string ext = get_extension(book_file);
            bool button_enabled = false;
            String book_format_str;
            if (ext == BOOK_EXTENSION_NODOT) {
                book_format_str = language.get("book", "egaroucid_format");
                go_button.enable();
                go_with_level_button.enable();
                button_enabled = true;
            } else if (ext == "dat") {
                book_format_str = language.get("book", "edax_format");
                go_button.enable();
                go_with_level_button.enable();
                button_enabled = true;
            } else {
                book_format_str = language.get("book", "undefined_format");
                go_button.disable();
                go_with_level_button.disable();
            }
            getData().fonts.font(book_format_str).draw(18, Arg::topCenter(X_CENTER, sy + 142), getData().colors.white);

            level_bar.draw();

            if (level_bar.is_changeable()) {
                back_button.disable_notransparent();
                go_with_level_button.disable_notransparent();
                go_button.disable_notransparent();
            } else {
                back_button.enable();
                go_with_level_button.enable();
                go_button.enable();
            }
            back_button.draw();
            go_with_level_button.draw();
            go_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
            if (go_with_level_button.clicked()) {
                if (ext == BOOK_EXTENSION_NODOT) {
                    save_book_edax_future = std::async(std::launch::async, book_save_as_egaroucid, book_file, level);
                } else if (ext == "dat") {
                    save_book_edax_future = std::async(std::launch::async, book_save_as_edax, book_file, level);
                }
                book_exporting = true;
            } else if (go_button.clicked() || (return_pressed && button_enabled)) {
                if (ext == BOOK_EXTENSION_NODOT) {
                    save_book_edax_future = std::async(std::launch::async, book_save_as_egaroucid, book_file, LEVEL_UNDEFINED);
                } else if (ext == "dat") {
                    save_book_edax_future = std::async(std::launch::async, book_save_as_edax, book_file, LEVEL_UNDEFINED);
                }
                book_exporting = true;
            }
        } else {
            getData().fonts.font(language.get("book", "exporting")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            if (save_book_edax_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                save_book_edax_future.get();
                done = true;
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        }
    }

    void draw() const override {

    }
};

class Merge_book : public App::Scene {
private:
    std::future<bool> import_book_future;
    Button back_button;
    Button go_button;
    std::string book_file;
    bool importing;
    bool imported;
    bool failed;
    TextAreaEditState text_area;

public:
    Merge_book(const InitData& init) : IScene{ init } {
        back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        go_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("book", "merge"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        importing = false;
        imported = false;
        failed = false;
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        const int icon_width = SCENE_ICON_WIDTH;
        getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
        getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
        int sy = 20 + icon_width + 40;
        if (!importing) {
            getData().fonts.font(language.get("book", "book_merge")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            getData().fonts.font(language.get("book", "input_book_path")).draw(15, Arg::topCenter(X_CENTER, sy + 50), getData().colors.white);
            text_area.active = true;
            SimpleGUI::TextArea(text_area, Vec2{X_CENTER - 300, sy + 80}, SizeF{600, 100}, TEXTBOX_MAX_CHARS);
            bool return_pressed = false;
            if (text_area.text.size()) {
                if (text_area.text[text_area.text.size() - 1] == '\n') {
                    return_pressed = true;
                }
            }
            bool file_dragged = false;
            if (DragDrop::HasNewFilePaths()) {
                text_area.text = DragDrop::GetDroppedFilePaths()[0].path;
                text_area.cursorPos = 0;
                text_area.scrollY = 0.0;
                text_area.textChanged = true;
                file_dragged = true;
            }
            book_file = text_area.text.replaced(U"\r", U"").replaced(U"\n", U"").narrow();
            std::string ext = get_extension(book_file);
            bool formatted_file = false;
            if (ext == BOOK_EXTENSION_NODOT || ext == "dat") {
                go_button.enable();
                formatted_file = true;
            } else {
                go_button.disable();
                getData().fonts.font(language.get("book", "wrong_extension") + U" " + language.get("book", "legal_extension2")).draw(15, Arg::topCenter(X_CENTER, sy + 190), getData().colors.white);
            }
            back_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                reset_book_additional_information();
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
            go_button.draw();
            if (formatted_file && (go_button.clicked() || KeyEnter.pressed() || file_dragged)) {
                import_book_future = std::async(std::launch::async, import_book, book_file);
                importing = true;
            }
        } else if (!imported) {
            getData().fonts.font(language.get("book", "loading")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            if (import_book_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                failed = !import_book_future.get();
                imported = true;
            }
        } else {
            if (failed) {
                getData().fonts.font(language.get("book", "import_failed")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
                back_button.draw();
                if (back_button.clicked() || KeyEscape.pressed()) {
                    reset_book_additional_information();
                    getData().graph_resources.need_init = false;
                    changeScene(U"Main_scene", SCENE_FADE_TIME);
                }
            } else {
                reset_book_additional_information();
                getData().book_information.changed = true;
                getData().graph_resources.need_init = false;
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
    std::string book_file;
    std::future<void> delete_book_future;
    std::future<bool> import_book_future;
    bool book_deleting;
    bool book_importing;
    bool failed;
    bool done;
    TextAreaEditState text_area;

public:
    Refer_book(const InitData& init) : IScene{ init } {
        single_back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        default_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("book", "use_default"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        go_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("book", "import"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        text_area.text = Unicode::Widen(getData().settings.book_file);
        text_area.cursorPos = text_area.text.size();
        text_area.rebuildGlyphs();
        book_deleting = false;
        book_importing = false;
        failed = false;
        done = false;
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        const int icon_width = SCENE_ICON_WIDTH;
        getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
        getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
        int sy = 20 + icon_width + 40;
        if (!book_deleting && !book_importing && !failed && !done) {
            getData().fonts.font(language.get("book", "book_reference")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            getData().fonts.font(language.get("book", "input_book_path")).draw(15, Arg::topCenter(X_CENTER, sy + 50), getData().colors.white);
            text_area.active = true;
            SimpleGUI::TextArea(text_area, Vec2{X_CENTER - 300, sy + 80}, SizeF{600, 100}, TEXTBOX_MAX_CHARS);
            bool return_pressed = false;
            if (text_area.text.size()) {
                if (text_area.text[text_area.text.size() - 1] == '\n') {
                    return_pressed = true;
                }
            }
            bool file_dragged = false;
            if (DragDrop::HasNewFilePaths()) {
                text_area.text = DragDrop::GetDroppedFilePaths()[0].path;
                text_area.cursorPos = 0;
                text_area.scrollY = 0.0;
                text_area.textChanged = true;
                file_dragged = true;
            }
            book_file = text_area.text.replaced(U"\r", U"").replaced(U"\n", U"").narrow();
            std::string ext = get_extension(book_file);
            bool formatted_book = false;
            if (ext == BOOK_EXTENSION_NODOT) {
                go_button.enable();
                formatted_book = true;
            } else {
                go_button.disable();
                getData().fonts.font(language.get("book", "wrong_extension") + U" " + language.get("book", "legal_extension1")).draw(15, Arg::topCenter(X_CENTER, sy + 190), getData().colors.white);
            }
            back_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                reset_book_additional_information();
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
            default_button.draw();
            if (default_button.clicked()) {
                text_area.text = Unicode::Widen(getData().directories.document_dir + "book" + BOOK_EXTENSION);
                text_area.cursorPos = text_area.text.size();
                text_area.scrollY = 0.0;
                text_area.rebuildGlyphs();
            }
            go_button.draw();
            if (formatted_book && (go_button.clicked() || return_pressed || file_dragged)) {
                //getData().book_information.changed = true;
                getData().settings.book_file = book_file;
                std::cerr << "book reference changed to " << book_file << std::endl;
                delete_book_future = std::async(std::launch::async, delete_book);
                book_deleting = true;
            }
        } else if (book_deleting || book_importing) {
            getData().fonts.font(language.get("book", "loading")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            if (book_deleting) {
                if (delete_book_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    delete_book_future.get();
                    book_deleting = false;
                    import_book_future = std::async(std::launch::async, import_book, getData().settings.book_file);
                    book_importing = true;
                }
            } else if (book_importing) {
                if (import_book_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    failed = !import_book_future.get();
                    //if (getData().settings.book_file.size() < 6 || getData().settings.book_file.substr(getData().settings.book_file.size() - 6, 6) != BOOK_EXTENSION)
                    //    getData().settings.book_file += BOOK_EXTENSION;
                    book_importing = false;
                    done = true;
                }
            }
        } else if (done) {
            if (failed) {
                getData().fonts.font(language.get("book", "import_failed")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
                single_back_button.draw();
                if (single_back_button.clicked() || KeyEscape.pressed()) {
                    reset_book_additional_information();
                    getData().graph_resources.need_init = false;
                    changeScene(U"Main_scene", SCENE_FADE_TIME);
                }
            } else {
                reset_book_additional_information();
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        }
    }

    void draw() const override {

    }
};





class Enhance_book : public App::Scene {
private:
    Button start_button;
    Button stop_button;
    Button back_button;
    History_elem history_elem;
    bool book_learning;
    bool done;
    bool before_start;
    std::future<void> book_learn_future;
    Board root_board;
    int depth;
    int error_per_move;
    int error_sum;
    int error_leaf;

public:
    Enhance_book(const InitData& init) : IScene{ init } {
        start_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY - 65, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("book", "start"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        stop_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("book", "stop_learn"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        root_board = getData().history_elem.board;
        history_elem = getData().history_elem;
        history_elem.policy = -1;
        book_learning = false;
        done = false;
        before_start = true;
        depth = getData().menu_elements.book_learn_depth;
        if (!getData().menu_elements.use_book_learn_depth) {
            depth = BOOK_DEPTH_INF;
        }
        error_per_move = getData().menu_elements.book_learn_error_per_move;
        if (!getData().menu_elements.use_book_learn_error_per_move) {
            error_per_move = BOOK_ERROR_INF;
        }
        error_sum = getData().menu_elements.book_learn_error_sum;
        if (!getData().menu_elements.use_book_learn_error_sum) {
            error_sum = BOOK_ERROR_INF;
        }
        error_leaf = getData().menu_elements.book_learn_error_leaf;
        if (!getData().menu_elements.use_book_learn_error_leaf) {
            error_leaf = BOOK_ERROR_INF;
        }
    }

    void update() override {
        if ((before_start || done) && (System::GetUserActions() & UserAction::CloseButtonClicked)) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        draw_board(getData().fonts, getData().colors, history_elem);
        draw_info(getData().colors, history_elem, getData().fonts, getData().menu_elements, false, "");
        getData().fonts.font(language.get("book", "book_deviate")).draw(25, 480, 200, getData().colors.white);
        draw_book_settings(getData().fonts.font, getData().colors.white, depth, error_per_move, error_sum, error_leaf);
        if (book_learning) {
            getData().fonts.font(language.get("book", "learning")).draw(20, 480, 230, getData().colors.white);
            stop_button.draw();
            if (stop_button.clicked()) {
                global_searching = false;
                book_learning = false;
            }
        } else if (before_start) {
            start_button.draw();
            if (start_button.clicked()) {
                before_start = false;
                book_learning = true;
                int fork_n_discs = -1;
                if (getData().graph_resources.branch == 1) {
                    fork_n_discs = getData().graph_resources.nodes[1][0].board.n_discs();
                }
                int registered_value = 0; // from black
                for (History_elem history_elem: getData().graph_resources.nodes[0]) {
                    if (history_elem.board.n_discs() == fork_n_discs) {
                        break;
                    }
                    if (history_elem.board.n_discs() >= root_board.n_discs()) {
                        break;
                    }
                    if (book.contain(&history_elem.board)) {
                        registered_value = book.get(history_elem.board).value;
                        if (history_elem.player == WHITE) {
                            registered_value *= -1;
                        }
                    } else {
                        if (history_elem.player == BLACK) {
                            book.change(&history_elem.board, registered_value, getData().menu_elements.level);
                        } else if (history_elem.player == WHITE) {
                            book.change(&history_elem.board, -registered_value, getData().menu_elements.level);
                        }
                    }
                }
                if (getData().graph_resources.branch == 1) {
                    for (History_elem history_elem: getData().graph_resources.nodes[1]) {
                        if (history_elem.board.n_discs() >= root_board.n_discs()) {
                            break;
                        }
                        if (book.contain(&history_elem.board)) {
                            registered_value = book.get(history_elem.board).value;
                            if (history_elem.player == WHITE) {
                                registered_value *= -1;
                            }
                        } else {
                            if (history_elem.player == BLACK) {
                                book.change(&history_elem.board, registered_value, getData().menu_elements.level);
                            } else if (history_elem.player == WHITE) {
                                book.change(&history_elem.board, -registered_value, getData().menu_elements.level);
                            }
                        }
                    }
                }
                book_learn_future = std::async(std::launch::async, book_deviate, root_board, getData().menu_elements.level, depth, error_per_move, error_sum, error_leaf, BOOK_DEVIATE_MAX_N_LOOPS_INF, &history_elem.board, &history_elem.player, getData().settings.book_file, getData().settings.book_file + ".bak", &book_learning);
            }
            back_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        } else if (!done) {
            getData().fonts.font(language.get("book", "stopping")).draw(20, 480, 230, getData().colors.white);
            if (book_learn_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                book_learn_future.get();
                book_learning = false;
                done = true;
                global_searching = true;
            }
        } else {
            getData().fonts.font(language.get("book", "complete")).draw(20, 480, 230, getData().colors.white);
            back_button.draw();
            if (back_button.clicked()) {
                reset_book_additional_information();
                getData().book_information.changed = true;
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        }
    }

    void draw() const override {

    }
};



class Deviate_book_transcript : public App::Scene {
private:
    Button single_back_button;
    Button back_button;
    Button start_button;
    Button start_with_max_n_loops_button;
    Slidebar n_loops_bar;
    bool transcript_done;
    bool transcript_failed;
    Board board;
    bool file_dragged;
    TextAreaEditState text_area;
    std::string raw_transcripts;
    std::vector<Board> board_list;
    std::vector<std::string> transcript_list;
    std::vector<int> error_lines;

    Button stop_button;
    Button back_button_deviating;
    History_elem history_elem;
    bool book_learning;
    bool learning_done;
    bool stop_button_pressed;
    int board_idx = 0;
    std::future<void> book_learn_future;
    int depth;
    int error_per_move;
    int error_sum;
    int error_leaf;
    int max_n_loops;
    int max_n_loops_used;
	uint64_t last_saved_time;

    bool all_done;

public:
    Deviate_book_transcript(const InitData& init) : IScene{ init } {
        single_back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        start_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("book", "start"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        start_with_max_n_loops_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("book", "start_with_max_n_loops"), 15, getData().fonts.font, getData().colors.white, getData().colors.black);
        file_dragged = false;
        transcript_done = false;
        transcript_failed = false;

        stop_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("book", "stop_learn"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button_deviating.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        history_elem = getData().history_elem;
        history_elem.policy = -1;
        book_learning = false;
        learning_done = false;
        stop_button_pressed = false;
        depth = getData().menu_elements.book_learn_depth;
        if (!getData().menu_elements.use_book_learn_depth) {
            depth = BOOK_DEPTH_INF;
        }
        error_per_move = getData().menu_elements.book_learn_error_per_move;
        if (!getData().menu_elements.use_book_learn_error_per_move) {
            error_per_move = BOOK_ERROR_INF;
        }
        error_sum = getData().menu_elements.book_learn_error_sum;
        if (!getData().menu_elements.use_book_learn_error_sum) {
            error_sum = BOOK_ERROR_INF;
        }
        error_leaf = getData().menu_elements.book_learn_error_leaf;
        if (!getData().menu_elements.use_book_learn_error_leaf) {
            error_leaf = BOOK_ERROR_INF;
        }
        max_n_loops = 15;
        n_loops_bar.init(X_CENTER - 220, 345, 440, 20, language.get("book", "max_n_loops"), 15, getData().colors.white, getData().fonts.font, 1, 30, &max_n_loops);

        all_done = false;
    }

    void update() override {
        if ((!transcript_done || transcript_failed || all_done) && System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        if (!transcript_done) { // transcript
            int sy = 20;
            getData().fonts.font(language.get("book", "book_deviate_with_transcript")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            sy += 45;
            if (DragDrop::HasNewFilePaths()) {
                std::string path = DragDrop::GetDroppedFilePaths()[0].path.narrow();
                std::ifstream ifs(path);
                if (ifs.fail()) {
                    std::cerr << "can't open " << path << std::endl;
                } else {
                    std::istreambuf_iterator<char> it(ifs);
                    std::istreambuf_iterator<char> last;
                    std::string str(it, last);
                    raw_transcripts = str;
                    file_dragged = true;
                }
            }
            if (file_dragged) {
                std::stringstream ss{raw_transcripts};
                std::string line_preview;
                bool need_to_write_more = true;
                std::string preview;
                for (int i = 0; i < 12; ++i) {
                    if (!getline(ss, line_preview)) {
                        need_to_write_more = false;
                        break;
                    } else {
                        preview += line_preview + "\n";
                    }
                }
                if (need_to_write_more) {
                    preview += "...";
                }
                getData().fonts.font(Unicode::Widen(preview)).draw(15, X_CENTER - 350, sy, getData().colors.white);
            } else {
                text_area.active = true;
                SimpleGUI::TextArea(text_area, Vec2{X_CENTER - 350, sy}, SizeF{700, 270}, TEXTBOX_MAX_CHARS);
                raw_transcripts = text_area.text.narrow();
            }
            n_loops_bar.draw();
            sy += 305;
            getData().fonts.font(language.get("in_out", "you_can_paste_with_ctrl_v")).draw(13, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            if (!file_dragged) {
                getData().fonts.font(Format(text_area.text.size()) + U"/" + Format(TEXTBOX_MAX_CHARS) + U" " + language.get("common", "characters")).draw(13, Arg::topRight(X_CENTER + 350, sy), getData().colors.white);
            }
            sy += 20;
            getData().fonts.font(language.get("book", "input_transcripts_with_line_breaks")).draw(13, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            if (n_loops_bar.is_changeable()) {
                back_button.disable_notransparent();
                start_button.disable_notransparent();
                start_with_max_n_loops_button.disable_notransparent();
            } else {
                back_button.enable();
                start_button.enable();
                start_with_max_n_loops_button.enable();
            }
            back_button.draw();
            start_button.draw();
            start_with_max_n_loops_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
            if ((start_button.clicked() || start_with_max_n_loops_button.clicked()) && raw_transcripts.size() && !n_loops_bar.is_changeable()) {
                if (start_button.clicked()) {
                    max_n_loops_used = BOOK_DEVIATE_MAX_N_LOOPS_INF;
                } else {
                    max_n_loops_used = max_n_loops;
                }
                transcript_failed = import_transcript_processing();
                if (!transcript_failed) {
                    board_idx = 0;
                    book_learn_future = std::async(std::launch::async, book_deviate, board_list[board_idx], getData().menu_elements.level, depth, error_per_move, error_sum, error_leaf, max_n_loops_used, &history_elem.board, &history_elem.player, getData().settings.book_file, getData().settings.book_file + ".bak", &book_learning);
					last_saved_time = tim();
                    book_learning = true;
                }
                transcript_done = true;
            }
        } else if (transcript_failed) { // error in transcript list
            int sy = 20;
            getData().fonts.font(language.get("book", "book_deviate_with_transcript")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            sy += 45;
            getData().fonts.font(language.get("book", "transcript_error")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            sy += 45;
            String error_lines_str = U"Line: ";
            for (int i = 0; i < std::min(250, (int)error_lines.size()); ++i) {
                error_lines_str += Format(error_lines[i]);
                if (i != (int)error_lines.size() - 1) {
                    error_lines_str += U", ";
                }
                if ((i + 1) % 20 == 0) {
                    error_lines_str += U"\n";
                }
            }
            if (error_lines.size() > 250) {
                error_lines_str += U"...";
            }
            getData().fonts.font(error_lines_str).draw(15, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            single_back_button.draw();
            if (single_back_button.clicked()) {
                error_lines.clear();
                board_list.clear();
                transcript_done = false;
                transcript_failed = false;
                file_dragged = false;
            }
        } else { // training
            draw_board(getData().fonts, getData().colors, history_elem);
            draw_info(getData().colors, history_elem, getData().fonts, getData().menu_elements, false, "");
            getData().fonts.font(language.get("book", "book_deviate_with_transcript")).draw(20, 480, 200, getData().colors.white);
            String depth_str = Format(depth);
            if (depth == BOOK_DEPTH_INF) {
                depth_str = language.get("book", "unlimited");
            }
            getData().fonts.font(language.get("book", "depth") + U": " + depth_str).draw(15, 480, 260, getData().colors.white);
            String error_per_move_str = Format(error_per_move);
            if (error_per_move == BOOK_ERROR_INF) {
                error_per_move_str = language.get("book", "unlimited");
            }
            getData().fonts.font(language.get("book", "error_per_move") + U": " + error_per_move_str).draw(15, 480, 280, getData().colors.white);
            String error_sum_str = Format(error_sum);
            if (error_sum == BOOK_ERROR_INF) {
                error_sum_str = language.get("book", "unlimited");
            }
            getData().fonts.font(language.get("book", "error_sum") + U": " + error_sum_str).draw(15, 480, 300, getData().colors.white);
            String error_leaf_str = Format(error_leaf);
            if (error_leaf == BOOK_ERROR_INF) {
                error_leaf_str = language.get("book", "unlimited");
            }
            getData().fonts.font(language.get("book", "error_leaf") + U": " + error_leaf_str).draw(15, 480, 320, getData().colors.white);
            if (book_learning) { // learning
                getData().fonts.font(language.get("book", "learning")).draw(20, 480, 230, getData().colors.white);
                getData().fonts.font(U"Line: " + Format(board_idx + 1) + U"/" + Format(board_list.size())).draw(15, 480, 340, getData().colors.white);
                std::string transcript_str = transcript_list[board_idx];
                if (transcript_str.size() > 34) {
                    if (transcript_str.size() > 34 * 2) {
                        transcript_str = transcript_str.substr(0, 34 * 2 - 2) + "...";
                    }
                    transcript_str = transcript_str.substr(0, 34) + "\n" + transcript_str.substr(34, transcript_str.size() - 34);
                }
                getData().fonts.font(Unicode::Widen(transcript_str)).draw(15, 480, 360, getData().colors.white);
                stop_button.draw();
                if (stop_button.clicked()) {
                    global_searching = false;
                    book_learning = false;
                    stop_button_pressed = true;
                }
            } else if (!learning_done) { // stop button pressed or completed
                if (stop_button_pressed) {
                    getData().fonts.font(language.get("book", "stopping")).draw(20, 480, 230, getData().colors.white);
                }
                if (book_learn_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    book_learn_future.get();
                    ++board_idx;
                    global_searching = true;
                    if (board_idx < (int)board_list.size() && !stop_button_pressed) { // next board
						if (tim() - last_saved_time >= AUTO_BOOK_SAVE_TIME) {
							book.save_egbk3(getData().settings.book_file, getData().settings.book_file + ".bak");
							last_saved_time = tim();
						}
                        book_learn_future = std::async(std::launch::async, book_deviate, board_list[board_idx], getData().menu_elements.level, depth, error_per_move, error_sum, error_leaf, max_n_loops_used, &history_elem.board, &history_elem.player, getData().settings.book_file, getData().settings.book_file + ".bak", &book_learning);
                        book_learning = true;
                        learning_done = false;
                    } else { // all boards done
                        book_learning = false;
                        learning_done = true;
                    }
                }
            } else {
                all_done = true;
                getData().fonts.font(language.get("book", "complete")).draw(20, 480, 230, getData().colors.white);
                back_button_deviating.draw();
                if (back_button_deviating.clicked()) {
                    reset_book_additional_information();
                    getData().book_information.changed = true;
                    getData().graph_resources.need_init = false;
                    changeScene(U"Main_scene", SCENE_FADE_TIME);
                }
            }
        }
    }

    void draw() const override {

    }

private:
    bool import_transcript_processing() {
        bool error_found = false;
        std::string str = raw_transcripts;
        std::stringstream ss{str};
        std::string transcript;
        int line_idx = 1;
        while (getline(ss, transcript)) {
            if (transcript.size() == 0) {
                continue;
            }
            Board board;
            Flip flip;
            bool error_found_line = false;
            board.reset();
            std::vector<std::pair<Board, int>> boards_to_be_registered;
            int registered_value = 0;
            for (int i = 0; i < transcript.size(); i += 2) {
                if (book.contain(&board)) {
                    registered_value = book.get(&board).value;
                }
                int x = (int)(transcript[i] - 'a');
                if (x < 0 || HW <= x) {
                    x = (int)(transcript[i] - 'A');
                }
                if (transcript.size() <= i + 1) {
                    error_found = true;
                    error_found_line = true;
                    error_lines.emplace_back(line_idx);
                    std::cerr << "transcript size error at line " << line_idx << " " << transcript << std::endl;
                    break;
                }
                int y = (int)(transcript[i + 1] - '1');
                if (x < 0 || HW <= x || y < 0 || HW <= y) {
                    error_found = true;
                    error_found_line = true;
                    error_lines.emplace_back(line_idx);
                    std::cerr << "coord out of range at line " << line_idx << " " << transcript << std::endl;
                    break;
                }
                int policy = HW2_M1 - (y * HW + x);
                if ((1 & (board.get_legal() >> policy)) == 0) {
                    error_found = true;
                    error_found_line = true;
                    error_lines.emplace_back(line_idx);
                    std::cerr << "illegal move at line " << line_idx << " " << transcript << std::endl;
                    break;
                }
                calc_flip(&flip, &board, policy);
                board.move_board(&flip);
                registered_value *= -1;
                if (board.get_legal() == 0) {
                    if (board.is_end()) {
                        error_found = true;
                        error_found_line = true;
                        error_lines.emplace_back(line_idx);
                        std::cerr << "game over at line " << line_idx << " " << transcript << std::endl;
                        break;
                    }
                    board.pass();
                    registered_value *= -1;
                }
                if (!book.contain(&board)) {
                    boards_to_be_registered.emplace_back(std::make_pair(board, registered_value));
                }
            }
            if (!error_found_line) {
                for (std::pair<Board, int> board_to_be_registered: boards_to_be_registered) {
                    if (board_to_be_registered.first.n_discs() >= board.n_discs()) {
                        break;
                    }
                    book.change(&board_to_be_registered.first, board_to_be_registered.second, getData().menu_elements.level);
                }
                board_list.emplace_back(board);
                transcript_list.emplace_back(transcript);
            }
            //if (error_found_line) {
            //    std::cerr << "error found in line " << line_idx << " " << transcript << std::endl;
            //}
            ++line_idx;
        }
        return error_found;
    }
};



class Fix_book : public App::Scene {
private:
    Button start_button;
    Button back_button;
    Button stop_button;
    bool before_start;
    bool done;
    bool stop;
    std::future<void> task_future;

public:
    Fix_book(const InitData& init) : IScene{ init } {
        start_button.init(BACK_BUTTON_SX, BUTTON2_VERTICAL_1_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("book", "start"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(BACK_BUTTON_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        stop_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("book", "force_stop"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        before_start = true;
        done = false;
        stop = false;
    }

    void update() override {
        if ((before_start || done) && (System::GetUserActions() & UserAction::CloseButtonClicked)) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        getData().fonts.font(language.get("book", "book_fix")).draw(25, 50, 50, getData().colors.white);
        if (before_start) {
            start_button.draw();
            if (start_button.clicked()) {
                before_start = false;
                task_future = std::async(std::launch::async, book_fix, &stop);
            }
            back_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        } else if (!done) {
            stop_button.draw();
            if (stop_button.clicked()) {
                stop = true;
            }
            if (task_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                task_future.get();
                done = true;
                global_searching = true;
            }
        } else {
            reset_book_additional_information();
            getData().book_information.changed = true;
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
    }

    void draw() const override {

    }
};




class Fix_book_edax : public App::Scene {
private:
    Button start_button;
    Button back_button;
    Button stop_button;
    bool before_start;
    bool done;
    bool stop;
    std::future<void> task_future;

public:
    Fix_book_edax(const InitData& init) : IScene{ init } {
        start_button.init(BACK_BUTTON_SX, BUTTON2_VERTICAL_1_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("book", "start"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(BACK_BUTTON_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        stop_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("book", "force_stop"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        before_start = true;
        done = false;
        stop = false;
    }

    void update() override {
        if ((before_start || done) && (System::GetUserActions() & UserAction::CloseButtonClicked)) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        getData().fonts.font(language.get("book", "book_fix_edax")).draw(25, 50, 50, getData().colors.white);
        if (before_start) {
            start_button.draw();
            if (start_button.clicked()) {
                before_start = false;
                task_future = std::async(std::launch::async, book_fix_edax, &stop);
            }
            back_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        } else if (!done) {
            stop_button.draw();
            if (stop_button.clicked()) {
                stop = true;
            }
            if (task_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                task_future.get();
                done = true;
                global_searching = true;
            }
        } else {
            reset_book_additional_information();
            getData().book_information.changed = true;
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
    }

    void draw() const override {

    }
};



class Reduce_book : public App::Scene {
private:
    Button start_button;
    Button stop_button;
    Button back_button;
    History_elem history_elem;
    bool book_learning;
    bool done;
    bool before_start;
    std::future<void> book_learn_future;
    Board root_board;
    int depth;
    int error_per_move;
    int error_sum;

public:
    Reduce_book(const InitData& init) : IScene{ init } {
        start_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY - 65, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("book", "start"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        stop_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("book", "force_stop"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        root_board = getData().history_elem.board;
        history_elem = getData().history_elem;
        history_elem.policy = -1;
        book_learning = false;
        done = false;
        before_start = true;
        depth = getData().menu_elements.book_learn_depth;
        if (!getData().menu_elements.use_book_learn_depth) {
            depth = BOOK_DEPTH_INF;
        }
        error_per_move = getData().menu_elements.book_learn_error_per_move;
        if (!getData().menu_elements.use_book_learn_error_per_move) {
            error_per_move = BOOK_ERROR_INF;
        }
        error_sum = getData().menu_elements.book_learn_error_sum;
        if (!getData().menu_elements.use_book_learn_error_sum) {
            error_sum = BOOK_ERROR_INF;
        }
    }

    void update() override {
        if ((before_start || done) && (System::GetUserActions() & UserAction::CloseButtonClicked)) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        draw_board(getData().fonts, getData().colors, history_elem);
        draw_info(getData().colors, history_elem, getData().fonts, getData().menu_elements, false, "");
        getData().fonts.font(language.get("book", "book_reduce")).draw(25, 480, 200, getData().colors.white);
        draw_book_settings(getData().fonts.font, getData().colors.white, depth, error_per_move, error_sum);
        if (book_learning) {
            getData().fonts.font(language.get("book", "reducing")).draw(20, 480, 230, getData().colors.white);
            stop_button.draw();
            if (stop_button.clicked()) {
                global_searching = false;
                book_learning = false;
            }
        } else if (before_start) {
            start_button.draw();
            if (start_button.clicked()) {
                before_start = false;
                book_learning = true;
                book_learn_future = std::async(std::launch::async, book_reduce, root_board, depth, error_per_move, error_sum, &book_learning);
            }
            back_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        } else if (!done) {
            getData().fonts.font(language.get("book", "stopping")).draw(20, 480, 230, getData().colors.white);
            if (book_learn_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                book_learn_future.get();
                book_learning = false;
                done = true;
                global_searching = true;
            }
        } else {
            getData().fonts.font(language.get("book", "complete")).draw(20, 480, 230, getData().colors.white);
            back_button.draw();
            if (back_button.clicked()) {
                reset_book_additional_information();
                getData().book_information.changed = true;
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        }
    }

    void draw() const override {

    }
};

class Leaf_recalculate_book : public App::Scene {
private:
    Button start_button;
    Button stop_button;
    Button back_button;
    History_elem history_elem;
    bool book_learning;
    bool done;
    bool before_start;
    std::future<void> book_learn_future;
    Board root_board;
    int depth;
    int error_per_move;
    int error_sum;

public:
    Leaf_recalculate_book(const InitData& init) : IScene{ init } {
        start_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY - 65, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("book", "start"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        stop_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("book", "stop_learn"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        root_board = getData().history_elem.board;
        history_elem = getData().history_elem;
        history_elem.policy = -1;
        book_learning = false;
        done = false;
        before_start = true;
        depth = getData().menu_elements.book_learn_depth;
        if (!getData().menu_elements.use_book_learn_depth) {
            depth = BOOK_DEPTH_INF;
        }
        error_per_move = getData().menu_elements.book_learn_error_per_move;
        if (!getData().menu_elements.use_book_learn_error_per_move) {
            error_per_move = BOOK_ERROR_INF;
        }
        error_sum = getData().menu_elements.book_learn_error_sum;
        if (!getData().menu_elements.use_book_learn_error_sum) {
            error_sum = BOOK_ERROR_INF;
        }
    }

    void update() override {
        if ((before_start || done) && (System::GetUserActions() & UserAction::CloseButtonClicked)) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        draw_board(getData().fonts, getData().colors, history_elem);
        draw_info(getData().colors, history_elem, getData().fonts, getData().menu_elements, false, "");
        getData().fonts.font(language.get("book", "book_recalculate_leaf")).draw(25, 480, 200, getData().colors.white);
        draw_book_settings(getData().fonts.font, getData().colors.white, depth, error_per_move, error_sum);
        if (book_learning) {
            getData().fonts.font(language.get("book", "learning")).draw(20, 480, 230, getData().colors.white);
            stop_button.draw();
            if (stop_button.clicked()) {
                global_searching = false;
                book_learning = false;
            }
        } else if (before_start) {
            start_button.draw();
            if (start_button.clicked()) {
                before_start = false;
                book_learning = true;
                book_learn_future = std::async(std::launch::async, book_recalculate_leaf, root_board, getData().menu_elements.level, depth, error_per_move, error_sum, &history_elem.board, &history_elem.player, &book_learning, false, tim());
            }
            back_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        } else if (!done) {
            getData().fonts.font(language.get("book", "stopping")).draw(20, 480, 230, getData().colors.white);
            if (book_learn_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                book_learn_future.get();
                book_learning = false;
                done = true;
                global_searching = true;
            }
        } else {
            getData().fonts.font(language.get("book", "complete")).draw(20, 480, 230, getData().colors.white);
            back_button.draw();
            if (back_button.clicked()) {
                reset_book_additional_information();
                getData().book_information.changed = true;
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        }
    }

    void draw() const override {

    }
};





class N_lines_recalculate_book : public App::Scene {
private:
    Button start_button;
    Button back_button;
    Button stop_button;
    bool before_start;
    bool done;
    bool stop;
    std::future<void> task_future;

public:
    N_lines_recalculate_book(const InitData& init) : IScene{ init } {
        start_button.init(BACK_BUTTON_SX, BUTTON2_VERTICAL_1_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("book", "start"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(BACK_BUTTON_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        stop_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("book", "force_stop"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        before_start = true;
        done = false;
        stop = false;
    }

    void update() override {
        if ((before_start || done) && (System::GetUserActions() & UserAction::CloseButtonClicked)) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        getData().fonts.font(language.get("book", "book_recalculate_n_lines")).draw(25, 50, 50, getData().colors.white);
        if (before_start) {
            start_button.draw();
            if (start_button.clicked()) {
                before_start = false;
                task_future = std::async(std::launch::async, book_recalculate_n_lines_all, &stop);
            }
            back_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        } else if (!done) {
            stop_button.draw();
            if (stop_button.clicked())
                stop = true;
            if (task_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                task_future.get();
                done = true;
                global_searching = true;
            }
        } else {
            reset_book_additional_information();
            getData().book_information.changed = true;
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
    }

    void draw() const override {

    }
};




/*
class Upgrade_better_leaves_book : public App::Scene {
private:
    Button start_button;
    Button back_button;
    Button stop_button;
    bool before_start;
    bool done;
    bool stop;
    std::future<void> task_future;

public:
    Upgrade_better_leaves_book(const InitData& init) : IScene{ init } {
        start_button.init(BACK_BUTTON_SX, BUTTON2_VERTICAL_1_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("book", "start"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        back_button.init(BACK_BUTTON_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        stop_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("book", "force_stop"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        before_start = true;
        done = false;
        stop = false;
    }

    void update() override {
        if ((before_start || done) && (System::GetUserActions() & UserAction::CloseButtonClicked)) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        getData().fonts.font(language.get("book", "book_upgrade_better_leaves")).draw(25, 50, 50, getData().colors.white);
        if (before_start) {
            start_button.draw();
            if (start_button.clicked()) {
                before_start = false;
                task_future = std::async(std::launch::async, book_upgrade_better_leaves_all, &stop);
            }
            back_button.draw();
            if (back_button.clicked() || KeyEscape.pressed()) {
                getData().graph_resources.need_init = false;
                changeScene(U"Main_scene", SCENE_FADE_TIME);
            }
        } else if (!done) {
            stop_button.draw();
            if (stop_button.clicked())
                stop = true;
            if (task_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                task_future.get();
                done = true;
                global_searching = true;
            }
        } else {
            reset_book_additional_information();
            getData().book_information.changed = true;
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
    }

    void draw() const override {

    }
};
*/



class Show_book_info : public App::Scene {
private:
    Button back_button;
    std::future<Book_info> book_info_future;
    Book_info book_info;
    bool book_info_calculating;

public:
    Show_book_info(const InitData& init) : IScene{ init } {
        back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        book_info_calculating = true;
        book_info_future = std::async(std::launch::async, calculate_book_info, &book_info_calculating);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        getData().fonts.font(language.get("book", "show_book_info")).draw(25, 40, 30, getData().colors.white);
        if (book_info_calculating) {
            getData().fonts.font(language.get("book", "calculating")).draw(25, 50, 67, getData().colors.white);
            if (book_info_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                book_info = book_info_future.get();
                book_info_calculating = false;
            }
        } else {
            getData().fonts.font(language.get("book", "n_registered") + U": " + Format(book_info.n_boards)).draw(15, 50, 67, getData().colors.white);
            int sy = 90;
            int sx = 50;
            for (int level = 1; level < N_LEVEL; ++level) {
                String str = U"Lv.";
                if (level < 10) {
                    str += U" ";
                }
                str += Format(level) + U": " + Format(book_info.n_boards_in_level[level]);
                getData().fonts.font(str).draw(13, sx, sy, getData().colors.white);
                sy += 15;
                if (level % 20 == 0 && level != N_LEVEL - 1) {
                    sy = 90;
                    sx += 120;
                }
            }
            String str = U"Lv. S: " + Format(book_info.n_boards_in_level[LEVEL_HUMAN]);
            getData().fonts.font(str).draw(13, sx, sy, getData().colors.white);
            sx += 130;
            sy = 90;
            for (int ply = 0; ply <= HW2 - 4; ++ply) {
                String str = U"Ply.";
                if (ply < 10) {
                    str += U" ";
                }
                str += Format(ply) + U": " + Format(book_info.n_boards_in_ply[ply]);
                getData().fonts.font(str).draw(13, sx, sy, getData().colors.white);
                sy += 15;
                if (ply % 21 == 20) {
                    sy = 90;
                    sx += 120;
                }
            }
        }
        back_button.draw();
        if (back_button.clicked() || KeyEscape.pressed()) {
            if (book_info_calculating) {
                book_info_calculating = false;
                book_info_future.get();
            }
            reset_book_additional_information();
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
    }

    void draw() const override {

    }
};