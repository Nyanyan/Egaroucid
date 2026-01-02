/*
    Egaroucid Project

    @file shortcut_key.hpp
        Shortcut Key Manager
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <Siv3D.hpp>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "language.hpp"

#define SHORTCUT_KEY_UNDEFINED U"undefined"

struct Shortcut_key_elem {
    String name;
    std::vector<String> keys;
    std::vector<std::vector<std::string>> description_keys;
};

std::vector<Shortcut_key_elem> shortcut_keys_default = {
    // buttons
    {U"start_game",             {U"Space"},             {{"play", "start_game"}}},
    {U"pass",                   {},                     {{"play", "pass"}}},

    // game
    {U"new_game",               {U"Ctrl", U"N"},        {{"play", "game"}, {"play", "new_game"}}},
    {U"new_game_human_black",   {},                     {{"play", "game"}, {"play", "new_game_human_black"}}},
    {U"new_game_human_white",   {},                     {{"play", "game"}, {"play", "new_game_human_white"}}},
    {U"new_selfplay",           {},                     {{"play", "game"}, {"play", "new_selfplay"}}},
    {U"analyze",                {U"A"},                 {{"play", "game"}, {"play", "analyze"}}},
    {U"game_information",       {},                     {{"play", "game"}, {"play", "game_information"}}},

    // settings
    {U"use_book",               {},                     {{"settings", "settings"}, {"ai_settings", "use_book"}}},
    {U"accept_ai_loss",         {},                     {{"settings", "settings"}, {"ai_settings", "accept_ai_loss"}}}, 
    {U"ai_put_black",           {U"B"},                 {{"settings", "settings"}, {"settings", "play", "ai_put_black"}}},
    {U"ai_put_white",           {U"W"},                 {{"settings", "settings"}, {"settings", "play", "ai_put_white"}}},
    {U"pause_when_pass",        {},                     {{"settings", "settings"}, {"settings", "play", "pause_when_pass"}}},
    {U"force_specified_openings", {},                   {{"settings", "settings"}, {"settings", "play", "force_specified_openings"}}},
    {U"opening_setting",        {},                     {{"settings", "settings"}, {"settings", "play", "opening_setting"}}},
    {U"shortcut_key_setting",   {},                     {{"settings", "settings"}, {"settings", "shortcut_keys", "settings"}}},

    // display
    // on cell
    {U"show_legal",             {},                     {{"display", "display"}, {"display", "cell", "display_on_cell"}, {"display", "cell", "legal"}}},
    {U"show_disc_hint",         {U"V"},                 {{"display", "display"}, {"display", "cell", "display_on_cell"}, {"display", "cell", "disc_value"}}},
    {U"show_hint_level",        {},                     {{"display", "display"}, {"display", "cell", "display_on_cell"}, {"display", "cell", "disc_value"}, {"display", "cell", "show_hint_level"}}},
    {U"show_value_when_ai_calculating", {},             {{"display", "display"}, {"display", "cell", "display_on_cell"}, {"display", "cell", "disc_value"}, {"display", "cell", "show_value_when_ai_calculating"}}},
    {U"show_book_accuracy",     {},                     {{"display", "display"}, {"display", "cell", "display_on_cell"}, {"display", "cell", "disc_value"}, {"display", "cell", "show_book_accuracy"}}},
    {U"hint_colorize",          {},                     {{"display", "display"}, {"display", "cell", "display_on_cell"}, {"display", "cell", "disc_value"}, {"display", "cell", "hint_colorize"}}},
    {U"show_umigame_value",     {U"U"},                 {{"display", "display"}, {"display", "cell", "display_on_cell"}, {"display", "cell", "umigame_value"}}},
    {U"show_opening_on_cell",   {},                     {{"display", "display"}, {"display", "cell", "display_on_cell"}, {"display", "cell", "opening"}}},
    {U"show_next_move",         {},                     {{"display", "display"}, {"display", "cell", "display_on_cell"}, {"display", "cell", "next_move"}}},
    // on discs
    {U"show_last_move",         {},                     {{"display", "display"}, {"display", "disc", "display_on_disc"}, {"display", "disc", "last_move"}}},
    {U"show_stable_discs",      {},                     {{"display", "display"}, {"display", "disc", "display_on_disc"}, {"display", "disc", "stable"}}},
    {U"show_play_ordering",     {},                     {{"display", "display"}, {"display", "disc", "display_on_disc"}, {"display", "disc", "play_ordering"}}},
    {U"play_ordering_board_format", {},                 {{"display", "display"}, {"display", "disc", "display_on_disc"}, {"display", "disc", "play_ordering"}, {"display", "disc", "play_ordering_board_format"}}},
    {U"play_ordering_transcript_format",    {},         {{"display", "display"}, {"display", "disc", "display_on_disc"}, {"display", "disc", "play_ordering"}, {"display", "disc", "play_ordering_transcript_format"}}},
    // info area
    {U"show_opening_name",      {},                     {{"display", "display"}, {"display", "info", "display_on_info_area"}, {"display", "info", "opening_name"}}},
    {U"show_principal_variation",   {},                 {{"display", "display"}, {"display", "info", "display_on_info_area"}, {"display", "info", "principal_variation"}}},
    // graph area
    {U"show_graph",             {},                     {{"display", "display"}, {"display", "graph", "display_on_graph_area"}, {"display", "graph", "graph"}}},
    {U"show_graph_value",       {U"D"},                 {{"display", "display"}, {"display", "graph", "display_on_graph_area"}, {"display", "graph", "graph"}, {"display", "graph", "value"}}},
    {U"show_graph_sum_of_loss", {U"S"},                 {{"display", "display"}, {"display", "graph", "display_on_graph_area"}, {"display", "graph", "graph"}, {"display", "graph", "sum_of_loss"}}},
    {U"show_endgame_error",     {},                     {{"display", "display"}, {"display", "graph", "display_on_graph_area"}, {"display", "graph", "endgame_error"}}},
    // others
    {U"show_ai_focus",          {},                     {{"display", "display"}, {"display", "ai_focus"}}},
    {U"show_laser_pointer",     {U"P"},                 {{"display", "display"}, {"display", "laser_pointer"}}},
    {U"show_log",               {},                     {{"display", "display"}, {"display", "log"}}},
    {U"change_color_type",      {},                     {{"display", "display"}, {"display", "change_color_type"}}},

    // operate
    {U"put_1_move_by_ai",       {U"G"},                 {{"operation", "operation"}, {"operation", "put_1_move_by_ai"}}},
    {U"forward",                {U"Right"},             {{"operation", "operation"}, {"operation", "forward"}}},
    {U"backward",               {U"Left"},              {{"operation", "operation"}, {"operation", "backward"}}},
    {U"undo",                   {U"Backspace"},         {{"operation", "operation"}, {"operation", "undo"}}},
    {U"save_this_branch",       {U"Ctrl", U"L"},        {{"operation", "operation"}, {"operation", "save_this_branch"}}},
    {U"generate_random_board",  {U"Ctrl", U"R"},        {{"operation", "operation"}, {"operation", "generate_random_board", "generate"}}},
    {U"convert_180",            {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "rotate_180"}}},
    {U"convert_90_clock",       {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "rotate_90_clock"}}},
    {U"convert_90_anti_clock",  {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "rotate_90_anti_clock"}}},
    {U"convert_blackline",      {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "black_line"}}},
    {U"convert_whiteline",      {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "white_line"}}},
    {U"convert_horizontal",     {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "horizontal"}}},
    {U"convert_vertical",       {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "vertical"}}},
    {U"stop_calculating",       {U"Q"},                 {{"operation", "operation"}, {"operation", "ai_operation", "ai_operation"}, {"operation", "ai_operation", "stop_calculating"}}},
    {U"cache_clear",            {},                     {{"operation", "operation"}, {"operation", "ai_operation", "ai_operation"}, {"operation", "ai_operation", "cache_clear"}}},

    // input / output
    // input
    {U"input_from_clipboard",   {U"Ctrl", U"V"},        {{"in_out", "in_out"}, {"in_out", "in"}, {"in_out", "input_from_clipboard"}}},
    {U"input_text",             {},                     {{"in_out", "in_out"}, {"in_out", "in"}, {"in_out", "input_text"}}},
    {U"edit_board",             {U"Ctrl", U"E"},        {{"in_out", "in_out"}, {"in_out", "in"}, {"in_out", "edit_board"}}},
    {U"input_game",             {},                     {{"in_out", "in_out"}, {"in_out", "in"}, {"in_out", "input_game"}}},
    {U"input_bitboard",         {},                     {{"in_out", "in_out"}, {"in_out", "in"}, {"in_out", "input_bitboard"}}},
    // output
    {U"output_transcript",      {U"Ctrl", U"C"},        {{"in_out", "in_out"}, {"in_out", "out"}, {"in_out", "output_transcript"}}},
    {U"output_board",           {},                     {{"in_out", "in_out"}, {"in_out", "out"}, {"in_out", "output_board"}}},
    {U"screen_shot",            {U"Ctrl", U"S"},        {{"in_out", "in_out"}, {"in_out", "out"}, {"in_out", "screen_shot"}}},
    {U"change_screenshot_saving_dir",       {},         {{"in_out", "in_out"}, {"in_out", "out"}, {"in_out", "screen_shot"}, {"in_out", "change_screenshot_saving_dir"}}},
    {U"board_image",            {},                     {{"in_out", "in_out"}, {"in_out", "out"}, {"in_out", "board_image"}}},
    {U"save_game",              {},                     {{"in_out", "in_out"}, {"in_out", "out"}, {"in_out", "output_game"}}},
    {U"output_bitboard_player_opponent",    {},         {{"in_out", "in_out"}, {"in_out", "out"}, {"in_out", "output_bitboard"}, {"in_out", "player_opponent"}}},
    {U"output_bitboard_black_white",        {},         {{"in_out", "in_out"}, {"in_out", "out"}, {"in_out", "output_bitboard"}, {"in_out", "black_white"}}},

    // book
    // book settings
    // book operation
    {U"change_book_by_right_click", {},                 {{"book", "book"}, {"book", "book_operation"}, {"book", "right_click_to_modify"}}},
    {U"book_start_deviate",     {},                     {{"book", "book"}, {"book", "book_operation"}, {"book", "book_deviate"}}},
    {U"book_start_deviate_with_transcript", {},         {{"book", "book"}, {"book", "book_operation"}, {"book", "book_deviate_with_transcript"}}},
    {U"book_start_store",       {},                     {{"book", "book"}, {"book", "book_operation"}, {"book", "book_store"}}},
    {U"book_start_fix",         {},                     {{"book", "book"}, {"book", "book_operation"}, {"book", "book_fix"}}},
    // {U"book_start_fix_edax",    {},                     {{"book", "book"}, {"book", "book_operation"}, {"book", "book_fix_edax"}}},
    {U"book_start_reducing",    {},                     {{"book", "book"}, {"book", "book_operation"}, {"book", "book_reduce"}}},
    {U"book_start_recalculate_leaf",        {},         {{"book", "book"}, {"book", "book_operation"}, {"book", "book_recalculate_leaf"}}},
    {U"book_start_recalculate_n_lines",     {},         {{"book", "book"}, {"book", "book_operation"}, {"book", "book_recalculate_n_lines"}}},
    //{U"book_start_upgrade_better_leaves",   {},         {{"book", "book"}, {"book", "book_operation"}, {"book", "book_upgrade_better_leaves"}}},
    // file operation
    {U"import_book",            {},                     {{"book", "book"}, {"book", "file_operation"}, {"book", "import_book"}}},
    {U"export_book",            {},                     {{"book", "book"}, {"book", "file_operation"}, {"book", "export_book"}}},
    {U"book_merge",             {},                     {{"book", "book"}, {"book", "file_operation"}, {"book", "book_merge"}}},
    {U"book_reference",         {},                     {{"book", "book"}, {"book", "file_operation"}, {"book", "book_reference"}}},
    // others
    {U"show_book_info",         {},                     {{"book", "book"}, {"book", "show_book_info"}}},

    // help
    {U"open_usage",             {},                     {{"help", "help"}, {"help", "usage"}}},
    {U"open_website",           {},                     {{"help", "help"}, {"help", "website"}}},
    {U"bug_report",             {},                     {{"help", "help"}, {"help", "bug_report"}}},
    {U"update_check",           {},                     {{"help", "help"}, {"help", "update_check"}}},
    {U"auto_update_check",      {},                     {{"help", "help"}, {"help", "auto_update_check"}}},
    {U"license",                {},                     {{"help", "help"}, {"help", "license"}}},
};

// Enter and Left / Right keys are ignored
const HashSet<String> ignore_keys = {
    U"Enter",
    U"Left Command",
    U"Right Command",
    U"Left Ctrl",
    U"Left Shift",
    U"Right Shift",
};

String generate_key_str(std::vector<String> keys) {
    String res;
    for (int i = 0; i < (int)keys.size(); ++i) {
        if (keys[i] == U"Right") {
            res += U"->";
        } else if (keys[i] == U"Left") {
            res += U"<-";
        } else if (keys[i] == U"0x5b") {
            res += U"Windows";
        } else{
            res += keys[i];
        }
        if (i < (int)keys.size() - 1) {
            res += U"+";
        }
    }
    return res;
}

std::vector<String> get_all_inputs(bool *down_found) {
    const Array<Input> raw_keys = Keyboard::GetAllInputs();
    *down_found = false;
    std::unordered_set<String> keys;
    for (const auto& key : raw_keys) {
        // ignored keys
        if (ignore_keys.contains(key.name())) {
            continue;
        }
        *down_found |= key.down();
        keys.emplace(key.name());
    }
    std::vector<String> res;
    if (keys.find(U"Ctrl") != keys.end()) {
        res.emplace_back(U"Ctrl");
        keys.erase(U"Ctrl");
    }
    if (keys.find(U"Shift") != keys.end()) {
        res.emplace_back(U"Shift");
        keys.erase(U"Shift");
    }
    if (keys.find(U"Alt") != keys.end()) {
        res.emplace_back(U"Alt");
        keys.erase(U"Alt");
    }
    for (String key: keys) {
        res.emplace_back(key);
    }
    return res;
}

class Shortcut_keys {
public:
    std::vector<Shortcut_key_elem> shortcut_keys;
public:
    void set_default() {
        shortcut_keys = shortcut_keys_default;
    }

    void set_empty() {
        shortcut_keys = shortcut_keys_default;
        for (Shortcut_key_elem &elem: shortcut_keys) {
            elem.keys.clear();
        }
    }

    void init(String file) {
        set_empty();
        JSON json = JSON::Load(file);
        if (not json) {
            set_default();
        } else {
            std::unordered_set<String> name_list;
            for (Shortcut_key_elem &elem: shortcut_keys_default) {
                name_list.emplace(elem.name);
            }
            for (const auto& object: json) {
                if (name_list.find(object.key) == name_list.end()) {
                    std::cerr << "ERR shortcut key name not found " << object.key.narrow() << std::endl;
                    continue;
                }
                for (int i = 0; i < (int)shortcut_keys.size(); ++i) {
                    if (shortcut_keys[i].name == object.key) {
                        shortcut_keys[i].keys.clear();
                        for (const auto &key_name: object.value.arrayView()) {
                            shortcut_keys[i].keys.emplace_back(key_name.getString());
                        }
                    }
                }
            }
            for (int i = 0; i < (int)shortcut_keys.size(); ++i) {
                if (shortcut_keys[i].keys.size()) {
                    for (int j = i + 1; j < (int)shortcut_keys.size(); ++j) {
                        if (shortcut_keys[j].keys.size() == shortcut_keys[i].keys.size()) {
                            bool duplicate = true;
                            for (String &key: shortcut_keys[j].keys) {
                                if (std::find(shortcut_keys[i].keys.begin(), shortcut_keys[i].keys.end(), key) == shortcut_keys[i].keys.end()) {
                                    duplicate = false;
                                }
                            }
                            if (duplicate) {
                                shortcut_keys[j].keys.clear();
                                std::cerr << "shortcut key duplication found: " << shortcut_keys[i].name.narrow() << " " << shortcut_keys[j].name.narrow() << " deleted " << shortcut_keys[j].name.narrow() << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }

    void save_settings(String file) {
        JSON json;
        for (const Shortcut_key_elem &elem: shortcut_keys) {
            Array<JSON> arrayJSON;
            for (String key: elem.keys) {
                arrayJSON << key;
            }
            json[elem.name] = arrayJSON;
        }
        json.save(file);
    }

    void check_shortcut_key(String *shortcut_name_down, String *shortcut_name_pressed) {
        bool down_found = false;
        std::vector<String> keys = get_all_inputs(&down_found);
        *shortcut_name_down = SHORTCUT_KEY_UNDEFINED;
        *shortcut_name_pressed = SHORTCUT_KEY_UNDEFINED;
        for (const Shortcut_key_elem &elem: shortcut_keys) {
            if (keys.size() && keys.size() == elem.keys.size()) {
                bool matched = true;
                for (const String& key : keys) {
                    //std::cerr << key.narrow() << " " << (std::find(elem.keys.begin(), elem.keys.end(), key) == elem.keys.end()) << std::endl;
                    if (std::find(elem.keys.begin(), elem.keys.end(), key) == elem.keys.end()) {
                        matched = false;
                    }
                }
                if (matched) {
                    if (down_found) {
                        *shortcut_name_down = elem.name;
                    }
                    *shortcut_name_pressed = elem.name;
                    return;
                }
            }
        }
    }

    String get_shortcut_key_str(String name) {
        for (const Shortcut_key_elem &elem: shortcut_keys) {
            if (elem.name == name) {
                return generate_key_str(elem.keys);
            }
        }
        return SHORTCUT_KEY_UNDEFINED;
    }

    String get_shortcut_key_description(String name) {
        for (const Shortcut_key_elem &elem: shortcut_keys) {
            if (elem.name == name) {
                String res;
                for (int i = 0; i < (int)elem.description_keys.size(); ++i) {
                    res += language.get(elem.description_keys[i]);
                    if (i < (int)elem.description_keys.size() - 1) {
                        res += U"> ";
                    }
                }
                return res;
            }
        }
        return SHORTCUT_KEY_UNDEFINED;
    }

    void change(int idx, std::vector<String> keys) {
        shortcut_keys[idx].keys.clear();
        for (String key: keys) {
            shortcut_keys[idx].keys.emplace_back(key);
        }
    }

    void del(int idx) {
        shortcut_keys[idx].keys.clear();
    }
};

Shortcut_keys shortcut_keys;