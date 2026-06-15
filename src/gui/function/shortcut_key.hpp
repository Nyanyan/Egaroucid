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
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "const/gui_common.hpp"
#include "ai_profile.hpp"
#include "display_profile.hpp"
#include "language.hpp"
#if SIV3D_PLATFORM(WINDOWS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#define SHORTCUT_KEY_UNDEFINED U"undefined"
#define SHORTCUT_KEY_AI_PROFILE_PREFIX U"ai_profile:"
#define SHORTCUT_KEY_DISPLAY_PROFILE_PREFIX U"display_profile:"

struct Shortcut_key_elem {
    String name;
    std::vector<String> keys;
    std::vector<std::vector<std::string>> description_keys;
    String description_suffix;
};

inline String get_ai_profile_shortcut_name(const String& profile_file_name) {
    return String(SHORTCUT_KEY_AI_PROFILE_PREFIX) + profile_file_name;
}

inline bool is_ai_profile_shortcut_name(const String& shortcut_name) {
    return shortcut_name.starts_with(String(SHORTCUT_KEY_AI_PROFILE_PREFIX));
}

inline String get_ai_profile_file_name_from_shortcut_name(const String& shortcut_name) {
    if (!is_ai_profile_shortcut_name(shortcut_name)) {
        return U"";
    }
    return shortcut_name.substr(String(SHORTCUT_KEY_AI_PROFILE_PREFIX).size());
}

inline String get_display_profile_shortcut_name(const String& profile_file_name) {
    return String(SHORTCUT_KEY_DISPLAY_PROFILE_PREFIX) + profile_file_name;
}

inline bool is_display_profile_shortcut_name(const String& shortcut_name) {
    return shortcut_name.starts_with(String(SHORTCUT_KEY_DISPLAY_PROFILE_PREFIX));
}

inline String get_display_profile_file_name_from_shortcut_name(const String& shortcut_name) {
    if (!is_display_profile_shortcut_name(shortcut_name)) {
        return U"";
    }
    return shortcut_name.substr(String(SHORTCUT_KEY_DISPLAY_PROFILE_PREFIX).size());
}

inline Shortcut_key_elem create_ai_profile_shortcut_key_elem(const FilePath& path) {
    String profile_name;
    AI_profile_values values;
    if (!load_ai_profile_values(path, &values, &profile_name) || profile_name.trimmed().isEmpty()) {
        profile_name = FileSystem::FileName(path);
    }

    Shortcut_key_elem elem;
    elem.name = get_ai_profile_shortcut_name(FileSystem::FileName(path));
    elem.description_keys = {{"settings", "settings"}, {"settings", "profile", "profile"}};
    elem.description_suffix = profile_name;
    return elem;
}

inline Shortcut_key_elem create_display_profile_shortcut_key_elem(const FilePath& path) {
    String profile_name;
    Display_profile_values values;
    if (!load_display_profile_values(path, &values, &profile_name) || profile_name.trimmed().isEmpty()) {
        profile_name = FileSystem::FileName(path);
    }

    Shortcut_key_elem elem;
    elem.name = get_display_profile_shortcut_name(FileSystem::FileName(path));
    elem.description_keys = {{"display", "display"}, {"display", "profile", "profile"}};
    elem.description_suffix = profile_name;
    return elem;
}

inline void append_ai_profile_shortcut_key_elems(std::vector<Shortcut_key_elem>* shortcut_key_elems, const Directories* directories) {
    if (directories == nullptr) {
        return;
    }
    for (const auto& path : enumerate_ai_profile_files(*directories)) {
        shortcut_key_elems->emplace_back(create_ai_profile_shortcut_key_elem(path));
    }
}

inline void append_display_profile_shortcut_key_elems(std::vector<Shortcut_key_elem>* shortcut_key_elems, const Directories* directories) {
    if (directories == nullptr) {
        return;
    }
    for (const auto& path : enumerate_display_profile_files(*directories)) {
        shortcut_key_elems->emplace_back(create_display_profile_shortcut_key_elem(path));
    }
}

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
    {U"ai_profile_load",        {},                     {{"settings", "settings"}, {"settings", "profile", "profile"}}},
    {U"shortcut_key_setting",   {},                     {{"settings", "settings"}, {"settings", "shortcut_keys", "settings"}}},
    {U"shortcut_button_setting",{},                     {{"settings", "settings"}, {"settings", "shortcut_buttons", "settings"}}},
    {U"mouse_additional_button_setting",{},             {{"settings", "settings"}, {"settings", "mouse_additional_buttons", "settings"}}},

    // display
    {U"display_profile_load",  {},                      {{"display", "display"}, {"display", "profile", "profile"}}},
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
    {U"show_timer",             {},                     {{"display", "display"}, {"display", "info", "display_on_info_area"}, {"display", "info", "timer"}}},
    // graph area
    {U"show_graph",             {},                     {{"display", "display"}, {"display", "graph", "display_on_graph_area"}, {"display", "graph", "graph"}}},
    {U"show_graph_value",       {U"D"},                 {{"display", "display"}, {"display", "graph", "display_on_graph_area"}, {"display", "graph", "graph"}, {"display", "graph", "value"}}},
    {U"show_graph_sum_of_loss", {U"S"},                 {{"display", "display"}, {"display", "graph", "display_on_graph_area"}, {"display", "graph", "graph"}, {"display", "graph", "sum_of_loss"}}},
    {U"show_random_board_graph", {},                    {{"display", "display"}, {"display", "graph", "display_on_graph_area"}, {"display", "graph", "graph"}, {"display", "graph", "show_random_board_graph"}}},
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
    {U"go_to_first_position",   {U"Home"},              {{"operation", "operation"}, {"operation", "go_to_first_position"}}},
    {U"go_to_last_position",    {U"End"},               {{"operation", "operation"}, {"operation", "go_to_last_position"}}},
    {U"go_to_random_generated_position", {U"Shift", U"Home"}, {{"operation", "operation"}, {"operation", "go_to_random_generated_position"}}},
    {U"save_this_branch",       {U"Ctrl", U"L"},        {{"operation", "operation"}, {"operation", "save_this_branch"}}},
    {U"generate_xot_board",     {},                     {{"operation", "operation"}, {"operation", "generate_xot_board"}}},
    {U"generate_random_board",  {U"Ctrl", U"R"},        {{"operation", "operation"}, {"operation", "generate_random_board", "generate"}}},
    {U"convert_180",            {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "rotate_180"}}},
    {U"convert_90_clock",       {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "rotate_90_clock"}}},
    {U"convert_90_anti_clock",  {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "rotate_90_anti_clock"}}},
    {U"convert_blackline",      {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "black_line"}}},
    {U"convert_whiteline",      {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "white_line"}}},
    {U"convert_horizontal",     {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "horizontal"}}},
    {U"convert_vertical",       {},                     {{"operation", "operation"}, {"operation", "convert", "convert"}, {"operation", "convert", "vertical"}}},
    {U"stop_calculating",       {U"Q"},                 {{"operation", "operation"}, {"operation", "ai_operation", "ai_operation"}, {"operation", "ai_operation", "stop_calculating"}}},
    {U"resume_calculating",     {},                     {{"operation", "operation"}, {"operation", "ai_operation", "ai_operation"}, {"operation", "ai_operation", "resume_calculating"}}},
    {U"cache_clear",            {},                     {{"operation", "operation"}, {"operation", "ai_operation", "ai_operation"}, {"operation", "ai_operation", "cache_clear"}}},

    // input / output
    // input
    {U"input_from_clipboard",   {U"Ctrl", U"V"},        {{"in_out", "in_out"}, {"in_out", "in"}, {"in_out", "input_from_clipboard"}}},
    {U"input_text",             {},                     {{"in_out", "in_out"}, {"in_out", "in"}, {"in_out", "input_text"}}},
    {U"input_othello_quest",     {},                     {{"in_out", "in_out"}, {"in_out", "in"}, {"in_out", "input_othello_quest"}}},
    {U"edit_board",             {U"Ctrl", U"E"},        {{"in_out", "in_out"}, {"in_out", "in"}, {"in_out", "edit_board"}}},
    {U"input_bitboard",         {},                     {{"in_out", "in_out"}, {"in_out", "in"}, {"in_out", "input_bitboard"}}},
    {U"game_library",           {},                     {{"in_out", "in_out"}, {"in_out", "game_library"}}},
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
        } else if (keys[i] == U"0x5b" || keys[i] == U"0x5c") {
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

#if SIV3D_PLATFORM(WINDOWS)
inline bool windows_shortcut_modifier_vk(const int vk) {
    return vk == VK_SHIFT || vk == VK_LSHIFT || vk == VK_RSHIFT ||
        vk == VK_CONTROL || vk == VK_LCONTROL || vk == VK_RCONTROL ||
        vk == VK_MENU || vk == VK_LMENU || vk == VK_RMENU;
}

inline bool windows_shortcut_scannable_vk(const int vk) {
    if (vk < VK_BACK || 0xFE < vk) {
        return false;
    }
#ifdef VK_PROCESSKEY
    if (vk == VK_PROCESSKEY) {
        return false;
    }
#endif
#ifdef VK_PACKET
    if (vk == VK_PACKET) {
        return false;
    }
#endif
    return true;
}

inline bool windows_shortcut_vk_to_key_name(const int vk, String* key_name) {
    if ('A' <= vk && vk <= 'Z') {
        *key_name = Unicode::Widen(std::string(1, static_cast<char>(vk)));
        return true;
    }
    if ('0' <= vk && vk <= '9') {
        *key_name = Unicode::Widen(std::string(1, static_cast<char>(vk)));
        return true;
    }
    if (VK_NUMPAD0 <= vk && vk <= VK_NUMPAD9) {
        *key_name = U"Num" + Format(vk - VK_NUMPAD0);
        return true;
    }
    if (VK_F1 <= vk && vk <= VK_F24) {
        *key_name = U"F" + Format(vk - VK_F1 + 1);
        return true;
    }

    switch (vk) {
    case VK_CANCEL: *key_name = U"Cancel"; return true;
    case VK_BACK: *key_name = U"Backspace"; return true;
    case VK_TAB: *key_name = U"Tab"; return true;
    case VK_CLEAR: *key_name = U"Clear"; return true;
    case VK_RETURN: *key_name = U"Enter"; return true;
    case VK_SHIFT: *key_name = U"Shift"; return true;
    case VK_CONTROL: *key_name = U"Ctrl"; return true;
    case VK_MENU: *key_name = U"Alt"; return true;
    case VK_PAUSE: *key_name = U"Pause"; return true;
    case VK_CAPITAL: *key_name = U"CapsLock"; return true;
    case VK_KANA: *key_name = U"Kana"; return true;
#ifdef VK_IME_ON
    case VK_IME_ON: *key_name = U"IMEOn"; return true;
#endif
    case VK_JUNJA: *key_name = U"Junja"; return true;
    case VK_FINAL: *key_name = U"Final"; return true;
    case VK_HANJA: *key_name = U"Kanji"; return true;
#ifdef VK_IME_OFF
    case VK_IME_OFF: *key_name = U"IMEOff"; return true;
#endif
    case VK_ESCAPE: *key_name = U"Escape"; return true;
    case VK_CONVERT: *key_name = U"Convert"; return true;
    case VK_NONCONVERT: *key_name = U"NonConvert"; return true;
    case VK_ACCEPT: *key_name = U"Accept"; return true;
    case VK_MODECHANGE: *key_name = U"ModeChange"; return true;
    case VK_SPACE: *key_name = U"Space"; return true;
    case VK_PRIOR: *key_name = U"PageUp"; return true;
    case VK_NEXT: *key_name = U"PageDown"; return true;
    case VK_END: *key_name = U"End"; return true;
    case VK_HOME: *key_name = U"Home"; return true;
    case VK_LEFT: *key_name = U"Left"; return true;
    case VK_UP: *key_name = U"Up"; return true;
    case VK_RIGHT: *key_name = U"Right"; return true;
    case VK_DOWN: *key_name = U"Down"; return true;
    case VK_SELECT: *key_name = U"Select"; return true;
    case VK_PRINT: *key_name = U"Print"; return true;
    case VK_EXECUTE: *key_name = U"Execute"; return true;
    case VK_SNAPSHOT: *key_name = U"PrintScreen"; return true;
    case VK_INSERT: *key_name = U"Insert"; return true;
    case VK_DELETE: *key_name = U"Delete"; return true;
    case VK_HELP: *key_name = U"Help"; return true;
    case VK_LWIN: *key_name = U"0x5b"; return true;
    case VK_RWIN: *key_name = U"0x5c"; return true;
    case VK_APPS: *key_name = U"Apps"; return true;
    case VK_SLEEP: *key_name = U"Sleep"; return true;
    case VK_MULTIPLY: *key_name = U"NumMultiply"; return true;
    case VK_ADD: *key_name = U"NumAdd"; return true;
    case VK_SEPARATOR: *key_name = U"NumSeparator"; return true;
    case VK_SUBTRACT: *key_name = U"NumSubtract"; return true;
    case VK_DECIMAL: *key_name = U"NumDecimal"; return true;
    case VK_DIVIDE: *key_name = U"NumDivide"; return true;
    case VK_NUMLOCK: *key_name = U"NumLock"; return true;
    case VK_SCROLL: *key_name = U"ScrollLock"; return true;
    case VK_LSHIFT: *key_name = U"Left Shift"; return true;
    case VK_RSHIFT: *key_name = U"Right Shift"; return true;
    case VK_LCONTROL: *key_name = U"Left Ctrl"; return true;
    case VK_RCONTROL: *key_name = U"Right Ctrl"; return true;
    case VK_LMENU: *key_name = U"Left Alt"; return true;
    case VK_RMENU: *key_name = U"Right Alt"; return true;
    case VK_BROWSER_BACK: *key_name = U"BrowserBack"; return true;
    case VK_BROWSER_FORWARD: *key_name = U"BrowserForward"; return true;
    case VK_BROWSER_REFRESH: *key_name = U"BrowserRefresh"; return true;
    case VK_BROWSER_STOP: *key_name = U"BrowserStop"; return true;
    case VK_BROWSER_SEARCH: *key_name = U"BrowserSearch"; return true;
    case VK_BROWSER_FAVORITES: *key_name = U"BrowserFavorites"; return true;
    case VK_BROWSER_HOME: *key_name = U"BrowserHome"; return true;
    case VK_VOLUME_MUTE: *key_name = U"VolumeMute"; return true;
    case VK_VOLUME_DOWN: *key_name = U"VolumeDown"; return true;
    case VK_VOLUME_UP: *key_name = U"VolumeUp"; return true;
    case VK_MEDIA_NEXT_TRACK: *key_name = U"NextTrack"; return true;
    case VK_MEDIA_PREV_TRACK: *key_name = U"PreviousTrack"; return true;
    case VK_MEDIA_STOP: *key_name = U"StopMedia"; return true;
    case VK_MEDIA_PLAY_PAUSE: *key_name = U"PlayPauseMedia"; return true;
    case VK_LAUNCH_MAIL: *key_name = U"LaunchMail"; return true;
    case VK_LAUNCH_MEDIA_SELECT: *key_name = U"LaunchMediaSelect"; return true;
    case VK_LAUNCH_APP1: *key_name = U"LaunchApp1"; return true;
    case VK_LAUNCH_APP2: *key_name = U"LaunchApp2"; return true;
    case VK_OEM_1: *key_name = U"Semicolon"; return true;
    case VK_OEM_PLUS: *key_name = U"Equal"; return true;
    case VK_OEM_COMMA: *key_name = U"Comma"; return true;
    case VK_OEM_MINUS: *key_name = U"Minus"; return true;
    case VK_OEM_PERIOD: *key_name = U"Period"; return true;
    case VK_OEM_2: *key_name = U"Slash"; return true;
    case VK_OEM_3: *key_name = U"GraveAccent"; return true;
    case VK_OEM_4: *key_name = U"LBracket"; return true;
    case VK_OEM_5: *key_name = U"Backslash"; return true;
    case VK_OEM_6: *key_name = U"RBracket"; return true;
    case VK_OEM_7: *key_name = U"Apostrophe"; return true;
    case VK_OEM_8: *key_name = U"OEM8"; return true;
    case VK_OEM_102: *key_name = U"OEM102"; return true;
    case VK_ATTN: *key_name = U"Attn"; return true;
    case VK_CRSEL: *key_name = U"CrSel"; return true;
    case VK_EXSEL: *key_name = U"ExSel"; return true;
    case VK_EREOF: *key_name = U"EraseEOF"; return true;
    case VK_PLAY: *key_name = U"Play"; return true;
    case VK_ZOOM: *key_name = U"Zoom"; return true;
    case VK_OEM_CLEAR: *key_name = U"OEMClear"; return true;
    default:
        return false;
    }
}
#endif

std::vector<String> get_all_inputs(bool *down_found) {
    const Array<Input> raw_keys = Keyboard::GetAllInputs();
    *down_found = false;
    std::unordered_set<String> keys;
    for (const auto& key : raw_keys) {
        String key_name = key.name();
#if SIV3D_PLATFORM(WINDOWS)
        String windows_key_name;
        if (key.deviceType() == InputDeviceType::Keyboard &&
            windows_shortcut_vk_to_key_name(static_cast<int>(key.code()), &windows_key_name)) {
            key_name = windows_key_name;
        }
        if (key_name == U"Left Ctrl" || key_name == U"Right Ctrl") {
            key_name = U"Ctrl";
        } else if (key_name == U"Left Shift" || key_name == U"Right Shift") {
            key_name = U"Shift";
        } else if (key_name == U"Left Alt" || key_name == U"Right Alt") {
            key_name = U"Alt";
        }
#endif
        // ignored keys
        if (ignore_keys.contains(key_name)) {
            continue;
        }
        *down_found |= key.down();
        keys.emplace(key_name);
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

inline std::string shortcut_key_list_to_string(const std::vector<String>& keys) {
    std::string res;
    for (int i = 0; i < (int)keys.size(); ++i) {
        if (i) {
            res += "+";
        }
        res += keys[i].narrow();
    }
    return res;
}

#if SIV3D_PLATFORM(WINDOWS)
inline String windows_shortcut_key_token(const String& key_name) {
    const String trimmed = key_name.trimmed();
    String token;
    for (const char32 ch : trimmed) {
        if (ch == U' ' || ch == U'_') {
            continue;
        }
        if (U'a' <= ch && ch <= U'z') {
            token += static_cast<char32>(U'A' + (ch - U'a'));
        } else {
            token += ch;
        }
    }
    return token;
}

inline bool windows_shortcut_parse_hex_vk(const String& key_name, int* vk) {
    const String trimmed = key_name.trimmed();
    if (trimmed.size() < 3 || trimmed[0] != U'0' || (trimmed[1] != U'x' && trimmed[1] != U'X')) {
        return false;
    }

    int value = 0;
    for (size_t i = 2; i < trimmed.size(); ++i) {
        const char32 ch = trimmed[i];
        int digit = -1;
        if (U'0' <= ch && ch <= U'9') {
            digit = static_cast<int>(ch - U'0');
        } else if (U'a' <= ch && ch <= U'f') {
            digit = 10 + static_cast<int>(ch - U'a');
        } else if (U'A' <= ch && ch <= U'F') {
            digit = 10 + static_cast<int>(ch - U'A');
        } else {
            return false;
        }
        value = value * 16 + digit;
        if (value > 0xFE) {
            return false;
        }
    }

    if (!windows_shortcut_scannable_vk(value)) {
        return false;
    }
    *vk = value;
    return true;
}

inline bool shortcut_key_name_to_windows_vk(const String& key_name, int* vk) {
    const String trimmed = key_name.trimmed();
    if (trimmed.size() == 1) {
        const char32 ch = trimmed[0];
        if (U'A' <= ch && ch <= U'Z') {
            *vk = static_cast<int>('A' + (ch - U'A'));
            return true;
        }
        if (U'a' <= ch && ch <= U'z') {
            *vk = static_cast<int>('A' + (ch - U'a'));
            return true;
        }
        if (U'0' <= ch && ch <= U'9') {
            *vk = static_cast<int>('0' + (ch - U'0'));
            return true;
        }
        switch (ch) {
        case U';':
        case U':':
            *vk = VK_OEM_1;
            return true;
        case U'=':
        case U'+':
            *vk = VK_OEM_PLUS;
            return true;
        case U',':
        case U'<':
            *vk = VK_OEM_COMMA;
            return true;
        case U'-':
            *vk = VK_OEM_MINUS;
            return true;
        case U'.':
        case U'>':
            *vk = VK_OEM_PERIOD;
            return true;
        case U'/':
        case U'?':
            *vk = VK_OEM_2;
            return true;
        case U'`':
        case U'~':
            *vk = VK_OEM_3;
            return true;
        case U'[':
        case U'{':
            *vk = VK_OEM_4;
            return true;
        case U'\\':
        case U'|':
            *vk = VK_OEM_5;
            return true;
        case U']':
        case U'}':
            *vk = VK_OEM_6;
            return true;
        case U'\'':
        case U'"':
        case U'^':
            *vk = VK_OEM_7;
            return true;
        default:
            break;
        }
    }

    if (windows_shortcut_parse_hex_vk(trimmed, vk)) {
        return true;
    }

    const String token = windows_shortcut_key_token(trimmed);
    if (token.size() >= 2 && token[0] == U'F') {
        int function_key_number = 0;
        bool valid_function_key_name = true;
        for (size_t i = 1; i < token.size(); ++i) {
            const char32 ch = token[i];
            if (ch < U'0' || U'9' < ch) {
                valid_function_key_name = false;
                break;
            }
            function_key_number = function_key_number * 10 + static_cast<int>(ch - U'0');
        }
        if (valid_function_key_name && 1 <= function_key_number && function_key_number <= 24) {
            *vk = VK_F1 + function_key_number - 1;
            return true;
        }
    }

    auto parse_numpad_digit = [&](const String& prefix) {
        if (!token.starts_with(prefix) || token.size() != prefix.size() + 1) {
            return false;
        }
        const char32 digit = token[prefix.size()];
        if (digit < U'0' || U'9' < digit) {
            return false;
        }
        *vk = VK_NUMPAD0 + static_cast<int>(digit - U'0');
        return true;
    };
    if (parse_numpad_digit(U"NUM") || parse_numpad_digit(U"NUMPAD") || parse_numpad_digit(U"KEYNUM")) {
        return true;
    }

    static const std::unordered_map<String, int> key_map = {
        {U"Ctrl", VK_CONTROL},
        {U"CTRL", VK_CONTROL},
        {U"CONTROL", VK_CONTROL},
        {U"Left Ctrl", VK_LCONTROL},
        {U"LEFTCTRL", VK_LCONTROL},
        {U"LCTRL", VK_LCONTROL},
        {U"LCONTROL", VK_LCONTROL},
        {U"Right Ctrl", VK_RCONTROL},
        {U"RIGHTCTRL", VK_RCONTROL},
        {U"RCTRL", VK_RCONTROL},
        {U"RCONTROL", VK_RCONTROL},
        {U"Shift", VK_SHIFT},
        {U"SHIFT", VK_SHIFT},
        {U"Left Shift", VK_LSHIFT},
        {U"LEFTSHIFT", VK_LSHIFT},
        {U"LSHIFT", VK_LSHIFT},
        {U"Right Shift", VK_RSHIFT},
        {U"RIGHTSHIFT", VK_RSHIFT},
        {U"RSHIFT", VK_RSHIFT},
        {U"Alt", VK_MENU},
        {U"ALT", VK_MENU},
        {U"MENU", VK_MENU},
        {U"Left Alt", VK_LMENU},
        {U"LEFTALT", VK_LMENU},
        {U"LALT", VK_LMENU},
        {U"LMENU", VK_LMENU},
        {U"Right Alt", VK_RMENU},
        {U"RIGHTALT", VK_RMENU},
        {U"RALT", VK_RMENU},
        {U"RMENU", VK_RMENU},
        {U"Space", VK_SPACE},
        {U"SPACE", VK_SPACE},
        {U"Backspace", VK_BACK},
        {U"BACKSPACE", VK_BACK},
        {U"BACK", VK_BACK},
        {U"Tab", VK_TAB},
        {U"TAB", VK_TAB},
        {U"Enter", VK_RETURN},
        {U"ENTER", VK_RETURN},
        {U"RETURN", VK_RETURN},
        {U"Escape", VK_ESCAPE},
        {U"ESCAPE", VK_ESCAPE},
        {U"ESC", VK_ESCAPE},
        {U"Left", VK_LEFT},
        {U"LEFT", VK_LEFT},
        {U"Right", VK_RIGHT},
        {U"RIGHT", VK_RIGHT},
        {U"Up", VK_UP},
        {U"UP", VK_UP},
        {U"Down", VK_DOWN},
        {U"DOWN", VK_DOWN},
        {U"Home", VK_HOME},
        {U"HOME", VK_HOME},
        {U"End", VK_END},
        {U"END", VK_END},
        {U"PageUp", VK_PRIOR},
        {U"PAGEUP", VK_PRIOR},
        {U"PGUP", VK_PRIOR},
        {U"PRIOR", VK_PRIOR},
        {U"PageDown", VK_NEXT},
        {U"PAGEDOWN", VK_NEXT},
        {U"PGDN", VK_NEXT},
        {U"NEXT", VK_NEXT},
        {U"Insert", VK_INSERT},
        {U"INSERT", VK_INSERT},
        {U"INS", VK_INSERT},
        {U"Delete", VK_DELETE},
        {U"DELETE", VK_DELETE},
        {U"DEL", VK_DELETE},
        {U"Cancel", VK_CANCEL},
        {U"CANCEL", VK_CANCEL},
        {U"Clear", VK_CLEAR},
        {U"CLEAR", VK_CLEAR},
        {U"Pause", VK_PAUSE},
        {U"PAUSE", VK_PAUSE},
        {U"CapsLock", VK_CAPITAL},
        {U"CAPSLOCK", VK_CAPITAL},
        {U"Kana", VK_KANA},
        {U"KANA", VK_KANA},
        {U"HANGUL", VK_KANA},
#ifdef VK_IME_ON
        {U"IMEOn", VK_IME_ON},
        {U"IMEON", VK_IME_ON},
#endif
        {U"Junja", VK_JUNJA},
        {U"JUNJA", VK_JUNJA},
        {U"Final", VK_FINAL},
        {U"FINAL", VK_FINAL},
        {U"Kanji", VK_HANJA},
        {U"KANJI", VK_HANJA},
        {U"HANJA", VK_HANJA},
#ifdef VK_IME_OFF
        {U"IMEOff", VK_IME_OFF},
        {U"IMEOFF", VK_IME_OFF},
#endif
        {U"Convert", VK_CONVERT},
        {U"CONVERT", VK_CONVERT},
        {U"NonConvert", VK_NONCONVERT},
        {U"NONCONVERT", VK_NONCONVERT},
        {U"Accept", VK_ACCEPT},
        {U"ACCEPT", VK_ACCEPT},
        {U"ModeChange", VK_MODECHANGE},
        {U"MODECHANGE", VK_MODECHANGE},
        {U"Print", VK_PRINT},
        {U"PRINT", VK_PRINT},
        {U"PrintScreen", VK_SNAPSHOT},
        {U"PRINTSCREEN", VK_SNAPSHOT},
        {U"PRTSC", VK_SNAPSHOT},
        {U"Select", VK_SELECT},
        {U"SELECT", VK_SELECT},
        {U"Execute", VK_EXECUTE},
        {U"EXECUTE", VK_EXECUTE},
        {U"Help", VK_HELP},
        {U"HELP", VK_HELP},
        {U"0x5b", VK_LWIN},
        {U"0X5B", VK_LWIN},
        {U"Left Windows", VK_LWIN},
        {U"LEFTWINDOWS", VK_LWIN},
        {U"LWIN", VK_LWIN},
        {U"Left Command", VK_LWIN},
        {U"LEFTCOMMAND", VK_LWIN},
        {U"0x5c", VK_RWIN},
        {U"0X5C", VK_RWIN},
        {U"Right Windows", VK_RWIN},
        {U"RIGHTWINDOWS", VK_RWIN},
        {U"RWIN", VK_RWIN},
        {U"Right Command", VK_RWIN},
        {U"RIGHTCOMMAND", VK_RWIN},
        {U"Apps", VK_APPS},
        {U"APPS", VK_APPS},
        {U"Sleep", VK_SLEEP},
        {U"SLEEP", VK_SLEEP},
        {U"NumMultiply", VK_MULTIPLY},
        {U"NUMMULTIPLY", VK_MULTIPLY},
        {U"NUMSTAR", VK_MULTIPLY},
        {U"NumAdd", VK_ADD},
        {U"NUMADD", VK_ADD},
        {U"NUMPLUS", VK_ADD},
        {U"NumSeparator", VK_SEPARATOR},
        {U"NUMSEPARATOR", VK_SEPARATOR},
        {U"NumSubtract", VK_SUBTRACT},
        {U"NUMSUBTRACT", VK_SUBTRACT},
        {U"NUMMINUS", VK_SUBTRACT},
        {U"NumDecimal", VK_DECIMAL},
        {U"NUMDECIMAL", VK_DECIMAL},
        {U"NUMPERIOD", VK_DECIMAL},
        {U"NumDivide", VK_DIVIDE},
        {U"NUMDIVIDE", VK_DIVIDE},
        {U"NUMSLASH", VK_DIVIDE},
        {U"NumLock", VK_NUMLOCK},
        {U"NUMLOCK", VK_NUMLOCK},
        {U"ScrollLock", VK_SCROLL},
        {U"SCROLLLOCK", VK_SCROLL},
        {U"NextTrack", VK_MEDIA_NEXT_TRACK},
        {U"NEXTTRACK", VK_MEDIA_NEXT_TRACK},
        {U"PreviousTrack", VK_MEDIA_PREV_TRACK},
        {U"PREVIOUSTRACK", VK_MEDIA_PREV_TRACK},
        {U"StopMedia", VK_MEDIA_STOP},
        {U"STOPMEDIA", VK_MEDIA_STOP},
        {U"PlayPauseMedia", VK_MEDIA_PLAY_PAUSE},
        {U"PLAYPAUSEMEDIA", VK_MEDIA_PLAY_PAUSE},
        {U"BrowserBack", VK_BROWSER_BACK},
        {U"BROWSERBACK", VK_BROWSER_BACK},
        {U"BrowserForward", VK_BROWSER_FORWARD},
        {U"BROWSERFORWARD", VK_BROWSER_FORWARD},
        {U"BrowserRefresh", VK_BROWSER_REFRESH},
        {U"BROWSERREFRESH", VK_BROWSER_REFRESH},
        {U"BrowserStop", VK_BROWSER_STOP},
        {U"BROWSERSTOP", VK_BROWSER_STOP},
        {U"BrowserSearch", VK_BROWSER_SEARCH},
        {U"BROWSERSEARCH", VK_BROWSER_SEARCH},
        {U"BrowserFavorites", VK_BROWSER_FAVORITES},
        {U"BROWSERFAVORITES", VK_BROWSER_FAVORITES},
        {U"BrowserHome", VK_BROWSER_HOME},
        {U"BROWSERHOME", VK_BROWSER_HOME},
        {U"VolumeMute", VK_VOLUME_MUTE},
        {U"VOLUMEMUTE", VK_VOLUME_MUTE},
        {U"VolumeDown", VK_VOLUME_DOWN},
        {U"VOLUMEDOWN", VK_VOLUME_DOWN},
        {U"VolumeUp", VK_VOLUME_UP},
        {U"VOLUMEUP", VK_VOLUME_UP},
        {U"LaunchMail", VK_LAUNCH_MAIL},
        {U"LAUNCHMAIL", VK_LAUNCH_MAIL},
        {U"LaunchMediaSelect", VK_LAUNCH_MEDIA_SELECT},
        {U"LAUNCHMEDIASELECT", VK_LAUNCH_MEDIA_SELECT},
        {U"LaunchApp1", VK_LAUNCH_APP1},
        {U"LAUNCHAPP1", VK_LAUNCH_APP1},
        {U"LaunchApp2", VK_LAUNCH_APP2},
        {U"LAUNCHAPP2", VK_LAUNCH_APP2},
        {U"Semicolon", VK_OEM_1},
        {U"SEMICOLON", VK_OEM_1},
        {U"Semicolon_US", VK_OEM_1},
        {U"SEMICOLONUS", VK_OEM_1},
        {U"Colon_JIS", VK_OEM_1},
        {U"COLONJIS", VK_OEM_1},
        {U"Equal", VK_OEM_PLUS},
        {U"EQUAL", VK_OEM_PLUS},
        {U"Equal_US", VK_OEM_PLUS},
        {U"EQUALUS", VK_OEM_PLUS},
        {U"Semicolon_JIS", VK_OEM_PLUS},
        {U"SEMICOLONJIS", VK_OEM_PLUS},
        {U"Comma", VK_OEM_COMMA},
        {U"COMMA", VK_OEM_COMMA},
        {U"Minus", VK_OEM_MINUS},
        {U"MINUS", VK_OEM_MINUS},
        {U"Period", VK_OEM_PERIOD},
        {U"PERIOD", VK_OEM_PERIOD},
        {U"Slash", VK_OEM_2},
        {U"SLASH", VK_OEM_2},
        {U"GraveAccent", VK_OEM_3},
        {U"GRAVEACCENT", VK_OEM_3},
        {U"LBracket", VK_OEM_4},
        {U"LBRACKET", VK_OEM_4},
        {U"LeftBracket", VK_OEM_4},
        {U"LEFTBRACKET", VK_OEM_4},
        {U"Backslash", VK_OEM_5},
        {U"BACKSLASH", VK_OEM_5},
        {U"Yen_JIS", VK_OEM_5},
        {U"YENJIS", VK_OEM_5},
        {U"RBracket", VK_OEM_6},
        {U"RBRACKET", VK_OEM_6},
        {U"RightBracket", VK_OEM_6},
        {U"RIGHTBRACKET", VK_OEM_6},
        {U"Apostrophe", VK_OEM_7},
        {U"APOSTROPHE", VK_OEM_7},
        {U"Apostrophe_US", VK_OEM_7},
        {U"APOSTROPHEUS", VK_OEM_7},
        {U"Caret_JIS", VK_OEM_7},
        {U"CARETJIS", VK_OEM_7},
        {U"OEM8", VK_OEM_8},
        {U"OEM102", VK_OEM_102},
        {U"Underscore_JIS", VK_OEM_102},
        {U"UNDERSCOREJIS", VK_OEM_102},
        {U"Attn", VK_ATTN},
        {U"ATTN", VK_ATTN},
        {U"CrSel", VK_CRSEL},
        {U"CRSEL", VK_CRSEL},
        {U"ExSel", VK_EXSEL},
        {U"EXSEL", VK_EXSEL},
        {U"EraseEOF", VK_EREOF},
        {U"ERASEEOF", VK_EREOF},
        {U"Play", VK_PLAY},
        {U"PLAY", VK_PLAY},
        {U"Zoom", VK_ZOOM},
        {U"ZOOM", VK_ZOOM},
        {U"OEMClear", VK_OEM_CLEAR},
        {U"OEMCLEAR", VK_OEM_CLEAR},
    };

    auto it = key_map.find(trimmed);
    if (it == key_map.end()) {
        it = key_map.find(token);
    }
    if (it == key_map.end() || !windows_shortcut_scannable_vk(it->second)) {
        return false;
    }
    *vk = it->second;
    return true;
}

inline bool windows_vk_pressed(int vk) {
    return (GetAsyncKeyState(vk) & 0x8000) != 0;
}

struct Windows_shortcut_keyboard_state {
    bool focused = false;
    std::unordered_set<int> pressed_vks;
    std::unordered_set<int> down_vks;
};

inline Windows_shortcut_keyboard_state get_windows_shortcut_keyboard_state() {
    static std::unordered_set<int> previous_pressed_vks;
    static bool previous_focused = false;

    Windows_shortcut_keyboard_state state;
    state.focused = Window::GetState().focused;
    for (int vk = VK_BACK; vk <= 0xFE; ++vk) {
        if (windows_shortcut_scannable_vk(vk) && windows_vk_pressed(vk)) {
            state.pressed_vks.emplace(vk);
        }
    }

    if (state.focused && previous_focused) {
        for (const int vk : state.pressed_vks) {
            if (previous_pressed_vks.find(vk) == previous_pressed_vks.end()) {
                state.down_vks.emplace(vk);
            }
        }
    }

    previous_pressed_vks = state.pressed_vks;
    previous_focused = state.focused;
    return state;
}

inline bool windows_shortcut_vk_pressed(const Windows_shortcut_keyboard_state& keyboard_state, const int vk) {
    return keyboard_state.pressed_vks.find(vk) != keyboard_state.pressed_vks.end();
}

inline bool windows_shortcut_vk_down(const Windows_shortcut_keyboard_state& keyboard_state, const int vk) {
    return keyboard_state.down_vks.find(vk) != keyboard_state.down_vks.end();
}

inline bool windows_shortcut_key_name_found(const std::vector<String>& keys, const String& key_name) {
    const String expected_token = windows_shortcut_key_token(key_name);
    for (const String& key : keys) {
        if (windows_shortcut_key_token(key) == expected_token) {
            return true;
        }
    }
    return false;
}

inline bool windows_shortcut_modifier_family_matches(
    const Windows_shortcut_keyboard_state& keyboard_state,
    const std::vector<String>& keys,
    const String& generic_name,
    const String& left_name,
    const String& right_name,
    const int generic_vk,
    const int left_vk,
    const int right_vk
) {
    const bool needs_generic = windows_shortcut_key_name_found(keys, generic_name);
    const bool needs_left = windows_shortcut_key_name_found(keys, left_name);
    const bool needs_right = windows_shortcut_key_name_found(keys, right_name);
    const bool generic_pressed = windows_shortcut_vk_pressed(keyboard_state, generic_vk);
    const bool left_pressed = windows_shortcut_vk_pressed(keyboard_state, left_vk);
    const bool right_pressed = windows_shortcut_vk_pressed(keyboard_state, right_vk);
    const bool any_pressed = generic_pressed || left_pressed || right_pressed;

    if (!needs_generic && !needs_left && !needs_right) {
        return !any_pressed;
    }
    if (needs_left || needs_right) {
        return left_pressed == needs_left && right_pressed == needs_right;
    }

    const int physical_count = (left_pressed ? 1 : 0) + (right_pressed ? 1 : 0);
    if (physical_count > 1) {
        return false;
    }
    return any_pressed;
}

inline bool windows_shortcut_modifier_state_matches(
    const Windows_shortcut_keyboard_state& keyboard_state,
    const std::vector<String>& keys
) {
    return windows_shortcut_modifier_family_matches(keyboard_state, keys, U"Ctrl", U"Left Ctrl", U"Right Ctrl", VK_CONTROL, VK_LCONTROL, VK_RCONTROL) &&
        windows_shortcut_modifier_family_matches(keyboard_state, keys, U"Shift", U"Left Shift", U"Right Shift", VK_SHIFT, VK_LSHIFT, VK_RSHIFT) &&
        windows_shortcut_modifier_family_matches(keyboard_state, keys, U"Alt", U"Left Alt", U"Right Alt", VK_MENU, VK_LMENU, VK_RMENU);
}

inline bool windows_shortcut_non_modifier_state_matches(
    const Windows_shortcut_keyboard_state& keyboard_state,
    const std::unordered_set<int>& expected_vks
) {
    for (const int vk : keyboard_state.pressed_vks) {
        if (!windows_shortcut_modifier_vk(vk) && expected_vks.find(vk) == expected_vks.end()) {
            return false;
        }
    }
    return true;
}

inline bool windows_shortcut_expected_key_down_found(
    const Windows_shortcut_keyboard_state& keyboard_state,
    const std::unordered_set<int>& expected_vks
) {
    for (const int vk : expected_vks) {
        if (windows_shortcut_vk_down(keyboard_state, vk)) {
            return true;
        }
    }
    return false;
}

inline bool check_windows_shortcut_key_state(
    const Shortcut_key_elem& elem,
    const Windows_shortcut_keyboard_state& keyboard_state,
    bool* down_found,
    bool* pressed_found
) {
    *down_found = false;
    *pressed_found = false;
    if (elem.keys.empty()) {
        return false;
    }

    std::unordered_set<int> expected_vks;
    for (const String& key : elem.keys) {
        int vk = 0;
        if (!shortcut_key_name_to_windows_vk(key, &vk)) {
            return false;
        }
        expected_vks.emplace(vk);
    }

    if (!keyboard_state.focused) {
        return true;
    }

    bool pressed = windows_shortcut_modifier_state_matches(keyboard_state, elem.keys) &&
        windows_shortcut_non_modifier_state_matches(keyboard_state, expected_vks);
    for (const String& key : elem.keys) {
        int vk = 0;
        shortcut_key_name_to_windows_vk(key, &vk);
        pressed &= windows_shortcut_vk_pressed(keyboard_state, vk);
    }

    *pressed_found = pressed;
    *down_found = pressed && windows_shortcut_expected_key_down_found(keyboard_state, expected_vks);
    return true;
}

inline std::vector<String> get_windows_shortcut_inputs_for_diagnostics() {
    std::vector<String> keys;
    auto append_if_pressed = [&](const String& name, const int vk) {
        if (windows_vk_pressed(vk)) {
            keys.emplace_back(name);
        }
    };

    append_if_pressed(U"Ctrl", VK_CONTROL);
    append_if_pressed(U"Shift", VK_SHIFT);
    append_if_pressed(U"Alt", VK_MENU);

    for (int vk = VK_BACK; vk <= 0xFE; ++vk) {
        if (!windows_shortcut_scannable_vk(vk) || windows_shortcut_modifier_vk(vk) || !windows_vk_pressed(vk)) {
            continue;
        }
        String key_name;
        if (windows_shortcut_vk_to_key_name(vk, &key_name)) {
            keys.emplace_back(key_name);
        }
    }
    return keys;
}
#endif

inline void shortcut_diagnostic_log(
    const std::vector<String>& raw_keys,
    const bool raw_down_found,
    const String& shortcut_name_down,
    const String& shortcut_name_pressed
) {
#if SIV3D_PLATFORM(WINDOWS)
    char* env_path = nullptr;
    size_t env_path_size = 0;
    if (_dupenv_s(&env_path, &env_path_size, "EGAROUCID_SHORTCUT_DIAG_LOG") != 0 || env_path == nullptr || env_path[0] == '\0') {
        if (env_path) {
            free(env_path);
        }
        return;
    }
    const std::string path = env_path;
    free(env_path);
#else
    const char* env_path = std::getenv("EGAROUCID_SHORTCUT_DIAG_LOG");
    if (env_path == nullptr || env_path[0] == '\0') {
        return;
    }
    const std::string path = env_path;
#endif

    std::string raw = shortcut_key_list_to_string(raw_keys);
#if SIV3D_PLATFORM(WINDOWS)
    std::string normalized = shortcut_key_list_to_string(get_windows_shortcut_inputs_for_diagnostics());
#else
    std::string normalized = raw;
#endif
    if (raw.empty() && normalized.empty() &&
        shortcut_name_down == SHORTCUT_KEY_UNDEFINED &&
        shortcut_name_pressed == SHORTCUT_KEY_UNDEFINED) {
        return;
    }

    static std::string previous_line;
    std::ostringstream oss;
    oss << "raw=" << raw
        << "\trawDown=" << (raw_down_found ? "1" : "0")
        << "\twin=" << normalized
        << "\tdown=" << shortcut_name_down.narrow()
        << "\tpressed=" << shortcut_name_pressed.narrow();

    const std::string line = oss.str();
    if (line == previous_line) {
        return;
    }
    previous_line = line;

    std::ofstream fout(path, std::ios::app);
    if (fout) {
        fout << line << '\n';
    }
}

inline bool shortcut_diagnostic_only_mode() {
#if SIV3D_PLATFORM(WINDOWS)
    char* value = nullptr;
    size_t value_size = 0;
    const bool enabled = (_dupenv_s(&value, &value_size, "EGAROUCID_SHORTCUT_DIAG_ONLY") == 0 &&
        value != nullptr && value[0] == '1' && value[1] == '\0');
    if (value) {
        free(value);
    }
    return enabled;
#else
    const char* value = std::getenv("EGAROUCID_SHORTCUT_DIAG_ONLY");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
#endif
}

inline void clear_shortcut_result_for_diagnostic_only(String* shortcut_name_down, String* shortcut_name_pressed) {
    if (shortcut_diagnostic_only_mode()) {
        *shortcut_name_down = SHORTCUT_KEY_UNDEFINED;
        *shortcut_name_pressed = SHORTCUT_KEY_UNDEFINED;
    }
}

class Shortcut_keys {
public:
    std::vector<Shortcut_key_elem> shortcut_keys;
private:
    const Directories* directories = nullptr;
    std::vector<Shortcut_key_elem> current_default_shortcut_keys;
public:
    void set_default() {
        refresh_default_shortcut_keys();
        shortcut_keys = current_default_shortcut_keys;
    }

    void set_empty() {
        refresh_default_shortcut_keys();
        shortcut_keys = current_default_shortcut_keys;
        for (Shortcut_key_elem &elem: shortcut_keys) {
            elem.keys.clear();
        }
    }

    void init(String file, const Directories* directories_) {
        directories = directories_;
        set_empty();
        JSON json = JSON::Load(file);
        if (not json) {
            set_default();
        } else {
            std::unordered_set<String> name_list;
            std::unordered_set<String> loaded_name_list;
            for (Shortcut_key_elem &elem: current_default_shortcut_keys) {
                name_list.emplace(elem.name);
            }
            for (const auto& object: json) {
                if (name_list.find(object.key) == name_list.end()) {
                    std::cerr << "ERR shortcut key name not found " << object.key.narrow() << std::endl;
                    continue;
                }
                loaded_name_list.emplace(object.key);
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
            auto shortcut_conflicts = [&](const std::vector<String>& lhs, const std::vector<String>& rhs) {
                if (lhs.size() != rhs.size()) {
                    return false;
                }
                for (const String& key : lhs) {
                    if (std::find(rhs.begin(), rhs.end(), key) == rhs.end()) {
                        return false;
                    }
                }
                return true;
            };
            for (int i = 0; i < (int)shortcut_keys.size(); ++i) {
                if (loaded_name_list.find(shortcut_keys[i].name) != loaded_name_list.end()) {
                    continue;
                }
                const std::vector<String>& default_keys = current_default_shortcut_keys[i].keys;
                if (default_keys.empty()) {
                    continue;
                }
                bool conflicted = false;
                for (int j = 0; j < (int)shortcut_keys.size(); ++j) {
                    if (i == j || shortcut_keys[j].keys.empty()) {
                        continue;
                    }
                    if (shortcut_conflicts(default_keys, shortcut_keys[j].keys)) {
                        conflicted = true;
                        break;
                    }
                }
                if (!conflicted) {
                    shortcut_keys[i].keys = default_keys;
                }
            }
        }
    }

    void init(String file) {
        init(file, directories);
    }

    void sync_dynamic_shortcut_keys(const Directories* directories_) {
        directories = directories_;
        std::unordered_map<String, std::vector<String>> current_keys;
        for (const Shortcut_key_elem& elem : shortcut_keys) {
            current_keys[elem.name] = elem.keys;
        }

        refresh_default_shortcut_keys();
        shortcut_keys = current_default_shortcut_keys;
        for (Shortcut_key_elem& elem : shortcut_keys) {
            auto it = current_keys.find(elem.name);
            if (it != current_keys.end()) {
                elem.keys = it->second;
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
#if SIV3D_PLATFORM(WINDOWS)
        const Windows_shortcut_keyboard_state windows_keyboard_state = get_windows_shortcut_keyboard_state();
#endif
        for (const Shortcut_key_elem &elem: shortcut_keys) {
#if SIV3D_PLATFORM(WINDOWS)
            bool windows_down_found = false;
            bool windows_pressed_found = false;
            if (check_windows_shortcut_key_state(elem, windows_keyboard_state, &windows_down_found, &windows_pressed_found)) {
                if (windows_pressed_found) {
                    if (windows_down_found) {
                        *shortcut_name_down = elem.name;
                    }
                    *shortcut_name_pressed = elem.name;
                    shortcut_diagnostic_log(keys, down_found, *shortcut_name_down, *shortcut_name_pressed);
                    clear_shortcut_result_for_diagnostic_only(shortcut_name_down, shortcut_name_pressed);
                    return;
                }
                continue;
            }
#endif
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
                    shortcut_diagnostic_log(keys, down_found, *shortcut_name_down, *shortcut_name_pressed);
                    clear_shortcut_result_for_diagnostic_only(shortcut_name_down, shortcut_name_pressed);
                    return;
                }
            }
        }
        shortcut_diagnostic_log(keys, down_found, *shortcut_name_down, *shortcut_name_pressed);
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
                if (!elem.description_suffix.isEmpty()) {
                    if (!res.isEmpty()) {
                        res += U"> ";
                    }
                    res += elem.description_suffix;
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

private:
    void refresh_default_shortcut_keys() {
        current_default_shortcut_keys = shortcut_keys_default;
        std::vector<Shortcut_key_elem> ai_profile_shortcut_keys;
        append_ai_profile_shortcut_key_elems(&ai_profile_shortcut_keys, directories);
        auto insert_pos = std::find_if(
            current_default_shortcut_keys.begin(),
            current_default_shortcut_keys.end(),
            [](const Shortcut_key_elem& elem) { return elem.name == U"ai_profile_load"; });
        if (insert_pos != current_default_shortcut_keys.end()) {
            ++insert_pos;
        }
        current_default_shortcut_keys.insert(insert_pos, ai_profile_shortcut_keys.begin(), ai_profile_shortcut_keys.end());

        std::vector<Shortcut_key_elem> display_profile_shortcut_keys;
        append_display_profile_shortcut_key_elems(&display_profile_shortcut_keys, directories);
        auto display_insert_pos = std::find_if(
            current_default_shortcut_keys.begin(),
            current_default_shortcut_keys.end(),
            [](const Shortcut_key_elem& elem) { return elem.name == U"display_profile_load"; });
        if (display_insert_pos != current_default_shortcut_keys.end()) {
            ++display_insert_pos;
        }
        current_default_shortcut_keys.insert(display_insert_pos, display_profile_shortcut_keys.begin(), display_profile_shortcut_keys.end());
    }
};

Shortcut_keys shortcut_keys;
