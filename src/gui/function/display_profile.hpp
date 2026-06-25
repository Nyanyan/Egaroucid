/*
    Egaroucid Project

    @file display_profile.hpp
        Display profile utility
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once

#include <algorithm>
#include <functional>
#include "const/gui_common.hpp"

struct Display_profile_values {
    bool use_disc_hint{ true };
    int n_disc_hint{ SHOW_ALL_HINT };
    bool show_hint_level{ true };
    bool use_umigame_value{ false };
    int umigame_value_depth{ 60 };
    int umigame_value_score_min{ UMIGAME_VALUE_SCORE_MIN };
    int umigame_value_score_max{ UMIGAME_VALUE_SCORE_MAX };
    int umigame_value_integration_error{ 0 };
    bool show_legal{ true };
    bool show_graph{ true };
    bool show_opening_on_cell{ true };
    bool show_stable_discs{ false };
    bool show_to_be_flipped_discs{ false };
    bool show_last_flipped_discs{ false };
    bool show_play_ordering{ false };
    bool play_ordering_board_format{ true };
    bool play_ordering_transcript_format{ false };
    bool show_laser_pointer{ false };
    bool show_log{ true };
    bool show_last_move{ true };
    bool show_next_move{ true };
    bool show_next_move_change_view{ false };
    bool change_color_type{ false };
    bool show_book_accuracy{ false };
    bool show_graph_value{ true };
    bool show_graph_sum_of_loss{ false };
    bool show_random_board_graph{ false };
    bool show_opening_name{ true };
    bool show_principal_variation{ true };
    bool show_timer{ false };
    bool show_ai_focus{ false };
    int pv_length{ 7 };
    bool show_value_when_ai_calculating{ false };
    bool show_endgame_error{ false };
    bool show_endgame_error_40_to_60{ true };
    bool show_endgame_error_41_to_60{ false };
    bool hint_colorize{ false };
};

inline String get_display_settings_dir(const Directories& directories) {
    return Unicode::Widen(directories.appdata_dir) + U"display_settings/";
}

inline String get_display_settings_file_path(const Directories& directories, const String& file_name) {
    return get_display_settings_dir(directories) + file_name;
}

inline void ensure_display_settings_dir(const Directories& directories) {
    const String dir = get_display_settings_dir(directories);
    if (!FileSystem::Exists(dir)) {
        FileSystem::CreateDirectories(dir);
    }
}

inline void normalize_display_profile_values(Display_profile_values* values) {
    values->n_disc_hint = std::clamp(values->n_disc_hint, 1, SHOW_ALL_HINT);
    values->umigame_value_depth = std::clamp(values->umigame_value_depth, 1, 60);
    values->umigame_value_score_min = normalize_umigame_score_slider_value(values->umigame_value_score_min);
    values->umigame_value_score_max = normalize_umigame_score_slider_value(values->umigame_value_score_max);
    values->umigame_value_score_min = std::clamp(values->umigame_value_score_min, UMIGAME_VALUE_SCORE_MIN, UMIGAME_VALUE_SCORE_MAX - UMIGAME_VALUE_SCORE_MIN_GAP);
    values->umigame_value_score_max = std::clamp(values->umigame_value_score_max, UMIGAME_VALUE_SCORE_MIN + UMIGAME_VALUE_SCORE_MIN_GAP, UMIGAME_VALUE_SCORE_MAX);
    if (values->umigame_value_score_min > values->umigame_value_score_max - UMIGAME_VALUE_SCORE_MIN_GAP) {
        values->umigame_value_score_min = values->umigame_value_score_max - UMIGAME_VALUE_SCORE_MIN_GAP;
    }
    values->umigame_value_integration_error = std::clamp(values->umigame_value_integration_error, UMIGAME_VALUE_INTEGRATION_ERROR_MIN, UMIGAME_VALUE_INTEGRATION_ERROR_MAX);
    values->pv_length = std::clamp(values->pv_length, PV_LENGTH_SETTING_MIN, PV_LENGTH_SETTING_MAX);

    if (values->play_ordering_board_format == values->play_ordering_transcript_format) {
        values->play_ordering_board_format = true;
        values->play_ordering_transcript_format = false;
    }
    if (values->show_graph_value == values->show_graph_sum_of_loss) {
        values->show_graph_value = true;
        values->show_graph_sum_of_loss = false;
    }
    if (values->show_endgame_error_40_to_60 == values->show_endgame_error_41_to_60) {
        values->show_endgame_error_40_to_60 = true;
        values->show_endgame_error_41_to_60 = false;
    }
}

inline Display_profile_values to_display_profile_values(const Settings& settings) {
    Display_profile_values values;
    values.use_disc_hint = settings.use_disc_hint;
    values.n_disc_hint = settings.n_disc_hint;
    values.show_hint_level = settings.show_hint_level;
    values.use_umigame_value = settings.use_umigame_value;
    values.umigame_value_depth = settings.umigame_value_depth;
    values.umigame_value_score_min = settings.umigame_value_score_min;
    values.umigame_value_score_max = settings.umigame_value_score_max;
    values.umigame_value_integration_error = settings.umigame_value_integration_error;
    values.show_legal = settings.show_legal;
    values.show_graph = settings.show_graph;
    values.show_opening_on_cell = settings.show_opening_on_cell;
    values.show_stable_discs = settings.show_stable_discs;
    values.show_to_be_flipped_discs = settings.show_to_be_flipped_discs;
    values.show_last_flipped_discs = settings.show_last_flipped_discs;
    values.show_play_ordering = settings.show_play_ordering;
    values.play_ordering_board_format = settings.play_ordering_board_format;
    values.play_ordering_transcript_format = settings.play_ordering_transcript_format;
    values.show_laser_pointer = settings.show_laser_pointer;
    values.show_log = settings.show_log;
    values.show_last_move = settings.show_last_move;
    values.show_next_move = settings.show_next_move;
    values.show_next_move_change_view = settings.show_next_move_change_view;
    values.change_color_type = settings.change_color_type;
    values.show_book_accuracy = settings.show_book_accuracy;
    values.show_graph_value = settings.show_graph_value;
    values.show_graph_sum_of_loss = settings.show_graph_sum_of_loss;
    values.show_random_board_graph = settings.show_random_board_graph;
    values.show_opening_name = settings.show_opening_name;
    values.show_principal_variation = settings.show_principal_variation;
    values.show_timer = settings.show_timer;
    values.show_ai_focus = settings.show_ai_focus;
    values.pv_length = settings.pv_length;
    values.show_value_when_ai_calculating = settings.show_value_when_ai_calculating;
    values.show_endgame_error = settings.show_endgame_error;
    values.show_endgame_error_40_to_60 = settings.show_endgame_error_40_to_60;
    values.show_endgame_error_41_to_60 = settings.show_endgame_error_41_to_60;
    values.hint_colorize = settings.hint_colorize;
    normalize_display_profile_values(&values);
    return values;
}

inline Display_profile_values to_display_profile_values(const Menu_elements& menu_elements) {
    Display_profile_values values;
    values.use_disc_hint = menu_elements.use_disc_hint;
    values.n_disc_hint = menu_elements.n_disc_hint;
    values.show_hint_level = menu_elements.show_hint_level;
    values.use_umigame_value = menu_elements.use_umigame_value;
    values.umigame_value_depth = menu_elements.umigame_value_depth;
    values.umigame_value_score_min = menu_elements.umigame_value_score_min;
    values.umigame_value_score_max = menu_elements.umigame_value_score_max;
    values.umigame_value_integration_error = menu_elements.umigame_value_integration_error;
    values.show_legal = menu_elements.show_legal;
    values.show_graph = menu_elements.show_graph;
    values.show_opening_on_cell = menu_elements.show_opening_on_cell;
    values.show_stable_discs = menu_elements.show_stable_discs;
    values.show_to_be_flipped_discs = menu_elements.show_to_be_flipped_discs;
    values.show_last_flipped_discs = menu_elements.show_last_flipped_discs;
    values.show_play_ordering = menu_elements.show_play_ordering;
    values.play_ordering_board_format = menu_elements.play_ordering_board_format;
    values.play_ordering_transcript_format = menu_elements.play_ordering_transcript_format;
    values.show_laser_pointer = menu_elements.show_laser_pointer;
    values.show_log = menu_elements.show_log;
    values.show_last_move = menu_elements.show_last_move;
    values.show_next_move = menu_elements.show_next_move;
    values.show_next_move_change_view = menu_elements.show_next_move_change_view;
    values.change_color_type = menu_elements.change_color_type;
    values.show_book_accuracy = menu_elements.show_book_accuracy;
    values.show_graph_value = menu_elements.show_graph_value;
    values.show_graph_sum_of_loss = menu_elements.show_graph_sum_of_loss;
    values.show_random_board_graph = menu_elements.show_random_board_graph;
    values.show_opening_name = menu_elements.show_opening_name;
    values.show_principal_variation = menu_elements.show_principal_variation;
    values.show_timer = menu_elements.show_timer;
    values.show_ai_focus = menu_elements.show_ai_focus;
    values.pv_length = menu_elements.pv_length;
    values.show_value_when_ai_calculating = menu_elements.show_value_when_ai_calculating;
    values.show_endgame_error = menu_elements.show_endgame_error;
    values.show_endgame_error_40_to_60 = menu_elements.show_endgame_error_40_to_60;
    values.show_endgame_error_41_to_60 = menu_elements.show_endgame_error_41_to_60;
    values.hint_colorize = menu_elements.hint_colorize;
    normalize_display_profile_values(&values);
    return values;
}

inline void apply_display_profile_values(const Display_profile_values& values, Settings* settings) {
    Display_profile_values normalized_values = values;
    normalize_display_profile_values(&normalized_values);
    settings->use_disc_hint = normalized_values.use_disc_hint;
    settings->n_disc_hint = normalized_values.n_disc_hint;
    settings->show_hint_level = normalized_values.show_hint_level;
    settings->use_umigame_value = normalized_values.use_umigame_value;
    settings->umigame_value_depth = normalized_values.umigame_value_depth;
    settings->umigame_value_score_min = normalized_values.umigame_value_score_min;
    settings->umigame_value_score_max = normalized_values.umigame_value_score_max;
    settings->umigame_value_integration_error = normalized_values.umigame_value_integration_error;
    settings->show_legal = normalized_values.show_legal;
    settings->show_graph = normalized_values.show_graph;
    settings->show_opening_on_cell = normalized_values.show_opening_on_cell;
    settings->show_stable_discs = normalized_values.show_stable_discs;
    settings->show_to_be_flipped_discs = normalized_values.show_to_be_flipped_discs;
    settings->show_last_flipped_discs = normalized_values.show_last_flipped_discs;
    settings->show_play_ordering = normalized_values.show_play_ordering;
    settings->play_ordering_board_format = normalized_values.play_ordering_board_format;
    settings->play_ordering_transcript_format = normalized_values.play_ordering_transcript_format;
    settings->show_laser_pointer = normalized_values.show_laser_pointer;
    settings->show_log = normalized_values.show_log;
    settings->show_last_move = normalized_values.show_last_move;
    settings->show_next_move = normalized_values.show_next_move;
    settings->show_next_move_change_view = normalized_values.show_next_move_change_view;
    settings->change_color_type = normalized_values.change_color_type;
    settings->show_book_accuracy = normalized_values.show_book_accuracy;
    settings->show_graph_value = normalized_values.show_graph_value;
    settings->show_graph_sum_of_loss = normalized_values.show_graph_sum_of_loss;
    settings->show_random_board_graph = normalized_values.show_random_board_graph;
    settings->show_opening_name = normalized_values.show_opening_name;
    settings->show_principal_variation = normalized_values.show_principal_variation;
    settings->show_timer = normalized_values.show_timer;
    settings->show_ai_focus = normalized_values.show_ai_focus;
    settings->pv_length = normalized_values.pv_length;
    settings->show_value_when_ai_calculating = normalized_values.show_value_when_ai_calculating;
    settings->show_endgame_error = normalized_values.show_endgame_error;
    settings->show_endgame_error_40_to_60 = normalized_values.show_endgame_error_40_to_60;
    settings->show_endgame_error_41_to_60 = normalized_values.show_endgame_error_41_to_60;
    settings->hint_colorize = normalized_values.hint_colorize;
}

inline void apply_display_profile_values(const Display_profile_values& values, Menu_elements* menu_elements) {
    Display_profile_values normalized_values = values;
    normalize_display_profile_values(&normalized_values);
    menu_elements->use_disc_hint = normalized_values.use_disc_hint;
    menu_elements->n_disc_hint = normalized_values.n_disc_hint;
    menu_elements->show_hint_level = normalized_values.show_hint_level;
    menu_elements->use_umigame_value = normalized_values.use_umigame_value;
    menu_elements->umigame_value_depth = normalized_values.umigame_value_depth;
    menu_elements->umigame_value_score_min = normalized_values.umigame_value_score_min;
    menu_elements->umigame_value_score_max = normalized_values.umigame_value_score_max;
    menu_elements->umigame_value_integration_error = normalized_values.umigame_value_integration_error;
    menu_elements->show_legal = normalized_values.show_legal;
    menu_elements->show_graph = normalized_values.show_graph;
    menu_elements->show_opening_on_cell = normalized_values.show_opening_on_cell;
    menu_elements->show_stable_discs = normalized_values.show_stable_discs;
    menu_elements->show_to_be_flipped_discs = normalized_values.show_to_be_flipped_discs;
    menu_elements->show_last_flipped_discs = normalized_values.show_last_flipped_discs;
    menu_elements->show_play_ordering = normalized_values.show_play_ordering;
    menu_elements->play_ordering_board_format = normalized_values.play_ordering_board_format;
    menu_elements->play_ordering_transcript_format = normalized_values.play_ordering_transcript_format;
    menu_elements->show_laser_pointer = normalized_values.show_laser_pointer;
    menu_elements->show_log = normalized_values.show_log;
    menu_elements->show_last_move = normalized_values.show_last_move;
    menu_elements->show_next_move = normalized_values.show_next_move;
    menu_elements->show_next_move_change_view = normalized_values.show_next_move_change_view;
    menu_elements->change_color_type = normalized_values.change_color_type;
    menu_elements->show_book_accuracy = normalized_values.show_book_accuracy;
    menu_elements->show_graph_value = normalized_values.show_graph_value;
    menu_elements->show_graph_sum_of_loss = normalized_values.show_graph_sum_of_loss;
    menu_elements->show_random_board_graph = normalized_values.show_random_board_graph;
    menu_elements->show_opening_name = normalized_values.show_opening_name;
    menu_elements->show_principal_variation = normalized_values.show_principal_variation;
    menu_elements->show_timer = normalized_values.show_timer;
    menu_elements->show_ai_focus = normalized_values.show_ai_focus;
    menu_elements->pv_length = normalized_values.pv_length;
    menu_elements->show_value_when_ai_calculating = normalized_values.show_value_when_ai_calculating;
    menu_elements->show_endgame_error = normalized_values.show_endgame_error;
    menu_elements->show_endgame_error_40_to_60 = normalized_values.show_endgame_error_40_to_60;
    menu_elements->show_endgame_error_41_to_60 = normalized_values.show_endgame_error_41_to_60;
    menu_elements->hint_colorize = normalized_values.hint_colorize;
}

inline bool import_display_profile_bool(JSON& json, const String& key, bool* value) {
    if (json[key].getType() != JSONValueType::Bool) {
        return false;
    }
    *value = json[key].get<bool>();
    return true;
}

inline bool import_display_profile_int(JSON& json, const String& key, int* value) {
    if (json[key].getType() != JSONValueType::Number) {
        return false;
    }
    *value = static_cast<int>(json[key].get<double>());
    return true;
}

inline bool import_display_profile_name(JSON& json, String* name) {
    if (json[U"name"].getType() != JSONValueType::String) {
        return false;
    }
    *name = json[U"name"].getString();
    return true;
}

inline void export_display_profile_json(JSON& json, const Display_profile_values& values, const String& name) {
    Display_profile_values normalized_values = values;
    normalize_display_profile_values(&normalized_values);
    json[U"name"] = name;
    json[U"use_disc_hint"] = normalized_values.use_disc_hint;
    json[U"n_disc_hint"] = normalized_values.n_disc_hint;
    json[U"show_hint_level"] = normalized_values.show_hint_level;
    json[U"use_umigame_value"] = normalized_values.use_umigame_value;
    json[U"umigame_value_depth"] = normalized_values.umigame_value_depth;
    json[U"umigame_value_score_slider_version"] = UMIGAME_VALUE_SCORE_SLIDER_VERSION;
    json[U"umigame_value_score_min"] = normalized_values.umigame_value_score_min;
    json[U"umigame_value_score_max"] = normalized_values.umigame_value_score_max;
    json[U"umigame_value_integration_error"] = normalized_values.umigame_value_integration_error;
    json[U"show_legal"] = normalized_values.show_legal;
    json[U"show_graph"] = normalized_values.show_graph;
    json[U"show_opening_on_cell"] = normalized_values.show_opening_on_cell;
    json[U"show_stable_discs"] = normalized_values.show_stable_discs;
    json[U"show_to_be_flipped_discs"] = normalized_values.show_to_be_flipped_discs;
    json[U"show_last_flipped_discs"] = normalized_values.show_last_flipped_discs;
    json[U"show_play_ordering"] = normalized_values.show_play_ordering;
    json[U"play_ordering_board_format"] = normalized_values.play_ordering_board_format;
    json[U"play_ordering_transcript_format"] = normalized_values.play_ordering_transcript_format;
    json[U"show_laser_pointer"] = normalized_values.show_laser_pointer;
    json[U"show_log"] = normalized_values.show_log;
    json[U"show_last_move"] = normalized_values.show_last_move;
    json[U"show_next_move"] = normalized_values.show_next_move;
    json[U"show_next_move_change_view"] = normalized_values.show_next_move_change_view;
    json[U"change_color_type"] = normalized_values.change_color_type;
    json[U"show_book_accuracy"] = normalized_values.show_book_accuracy;
    json[U"show_graph_value"] = normalized_values.show_graph_value;
    json[U"show_graph_sum_of_loss"] = normalized_values.show_graph_sum_of_loss;
    json[U"show_random_board_graph"] = normalized_values.show_random_board_graph;
    json[U"show_opening_name"] = normalized_values.show_opening_name;
    json[U"show_principal_variation"] = normalized_values.show_principal_variation;
    json[U"show_timer"] = normalized_values.show_timer;
    json[U"show_ai_focus"] = normalized_values.show_ai_focus;
    json[U"pv_length"] = normalized_values.pv_length;
    json[U"show_value_when_ai_calculating"] = normalized_values.show_value_when_ai_calculating;
    json[U"show_endgame_error"] = normalized_values.show_endgame_error;
    json[U"show_endgame_error_40_to_60"] = normalized_values.show_endgame_error_40_to_60;
    json[U"show_endgame_error_41_to_60"] = normalized_values.show_endgame_error_41_to_60;
    json[U"hint_colorize"] = normalized_values.hint_colorize;
}

inline bool load_display_profile_values(const FilePath& path, Display_profile_values* values, String* profile_name) {
    JSON json = JSON::Load(path);
    if (json.size() == 0) {
        return false;
    }

    import_display_profile_bool(json, U"use_disc_hint", &values->use_disc_hint);
    import_display_profile_int(json, U"n_disc_hint", &values->n_disc_hint);
    import_display_profile_bool(json, U"show_hint_level", &values->show_hint_level);
    import_display_profile_bool(json, U"use_umigame_value", &values->use_umigame_value);
    import_display_profile_int(json, U"umigame_value_depth", &values->umigame_value_depth);
    import_display_profile_int(json, U"umigame_value_score_min", &values->umigame_value_score_min);
    import_display_profile_int(json, U"umigame_value_score_max", &values->umigame_value_score_max);
    if (json[U"umigame_value_score_slider_version"].getType() != JSONValueType::Number) {
        values->umigame_value_score_min = migrate_legacy_umigame_score_slider_value(values->umigame_value_score_min);
        values->umigame_value_score_max = migrate_legacy_umigame_score_slider_value(values->umigame_value_score_max);
    }
    import_display_profile_int(json, U"umigame_value_integration_error", &values->umigame_value_integration_error);
    import_display_profile_bool(json, U"show_legal", &values->show_legal);
    import_display_profile_bool(json, U"show_graph", &values->show_graph);
    import_display_profile_bool(json, U"show_opening_on_cell", &values->show_opening_on_cell);
    import_display_profile_bool(json, U"show_stable_discs", &values->show_stable_discs);
    import_display_profile_bool(json, U"show_to_be_flipped_discs", &values->show_to_be_flipped_discs);
    import_display_profile_bool(json, U"show_last_flipped_discs", &values->show_last_flipped_discs);
    import_display_profile_bool(json, U"show_play_ordering", &values->show_play_ordering);
    import_display_profile_bool(json, U"play_ordering_board_format", &values->play_ordering_board_format);
    import_display_profile_bool(json, U"play_ordering_transcript_format", &values->play_ordering_transcript_format);
    import_display_profile_bool(json, U"show_laser_pointer", &values->show_laser_pointer);
    import_display_profile_bool(json, U"show_log", &values->show_log);
    import_display_profile_bool(json, U"show_last_move", &values->show_last_move);
    import_display_profile_bool(json, U"show_next_move", &values->show_next_move);
    import_display_profile_bool(json, U"show_next_move_change_view", &values->show_next_move_change_view);
    import_display_profile_bool(json, U"change_color_type", &values->change_color_type);
    import_display_profile_bool(json, U"show_book_accuracy", &values->show_book_accuracy);
    import_display_profile_bool(json, U"show_graph_value", &values->show_graph_value);
    import_display_profile_bool(json, U"show_graph_sum_of_loss", &values->show_graph_sum_of_loss);
    import_display_profile_bool(json, U"show_random_board_graph", &values->show_random_board_graph);
    import_display_profile_bool(json, U"show_opening_name", &values->show_opening_name);
    import_display_profile_bool(json, U"show_principal_variation", &values->show_principal_variation);
    import_display_profile_bool(json, U"show_timer", &values->show_timer);
    import_display_profile_bool(json, U"show_ai_focus", &values->show_ai_focus);
    import_display_profile_int(json, U"pv_length", &values->pv_length);
    import_display_profile_bool(json, U"show_value_when_ai_calculating", &values->show_value_when_ai_calculating);
    import_display_profile_bool(json, U"show_endgame_error", &values->show_endgame_error);
    import_display_profile_bool(json, U"show_endgame_error_40_to_60", &values->show_endgame_error_40_to_60);
    import_display_profile_bool(json, U"show_endgame_error_41_to_60", &values->show_endgame_error_41_to_60);
    import_display_profile_bool(json, U"hint_colorize", &values->hint_colorize);
    normalize_display_profile_values(values);

    if (profile_name) {
        if (!import_display_profile_name(json, profile_name) || profile_name->trimmed().isEmpty()) {
            *profile_name = FileSystem::FileName(path);
        }
    }
    return true;
}

inline bool save_display_profile_values(const FilePath& path, const Display_profile_values& values, const String& profile_name) {
    JSON json;
    export_display_profile_json(json, values, profile_name);
    return json.save(path);
}

inline bool equals_display_profile_values(const Display_profile_values& lhs, const Display_profile_values& rhs) {
    if (lhs.use_disc_hint != rhs.use_disc_hint) return false;
    if (lhs.n_disc_hint != rhs.n_disc_hint) return false;
    if (lhs.show_hint_level != rhs.show_hint_level) return false;
    if (lhs.use_umigame_value != rhs.use_umigame_value) return false;
    if (lhs.umigame_value_depth != rhs.umigame_value_depth) return false;
    if (lhs.umigame_value_score_min != rhs.umigame_value_score_min) return false;
    if (lhs.umigame_value_score_max != rhs.umigame_value_score_max) return false;
    if (lhs.umigame_value_integration_error != rhs.umigame_value_integration_error) return false;
    if (lhs.show_legal != rhs.show_legal) return false;
    if (lhs.show_graph != rhs.show_graph) return false;
    if (lhs.show_opening_on_cell != rhs.show_opening_on_cell) return false;
    if (lhs.show_stable_discs != rhs.show_stable_discs) return false;
    if (lhs.show_to_be_flipped_discs != rhs.show_to_be_flipped_discs) return false;
    if (lhs.show_last_flipped_discs != rhs.show_last_flipped_discs) return false;
    if (lhs.show_play_ordering != rhs.show_play_ordering) return false;
    if (lhs.play_ordering_board_format != rhs.play_ordering_board_format) return false;
    if (lhs.play_ordering_transcript_format != rhs.play_ordering_transcript_format) return false;
    if (lhs.show_laser_pointer != rhs.show_laser_pointer) return false;
    if (lhs.show_log != rhs.show_log) return false;
    if (lhs.show_last_move != rhs.show_last_move) return false;
    if (lhs.show_next_move != rhs.show_next_move) return false;
    if (lhs.show_next_move_change_view != rhs.show_next_move_change_view) return false;
    if (lhs.change_color_type != rhs.change_color_type) return false;
    if (lhs.show_book_accuracy != rhs.show_book_accuracy) return false;
    if (lhs.show_graph_value != rhs.show_graph_value) return false;
    if (lhs.show_graph_sum_of_loss != rhs.show_graph_sum_of_loss) return false;
    if (lhs.show_random_board_graph != rhs.show_random_board_graph) return false;
    if (lhs.show_opening_name != rhs.show_opening_name) return false;
    if (lhs.show_principal_variation != rhs.show_principal_variation) return false;
    if (lhs.show_timer != rhs.show_timer) return false;
    if (lhs.show_ai_focus != rhs.show_ai_focus) return false;
    if (lhs.pv_length != rhs.pv_length) return false;
    if (lhs.show_value_when_ai_calculating != rhs.show_value_when_ai_calculating) return false;
    if (lhs.show_endgame_error != rhs.show_endgame_error) return false;
    if (lhs.show_endgame_error_40_to_60 != rhs.show_endgame_error_40_to_60) return false;
    if (lhs.show_endgame_error_41_to_60 != rhs.show_endgame_error_41_to_60) return false;
    if (lhs.hint_colorize != rhs.hint_colorize) return false;
    return true;
}

inline Array<FilePath> enumerate_display_profile_files(const Directories& directories) {
    ensure_display_settings_dir(directories);
    Array<FilePath> files;
    const String dir = get_display_settings_dir(directories);
    for (const auto& path : FileSystem::DirectoryContents(dir)) {
        if (FileSystem::IsFile(path) && path.ends_with(U".json")) {
            files << path;
        }
    }
    std::sort(files.begin(), files.end(), std::greater<FilePath>());
    return files;
}

inline String generate_unique_display_profile_filepath(const Directories& directories) {
    ensure_display_settings_dir(directories);
    const String dir = get_display_settings_dir(directories);
    String base_name = DateTime::Now().format(U"yyyyMMddHHmmss");
    String candidate = dir + base_name + U".json";
    int suffix = 1;
    while (FileSystem::Exists(candidate)) {
        candidate = dir + base_name + U"_" + Format(suffix) + U".json";
        ++suffix;
    }
    return candidate;
}

inline void ensure_default_display_profile(const Directories& directories, const Settings& settings, bool setting_json_exists) {
    ensure_display_settings_dir(directories);
    const String default_path = get_display_settings_file_path(directories, U"default.json");
    if (FileSystem::Exists(default_path)) {
        return;
    }

    if (setting_json_exists) {
        save_display_profile_values(default_path, to_display_profile_values(settings), U"default");
        return;
    }

    save_display_profile_values(default_path, to_display_profile_values(settings), U"default");
}

inline bool load_display_profile_into_settings(const Directories& directories, Settings* settings) {
    ensure_display_settings_dir(directories);
    String profile_file = Unicode::Widen(settings->display_profile_file);
    if (profile_file.isEmpty()) {
        profile_file = U"default.json";
    }

    Display_profile_values values = to_display_profile_values(*settings);
    String profile_name;
    String profile_path = get_display_settings_file_path(directories, profile_file);
    if (load_display_profile_values(profile_path, &values, &profile_name)) {
        apply_display_profile_values(values, settings);
        settings->display_profile_file = profile_file.narrow();
        settings->display_profile_name = profile_name.narrow();
        return true;
    }

    profile_file = U"default.json";
    profile_path = get_display_settings_file_path(directories, profile_file);
    if (load_display_profile_values(profile_path, &values, &profile_name)) {
        apply_display_profile_values(values, settings);
        settings->display_profile_file = profile_file.narrow();
        settings->display_profile_name = profile_name.narrow();
        return true;
    }

    settings->display_profile_file = profile_file.narrow();
    settings->display_profile_name = "default";
    return false;
}

inline bool is_display_profile_modified(const Directories& directories, const Settings& settings, const Menu_elements& menu_elements) {
    String profile_file = Unicode::Widen(settings.display_profile_file);
    if (profile_file.isEmpty()) {
        profile_file = U"default.json";
    }
    const String path = get_display_settings_file_path(directories, profile_file);
    Display_profile_values loaded_values;
    String profile_name;
    if (!load_display_profile_values(path, &loaded_values, &profile_name)) {
        return false;
    }
    const Display_profile_values current_values = to_display_profile_values(menu_elements);
    return !equals_display_profile_values(loaded_values, current_values);
}
