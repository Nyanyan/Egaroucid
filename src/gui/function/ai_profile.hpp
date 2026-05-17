/*
    Egaroucid Project

    @file ai_profile.hpp
        AI profile utility
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include "const/gui_common.hpp"

struct AI_profile_values {
    bool use_book{ true };
    bool accept_ai_loss{ false };
    int max_loss{ 2 };
    int loss_percentage{ 30 };
    AI_loss_graph_values max_loss_by_move = [] {
        AI_loss_graph_values values{};
        values.fill(2);
        return values;
    }();
    AI_loss_graph_values loss_percentage_by_move = [] {
        AI_loss_graph_values values{};
        values.fill(30);
        return values;
    }();
    int level{ DEFAULT_LEVEL };
    int n_threads{ 1 };
#if USE_CHANGEABLE_HASH_LEVEL
    int hash_level{ DEFAULT_HASH_LEVEL };
#endif
    bool ai_put_black{ false };
    bool ai_put_white{ false };
    bool pause_when_pass{ true };
    bool force_specified_openings{ false };
};

inline String get_ai_settings_dir(const Directories& directories) {
    return Unicode::Widen(directories.appdata_dir) + U"ai_settings/";
}

inline String get_ai_settings_file_path(const Directories& directories, const String& file_name) {
    return get_ai_settings_dir(directories) + file_name;
}

inline void ensure_ai_settings_dir(const Directories& directories) {
    const String dir = get_ai_settings_dir(directories);
    if (!FileSystem::Exists(dir)) {
        FileSystem::CreateDirectories(dir);
    }
}

inline int clamp_ai_max_loss_value(int value) {
    return std::clamp(value, 0, AI_MAX_LOSS_INF);
}

inline int clamp_ai_loss_percentage_value(int value) {
    return std::clamp(value, 0, AI_LOSS_PERCENTAGE_INF);
}

inline int snap_ai_max_loss_value(int value) {
    int clamped_value = clamp_ai_max_loss_value(value);
    int best = AI_MAX_LOSS_SNAP_VALUES[0];
    int best_diff = std::abs(clamped_value - best);
    for (int candidate : AI_MAX_LOSS_SNAP_VALUES) {
        int diff = std::abs(clamped_value - candidate);
        if (diff < best_diff) {
            best = candidate;
            best_diff = diff;
        }
    }
    return best;
}

inline int snap_ai_loss_percentage_value(int value) {
    int clamped_value = clamp_ai_loss_percentage_value(value);
    int snapped = static_cast<int>(std::round(clamped_value / 5.0)) * 5;
    return std::clamp(snapped, 0, AI_LOSS_PERCENTAGE_INF);
}

inline void fill_ai_loss_graph_values(AI_loss_graph_values* values, int value, bool is_percentage) {
    if (is_percentage) {
        values->fill(snap_ai_loss_percentage_value(value));
    } else {
        values->fill(snap_ai_max_loss_value(value));
    }
}

inline void clamp_ai_loss_graph_values(AI_loss_graph_values* values, bool is_percentage) {
    for (int& value : *values) {
        if (is_percentage) {
            value = snap_ai_loss_percentage_value(value);
        } else {
            value = snap_ai_max_loss_value(value);
        }
    }
}

inline int move_number_to_ai_loss_graph_idx(int move_number) {
    const int clamped_move_number = std::clamp(move_number, 1, HW2 - 4);
    const int idx = (clamped_move_number - 1) / AI_LOSS_GRAPH_INTERVAL;
    return std::clamp(idx, 0, AI_LOSS_GRAPH_POINT_COUNT - 1);
}

inline int ai_loss_graph_idx_to_move_number(int idx) {
    const int clamped_idx = std::clamp(idx, 0, AI_LOSS_GRAPH_POINT_COUNT - 1);
    return std::min(HW2 - 4, (clamped_idx + 1) * AI_LOSS_GRAPH_INTERVAL);
}

inline int get_ai_loss_graph_value(const AI_loss_graph_values& values, int move_number) {
    const int idx = move_number_to_ai_loss_graph_idx(move_number);
    return values[idx];
}

inline void normalize_ai_profile_loss_values(AI_profile_values* values) {
    values->max_loss = clamp_ai_max_loss_value(values->max_loss);
    values->loss_percentage = clamp_ai_loss_percentage_value(values->loss_percentage);
    clamp_ai_loss_graph_values(&values->max_loss_by_move, false);
    clamp_ai_loss_graph_values(&values->loss_percentage_by_move, true);
    values->max_loss = values->max_loss_by_move[0];
    values->loss_percentage = values->loss_percentage_by_move[0];
}

inline AI_profile_values to_ai_profile_values(const Settings& settings) {
    AI_profile_values values;
    values.use_book = settings.use_book;
    values.accept_ai_loss = settings.accept_ai_loss;
    values.max_loss_by_move = settings.max_loss_by_move;
    values.loss_percentage_by_move = settings.loss_percentage_by_move;
    values.max_loss = values.max_loss_by_move[0];
    values.loss_percentage = values.loss_percentage_by_move[0];
    values.level = settings.level;
    values.n_threads = settings.n_threads;
#if USE_CHANGEABLE_HASH_LEVEL
    values.hash_level = settings.hash_level;
#endif
    values.ai_put_black = settings.ai_put_black;
    values.ai_put_white = settings.ai_put_white;
    values.pause_when_pass = settings.pause_when_pass;
    values.force_specified_openings = settings.force_specified_openings;
    normalize_ai_profile_loss_values(&values);
    return values;
}

inline AI_profile_values to_ai_profile_values(const Menu_elements& menu_elements) {
    AI_profile_values values;
    values.use_book = menu_elements.use_book;
    values.accept_ai_loss = menu_elements.accept_ai_loss;
    values.max_loss_by_move = menu_elements.max_loss_by_move;
    values.loss_percentage_by_move = menu_elements.loss_percentage_by_move;
    values.max_loss = values.max_loss_by_move[0];
    values.loss_percentage = values.loss_percentage_by_move[0];
    values.level = menu_elements.level;
    values.n_threads = menu_elements.n_threads;
#if USE_CHANGEABLE_HASH_LEVEL
    values.hash_level = menu_elements.hash_level;
#endif
    values.ai_put_black = menu_elements.ai_put_black;
    values.ai_put_white = menu_elements.ai_put_white;
    values.pause_when_pass = menu_elements.pause_when_pass;
    values.force_specified_openings = menu_elements.force_specified_openings;
    normalize_ai_profile_loss_values(&values);
    return values;
}

inline void apply_ai_profile_values(const AI_profile_values& values, Settings* settings) {
    AI_profile_values normalized_values = values;
    normalize_ai_profile_loss_values(&normalized_values);
    settings->use_book = normalized_values.use_book;
    settings->accept_ai_loss = normalized_values.accept_ai_loss;
    settings->max_loss_by_move = normalized_values.max_loss_by_move;
    settings->loss_percentage_by_move = normalized_values.loss_percentage_by_move;
    settings->max_loss = normalized_values.max_loss;
    settings->loss_percentage = normalized_values.loss_percentage;
    settings->level = normalized_values.level;
    settings->n_threads = normalized_values.n_threads;
#if USE_CHANGEABLE_HASH_LEVEL
    settings->hash_level = normalized_values.hash_level;
#endif
    settings->ai_put_black = normalized_values.ai_put_black;
    settings->ai_put_white = normalized_values.ai_put_white;
    settings->pause_when_pass = normalized_values.pause_when_pass;
    settings->force_specified_openings = normalized_values.force_specified_openings;
}

inline void apply_ai_profile_values(const AI_profile_values& values, Menu_elements* menu_elements) {
    AI_profile_values normalized_values = values;
    normalize_ai_profile_loss_values(&normalized_values);
    menu_elements->use_book = normalized_values.use_book;
    menu_elements->accept_ai_loss = normalized_values.accept_ai_loss;
    menu_elements->max_loss_by_move = normalized_values.max_loss_by_move;
    menu_elements->loss_percentage_by_move = normalized_values.loss_percentage_by_move;
    menu_elements->max_loss = normalized_values.max_loss;
    menu_elements->loss_percentage = normalized_values.loss_percentage;
    menu_elements->level = normalized_values.level;
    menu_elements->n_threads = normalized_values.n_threads;
#if USE_CHANGEABLE_HASH_LEVEL
    menu_elements->hash_level = normalized_values.hash_level;
#endif
    menu_elements->ai_put_black = normalized_values.ai_put_black;
    menu_elements->ai_put_white = normalized_values.ai_put_white;
    menu_elements->pause_when_pass = normalized_values.pause_when_pass;
    menu_elements->force_specified_openings = normalized_values.force_specified_openings;
}

inline bool import_ai_profile_bool(JSON& json, const String& key, bool* value) {
    if (json[key].getType() != JSONValueType::Bool) {
        return false;
    }
    *value = json[key].get<bool>();
    return true;
}

inline bool import_ai_profile_int(JSON& json, const String& key, int* value) {
    if (json[key].getType() != JSONValueType::Number) {
        return false;
    }
    *value = static_cast<int>(json[key].get<double>());
    return true;
}

inline bool import_ai_profile_int_graph(JSON& json, const String& key, AI_loss_graph_values* values, bool is_percentage) {
    if (json[key].getType() != JSONValueType::Object) {
        return false;
    }
    for (int i = 0; i < AI_LOSS_GRAPH_POINT_COUNT; ++i) {
        const String move_key = Format(ai_loss_graph_idx_to_move_number(i));
        if (json[key][move_key].getType() != JSONValueType::Number) {
            return false;
        }
        int value = static_cast<int>(json[key][move_key].get<double>());
        (*values)[i] = is_percentage ? clamp_ai_loss_percentage_value(value) : clamp_ai_max_loss_value(value);
    }
    return true;
}

inline void export_ai_profile_int_graph(JSON& json, const String& key, const AI_loss_graph_values& values) {
    for (int i = 0; i < AI_LOSS_GRAPH_POINT_COUNT; ++i) {
        json[key][Format(ai_loss_graph_idx_to_move_number(i))] = values[i];
    }
}

inline bool import_ai_profile_name(JSON& json, String* name) {
    if (json[U"name"].getType() != JSONValueType::String) {
        return false;
    }
    *name = json[U"name"].getString();
    return true;
}

inline void export_ai_profile_json(JSON& json, const AI_profile_values& values, const String& name) {
    AI_profile_values normalized_values = values;
    normalize_ai_profile_loss_values(&normalized_values);
    json[U"name"] = name;
    json[U"use_book"] = normalized_values.use_book;
    json[U"accept_ai_loss"] = normalized_values.accept_ai_loss;
    json[U"max_loss"] = normalized_values.max_loss;
    json[U"loss_percentage"] = normalized_values.loss_percentage;
    export_ai_profile_int_graph(json, U"max_loss_by_move", normalized_values.max_loss_by_move);
    export_ai_profile_int_graph(json, U"loss_percentage_by_move", normalized_values.loss_percentage_by_move);
    json[U"level"] = normalized_values.level;
    json[U"n_threads"] = normalized_values.n_threads;
#if USE_CHANGEABLE_HASH_LEVEL
    json[U"hash_level"] = normalized_values.hash_level;
#endif
    json[U"ai_put_black"] = normalized_values.ai_put_black;
    json[U"ai_put_white"] = normalized_values.ai_put_white;
    json[U"pause_when_pass"] = normalized_values.pause_when_pass;
    json[U"force_specified_openings"] = normalized_values.force_specified_openings;
}

inline bool load_ai_profile_values(const FilePath& path, AI_profile_values* values, String* profile_name) {
    JSON json = JSON::Load(path);
    if (json.size() == 0) {
        return false;
    }

    import_ai_profile_bool(json, U"use_book", &values->use_book);
    import_ai_profile_bool(json, U"accept_ai_loss", &values->accept_ai_loss);
    import_ai_profile_int(json, U"max_loss", &values->max_loss);
    import_ai_profile_int(json, U"loss_percentage", &values->loss_percentage);
    const bool max_loss_curve_loaded = import_ai_profile_int_graph(json, U"max_loss_by_move", &values->max_loss_by_move, false);
    const bool loss_percentage_curve_loaded = import_ai_profile_int_graph(json, U"loss_percentage_by_move", &values->loss_percentage_by_move, true);
    if (!max_loss_curve_loaded) {
        fill_ai_loss_graph_values(&values->max_loss_by_move, values->max_loss, false);
    }
    if (!loss_percentage_curve_loaded) {
        fill_ai_loss_graph_values(&values->loss_percentage_by_move, values->loss_percentage, true);
    }
    import_ai_profile_int(json, U"level", &values->level);
    import_ai_profile_int(json, U"n_threads", &values->n_threads);
#if USE_CHANGEABLE_HASH_LEVEL
    import_ai_profile_int(json, U"hash_level", &values->hash_level);
#endif
    import_ai_profile_bool(json, U"ai_put_black", &values->ai_put_black);
    import_ai_profile_bool(json, U"ai_put_white", &values->ai_put_white);
    import_ai_profile_bool(json, U"pause_when_pass", &values->pause_when_pass);
    import_ai_profile_bool(json, U"force_specified_openings", &values->force_specified_openings);
    normalize_ai_profile_loss_values(values);

    if (profile_name) {
        if (!import_ai_profile_name(json, profile_name) || profile_name->trimmed().isEmpty()) {
            *profile_name = FileSystem::FileName(path);
        }
    }
    return true;
}

inline bool save_ai_profile_values(const FilePath& path, const AI_profile_values& values, const String& profile_name) {
    JSON json;
    export_ai_profile_json(json, values, profile_name);
    return json.save(path);
}

inline bool equals_ai_profile_values(const AI_profile_values& lhs, const AI_profile_values& rhs) {
    if (lhs.use_book != rhs.use_book) return false;
    if (lhs.accept_ai_loss != rhs.accept_ai_loss) return false;
    if (lhs.max_loss_by_move != rhs.max_loss_by_move) return false;
    if (lhs.loss_percentage_by_move != rhs.loss_percentage_by_move) return false;
    if (lhs.level != rhs.level) return false;
    if (lhs.n_threads != rhs.n_threads) return false;
#if USE_CHANGEABLE_HASH_LEVEL
    if (lhs.hash_level != rhs.hash_level) return false;
#endif
    if (lhs.ai_put_black != rhs.ai_put_black) return false;
    if (lhs.ai_put_white != rhs.ai_put_white) return false;
    if (lhs.pause_when_pass != rhs.pause_when_pass) return false;
    if (lhs.force_specified_openings != rhs.force_specified_openings) return false;
    return true;
}

inline Array<FilePath> enumerate_ai_profile_files(const Directories& directories) {
    ensure_ai_settings_dir(directories);
    Array<FilePath> files;
    const String dir = get_ai_settings_dir(directories);
    for (const auto& path : FileSystem::DirectoryContents(dir)) {
        if (FileSystem::IsFile(path) && path.ends_with(U".json")) {
            files << path;
        }
    }
    std::sort(files.begin(), files.end(), std::greater<FilePath>());
    return files;
}

inline String generate_unique_ai_profile_filepath(const Directories& directories) {
    ensure_ai_settings_dir(directories);
    const String dir = get_ai_settings_dir(directories);
    String base_name = DateTime::Now().format(U"yyyyMMddHHmmss");
    String candidate = dir + base_name + U".json";
    int suffix = 1;
    while (FileSystem::Exists(candidate)) {
        candidate = dir + base_name + U"_" + Format(suffix) + U".json";
        ++suffix;
    }
    return candidate;
}

inline void ensure_default_ai_profile(const Directories& directories, const Settings& settings, bool setting_json_exists) {
    ensure_ai_settings_dir(directories);
    const String default_path = get_ai_settings_file_path(directories, U"default.json");
    if (FileSystem::Exists(default_path)) {
        return;
    }

    if (setting_json_exists) {
        save_ai_profile_values(default_path, to_ai_profile_values(settings), U"default");
        return;
    }

    save_ai_profile_values(default_path, to_ai_profile_values(settings), U"default");
}

inline bool load_ai_profile_into_settings(const Directories& directories, Settings* settings) {
    ensure_ai_settings_dir(directories);
    String profile_file = Unicode::Widen(settings->ai_profile_file);
    if (profile_file.isEmpty()) {
        profile_file = U"default.json";
    }

    AI_profile_values values = to_ai_profile_values(*settings);
    String profile_name;
    String profile_path = get_ai_settings_file_path(directories, profile_file);
    if (load_ai_profile_values(profile_path, &values, &profile_name)) {
        apply_ai_profile_values(values, settings);
        settings->ai_profile_file = profile_file.narrow();
        settings->ai_profile_name = profile_name.narrow();
        return true;
    }

    profile_file = U"default.json";
    profile_path = get_ai_settings_file_path(directories, profile_file);
    if (load_ai_profile_values(profile_path, &values, &profile_name)) {
        apply_ai_profile_values(values, settings);
        settings->ai_profile_file = profile_file.narrow();
        settings->ai_profile_name = profile_name.narrow();
        return true;
    }

    settings->ai_profile_file = profile_file.narrow();
    settings->ai_profile_name = "default";
    return false;
}

inline bool is_ai_profile_modified(const Directories& directories, const Settings& settings, const Menu_elements& menu_elements) {
    String profile_file = Unicode::Widen(settings.ai_profile_file);
    if (profile_file.isEmpty()) {
        profile_file = U"default.json";
    }
    const String path = get_ai_settings_file_path(directories, profile_file);
    AI_profile_values loaded_values;
    String profile_name;
    if (!load_ai_profile_values(path, &loaded_values, &profile_name)) {
        return false;
    }
    const AI_profile_values current_values = to_ai_profile_values(menu_elements);
    return !equals_ai_profile_values(loaded_values, current_values);
}
