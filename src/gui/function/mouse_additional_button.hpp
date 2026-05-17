/*
    Egaroucid Project

    @file mouse_additional_button.hpp
        Mouse additional button assignment manager
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <Siv3D.hpp>
#include <array>
#include <vector>
#include "shortcut_key.hpp"

constexpr int MOUSE_ADDITIONAL_BUTTON_COUNT = 5;

class Mouse_additional_buttons {
public:
    std::array<String, MOUSE_ADDITIONAL_BUTTON_COUNT> functions;

    void set_default() {
        for (String& function_name : functions) {
            function_name.clear();
        }
        // Keep existing default behavior.
        functions[0] = U"backward";
        functions[1] = U"forward";
    }

    void init(const String& file) {
        set_default();
        JSON json = JSON::Load(file);
        if (!json) {
            return;
        }
        for (int i = 0; i < MOUSE_ADDITIONAL_BUTTON_COUNT; ++i) {
            String key = U"button_{}"_fmt(i + 1);
            if (json[key].getType() != JSONValueType::String) {
                continue;
            }
            String function_name = json[key].getString();
            if (function_name == U"") {
                functions[i].clear();
                continue;
            }
            if (!is_valid_function(function_name)) {
                continue;
            }
            if (find_assigned_button(function_name) != -1) {
                continue;
            }
            functions[i] = function_name;
        }
    }

    void save_settings(const String& file) const {
        JSON json;
        for (int i = 0; i < MOUSE_ADDITIONAL_BUTTON_COUNT; ++i) {
            String key = U"button_{}"_fmt(i + 1);
            json[key] = functions[i];
        }
        json.save(file);
    }

    String get_function(int button_idx) const {
        if (!is_valid_button_idx(button_idx)) {
            return U"";
        }
        return functions[button_idx];
    }

    void set_function(int button_idx, const String& function_name) {
        if (!is_valid_button_idx(button_idx)) {
            return;
        }
        if (!is_valid_function(function_name)) {
            return;
        }
        functions[button_idx] = function_name;
    }

    void clear_function(int button_idx) {
        if (!is_valid_button_idx(button_idx)) {
            return;
        }
        functions[button_idx].clear();
    }

    int find_assigned_button(const String& function_name, int ignore_button_idx = -1) const {
        for (int i = 0; i < MOUSE_ADDITIONAL_BUTTON_COUNT; ++i) {
            if (i == ignore_button_idx) {
                continue;
            }
            if (functions[i] == function_name) {
                return i;
            }
        }
        return -1;
    }

    int find_function_idx(const String& function_name) const {
        for (int i = 0; i < static_cast<int>(shortcut_keys.shortcut_keys.size()); ++i) {
            if (shortcut_keys.shortcut_keys[i].name == function_name) {
                return i;
            }
        }
        return -1;
    }

    String get_button_name(int button_idx) const {
        if (!is_valid_button_idx(button_idx)) {
            return U"?";
        }
        return U"X{}"_fmt(button_idx + 1);
    }

    bool is_button_down(int button_idx) const {
        const Input* input = get_button_input(button_idx);
        return (input && input->down());
    }

    bool is_button_pressed(int button_idx) const {
        const Input* input = get_button_input(button_idx);
        return (input && input->pressed());
    }

private:
    bool is_valid_button_idx(int button_idx) const {
        return (0 <= button_idx) && (button_idx < MOUSE_ADDITIONAL_BUTTON_COUNT);
    }

    bool is_valid_function(const String& function_name) const {
        return find_function_idx(function_name) != -1;
    }

    const Input* get_button_input(int button_idx) const {
        if (!is_valid_button_idx(button_idx)) {
            return nullptr;
        }
        static const std::array<Input, MOUSE_ADDITIONAL_BUTTON_COUNT> mouse_buttons = {
            MouseX1, MouseX2, MouseX3, MouseX4, MouseX5
        };
        return &mouse_buttons[button_idx];
    }
};

Mouse_additional_buttons mouse_additional_buttons;

