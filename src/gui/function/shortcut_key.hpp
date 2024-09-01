/*
    Egaroucid Project

    @file shortcut_key.hpp
        Shortcut Key Manager
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <Siv3D.hpp>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "language.hpp"

#define SHORTCUT_KEY_UNDEFINED U"undefined"

struct Shortcut_key_elem{
    String name;
    std::vector<String> keys;
    std::vector<std::vector<std::string>> function_str_keys;
};

std::vector<Shortcut_key_elem> shortcut_keys_default = {
    // buttons
    {U"start_game",             {U"Space"},             {{"play", "start_game"}}},

    // game
    {U"new_game",               {U"Ctrl", U"N"},        {{"play", "game"}, {"play", "new_game"}}},
    {U"new_game_human_black",   {},                     {{"play", "game"}, {"play", "new_game_human_black"}}},
    {U"new_game_human_white",   {},                     {{"play", "game"}, {"play", "new_game_human_white"}}},
    {U"new_selfplay",           {},                     {{"play", "game"}, {"play", "new_selfplay"}}},
    {U"analyze",                {U"A"},                 {{"play", "game"}, {"play", "analyze"}}},

    // settings
    {U"ai_put_black",           {U"B"},                 {{"settings", "settings"}, {"settings", "play", "ai_put_black"}}},
    {U"ai_put_white",           {U"W"},                 {{"settings", "settings"}, {"settings", "play", "ai_put_white"}}},

    // display
    {U"disc_value",             {U"V"},                 {{"display", "display"}, {"display", "cell", "disc_value"}}},
    {U"umigame_value",          {U"U"},                 {{"display", "display"}, {"display", "cell", "umigame_value"}}},
    {U"graph_value",            {U"D"},                 {{"display", "display"}, {"display", "graph", "value"}}},
    {U"graph_sum_of_loss",      {U"S"},                 {{"display", "display"}, {"display", "graph", "sum_of_loss"}}},
    {U"laser_pointer",          {U"P"},                 {{"display", "display"}, {"display", "laser_pointer"}}},

    // operate
    {U"put_1_move_by_ai",       {U"G"},                 {{"operation", "operation"}, {"operation", "put_1_move_by_ai"}}},
    {U"forward",                {U"Right"},             {{"operation", "operation"}, {"operation", "forward"}}},
    {U"backward",               {U"Left"},              {{"operation", "operation"}, {"operation", "backward"}}},
    {U"undo",                   {U"Backspace"},         {{"operation", "operation"}, {"operation", "undo"}}},
    {U"save_this_branch",       {U"Ctrl", U"L"},        {{"operation", "operation"}, {"operation", "save_this_branch"}}},
    {U"generate_random_board",  {U"Ctrl", U"R"},        {{"operation", "operation"}, {"operation", "generate_random_board", "generate"}}},
    {U"stop_calculating",       {U"Q"},                 {{"operation", "operation"}, {"operation", "ai_operation", "stop_calculating"}}},

    // input / output
    {U"input_transcript",       {U"Ctrl", U"T"},        {{"in_out", "in_out"}, {"in_out", "in"}, {"in_out", "input_transcript"}}},
    {U"edit_board",             {U"Ctrl", U"E"},        {{"in_out", "in_out"}, {"in_out", "in"}, {"in_out", "edit_board"}}},
    {U"output_transcript",      {U"Ctrl", U"C"},        {{"in_out", "in_out"}, {"in_out", "out"}, {"in_out", "output_transcript"}}},
    {U"screen_shot",            {U"Ctrl", U"S"},        {{"in_out", "in_out"}, {"in_out", "out"}, {"in_out", "screen_shot"}}},
};

class Shortcut_keys{
public:
    std::vector<Shortcut_key_elem> shortcut_keys;
public:
    void init(String file){
        shortcut_keys = shortcut_keys_default;
        /*
        JSON json = JSON::Load(file);
        std::unordered_set<String> name_list;
        for (Shortcut_key_dict_elem &elem: shortcut_key_str){
            name_list.emplace(elem.name);
        }
        shortcut_keys.clear();
        for (const auto& object: json){
            if (name_list.find(object.key) == name_list.end()){
                std::cerr << "ERR shortcut key name not found " << object.key.narrow() << std::endl;
                continue;
            }
            Shortcut_key_elem elem;
            elem.name = object.key;
            for (const auto &key_name: object.value[U"keys"].arrayView()){
                elem.keys.emplace_back(key_name.getString());
            }
            if (elem.keys.size()){
                //for (const auto &str_list: object.value[U"func_key"].arrayView()){
                //    std::vector<std::string> str_list_vector;
                //    for (const auto &str: str_list.arrayView()){
                //        str_list_vector.emplace_back(str.getString().narrow());
                //    }
                //    elem.function_str.emplace_back(language.get(str_list_vector));
                //}
                shortcut_keys.emplace_back(elem);
                std::cerr << elem.name.narrow() << " [";
                for (String &key: elem.keys){
                    std::cerr << key.narrow() << " ";
                }
                std::cerr << "]" << std::endl;
            }
        }
        */
    }

    void check_shortcut_key(String *shortcut_name_down, String *shortcut_name_pressed){
        const Array<Input> raw_keys = Keyboard::GetAllInputs();
        bool down_found = false;
        std::unordered_set<String> keys;
        for (const auto& key : raw_keys){
            down_found |= key.down();
            keys.emplace(key.name());
        }

        //std::cerr << "keys size " << keys.size() << " down found " << down_found << std::endl;
        //for (const String& key : keys){
        //    std::cerr << key.narrow() << " ";
        //}
        //std::cerr << std::endl;

        *shortcut_name_down = SHORTCUT_KEY_UNDEFINED;
        *shortcut_name_pressed = SHORTCUT_KEY_UNDEFINED;
        for (const Shortcut_key_elem &elem: shortcut_keys){
            if (keys.size() && keys.size() == elem.keys.size()){
                bool matched = true;
                for (const String& key : keys){
                    //std::cerr << key.narrow() << " " << (std::find(elem.keys.begin(), elem.keys.end(), key) == elem.keys.end()) << std::endl;
                    if (std::find(elem.keys.begin(), elem.keys.end(), key) == elem.keys.end()){
                        matched = false;
                    }
                }
                if (matched){
                    if (down_found){
                        *shortcut_name_down = elem.name;
                    }
                    *shortcut_name_pressed = elem.name;
                    return;
                }
            }
        }
    }

    String get_shortcut_key_list(String name){
        for (const Shortcut_key_elem &elem: shortcut_keys){
            if (elem.name == name){
                String res;
                for (int i = 0; i < (int)elem.keys.size(); ++i){
                    if (elem.keys[i] == U"Right"){
                        res += U"->";
                    } else if (elem.keys[i] == U"Left"){
                        res += U"<-";
                    } else{
                        res += elem.keys[i];
                    }
                    if (i < (int)elem.keys.size() - 1){
                        res += U"+";
                    }
                }
                return res;
            }
        }
        return SHORTCUT_KEY_UNDEFINED;
    }

    String get_shortcut_key_function_str(String name){
        for (const Shortcut_key_elem &elem: shortcut_keys){
            if (elem.name == name){
                String res;
                for (int i = 0; i < (int)elem.function_str_keys.size(); ++i){
                    res += language.get(elem.function_str_keys[i]);
                    if (i < (int)elem.function_str_keys.size() - 1){
                        res += U"> ";
                    }
                }
                return res;
            }
        }
        return SHORTCUT_KEY_UNDEFINED;
    }
};

Shortcut_keys shortcut_keys;
