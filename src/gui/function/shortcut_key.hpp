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

struct Shortcut_key_dict_elem{
    String name;
    std::vector<std::vector<std::string>> str_keys;
};

std::vector<Shortcut_key_dict_elem> shortcut_key_str = {
    {U"start_game",         {{"play", "start_game"}}},
    {U"new_game",           {{"play", "game"}, {"play", "new_game"}}},
    {U"analyze",            {{"play", "analyze"}}},
    {U"ai_put_black",       {{"settings", "play", "ai_put_black"}}},
    {U"ai_put_white",       {{"settings", "play", "ai_put_white"}}},
    {U"disc_value",         {{"display", "display"}, {"display", "cell", "disc_value"}}},
    {U"umigame_value",      {{"display", "display"}, {"display", "cell", "umigame_value"}}},
    {U"graph_value",        {{"display", "graph", "value"}}},
    {U"graph_sum_of_loss",  {{"display", "graph", "sum_of_loss"}}},
    {U"laser_pointer",      {{"display", "laser_pointer"}}},
};

struct Shortcut_key_elem{
    String name;
    std::vector<String> keys;
};

class Shortcut_keys{
private:
    std::vector<Shortcut_key_elem> shortcut_keys;
public:
    void init(const JSON json){
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
            /*
            for (const auto &str_list: object.value[U"func_key"].arrayView()){
                std::vector<std::string> str_list_vector;
                for (const auto &str: str_list.arrayView()){
                    str_list_vector.emplace_back(str.getString().narrow());
                }
                elem.function_str.emplace_back(language.get(str_list_vector));
            }
            */
            shortcut_keys.emplace_back(elem);
            std::cerr << elem.name.narrow() << " [";
            for (String &key: elem.keys){
                std::cerr << key.narrow() << " ";
            }
            std::cerr << "]" << std::endl;
            /*
            for (String &str: elem.function_str){
                std::cerr << str.narrow() << " ";
            }
            std::cerr << "]" << std::endl;
            */
        }
    }

    String check_shortcut_key(){
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

        if (down_found){
            for (const Shortcut_key_elem &elem: shortcut_keys){
                if (keys.size() == elem.keys.size()){
                    bool matched = true;
                    for (const String& key : keys){
                        //std::cerr << key.narrow() << " " << (std::find(elem.keys.begin(), elem.keys.end(), key) == elem.keys.end()) << std::endl;
                        if (std::find(elem.keys.begin(), elem.keys.end(), key) == elem.keys.end()){
                            matched = false;
                        }
                    }
                    if (matched){
                        return elem.name;
                    }
                }
            }
        }
        return SHORTCUT_KEY_UNDEFINED;
    }

    String get_shortcut_key_list(String name){
        for (const Shortcut_key_elem &elem: shortcut_keys){
            if (elem.name == name){
                String res;
                for (int i = 0; i < (int)elem.keys.size(); ++i){
                    res += elem.keys[i];
                    if (i < (int)elem.keys.size() - 1){
                        res += U"+";
                    }
                }
                return res;
            }
        }
        return U"";
    }
};

Shortcut_keys shortcut_keys;
