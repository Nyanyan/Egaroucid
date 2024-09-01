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
#include "language.hpp"

#define SHORTCUT_KEY_UNDEFINED U"undefined"

struct Shortcut_key_elem{
    String name;
    std::vector<String> keys;
    std::vector<String> function_str;
};

class Shortcut_keys{
private:
    std::vector<Shortcut_key_elem> shortcut_keys;
public:
    void init(const JSON json){
        shortcut_keys.clear();
        for (const auto& object: json){
            Shortcut_key_elem elem;
            elem.name = object.key;
            for (const auto &key_name: object.value[U"keys"].arrayView()){
                elem.keys.emplace_back(key_name.getString());
            }
            for (const auto &str_list: object.value[U"func_key"].arrayView()){
                std::vector<std::string> str_list_vector;
                for (const auto &str: str_list.arrayView()){
                    str_list_vector.emplace_back(str.getString().narrow());
                }
                elem.function_str.emplace_back(language.get(str_list_vector));
            }
            shortcut_keys.emplace_back(elem);
            std::cerr << elem.name.narrow() << " [";
            for (String &key: elem.keys){
                std::cerr << key.narrow() << " ";
            }
            std::cerr << "] [";
            for (String &str: elem.function_str){
                std::cerr << str.narrow() << " ";
            }
            std::cerr << "]" << std::endl;
        }
    }

    String get_shortcut_key(){
        const Array<Input> raw_keys = Keyboard::GetAllInputs();
        bool down_found = false;
        std::unordered_set<String> keys;
        for (const auto& key : raw_keys){
            down_found |= key.down();
            keys.emplace(key.name());
        }

        std::cerr << "keys size " << keys.size() << " down found " << down_found << std::endl;
        for (const String& key : keys){
            std::cerr << key.narrow() << " ";
        }
        std::cerr << std::endl;

        if (down_found){
            for (const Shortcut_key_elem &elem: shortcut_keys){
                if (keys.size() == elem.keys.size()){
                    bool matched = true;
                    for (const String& key : keys){
                        std::cerr << key.narrow() << " " << (std::find(elem.keys.begin(), elem.keys.end(), key) == elem.keys.end()) << std::endl;
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
};

Shortcut_keys shortcut_keys;
