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

#define N_SHORTCUT_KEYS 1

#define SHORTCUT_KEY_NEW_GAME 0

#define SHORTCUT_KEY_UNDEFINED -1

struct Shortcut_key_elem{
    std::vector<String> keys;
    std::vector<String> function_str;
};

class Shortcut_keys{
private:
    std::vector<Shortcut_key_elem> shortcut_keys;
public:
    void init(){
        shortcut_keys.clear();
        shortcut_keys.resize(N_SHORTCUT_KEYS);
        shortcut_keys[SHORTCUT_KEY_NEW_GAME].keys.emplace_back(U"Ctrl");
        shortcut_keys[SHORTCUT_KEY_NEW_GAME].keys.emplace_back(U"N");
        shortcut_keys[SHORTCUT_KEY_NEW_GAME].function_str.emplace_back(language.get("play", "game"));
        shortcut_keys[SHORTCUT_KEY_NEW_GAME].function_str.emplace_back(language.get("play", "new_game"));
    }

    int get_shortcut_key(){
        const Array<Input> raw_keys = Keyboard::GetAllInputs();
        std::unordered_set<String> keys;
        for (const auto& key : raw_keys){
            keys.emplace(key.name());
        }
        std::cerr << "keys size " << keys.size() << std::endl;
        for (const String& key : keys){
            std::cerr << key.narrow() << " ";
        }
        std::cerr << std::endl;
        for (int i = 0; i < N_SHORTCUT_KEYS; ++i){
            if (keys.size() == shortcut_keys[i].keys.size()){
                bool matched = true;
                for (const String& key : keys){
                    if (std::find(shortcut_keys[i].keys.begin(), shortcut_keys[i].keys.end(), key) == shortcut_keys[i].keys.end()){
                        matched = false;
                    }
                }
                if (matched){
                    return i;
                }
            }
        }
        return SHORTCUT_KEY_UNDEFINED;
    }
};

Shortcut_keys shortcut_keys;
