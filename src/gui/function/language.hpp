/*
    Egaroucid Project

    @file language.hpp
        Language manager
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once

#include <Siv3D.hpp>
#include <iostream>
#include <string>
#include <fstream>

class Language {
private:
    JSON lang;

public:

    bool init(std::string file) {
        lang.clear();
        lang = JSON::Load(Unicode::Widen(file));
        if (not lang) {
            std::cerr << "can't open `language_name`.json" << std::endl;
            return false;
        }
        return true;
    }

    String get(std::string v0) {
        String v0s = Unicode::Widen(v0);
        if (lang[v0s].getType() != JSONValueType::String)
            return U"?";
        return lang[v0s].getString();
    }

    String get(std::string v0, std::string v1) {
        String v0s = Unicode::Widen(v0);
        String v1s = Unicode::Widen(v1);
        if (lang[v0s][v1s].getType() != JSONValueType::String)
            return U"?";
        return lang[v0s][v1s].getString();
    }

    String get(std::string v0, std::string v1, std::string v2) {
        String v0s = Unicode::Widen(v0);
        String v1s = Unicode::Widen(v1);
        String v2s = Unicode::Widen(v2);
        if (lang[v0s][v1s][v2s].getType() != JSONValueType::String)
            return U"?";
        return lang[v0s][v1s][v2s].getString();
    }

    String get_random(std::string v0, std::string v1) {
        String v0s = Unicode::Widen(v0);
        String v1s = Unicode::Widen(v1);
        if (lang[v0s][v1s].getType() != JSONValueType::Array)
            return U"?";
        std::vector<String> arr;
        for (const auto& elem : lang[v0s][v1s].arrayView()){
            arr.emplace_back(elem.getString());
        }
        int idx = myrandrange(0, (int)arr.size());
        return arr[idx];
    }
};

Language language;

class Language_name {
private:
    JSON lang;

public:
    bool init(std::vector<std::string> &language_name_list) {
        lang = JSON::Load(U"resources/languages/languages.json");
        if (not lang) {
            std::cerr << "can't open languages.json" << std::endl;
            return false;
        }
        for (const auto& object : lang)
            language_name_list.emplace_back(object.key.narrow());
        for (int i = 1; i < (int)language_name_list.size(); ++i){
            if (language_name_list[i] == "japanese")
                std::swap(language_name_list[i], language_name_list[0]);
        }
        for (int i = 2; i < (int)language_name_list.size(); ++i){
            if (language_name_list[i] == "english")
                std::swap(language_name_list[i], language_name_list[1]);
        }
        return true;
    }

    String get(std::string v0) {
        String v0s = Unicode::Widen(v0);
        if (lang[v0s].getType() != JSONValueType::String)
            return U"?";
        return lang[v0s].getString();
    }

    int size(){
        return lang.size();
    }
};

Language_name language_name;