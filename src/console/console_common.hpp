/*
    Egaroucid Project

    @file console_common.hpp
        Common things
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <stdlib.h>
#include <string>
#ifdef _WIN64
    #include <windows.h>
#endif

#define MODE_HUMAN_AI 0
#define MODE_AI_HUMAN 1
#define MODE_AI_AI 2
#define MODE_HUMAN_HUMAN 3

std::string replace_backslash(std::string str){
    std::string res;
    for (int i = 0; i < (int)str.length(); ++i){
        if (str[i] == '\\')
            res += '/';
        else
            res += str[i];
    }
    return res;
}

std::string get_parent_path(char raw_path[]){
    char drive[MAX_PATH + 1],
        dir[MAX_PATH + 1],
        fname[MAX_PATH + 1],
        ext[MAX_PATH + 1];
    _splitpath_s(raw_path,drive,dir,fname,ext);
    std::string res = drive;
    res += dir;
    return res;
}

#ifdef _WIN64
    std::string get_binary_path(){
        std::string res;
        char raw_path[MAX_PATH + 1];
        if (GetModuleFileName(NULL, raw_path, MAX_PATH))
            res = replace_backslash(get_parent_path(raw_path));
        return res;
    }
#elif __APPLE__ // TBD
#else
    std::string get_binary_path(){
        char raw_path[PATH_MAX + 1];
        const size_t LINKSIZE = 100;
        char link[LINKSIZE];
        snprintf(link, LINKSIZE, "/proc/%d/exe", getpid());
        readlink(link, raw_path, PATH_MAX);
        std::cerr << raw_path << std::endl;
        std::string res = get_parent_path(raw_path);
        return res;
    }
#endif