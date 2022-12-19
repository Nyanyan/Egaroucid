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
//#include <stdlib.h>
#include <string>
#include <filesystem>
#ifdef _WIN64
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#elif __APPLE__ // TBD
#else
    #include <linux/limits.h>
    #include <sys/types.h>
    #include <unistd.h>
#endif

#define MODE_HUMAN_AI 0
#define MODE_AI_HUMAN 1
#define MODE_AI_AI 2
#define MODE_HUMAN_HUMAN 3

std::string get_parent_path(char raw_path[]){
    std::filesystem::path p = raw_path;
    //p = std::filesystem::canonical(p);
    std::string res = p.parent_path().string() + "/";
    return res;
}

std::string get_parent_path(wchar_t raw_path[]){
    std::filesystem::path p = raw_path;
    //p = std::filesystem::canonical(p);
    std::string res = p.parent_path().string() + "/";
    return res;
}

#ifdef _WIN64 // Windows
    std::string get_binary_path(){
        std::string res;
        #ifdef UNICODE
            wchar_t raw_path[MAX_PATH + 1];
        #else
            char raw_path[MAX_PATH + 1];
        #endif
        if (GetModuleFileName(NULL, raw_path, MAX_PATH))
            res = get_parent_path(raw_path);
        return res;
    }
#elif __APPLE__ // Mac TBD
#else // Linux
    std::string get_binary_path(){
        char raw_path[PATH_MAX + 1];
        const size_t LINKSIZE = 100;
        char link[LINKSIZE];
        snprintf(link, LINKSIZE, "/proc/%d/exe", getpid());
        readlink(link, raw_path, PATH_MAX);
        std::string res = get_parent_path(raw_path);
        return res;
    }
#endif
