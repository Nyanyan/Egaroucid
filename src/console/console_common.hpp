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
#ifdef _WIN64
    #include <windows.h>
#endif

#define MODE_HUMAN_AI 0
#define MODE_AI_HUMAN 1
#define MODE_AI_AI 2
#define MODE_HUMAN_HUMAN 3

#ifdef _WIN64
    std::string get_binary_path(){
        std::string res;
        char raw_path[MAX_PATH + 1];
        if (GetModuleFileName(NULL, raw_path, MAX_PATH)){
            char drive[MAX_PATH + 1],
                dir[MAX_PATH + 1],
                fname[MAX_PATH + 1],
                ext[MAX_PATH + 1];

            _splitpath_s(raw_path,drive,dir,fname,ext);
            res = drive;
            res += dir;
        }
        return res;
    }
#else
#endif