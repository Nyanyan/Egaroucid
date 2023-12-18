/*
    Egaroucid Project

    @file util.hpp
        GUI Utility
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/
#include <string>
#include <vector>

std::string get_extension(std::string file){
    std::string res;
    for (int i = (int)file.size() - 1; i >= 0; --i){
        if (file[i] == '.')
            break;
        res.insert(0, {file[i]});
    }
    return res;
}