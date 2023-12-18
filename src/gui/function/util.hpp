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
    std::vector<std::string> lst;
    auto offset = std::string::size_type(0);
    while (1) {
        auto pos = file.find(".", offset);
        if (pos == std::string::npos) {
            lst.push_back(file.substr(offset));
            break;
        }
        lst.push_back(file.substr(offset, pos - offset));
        offset = pos + 1;
    }
    return lst[lst.size() - 1];
}