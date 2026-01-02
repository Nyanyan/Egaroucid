/*
    Egaroucid Project

    @file opening_manager.hpp
        Opening manager for GUI
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <Siv3D.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "./../../engine/engine_all.hpp"

class Opening {
private:
    std::vector<std::pair<Board, std::string>> arr;
public:
    bool init(std::string file) {
        arr.clear();
        std::ifstream ifs(file);
        if (ifs.fail()) {
            std::cerr << "opening file " << file << " not found" << std::endl;
            return false;
        }
        std::string line;
        std::string name;
        int board_arr[HW2];
        int i;
        Board b;
        while (getline(ifs, line)) {
            for (i = 0; i < HW2; ++i) {
                if (is_black_like_char(line[i])) {
                    board_arr[i] = BLACK;
                } else if (is_white_like_char(line[i])) {
                    board_arr[i] = WHITE;
                } else {
                    board_arr[i] = VACANT;
                }
            }
            b.translate_from_arr(board_arr, BLACK);
            name = line.substr(65, line.size());
            arr.push_back(std::make_pair(b, name));
            b.board_black_line_mirror();
            arr.push_back(std::make_pair(b, name));
            b.board_rotate_180();
            arr.push_back(std::make_pair(b, name));
            b.board_black_line_mirror();
            arr.push_back(std::make_pair(b, name));
        }
        return true;
    }

    inline std::string get(Board b, int p) {
        int i, j;
        bool flag;
        for (i = 0; i < (int)arr.size(); ++i) {
            if (p == BLACK && arr[i].first.player == b.opponent && arr[i].first.opponent == b.player) {
                return arr[i].second;
            }
            if (p == WHITE && arr[i].first.player == b.player && arr[i].first.opponent == b.opponent) {
                return arr[i].second;
            }
        }
        return "";
    }

};

Opening opening;
Opening opening_many;

bool opening_init(std::string lang) {
    return 
        opening.init(EXE_DIRECTORY_PATH + "resources/openings/" + lang + "/openings.txt") && 
        opening_many.init(EXE_DIRECTORY_PATH + "resources/openings/" + lang + "/openings_fork.txt");
}