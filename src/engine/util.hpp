/*
    Egaroucid Project

    @file util.hpp
        Utility
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include "board.hpp"

/*
    @brief Input board from console

    '0' as black, '1' as white, '.' as empty

    @return board structure
*/
Board input_board(){
    Board res;
    char elem;
    int player;
    std::cin >> player;
    res.player = 0;
    res.opponent = 0;
    for (int i = 0; i < HW2; ++i){
        std::cin >> elem;
        if (elem == '0'){
            if (player == BLACK)
                res.player |= 1ULL << (HW2_M1 - i);
            else
                res.opponent |= 1ULL << (HW2_M1 - i);
        } else if (elem == '1'){
            if (player == WHITE)
                res.player |= 1ULL << (HW2_M1 - i);
            else
                res.opponent |= 1ULL << (HW2_M1 - i);
        }
    }
    return res;
}

// use Base81 from https://github.com/primenumber/issen/blob/f418af2c7decac8143dd699c7ee89579013987f7/README.md#base81
/*
    empty square:  0
    player disc:   1
    opponent disc: 2

    board coordinate:
        a  b  c  d  e  f  g  h
        ------------------------
    1| 0  1  2  3  4  5  6  7
    2| 8  9 10 11 12 13 14 15
    3|16 17 18 19 20 21 22 23
    4|24 25 26 27 28 29 30 31
    5|32 33 34 35 36 37 38 39
    6|40 41 42 43 44 45 46 47
    7|48 49 50 51 52 53 54 55
    8|56 57 58 59 60 61 62 63

    the 'i'th character of the string (length=16) is calculated as:
    char c = 
            '!' + 
            board[i * 4] + 
            board[i * 4 + 1] * 3 + 
            board[i * 4 + 2] * 9 + 
            board[i * 4 + 3] * 32
    
*/
bool input_board_base81(std::string board_str, Board *board) {
    if (board_str.length() != 16){
        std::cerr << "[ERROR] invalid argument" << std::endl;
        return true;
    }
    board->player = 0;
    board->opponent = 0;
    int idx, d;
    char c;
    for (int i = 0; i < 16; ++i){
        idx = i * 4 + 3;
        c = board_str[i] - '!';
        d = c / 32;
        if (d == 1){
            board->player |= 1ULL << idx;
        } else if (d == 2){
            board->opponent |= 1ULL << idx;
        }
        --idx;
        c %= 32;
        d = c / 9;
        if (d == 1){
            board->player |= 1ULL << idx;
        } else if (d == 2){
            board->opponent |= 1ULL << idx;
        }
        --idx;
        c %= 9;
        d = c / 3;
        if (d == 1){
            board->player |= 1ULL << idx;
        } else if (d == 2){
            board->opponent |= 1ULL << idx;
        }
        --idx;
        c %= 3;
        d = c;
        if (d == 1){
            board->player |= 1ULL << idx;
        } else if (d == 2){
            board->opponent |= 1ULL << idx;
        }
    }
    return false;
}

/*
    @brief Generate coordinate in string

    @param idx                  index of the coordinate
    @return coordinate as string
*/
std::string idx_to_coord(int idx){
    if (idx < 0 || HW2 <= idx)
        return "??";
    int y = HW_M1 - idx / HW;
    int x = HW_M1 - idx % HW;
    const std::string x_coord = "abcdefgh";
    return x_coord[x] + std::to_string(y + 1);
}

/*
    @brief Generate time in string

    @param t                    time in [ms]
    @return time with ms as string
*/
std::string ms_to_time(uint64_t t){
    std::string res;
    uint64_t hour = t / (1000 * 60 * 60);
    t %= 1000 * 60 * 60;
    uint64_t minute = t / (1000 * 60);
    t %= 1000 * 60;
    uint64_t second = t / 1000;
    uint64_t msecond = t % 1000;
    std::ostringstream hour_s;
    hour_s << std::right << std::setw(3) << std::setfill('0') << hour;
    res += hour_s.str();
    res += ":";
    std::ostringstream minute_s;
    minute_s << std::right << std::setw(2) << std::setfill('0') << minute;
    res += minute_s.str();
    res += ":";
    std::ostringstream second_s;
    second_s << std::right << std::setw(2) << std::setfill('0') << second;
    res += second_s.str();
    res += ".";
    std::ostringstream msecond_s;
    msecond_s << std::right << std::setw(3) << std::setfill('0') << msecond;
    res += msecond_s.str();
    return res;
}

/*
    @brief Generate time in string

    @param t                    time in [ms]
    @return time as string
*/
std::string ms_to_time_short(uint64_t t){
    std::string res;
    uint64_t hour = t / (1000 * 60 * 60);
    t -= hour * 1000 * 60 * 60;
    uint64_t minute = t / (1000 * 60);
    t -= minute * 1000 * 60;
    uint64_t second = t / 1000;
    t -= second * 1000;
    std::ostringstream hour_s;
    hour_s << std::right << std::setw(3) << std::setfill('0') << hour;
    res += hour_s.str();
    res += ":";
    std::ostringstream minute_s;
    minute_s << std::right << std::setw(2) << std::setfill('0') << minute;
    res += minute_s.str();
    res += ":";
    std::ostringstream second_s;
    second_s << std::right << std::setw(2) << std::setfill('0') << second;
    res += second_s.str();
    return res;
}