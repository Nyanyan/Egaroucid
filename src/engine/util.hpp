/*
    Egaroucid Project

    @file util.hpp
        Utility
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
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

/*
    @brief Generate coordinate in string

    @param idx                  index of the coordinate
    @return coordinate as string
*/
std::string idx_to_coord(int idx){
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
    res += ".";
    std::ostringstream msecond_s;
    msecond_s << std::left << std::setw(3) << std::setfill('0') << t;
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