/*
    Egaroucid Project

    @file bit_common.hpp
        Bit manipulation
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>

/*
    @brief print bits in reverse

    @param x                    an integer to print
*/
inline void bit_print_reverse(uint64_t x) {
    for (uint32_t i = 0; i < HW2; ++i) {
        std::cerr << (1 & (x >> i));
    }
    std::cerr << std::endl;
}

/*
    @brief print bits

    @param x                    an integer to print
*/
inline void bit_print(uint64_t x) {
    for (uint32_t i = 0; i < HW2; ++i) {
        std::cerr << (1 & (x >> (HW2_M1 - i)));
    }
    std::cerr << std::endl;
}

/*
    @brief print bits of uint8_t

    @param x                    an integer to print
*/
inline void bit_print_uchar(uint8_t x) {
    for (uint32_t i = 0; i < HW; ++i) {
        std::cerr << (1 & (x >> (HW_M1 - i)));
    }
    std::cerr << std::endl;
}

/*
    @brief print a board

    @param x                    an integer to print
*/
inline void bit_print_board(uint64_t x) {
    for (uint32_t i = 0; i < HW2; ++i) {
        std::cerr << (1 & (x >> (HW2_M1 - i)));
        if (i % HW == HW_M1) {
            std::cerr << std::endl;
        }
    }
    std::cerr << std::endl;
}

/*
    @brief print a board

    @param p                    an integer representing the player
    @param o                    an integer representing the opponent
*/
void print_board(uint64_t p, uint64_t o) {
    for (int i = 0; i < HW2; ++i) {
        if (1 & (p >> (HW2_M1 - i))) {
            std::cerr << '0';
        } else if (1 & (o >> (HW2_M1 - i))) {
            std::cerr << '1';
        } else {
            std::cerr << '.';
        }
        if (i % HW == HW_M1) {
            std::cerr << std::endl;
        }
    }
}