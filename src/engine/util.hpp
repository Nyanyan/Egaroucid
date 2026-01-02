/*
    Egaroucid Project

    @file util.hpp
        Utility
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include "board.hpp"
#include "search.hpp"

std::pair<Board, int> convert_board_from_str(std::string board_str) {
    Board board;
    board_str.erase(std::remove_if(board_str.begin(), board_str.end(), ::isspace), board_str.end());
    if (board_str.length() != HW2 + 1) {
        std::cerr << "[ERROR] invalid argument got length " << board_str.length() << " expected " << HW2 + 1 << std::endl;
        return std::make_pair(board, -1); // error
    }
    if (!board.from_str(board_str)) {
        return std::make_pair(board, -1); // error
    }
    int player = -1;
    if (is_black_like_char(board_str[HW2])) {
        player = BLACK;
    } else if (is_white_like_char(board_str[HW2])) {
        player = WHITE;
    }
    return std::make_pair(board, player);
}

/*
    @brief Input board from console

    '0' as black, '1' as white, '.' as empty

    @return board structure
*/
Board input_board() {
    Board res;
    char elem;
    int player;
    std::cin >> player;
    res.player = 0;
    res.opponent = 0;
    for (int i = 0; i < HW2; ++i) {
        std::cin >> elem;
        if (elem == '0') {
            if (player == BLACK)
                res.player |= 1ULL << (HW2_M1 - i);
            else
                res.opponent |= 1ULL << (HW2_M1 - i);
        } else if (elem == '1') {
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
    if (board_str.length() != 16) {
        std::cerr << "[ERROR] invalid argument" << std::endl;
        return true;
    }
    board->player = 0;
    board->opponent = 0;
    int idx, d;
    char c;
    for (int i = 0; i < 16; ++i) {
        idx = i * 4 + 3;
        c = board_str[i] - '!';
        d = c / 32;
        if (d == 1) {
            board->player |= 1ULL << idx;
        } else if (d == 2) {
            board->opponent |= 1ULL << idx;
        }
        --idx;
        c %= 32;
        d = c / 9;
        if (d == 1) {
            board->player |= 1ULL << idx;
        } else if (d == 2) {
            board->opponent |= 1ULL << idx;
        }
        --idx;
        c %= 9;
        d = c / 3;
        if (d == 1) {
            board->player |= 1ULL << idx;
        } else if (d == 2) {
            board->opponent |= 1ULL << idx;
        }
        --idx;
        c %= 3;
        d = c;
        if (d == 1) {
            board->player |= 1ULL << idx;
        } else if (d == 2) {
            board->opponent |= 1ULL << idx;
        }
    }
    return false;
}


inline bool compare_representative_board(Board *res, Board *cmp) {
    if (res->player > cmp->player || (res->player == cmp->player && res->opponent > cmp->opponent)) {
        res->player = cmp->player;
        res->opponent = cmp->opponent;
        return true;
    }
    return false;
}

inline Board representative_board(Board b) {
    Board res = b;
    Board bt = b;   bt.board_black_line_mirror();       compare_representative_board(&res, &bt);
    Board bv =      b.get_vertical_mirror();            compare_representative_board(&res, &bv);
    Board btv =     bt.get_vertical_mirror();           compare_representative_board(&res, &btv);
                    b.board_horizontal_mirror();        compare_representative_board(&res, &b);
                    bt.board_horizontal_mirror();       compare_representative_board(&res, &bt);
                    b.board_vertical_mirror();          compare_representative_board(&res, &b);
                    bt.board_vertical_mirror();         compare_representative_board(&res, &bt);
    return res;
}

inline Board representative_board(Board b, int *idx) {
    Board res = b;                                                                                      *idx = 0; // default
    Board bt = b;   bt.board_black_line_mirror();       if (compare_representative_board(&res, &bt))    *idx = 2; // black line
    Board bv =      b.get_vertical_mirror();            if (compare_representative_board(&res, &bv))    *idx = 1; // vertical
    Board btv =     bt.get_vertical_mirror();           if (compare_representative_board(&res, &btv))   *idx = 3; // black line + vertical
                    b.board_horizontal_mirror();        if (compare_representative_board(&res, &b))     *idx = 6; // horizontal
                    bt.board_horizontal_mirror();       if (compare_representative_board(&res, &bt))    *idx = 4; // black line + horizontal
                    b.board_vertical_mirror();          if (compare_representative_board(&res, &b))     *idx = 7; // horizontal + vertical
                    bt.board_vertical_mirror();         if (compare_representative_board(&res, &bt))    *idx = 5; // black line + horizontal + vertical
    return res;
}

inline Board representative_board(Board *b, int *idx) {
    return representative_board(b->copy(), idx);
}

inline Board representative_board(Board *b) {
    return representative_board(b->copy());
}

inline int convert_coord_from_representative_board(int cell, int idx) {
    int res;
    int y = cell / HW;
    int x = cell % HW;
    switch (idx) {
        case 0:
            res = cell;
            break;
        case 1:
            res = (HW_M1 - y) * HW + x; // vertical
            break;
        case 2:
            res = (HW_M1 - x) * HW + (HW_M1 - y); // black line
            break;
        case 3:
            res = (HW_M1 - x) * HW + y; // black line + vertical ( = rotate 90 clockwise)
            break;
        case 4:
            res = x * HW + (HW_M1 - y); // black line + horizontal ( = rotate 90 counterclockwise)
            break;
        case 5:
            res = x * HW + y; // black line + horizontal + vertical ( = white line)
            break;
        case 6:
            res = y * HW + (HW_M1 - x); // horizontal
            break;
        case 7:
            res = (HW_M1 - y) * HW + (HW_M1 - x); // horizontal + vertical ( = rotate180)
            break;
        default:
            res = MOVE_UNDEFINED;
            std::cerr << "converting coord error" << std::endl;
            break;
    }
    return res;
}

inline int convert_coord_to_representative_board(int cell, int idx) {
    int res;
    int y = cell / HW;
    int x = cell % HW;
    switch (idx) {
        case 0:
            res = cell;
            break;
        case 1:
            res = (HW_M1 - y) * HW + x; // vertical
            break;
        case 2:
            res = (HW_M1 - x) * HW + (HW_M1 - y); // black line
            break;
        case 3:
            res = x * HW + (HW_M1 - y); // black line + vertical ( = rotate 90 clockwise)
            break;
        case 4:
            res = (HW_M1 - x) * HW + y; // black line + horizontal ( = rotate 90 counterclockwise)
            break;
        case 5:
            res = x * HW + y; // black line + horizontal + vertical ( = white line)
            break;
        case 6:
            res = y * HW + (HW_M1 - x); // horizontal
            break;
        case 7:
            res = (HW_M1 - y) * HW + (HW_M1 - x); // horizontal + vertical ( = rotate180)
            break;
        default:
            res = MOVE_UNDEFINED;
            std::cerr << "converting coord error" << std::endl;
            break;
    }
    return res;
}

bool is_valid_transcript(std::string transcript) {
    if (transcript.size() % 2) {
        return false;
    }
    Board board;
    Flip flip;
    board.reset();
    for (int i = 0; i < transcript.size() - 1 && board.check_pass(); i += 2) {
        if (!is_coord_like_chars(transcript[i], transcript[i + 1])) {
            return false;
        }
        int coord = get_coord_from_chars(transcript[i], transcript[i + 1]);
        if ((board.get_legal() & (1ULL << coord)) == 0) {
            return false;
        }
        calc_flip(&flip, &board, coord);
        board.move_board(&flip);
    }
    return true;
}

std::vector<uint_fast8_t> transcript_to_arr(std::string transcript) {
    std::vector<uint_fast8_t> res;
    for (int i = 0; i < transcript.size() - 1; i += 2) {
        uint_fast8_t x = transcript[i] - 'a';
        res.emplace_back(get_coord_from_chars(transcript[i], transcript[i + 1]));
    }
    return res;
}