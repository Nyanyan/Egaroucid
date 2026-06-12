/*
    Egaroucid Project

    @file xot.hpp
        XOT opening identification
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "util.hpp"
#include "xot_keys.hpp"

constexpr int XOT_START_N_DISCS = 12;

inline bool is_xot_board_key(Board board) {
    board = representative_board(board);
    size_t left = 0;
    size_t right = XOT_BOARD_KEY_COUNT;
    while (left < right) {
        const size_t mid = left + (right - left) / 2;
        const XotBoardKey& key = XOT_BOARD_KEYS[mid];
        if (key.player < board.player || (key.player == board.player && key.opponent < board.opponent)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left < XOT_BOARD_KEY_COUNT &&
        XOT_BOARD_KEYS[left].player == board.player &&
        XOT_BOARD_KEYS[left].opponent == board.opponent;
}
