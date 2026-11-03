/*
    Egaroucid Project

    @file state.hpp
        State for Egaroucid
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/
#pragma once
#include "./../engine/engine_all.hpp"

// time control set by GTP time_settings
#define GTP_TIME_CONTROL_NONE 0 // search with level
#define GTP_TIME_CONTROL_PER_GAME 1 // main time for the whole game (sudden death)
#define GTP_TIME_CONTROL_PER_MOVE 2 // fixed time for each move (canadian byo-yomi without main time)

struct State {
    bool book_changed;
    uint64_t remaining_time_msec_black;
    uint64_t remaining_time_msec_white;
    std::future<std::vector<Ponder_elem>> ponder_future;
    bool ponder_searching;
    int gtp_time_control;
    uint64_t gtp_main_time_msec;
    uint64_t gtp_move_time_msec;

    State() {
        book_changed = false;
        remaining_time_msec_black = 0;
        remaining_time_msec_white = 0;
        ponder_searching = false;
        gtp_time_control = GTP_TIME_CONTROL_NONE;
        gtp_main_time_msec = 0;
        gtp_move_time_msec = 0;
    }
};