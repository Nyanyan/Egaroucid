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
#include "./../engine/contest_book.hpp"

struct State {
    bool book_changed;
    uint64_t remaining_time_msec_black;
    uint64_t remaining_time_msec_white;
    std::future<std::vector<Ponder_elem>> ponder_future;
    bool ponder_searching;
    Contest_book contest_book;
    std::string contest_book_start;

    State() {
        book_changed = false;
        remaining_time_msec_black = 0;
        remaining_time_msec_white = 0;
        ponder_searching = false;
        contest_book_start = "";
    }
};
