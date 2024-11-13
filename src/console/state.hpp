/*
    Egaroucid Project

    @file state.hpp
        State for Egaroucid
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/
#pragma once
#include "./../engine/engine_all.hpp"

struct State {
    bool book_changed;
    uint64_t remaining_time_msec;

    State() {
        book_changed = false;
        remaining_time_msec = 0;
    }
};