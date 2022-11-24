/*
    Egaroucid Project

    @file state.hpp
        State for Egaroucid
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/
#pragma once
#include "./../engine/engine_all.hpp"

struct State{
    bool book_changed;
    uint8_t date;

    State(){
        book_changed = false;
        date = INIT_DATE;
    }
};