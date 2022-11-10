/*
    Egaroucid Project

    @file state.hpp
        State for Egaroucid
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/
#pragma once

struct State{
    bool book_changed;

    State(){
        book_changed = false;
    }
};