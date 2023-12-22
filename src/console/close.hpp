/*
    Egaroucid Project

    @file close.hpp
        Closing process for Egaroucid
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include "./../engine/engine_all.hpp"
#include "option.hpp"
#include "state.hpp"

void close(State *state, Options *options){
    if (state->book_changed)
        book.save_egbk3(options->book_file, options->book_file + ".bak");
    std::exit(0);
}