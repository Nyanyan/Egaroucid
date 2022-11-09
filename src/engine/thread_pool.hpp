/*
    Egaroucid Project

    @file thread_pool.hpp
        Thread pool for Egaroucid
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include "setting.hpp"
#include "board.hpp"
#include "thread_pool/thread_pool_ctpl.hpp"

ctpl::thread_pool thread_pool(1);
