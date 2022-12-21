/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include "./../../engine/setting.hpp"

#if USE_SIMD
    // version definition
    #define EGAROUCID_VERSION U"6.1.0 SIMD"
#else
    // version definition
    #define EGAROUCID_VERSION U"6.1.0 Generic"
#endif