/*
    Egaroucid Project

    @file mobility.hpp
        Calculate legal moves
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include "setting.hpp"
#if USE_SIMD
    #include "mobility_simd.hpp"
#else
    #include "mobility_generic.hpp"
#endif