/*
    Egaroucid Project

    @file mobility.hpp
        Calculate legal moves
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include "setting.hpp"
#if USE_AVX2
    #include "mobility_avx2.hpp"
#else
    #include "mobility_generic.hpp"
#endif