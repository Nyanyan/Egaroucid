/*
    Egaroucid Project

    @file last_flip.hpp
        calculate number of flipped discs in the last move
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include "setting.hpp"
#if USE_AVX2
    #include "last_flip_avx2.hpp"
#else
    #include "last_flip_generic.hpp"
#endif