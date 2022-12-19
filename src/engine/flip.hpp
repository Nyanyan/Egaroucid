/*
    Egaroucid Project

    @file flip.hpp
        Flip calculation
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include "setting.hpp"
#if USE_AVX2
    #include "flip_avx2.hpp"
#else
    #include "flip_generic.hpp"
#endif