/*
    Egaroucid Project

    @file flip.hpp
        Flip calculation
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include "setting.hpp"
#if USE_SIMD
    #include "flip_simd.hpp"
#else
    #include "flip_generic.hpp"
#endif