/*
    Egaroucid Project

    @file flip.hpp
        Flip calculation
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include "setting.hpp"
#if USE_SIMD
#if USE_AVX512
#include "flip_avx512.hpp"
#else
#include "flip_simd.hpp"
#endif
#else
#include "flip_generic.hpp"
#endif