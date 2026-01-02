/*
    Egaroucid Project

    @file flip.hpp
        Flip calculation
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
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