/*
    Egaroucid Project

    @file last_flip.hpp
        calculate number of flipped discs in the last move
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "setting.hpp"
#if USE_SIMD
#include "last_flip_simd.hpp"
#else
#include "last_flip_generic.hpp"
#endif