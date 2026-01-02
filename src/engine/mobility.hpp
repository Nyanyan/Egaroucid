/*
    Egaroucid Project

    @file mobility.hpp
        Calculate legal moves
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "setting.hpp"
#if USE_SIMD
#include "mobility_simd.hpp"
#else
#include "mobility_generic.hpp"
#endif