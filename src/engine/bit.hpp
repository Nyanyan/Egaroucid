/*
    Egaroucid Project

    @file bit.hpp
        Bit manipulation
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "bit_common.hpp"
#if USE_SIMD
#include "bit_simd.hpp"
#else
#include "bit_generic.hpp"
#endif