/*
    Egaroucid Project

    @file bit.hpp
        Bit manipulation
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
    @notice I referred to codes written by others
*/

#pragma once
#include "setting.hpp"
#include "common.hpp"
#if USE_AVX2
    #include "bit_avx2.hpp"
#else
    #include "bit_generic.hpp"
#endif
