/*
    Egaroucid Project

    @file endsearch.hpp
        Search near endgame
        last2/3/4 imported from Edax AVX, (C) 1998 - 2018 Richard Delorme, 2014 - 23 Toshihiko Okuhara
    @date 2021-2024
    @author Takuto Yamana
    @author Toshihiko Okuhara
    @license GPL-3.0 license
*/

#pragma once

#if USE_SIMD
    #include "endsearch_last_simd.hpp"
#else
    #include "endsearch_last_generic.hpp"
#endif
#include "endsearch_nws.hpp"