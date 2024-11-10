/*
    Egaroucid Project

    @file evaluate.hpp
        Evaluation function
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include "setting.hpp"
#if USE_SIMD_EVALUATION
    #include "evaluate_simd.hpp"
#else
    #include "evaluate_generic.hpp"
#endif
