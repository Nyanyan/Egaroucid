/*
    Egaroucid Project

    @file evaluate.hpp
        Evaluation function
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "setting.hpp"
#if USE_SIMD_EVALUATION
// #include "evaluate_simd.hpp"
#include "evaluate_simd_fm.hpp"
#else
// #include "evaluate_generic.hpp"
#include "evaluate_generic_fm.hpp"
#endif