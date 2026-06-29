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

#if defined(EVALUATE_EXPERIMENT_7_7_BETA)
    #ifndef EVALUATE_SIMD_HEADER
        #define EVALUATE_SIMD_HEADER "evaluate_simd_experiment_7_7_beta.hpp"
    #endif
    #ifndef EVALUATE_GENERIC_HEADER
        #define EVALUATE_GENERIC_HEADER "evaluate_generic_experiment_7_7_beta.hpp"
    #endif
#endif

#if defined(EVALUATE_EXPERIMENT_EDAX_LINEAR)
    #ifndef EVALUATE_SIMD_HEADER
        #define EVALUATE_SIMD_HEADER "evaluate_simd_experiment_edax_linear.hpp"
    #endif
    #ifndef EVALUATE_GENERIC_HEADER
        #define EVALUATE_GENERIC_HEADER "evaluate_generic_experiment_edax_linear.hpp"
    #endif
#endif

#ifndef EVALUATE_SIMD_HEADER
    #define EVALUATE_SIMD_HEADER "evaluate_simd.hpp"
#endif

#ifndef EVALUATE_GENERIC_HEADER
    #define EVALUATE_GENERIC_HEADER "evaluate_generic.hpp"
#endif

#if USE_SIMD_EVALUATION
#include EVALUATE_SIMD_HEADER
#else
#include EVALUATE_GENERIC_HEADER
#endif
