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

#if defined(EVALUATE_EXPERIMENT_7_7_FM)
    #ifndef EVALUATE_SIMD_HEADER
        #define EVALUATE_SIMD_HEADER "evaluate_simd_experiment_7_7_fm.hpp"
    #endif
    #ifndef EVALUATE_GENERIC_HEADER
        #define EVALUATE_GENERIC_HEADER "evaluate_generic_experiment_7_7_fm.hpp"
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

#if defined(EVALUATE_EXPERIMENT_EDAX_FM)
    #ifndef EVALUATE_SIMD_HEADER
        #define EVALUATE_SIMD_HEADER "evaluate_simd_experiment_edax_fm.hpp"
    #endif
    #ifndef EVALUATE_GENERIC_HEADER
        #define EVALUATE_GENERIC_HEADER "evaluate_generic_experiment_edax_fm.hpp"
    #endif
#endif

#if defined(EVALUATE_EXPERIMENT_CURRENT_FM)
    #ifndef EVALUATE_SIMD_HEADER
        #define EVALUATE_SIMD_HEADER "evaluate_simd_experiment_current_fm.hpp"
    #endif
    #ifndef EVALUATE_GENERIC_HEADER
        #define EVALUATE_GENERIC_HEADER "evaluate_generic_experiment_current_fm.hpp"
    #endif
#endif

#if defined(EVALUATE_EXPERIMENT_CURRENT_FM_PATTERN_ONLY)
    #ifndef EVALUATE_SIMD_HEADER
        #define EVALUATE_SIMD_HEADER "evaluate_simd_experiment_current_fm_pattern_only.hpp"
    #endif
    #ifndef EVALUATE_GENERIC_HEADER
        #define EVALUATE_GENERIC_HEADER "evaluate_generic_experiment_current_fm_pattern_only.hpp"
    #endif
#endif

#if defined(EVALUATE_EXPERIMENT_CURRENT_FM_CROSS_TYPE)
    #ifndef EVALUATE_SIMD_HEADER
        #define EVALUATE_SIMD_HEADER "evaluate_simd_experiment_current_fm_cross_type.hpp"
    #endif
    #ifndef EVALUATE_GENERIC_HEADER
        #define EVALUATE_GENERIC_HEADER "evaluate_generic_experiment_current_fm_cross_type.hpp"
    #endif
#endif

#if defined(EVALUATE_EXPERIMENT_CURRENT_FM_SAME_TYPE)
    #ifndef EVALUATE_SIMD_HEADER
        #define EVALUATE_SIMD_HEADER "evaluate_simd_experiment_current_fm_same_type.hpp"
    #endif
    #ifndef EVALUATE_GENERIC_HEADER
        #define EVALUATE_GENERIC_HEADER "evaluate_generic_experiment_current_fm_same_type.hpp"
    #endif
#endif

#if defined(EVALUATE_EXPERIMENT_CURRENT_FM_DUAL_TYPE)
    #ifndef EVALUATE_SIMD_HEADER
        #define EVALUATE_SIMD_HEADER "evaluate_simd_experiment_current_fm_dual_type.hpp"
    #endif
    #ifndef EVALUATE_GENERIC_HEADER
        #define EVALUATE_GENERIC_HEADER "evaluate_generic_experiment_current_fm_dual_type.hpp"
    #endif
#endif

#if defined(EVALUATE_EXPERIMENT_CURRENT_FM_DUAL_TYPE_PHASE_WEIGHT)
    #ifndef EVALUATE_SIMD_HEADER
        #define EVALUATE_SIMD_HEADER "evaluate_simd_experiment_current_fm_dual_type_phase_weight.hpp"
    #endif
    #ifndef EVALUATE_GENERIC_HEADER
        #define EVALUATE_GENERIC_HEADER "evaluate_generic_experiment_current_fm_dual_type_phase_weight.hpp"
    #endif
#endif

#if defined(EVALUATE_EXPERIMENT_CURRENT_FM_LINEAR_ONLY)
    #ifndef EVALUATE_SIMD_HEADER
        #define EVALUATE_SIMD_HEADER "evaluate_simd_experiment_current_fm_linear_only.hpp"
    #endif
    #ifndef EVALUATE_GENERIC_HEADER
        #define EVALUATE_GENERIC_HEADER "evaluate_generic_experiment_current_fm_linear_only.hpp"
    #endif
#endif

#if defined(EVALUATE_EXPERIMENT_MOBILITY)
    #ifndef EVALUATE_SIMD_HEADER
        #define EVALUATE_SIMD_HEADER "evaluate_simd_experiment_mobility.hpp"
    #endif
    #ifndef EVALUATE_GENERIC_HEADER
        #define EVALUATE_GENERIC_HEADER "evaluate_generic_experiment_mobility.hpp"
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
