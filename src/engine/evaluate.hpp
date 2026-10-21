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
#if defined(EGAROUCID_EVAL_NNUE_PATTERNS) && (defined(USE_NNUE_EVALUATION) || !USE_SIMD_EVALUATION)
#error "EGAROUCID_EVAL_NNUE_PATTERNS needs the SIMD pattern evaluation (evaluate_simd.hpp)"
#endif
#if defined(USE_NNUE_EVALUATION)
#include "./../tools/evaluation/nnue/evaluate_nnue.hpp"
#elif USE_SIMD_EVALUATION
#include "evaluate_simd.hpp"
#else
#include "evaluate_generic.hpp"
#endif

#include "evaluate_move_ordering.hpp"
