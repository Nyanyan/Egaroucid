/*
    Egaroucid Project

    @file evaluate_nnue_search_impl.hpp
        NNUE evaluation functions that require Search definition
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once

inline int mid_evaluate_diff(Search *search) {
    return eval_nnue_forward_from_accumulator(search->phase(), search->eval.accumulator[search->eval.feature_idx]);
}

inline int mid_evaluate_move_ordering_dim0(Search *search) {
    return mid_evaluate_diff(search);
}

inline int mid_evaluate_dim0(Search *search) {
    return mid_evaluate_diff(search);
}

inline int mid_evaluate_move_ordering_end(Search *search) {
    return mid_evaluate_diff(search);
}
