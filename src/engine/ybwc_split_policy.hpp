/*
    Egaroucid Project

    @file ybwc_split_policy.hpp
        Shared split-depth policy for YBWC search
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once

#include <cstdint>
#include "setting.hpp"
#include "level.hpp"

constexpr int YBWC_MID_SPLIT_MIN_DEPTH = 6;

inline int ybwc_end_split_min_depth(const uint_fast8_t mpc_level) {
    return mpc_level == MPC_74_LEVEL
        ? YBWC_SELECTIVE_END_SPLIT_MIN_DEPTH
        : YBWC_END_SPLIT_MIN_DEPTH;
}

inline int ybwc_active_end_split_min_depth(const uint_fast8_t mpc_level) {
#if YBWC_ENFORCE_SELECTIVE_END_SPLIT_MIN_DEPTH
    return ybwc_end_split_min_depth(mpc_level);
#else
    (void)mpc_level;
    return YBWC_END_SPLIT_MIN_DEPTH;
#endif
}

inline bool ybwc_can_split_child(
    const uint_fast8_t mpc_level,
    const int child_depth,
    const bool is_end_search
) {
    return child_depth >= (is_end_search
        ? ybwc_active_end_split_min_depth(mpc_level)
        : YBWC_MID_SPLIT_MIN_DEPTH);
}
