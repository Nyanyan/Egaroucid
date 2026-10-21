/* Search-local choice of static ordering and shallow auxiliary models. */
#pragma once
#include <cstdint>
#include "level.hpp"

enum class Move_ordering_model : uint8_t { ordinary, primary, secondary };

struct Move_ordering_policy {
    bool automatic = false;
    Move_ordering_model static_model = Move_ordering_model::ordinary;
    Move_ordering_model auxiliary_model = Move_ordering_model::ordinary;
    int auxiliary_max_depth = 0;
    bool share_tt = false;

    bool operator==(const Move_ordering_policy &) const = default;
};

// Set only while initializing models, before any search threads start.
inline bool eval_move_ordering_auto_enabled = false;
#ifdef EGAROUCID_EVAL_NNUE_PATTERNS
inline bool eval_move_ordering_auto_requested = false;  // no Offset/Decay models for the NNUE pattern set
#else
inline bool eval_move_ordering_auto_requested = true;
#endif

// Defined after the ordinary evaluator in evaluate_move_ordering.hpp.
inline bool initialize_auto_move_ordering_models(const char *main_eval_file, bool show_log);

#ifndef EGAROUCID_MO_POLICY_VARIANT
#define EGAROUCID_MO_POLICY_VARIANT 0  // experiment builds only (see the end of select_move_ordering_policy)
#endif

// In automatic mode the primary bank is Offset, the secondary bank is Decay.
// Classify the root of each iteration, never its decreasing remaining depth.
constexpr Move_ordering_policy select_move_ordering_policy(int depth, bool is_end_search, uint_fast8_t mpc_level) {
    Move_ordering_policy policy;
    policy.automatic = true;
    if (!is_end_search) {
        if (depth >= 27) policy.static_model = Move_ordering_model::primary;
    } else if (mpc_level == MPC_100_LEVEL) {
        policy.static_model = policy.auxiliary_model = Move_ordering_model::secondary;
        policy.auxiliary_max_depth = 5;
    } else if (mpc_level != MPC_74_LEVEL) {
        policy.static_model = policy.auxiliary_model = Move_ordering_model::primary;
        policy.auxiliary_max_depth = 5;
        policy.share_tt = true;
    }
    // experiment variants: 1 no models, 2 static ordering only, 3 auxiliary leaves only, 4 auxiliary TT isolated
    if constexpr (EGAROUCID_MO_POLICY_VARIANT == 1 || EGAROUCID_MO_POLICY_VARIANT == 2) {
        if constexpr (EGAROUCID_MO_POLICY_VARIANT == 1) policy.static_model = Move_ordering_model::ordinary;
        policy.auxiliary_model = Move_ordering_model::ordinary;
        policy.auxiliary_max_depth = 0;
        policy.share_tt = false;
    } else if constexpr (EGAROUCID_MO_POLICY_VARIANT == 3) {
        policy.static_model = Move_ordering_model::ordinary;
    } else if constexpr (EGAROUCID_MO_POLICY_VARIANT == 4) {
        policy.share_tt = false;
    }
    return policy;
}
