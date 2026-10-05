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
    return policy;
}
