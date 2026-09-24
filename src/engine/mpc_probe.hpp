/* Egaroucid Project. SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <algorithm>
#include "search.hpp"

// Probe nesting follows the Search, including parallel copies. Precision is
// reduced only inside the configured nesting limit, and never labeled 100%
// while a recursive MPC is allowed. Restore all context on every return path.
// A probe from an endgame node splits like a midgame search, but more split
// tasks there only add nodes, so it keeps the tighter endgame-probe limit.
class Mpc_probe_scope {
    Search &search;
    const uint_fast8_t saved_level;
    const bool saved_dim0;
    const int saved_mid_split_task_limit;
public:
    Mpc_probe_scope(Search &s, bool dim0, bool from_endgame = false)
        : search(s), saved_level(s.mpc_level), saved_dim0(s.use_dim0_mpc_eval),
          saved_mid_split_task_limit(s.mid_split_task_limit) {
        ++search.mpc_probe_nesting;
        if (search.mpc_probe_nesting >= MPC_MAX_PROBE_NESTING) search.mpc_level = MPC_100_LEVEL;
        if (from_endgame) {
            search.mid_split_task_limit = std::min(search.mid_split_task_limit, YBWC_END_PROBE_MAX_SPLIT_TASKS);
        }
#if !USE_DIM0_ONLY_EVALUATION
        search.use_dim0_mpc_eval = dim0;
#else
        (void)dim0;
#endif
    }
    Mpc_probe_scope(const Mpc_probe_scope &) = delete;
    Mpc_probe_scope &operator=(const Mpc_probe_scope &) = delete;
    ~Mpc_probe_scope() {
        search.mpc_level = saved_level;
        search.use_dim0_mpc_eval = saved_dim0;
        search.mid_split_task_limit = saved_mid_split_task_limit;
        --search.mpc_probe_nesting;
    }
};
