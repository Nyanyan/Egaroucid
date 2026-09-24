/* Egaroucid Project. SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../engine/engine_all.hpp"
#include <cassert>
#include <iostream>

int main() {
    Search s;
    s.mpc_level = MPC_74_LEVEL;
    s.use_dim0_mpc_eval = true;
    {
        Mpc_probe_scope first(s, false);
        assert(s.mpc_probe_nesting == 1);
        assert(s.mpc_level == (MPC_MAX_PROBE_NESTING == 1 ? MPC_100_LEVEL : MPC_74_LEVEL));
        assert(mid_nws_lmr_reduction(&s, 20, 9, false) == 0);
        Search helper = s;
        {
            Mpc_probe_scope second(helper, true);
            assert(helper.mpc_probe_nesting == 2 && helper.mpc_level == MPC_100_LEVEL);
        }
        assert(helper.mpc_probe_nesting == 1 && helper.mpc_level == s.mpc_level);
        try {
            Mpc_probe_scope second(s, true);
            throw 1;
        } catch (int) {}
        assert(s.mpc_probe_nesting == 1);
    }
    assert(s.mpc_probe_nesting == 0 && s.mpc_level == MPC_74_LEVEL && s.use_dim0_mpc_eval);
    assert(mid_nws_lmr_reduction(&s, 20, 9, false) == 1);
    s.mid_split_task_limit = YBWC_MID_MAX_SPLIT_TASKS;
    {
        Mpc_probe_scope midgame(s, false);
        assert(s.mid_split_task_limit == YBWC_MID_MAX_SPLIT_TASKS);
        Mpc_probe_scope endgame(s, false, true);
        assert(s.mid_split_task_limit == std::min(YBWC_MID_MAX_SPLIT_TASKS, YBWC_END_PROBE_MAX_SPLIT_TASKS));
        Search helper = s;
        assert(helper.mid_split_task_limit == s.mid_split_task_limit);
    }
    assert(s.mid_split_task_limit == YBWC_MID_MAX_SPLIT_TASKS);
    std::cout << "PASS MPC probe nesting, helper inheritance, restoration, LMR exclusion and endgame probe split limit\n";
}
