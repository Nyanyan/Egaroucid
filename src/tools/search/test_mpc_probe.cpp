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
    std::cout << "PASS MPC probe nesting, helper inheritance, restoration and LMR exclusion\n";
}
