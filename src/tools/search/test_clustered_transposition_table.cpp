/* Egaroucid Project. SPDX-License-Identifier: GPL-3.0-or-later */
#define USE_CLUSTERED_TT 1
#include "../../engine/engine_all.hpp"
#include <barrier>
#include <cassert>
#include <iostream>
#include <thread>

Search position(uint64_t key, uint_fast8_t mpc = MPC_100_LEVEL) {
    Search s;
    s.board = Board(key, ~key);
    s.root_n_discs = s.n_discs = 64;
    s.mpc_level = mpc;
    return s;
}

int main() {
    Transposition_table tt;
    assert(tt.resize(2)); // Four deliberately colliding full-board records.
    auto a = position(1), b = position(2), c = position(3), d = position(4);
    tt.reg(&a, 0, 1, -64, 64, 2, 1);
    tt.reg(&b, 0, 1, -64, 64, 4, 2);
    tt.reg(&c, 0, 1, -64, 64, 8, 3);
    tt.reg(&d, 0, 8, 5, 6, 6, 4); // lower = 6 in the last lane
    tt.del(&a.board, 0); // A hole before the existing D must not steal its update.
    tt.reg(&d, 0, 8, 6, 7, 6, MOVE_UNDEFINED); // upper = 6
    int l = -64, u = 64;
    assert(tt.get_bounds(&d, 0, 8, &l, &u) && l == 6 && u == 6);
    auto selective = d;
    selective.mpc_level = MPC_74_LEVEL;
    tt.reg(&selective, 0, 12, -64, 64, 10, 5);
    assert(tt.get_bounds(&d, 0, 8, &l, &u) && l == 6 && u == 6);
    assert(!tt.get_bounds(&d, 0, 12, &l, &u));
    assert(tt.get_bounds(&selective, 0, 12, &l, &u) && l == 10 && u == 10);
    tt.reset_importance();
    assert(tt.get_bounds(&d, 0, 8, &l, &u) && l == 6 && u == 6);
    d.tt_nonexact_bound_min_ply = 1;
    tt.reg_overwrite(&d, 0, 8, 5, 6, 6, 4);
    assert(!tt.get_bounds(&d, 0, 8, &l, &u));
    tt.del(&d.board, 0);
    assert(!tt.has_node_any_level(&d, 0));
    assert(!tt.has_node_any_level(&selective, 0));

    for (int size : {0, 1, 2, 10}) {
        assert(tt.resize(size));
        tt.reg(&a, 0, 4, -64, 64, 2, 1);
        assert(tt.get_bounds(&a, 0, 4, &l, &u) && l == 2 && u == 2);
        tt.init();
        assert(!tt.has_node_any_level(&a, 0));
    }
    assert(tt.resize(2));
    std::barrier start(8);
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 8; ++worker) workers.emplace_back([&, worker] {
        start.arrive_and_wait();
        for (int iteration = 0; iteration < 100000; ++iteration) {
            const int key = 1 + (iteration * 17 + worker * 7) % 63;
            auto s = position(key);
            const int score = key - 32;
            tt.reg(&s, 0, 8, -64, 64, score, key);
            int low = -64, high = 64;
            uint_fast8_t moves[2] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
            tt.get(&s, 0, 8, &low, &high, moves);
            if (moves[0] != MOVE_UNDEFINED) {
                assert(moves[0] == key && low == score && high == score);
            } else assert(low == -64 && high == 64);
            if ((iteration & 1023) == 0) tt.reset_importance();
        }
    });
    for (auto &worker : workers) worker.join();
    std::cout << "PASS clustered bounds, collisions, precision, storage, concurrent coherence\n";
}
