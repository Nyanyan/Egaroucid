// Verify that early root completion does not erase the preliminary search.
#include <iostream>
#include "../../engine/engine_all.hpp"

int main() {
    bit_init(); mobility_init(); flip_init(); last_flip_init();
    endsearch_init(); stability_init();
    Board board(~uint64_t{3}, uint64_t{2}); // h8 flips the last opposing disc.
    const uint64_t legal = board.get_legal();
    if (legal != 1) return 2;
    bool searching = true;
    global_searching = true;
    uint64_t preliminary_nodes = 0;
    const auto clogs = first_clog_search(board, &preliminary_nodes, 1, legal, &searching);
    const auto result = tree_search_legal(board, -64, 64, 1, MPC_74_LEVEL,
        false, legal, false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
    if (preliminary_nodes == 0 || result.clog_nodes != preliminary_nodes || result.nodes != 1 || result.value != 64) {
        std::cerr << "FAIL preliminary=" << preliminary_nodes << " preserved=" << result.clog_nodes
                  << " main=" << result.nodes << " value=" << result.value << '\n';
        return 1;
    }
    std::cout << "PASS preliminary=" << result.clog_nodes << " main=" << result.nodes << '\n';
}
