// Root-restricted searches; no changes to the production engine.
#include "engine/engine_all.hpp"
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    const int threads = std::stoi(argv[1]);
    if (threads < 1 || threads > 28) return 2;
    const std::string resource = argv[2];
    thread_pool.resize(threads - 1);
    bit_init(); mobility_init(); flip_init(); last_flip_init(); endsearch_init();
#if USE_MPC_PRE_CALCULATION
    mpc_init();
#endif
    move_ordering_init(); stability_init();
    if (!hash_init(28, resource + "/../")) return 3;
    global_hash_level = 28;
#if USE_CRC32C_HASH
    global_hash_bit_mask = (1U << 28) - 1;
#endif
    if (!transposition_table.resize(28)) return 3;
    if (!evaluate_init(resource + "/eval.egev2", resource + "/eval_move_ordering_end.egev", true)) return 3;
    std::cout << "READY\n" << std::flush;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        int tag; std::string op;
        if (!(input >> tag >> op)) return 4;
        if (op == "clear") {
            transposition_table.init();
            std::cout << "{\"tag\":" << tag << ",\"kind\":\"clear\"}\n" << std::flush;
            continue;
        }
        int depth; double prob; std::string position, turn, move;
        if (op != "root" || !(input >> depth >> prob >> position >> turn >> move)) return 4;
        Board board;
        if (position.size() != 64 || (turn != "X" && turn != "O") || !board.from_str(position + " " + turn)) return 4;
        int level = -1;
        for (int i = 0; i < N_SELECTIVITY_LEVEL; ++i)
            if (std::abs(SELECTIVITY_PERCENTAGE[i] - prob) < 1e-4) level = i;
        if (level < 0 || depth < 1 || depth > 64 - board.n_discs()) return 4;
        uint64_t mask = board.get_legal();
        if (move != "all") {
            if (move.size() != 2 || move[0] < 'a' || move[0] > 'h' || move[1] < '1' || move[1] > '8') return 4;
            const int idx = 63 - ((move[1] - '1') * 8 + move[0] - 'a');
            mask &= 1ULL << idx;
        }
        if (!mask) return 4;
        global_searching = true; bool searching = true;
        const uint64_t start = tim();
        Search_result result = tree_search_legal(board, -SCORE_MAX, SCORE_MAX, depth, level,
            true, mask, threads > 1, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
        if (result.depth != depth || std::abs(result.probability - prob) > 1e-4 ||
            result.value < -64 || result.value > 64 || result.is_exact_lower_bound ||
            !(mask & (1ULL << result.policy))) return 5;
        std::cout << "{\"tag\":" << tag << ",\"kind\":\"root\",\"value\":" << result.value
            << ",\"depth\":" << result.depth << ",\"probability\":" << result.probability
            << ",\"end\":" << (result.is_end_search ? "true" : "false")
            << ",\"nodes\":" << result.nodes << ",\"time_ms\":" << (tim() - start)
            << ",\"move\":\"" << idx_to_coord(result.policy) << "\"}\n" << std::flush;
    }
}
