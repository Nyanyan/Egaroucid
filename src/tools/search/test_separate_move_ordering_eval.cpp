/*
    Verify separate ordering evaluation against the existing main evaluator
    loaded with the same file, including incremental moves, passes, sibling
    updates, scoped auxiliary leaves, shared/isolated TT behavior and
    unchanged exact search values.
    Usage: test.exe <resource_dir> <ordering.egev2>
*/
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../../engine/ai.hpp"

namespace {
void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
uint64_t random_state = 0x123456789ABCDEF0ULL;
uint64_t next_random() {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 7;
    random_state ^= random_state << 17;
    return random_state;
}
}

int main(int argc, char **argv) {
    try {
        require(argc == 3 || argc == 4, "usage: test.exe <resource_dir> <ordering.egev2> [auxiliary.egev2]");
        const std::string resources = argv[1];
        bit_init();
        mobility_init();
        flip_init();
        last_flip_init();
        endsearch_init();
        stability_init();
        move_ordering_init();
        mpc_init();
        require(hash_resize(DEFAULT_HASH_LEVEL, 16, resources + "/../", false), "hash initialization");
        // Exercise the explicit model mode independently of automatic policy.
        eval_move_ordering_auto_requested = false;
        require(evaluate_init(resources + "/eval.egev2", resources + "/eval_move_ordering_end.egev", false), "main evaluation initialization");
        Board initial;
        initial.reset();
        Search initial_search(&initial, MPC_100_LEVEL, false, false);
        const int initial_main_value = mid_evaluate_diff(&initial_search);
        require(!eval_move_ordering_mid_enabled, "separate evaluation defaults off");
        require(mid_evaluate_for_ordering(&initial_search) == initial_main_value, "disabled ordering evaluation matches main");
        require(load_eval_move_ordering_mid_file(argv[2], false), "separate evaluation initialization");
        require(mid_evaluate_diff(&initial_search) == initial_main_value, "loading separate evaluation preserves main");
        std::vector<std::pair<Board, int>> expected;
        std::vector<Board> exact_boards;
        uint64_t child_checks = 0;
        int passes = 0;
        for (int game = 0; game < 32; ++game) {
            Search search(&initial, MPC_100_LEVEL, false, false);
            bool passed = false;
            while (search.n_discs < 63) {
                expected.emplace_back(search.board, mid_evaluate_for_ordering(&search));
                Search fresh(&search.board, MPC_100_LEVEL, false, false);
                require(mid_evaluate_for_ordering(&fresh) == expected.back().second, "incremental features match fresh features");
                if (game < 4 && search.n_discs == 56) {
                    exact_boards.push_back(search.board);
                }
                uint64_t legal = search.board.get_legal();
                if (!legal) {
                    if (passed) break;
                    search.pass();
                    passed = true;
                    ++passes;
                    continue;
                }
                passed = false;
#if USE_SIMD
                Eval_features sibling_base;
                eval_prepare_move_sibling_base(&search.eval, &search.board, &sibling_base);
#endif
                uint64_t remaining = legal;
                for (int cell = first_bit(&remaining); remaining; cell = next_bit(&remaining)) {
                    Flip_value move;
                    calc_flip_value(&move, &search.board, cell);
                    search.move(&move.flip);
                    const int ordering = mid_evaluate_for_ordering(&search);
                    const int main = mid_evaluate_diff(&search);
                    move.static_eval = main;
                    require(move_ordering_mid_evaluate(&search, &move, true) == ordering, "ordering helper selects separate evaluation");
                    require(move.static_eval == SCORE_UNDEFINED, "ordering evaluation never enters main MPC cache");
                    expected.emplace_back(search.board, ordering);
                    search.undo(&move.flip);
                    for (int depth : {0, 1, 2}) {
                        Flip_value ordinary = move;
                        ordinary.static_eval = SCORE_UNDEFINED;
                        bool searching = true;
                        move_evaluate_nws(&search, &ordinary, -20, 20, depth, true, true, &searching);
                        require(ordinary.static_eval == SCORE_UNDEFINED, "NWS path preserves cache isolation");
#if USE_SIMD
                        Flip_value sibling = move;
                        sibling.static_eval = SCORE_UNDEFINED;
                        move_evaluate_nws_from_sibling_base(&search, &sibling, -20, 20, depth, true, true, &searching, &sibling_base);
                        require(ordinary.value == sibling.value && ordinary.n_legal == sibling.n_legal, "sibling ordering matches ordinary ordering");
                        require(sibling.static_eval == SCORE_UNDEFINED, "sibling path preserves cache isolation");
#endif
                    }
                    ++child_checks;
                }
                int choice = next_random() % pop_count_ull(legal);
                while (choice--) legal &= legal - 1;
                Flip move;
                calc_flip(&move, &search.board, first_bit(&legal));
                search.move(&move);
            }
        }
        for (Board board : exact_boards) {
            bool searching = true;
            transposition_table.init();
            eval_move_ordering_mid_enabled = false;
            auto baseline = tree_search_legal(board, -SCORE_MAX, SCORE_MAX, HW2 - board.n_discs(), MPC_100_LEVEL, false, board.get_legal(), false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
            transposition_table.init();
            eval_move_ordering_mid_enabled = true;
            auto candidate = tree_search_legal(board, -SCORE_MAX, SCORE_MAX, HW2 - board.n_discs(), MPC_100_LEVEL, false, board.get_legal(), false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
            require(baseline.value == candidate.value, "exact search value is unchanged");
        }
        auto sample = std::find_if(expected.begin(), expected.end(), [](auto &item) { return item.first.n_discs() == 24; });
        require(sample != expected.end(), "midgame sample available");
        Search auxiliary(&sample->first, MPC_100_LEVEL, false, false);
        auxiliary.use_multi_thread = true;
        const int normal_leaf = mid_evaluate_diff(&auxiliary);
        eval_move_ordering_mid_shallow_max_depth = 5;
        require(mid_evaluate_for_search(&auxiliary) == normal_leaf, "main leaf never switches merely because depth is small");
        {
            Move_ordering_eval_scope scope(auxiliary, 6);
            require(!search_uses_separate_shallow_evaluation(&auxiliary), "six-ply auxiliary search retains main leaves");
        }
        const uint32_t hash = auxiliary.board.hash();
        uint64_t legal = auxiliary.board.get_legal();
        const int policy = first_bit(&legal);
        transposition_table.init();
        transposition_table.reg(&auxiliary, hash, 5, -64, 64, 11, policy);
        {
            Move_ordering_eval_scope scope(auxiliary, 5);
            require(search_uses_separate_shallow_evaluation(&auxiliary), "five-ply auxiliary search switches leaves");
            require(mid_evaluate_for_search(&auxiliary) == sample->second, "auxiliary leaf uses supplied evaluation");
            require(!auxiliary.use_multi_thread, "shallow auxiliary searches stay serial");
            require(!auxiliary.can_use_tt_eval_bounds(), "isolated context forbids TT bounds");
            int lower = -64, upper = 64;
            require(!transposition_table.get_bounds(&auxiliary, hash, 5, &lower, &upper), "isolated auxiliary search ignores shared bounds");
            transposition_table.reg(&auxiliary, hash, 5, -64, 64, 13, policy);
            Flip_value hint;
            hint.static_eval = SCORE_UNDEFINED;
            move_ordering_mid_evaluate(&auxiliary, &hint, true);
            require(hint.static_eval == sample->second, "auxiliary MPC hint uses matching leaf model");
            bool searching = true;
            const int value = nega_scout_node(&auxiliary, -64, 64, 3, false, legal, false, SEARCH_NODE_NONPV, Search_cancellation_context{&searching, nullptr});
            require(value != SCORE_UNDEFINED, "isolated auxiliary search completes");
        }
        require(!search_uses_separate_shallow_evaluation(&auxiliary) && auxiliary.use_multi_thread, "scope restores model and parallel setting");
        int lower = -64, upper = 64;
        require(transposition_table.get_bounds(&auxiliary, hash, 5, &lower, &upper) && lower == 11 && upper == 11, "isolated search leaves main TT bounds unchanged");
        eval_move_ordering_mid_shallow_share_tt = true;
        transposition_table.init();
        int shallow_value;
        {
            Move_ordering_eval_scope scope(auxiliary, 5);
            require(auxiliary.can_use_tt_eval_bounds(), "shared mode permits ordinary TT bounds");
            bool searching = true;
            shallow_value = nega_scout_node(&auxiliary, -64, 64, 3, false, legal, false, SEARCH_NODE_NONPV, Search_cancellation_context{&searching, nullptr});
        }
        lower = -64; upper = 64;
        require(transposition_table.get_bounds(&auxiliary, hash, 3, &lower, &upper) && lower == shallow_value && upper == shallow_value,
                "normal context reuses separate auxiliary result with the same TT key and no model tag");
        eval_move_ordering_mid_shallow_max_depth = 0;
        eval_move_ordering_mid_shallow_share_tt = false;
        transposition_table.init();
        // Existing loader and evaluator serve as an independent reference for
        // both SIMD indexing and Generic player-orientation tables.
        require(load_eval_file(argv[2], false), "reference evaluation initialization");
        for (auto &sample : expected) {
            Search search(&sample.first, MPC_100_LEVEL, false, false);
            require(mid_evaluate_diff(&search) == sample.second, "separate evaluation matches existing main evaluator");
        }
        if (argc == 4) {
            require(load_eval_move_ordering_aux_file(argv[3], false), "auxiliary model initialization");
            require(load_eval_file(argv[3], false), "auxiliary reference model initialization");
            eval_move_ordering_mid_shallow_max_depth = 5;
            eval_move_ordering_mid_shallow_end_only = true;
            uint64_t differing_models = 0;
            for (auto &sample : expected) {
                Search mixed(&sample.first, MPC_100_LEVEL, false, false);
                mixed.use_multi_thread = true;
                const int auxiliary_reference = mid_evaluate_diff(&mixed);
                differing_models += auxiliary_reference != sample.second;
                require(mid_evaluate_for_ordering(&mixed) == sample.second, "loading auxiliary model preserves static ordering");
                {
                    Move_ordering_eval_scope mid_scope(mixed, 5, false);
                    require(!search_uses_separate_shallow_evaluation(&mixed), "end-only model disabled for midgame auxiliary search");
                    require(mixed.use_multi_thread && mixed.can_use_tt_eval_bounds(), "disabled scope preserves parallelism and TT access");
                }
                {
                    Move_ordering_eval_scope too_deep(mixed, 6, true);
                    require(!search_uses_separate_shallow_evaluation(&mixed), "six-ply endgame auxiliary search retains ordinary leaves");
                }
                {
                    Move_ordering_eval_scope end_scope(mixed, 5, true);
                    require(search_uses_separate_shallow_evaluation(&mixed), "endgame scope selects auxiliary model");
                    require(mid_evaluate_for_search(&mixed) == auxiliary_reference, "auxiliary leaves match independent reference model");
                    require(mid_evaluate_for_ordering(&mixed) == sample.second, "static model remains separate inside auxiliary search");
                    require(!mixed.can_use_tt_eval_bounds() && !mixed.use_multi_thread, "isolated auxiliary context disables bounds and parallelism");
                    Flip_value hint;
                    hint.static_eval = SCORE_UNDEFINED;
                    require(move_ordering_mid_evaluate(&mixed, &hint, true) == sample.second, "move ordering uses the static model");
                    require(hint.static_eval == auxiliary_reference, "MPC static hint matches the auxiliary leaf model");
                    {
                        Move_ordering_eval_scope nested(mixed, 3, false);
                        require(mid_evaluate_for_search(&mixed) == auxiliary_reference, "nested midgame probe inherits endgame auxiliary model");
                    }
                }
                require(!search_uses_separate_shallow_evaluation(&mixed) && mixed.use_multi_thread, "endgame scope restores leaf and parallel context");
            }
            require(differing_models > 0, "two supplied models exercise distinct evaluations");
            require(load_eval_file((resources + "/eval.egev2").c_str(), false), "restore ordinary main model");
            Search mixed(&sample->first, MPC_100_LEVEL, false, false);
            const int main_leaf = mid_evaluate_diff(&mixed);
            transposition_table.init();
            transposition_table.reg(&mixed, mixed.board.hash(), 5, -64, 64, 11, policy);
            {
                Move_ordering_eval_scope isolated(mixed, 5, true);
                bool searching = true;
                require(nega_scout_node(&mixed, -64, 64, 3, false, mixed.board.get_legal(), false, SEARCH_NODE_NONPV,
                                      Search_cancellation_context{&searching, nullptr}) != SCORE_UNDEFINED, "mixed isolated probe completes");
            }
            lower = -64; upper = 64;
            require(transposition_table.get_bounds(&mixed, mixed.board.hash(), 5, &lower, &upper) && lower == 11 && upper == 11,
                    "mixed isolated probe preserves ordinary TT bounds");
            eval_move_ordering_mid_shallow_share_tt = true;
            transposition_table.init();
            int mixed_value;
            {
                Move_ordering_eval_scope shared(mixed, 5, true);
                bool searching = true;
                require(mixed.can_use_tt_eval_bounds(), "mixed shared scope permits untagged ordinary TT bounds");
                mixed_value = nega_scout_node(&mixed, -64, 64, 3, false, mixed.board.get_legal(), false, SEARCH_NODE_NONPV,
                                              Search_cancellation_context{&searching, nullptr});
            }
            lower = -64; upper = 64;
            require(transposition_table.get_bounds(&mixed, mixed.board.hash(), 3, &lower, &upper) && lower == mixed_value && upper == mixed_value,
                    "ordinary search reuses mixed auxiliary result under the same key");
            require(mid_evaluate_for_search(&mixed) == main_leaf, "mixed auxiliary probes preserve ordinary main leaves");
            std::cout << "PASS mixed_model_samples=" << expected.size() << " distinct_values=" << differing_models << '\n';
        }
        require(load_eval_file((resources + "/eval.egev2").c_str(), false), "restore ordinary model for endgame checks");
        eval_move_ordering_mid_shallow_end_only = true;
        uint64_t end_only_exact_checks = 0;
        for (Board board : exact_boards) {
            bool searching = true;
            eval_move_ordering_mid_shallow_max_depth = 0;
            transposition_table.init();
            const auto baseline = tree_search_legal(board, -SCORE_MAX, SCORE_MAX, HW2 - board.n_discs(), MPC_100_LEVEL,
                false, board.get_legal(), false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
            for (bool sharing : {false, true}) {
                eval_move_ordering_mid_shallow_max_depth = 5;
                eval_move_ordering_mid_shallow_share_tt = sharing;
                transposition_table.init();
                const auto candidate = tree_search_legal(board, -SCORE_MAX, SCORE_MAX, HW2 - board.n_discs(), MPC_100_LEVEL,
                    false, board.get_legal(), false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
                require(candidate.value == baseline.value, "end-only auxiliary search preserves exact endgame value");
                ++end_only_exact_checks;
            }
        }
        std::cout << "PASS end_only_exact_checks=" << end_only_exact_checks << '\n';
        std::cout << "PASS samples=" << expected.size() << " child_checks=" << child_checks
                  << " passes=" << passes << " exact_positions=" << exact_boards.size() << '\n';
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
