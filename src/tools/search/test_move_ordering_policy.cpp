/* Regression checks for automatic Offset/Decay selection and TT isolation. */
#include <iostream>
#include <stdexcept>
#include <string>
#include <future>
#include "../../engine/engine_all.hpp"

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void clear_search_tables() {
    transposition_table.init();
    // The exact endgame cache is thread-local and survives main TT resets.
    for (auto &row : lttable) for (auto &entry : row) entry = LocalTTEntry{};
}

int main(int argc, char **argv) {
    try {
        require(argc == 2, "usage: test_move_ordering_policy <resource_dir>");
        const std::string resources = argv[1];
        thread_pool.resize(2);
        bit_init(); mobility_init(); flip_init(); last_flip_init(); endsearch_init();
#if USE_MPC_PRE_CALCULATION
        mpc_init();
#endif
        move_ordering_init(); stability_init();
        require(hash_resize(DEFAULT_HASH_LEVEL, 18, resources + "/../", false), "hash initialization");
        require(evaluate_init(resources + "/eval.egev2", resources + "/eval_move_ordering_end.egev", false), "evaluation initialization");
        Board board;
        board.reset();
        uint64_t random = 812731;
        while (board.n_discs() < 24) {
            uint64_t legal = board.get_legal();
            if (!legal) { board.pass(); continue; }
            random ^= random << 13; random ^= random >> 7; random ^= random << 17;
            int choice = random % pop_count_ull(legal);
            while (choice--) legal &= legal - 1;
            Flip flip; calc_flip(&flip, &board, first_bit(&legal)); board.move_board(&flip);
        }
#if USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION && !defined(USE_NNUE_EVALUATION)
        require(eval_move_ordering_auto_enabled && eval_move_ordering_mid_enabled && eval_move_ordering_aux_enabled, "automatic models initialized by default");
        require(load_eval_move_ordering_mid_file((resources + "/offset0_16.egev2").c_str(), false), "Offset bank");
        require(!eval_move_ordering_auto_enabled, "explicit model disables automatic selection");
        require(load_eval_move_ordering_aux_file((resources + "/decay_half16.egev2").c_str(), false), "Decay bank");
        Search reference(&board, MPC_100_LEVEL, false, false);
        const int ordinary = mid_evaluate_diff(&reference);
        require(load_eval_file((resources + "/offset0_16.egev2").c_str(), false), "Offset reference");
        const int offset = mid_evaluate_diff(&reference);
        require(load_eval_file((resources + "/decay_half16.egev2").c_str(), false), "Decay reference");
        const int decay = mid_evaluate_diff(&reference);
        require(load_eval_file((resources + "/eval.egev2").c_str(), false), "restore main evaluation");
        eval_move_ordering_auto_enabled = true;
        // Automatic selection must override unrelated manual knobs.
        eval_move_ordering_mid_shallow_max_depth = 0;
        eval_move_ordering_mid_shallow_share_tt = true;
        eval_move_ordering_mid_shallow_end_only = false;
        struct Case { int depth; bool end; uint_fast8_t mpc; Move_ordering_model model; bool auxiliary; bool sharing; };
        const Case cases[] = {
            {26, false, MPC_74_LEVEL, Move_ordering_model::ordinary, false, false},
            {27, false, MPC_74_LEVEL, Move_ordering_model::primary, false, false},
            {35, false, MPC_100_LEVEL, Move_ordering_model::primary, false, false},
            {40, true, MPC_74_LEVEL, Move_ordering_model::ordinary, false, false},
            {40, true, MPC_88_LEVEL, Move_ordering_model::primary, true, true},
            {40, true, MPC_93_LEVEL, Move_ordering_model::primary, true, true},
            {40, true, MPC_98_LEVEL, Move_ordering_model::primary, true, true},
            {40, true, MPC_99_LEVEL, Move_ordering_model::primary, true, true},
            {40, true, MPC_999_LEVEL, Move_ordering_model::primary, true, true},
            {40, true, MPC_100_LEVEL, Move_ordering_model::secondary, true, false},
        };
        for (const auto &item : cases) {
            Search search(&board, item.mpc, true, false);
            search.configure_move_ordering(item.depth, item.end);
            const int expected = item.model == Move_ordering_model::ordinary ? ordinary
                : item.model == Move_ordering_model::primary ? offset : decay;
            require(search.move_ordering_policy.static_model == item.model, "root chooses expected model");
            require(mid_evaluate_for_ordering(&search) == expected, "static evaluation matches independent model");
            require(mid_evaluate_for_search(&search) == ordinary, "main leaves stay ordinary");
            {
                Move_ordering_eval_scope scope(search, 5, false);
                require(!search_uses_separate_shallow_evaluation(&search), "automatic auxiliary evaluation is endgame-only");
            }
            {
                Move_ordering_eval_scope scope(search, 6, true);
                require(!search_uses_separate_shallow_evaluation(&search), "six-ply probes stay ordinary");
            }
            transposition_table.init();
            const auto hash = search.board.hash();
            uint64_t legal = search.board.get_legal();
            const int policy = first_bit(&legal);
            transposition_table.reg(&search, hash, 5, -64, 64, 11, policy);
            {
                Move_ordering_eval_scope scope(search, 5, true);
                require(search_uses_separate_shallow_evaluation(&search) == item.auxiliary, "five-ply probe eligibility");
                if (item.auxiliary) {
                    require(mid_evaluate_for_search(&search) == expected, "auxiliary uses root model");
                    require(search.can_use_tt_eval_bounds() == item.sharing, "automatic sharing/isolation selection");
                    require(!search.use_multi_thread, "auxiliary search stays serial");
                    int lower = -64, upper = 64;
                    require(transposition_table.get_bounds(&search, hash, 5, &lower, &upper) == item.sharing, "TT access follows policy");
                    transposition_table.reg(&search, hash, 5, -64, 64, 13, policy);
                    Flip_value hint;
                    hint.static_eval = SCORE_UNDEFINED;
                    require(move_ordering_mid_evaluate(&search, &hint, true) == expected && hint.static_eval == expected, "MPC hint uses matching model");
                    Move_ordering_eval_scope nested(search, 3, false);
                    require(mid_evaluate_for_search(&search) == expected, "nested probe inherits model");
                }
            }
            require(search.use_multi_thread && !search_uses_separate_shallow_evaluation(&search), "scope restores main search");
            require(mid_evaluate_for_search(&search) == ordinary, "main leaves restored");
            int lower = -64, upper = 64;
            require(transposition_table.get_bounds(&search, hash, 5, &lower, &upper), "main TT remains accessible");
            require(lower == (item.auxiliary && item.sharing ? 13 : 11), "isolated write preserves ordinary bounds");
            Search child = search;
            child.pass();
            require(child.move_ordering_policy == search.move_ordering_policy, "passing preserves root policy");
            require(std::async(std::launch::async, [child, search]() {
                return child.move_ordering_policy == search.move_ordering_policy;
            }).get(), "worker receives search-local policy");
        }
        // Compare automatic endgame roots with the independently selectable
        // manual configurations used by the experiment, including parallelism.
        Board end = board;
        while (end.n_discs() < 52) {
            uint64_t legal = end.get_legal();
            if (!legal) { end.pass(); legal = end.get_legal(); require(legal != 0, "endgame sample available"); }
            Flip flip; calc_flip(&flip, &end, first_bit(&legal)); end.move_board(&flip);
        }
        for (uint_fast8_t mpc : {uint_fast8_t(MPC_74_LEVEL), uint_fast8_t(MPC_88_LEVEL), uint_fast8_t(MPC_100_LEVEL)}) {
            require(load_eval_move_ordering_mid_file((resources + "/offset0_16.egev2").c_str(), false), "restore Offset bank");
            eval_move_ordering_auto_enabled = true;
            clear_search_tables();
            Search automatic(&end, mpc, false, false);
            bool searching = true;
            const auto actual = first_nega_scout_legal(&automatic, -64, 64, 12, true, {}, end.get_legal(), tim(), &searching);
            eval_move_ordering_auto_enabled = false;
            eval_move_ordering_aux_enabled = false;
            eval_move_ordering_mid_enabled = mpc != MPC_74_LEVEL;
            if (mpc == MPC_100_LEVEL) require(load_eval_move_ordering_mid_file((resources + "/decay_half16.egev2").c_str(), false), "manual Decay bank");
            eval_move_ordering_mid_shallow_max_depth = mpc == MPC_74_LEVEL ? 0 : 5;
            eval_move_ordering_mid_shallow_end_only = true;
            eval_move_ordering_mid_shallow_share_tt = mpc != MPC_100_LEVEL;
            clear_search_tables();
            Search manual(&end, mpc, false, false);
            const auto expected = first_nega_scout_legal(&manual, -64, 64, 12, true, {}, end.get_legal(), tim(), &searching);
            require(actual == expected, "automatic endgame agrees with manual model/TT configuration");
            require(automatic.n_nodes == manual.n_nodes, "automatic endgame visits same nodes as manual configuration");
            require(load_eval_move_ordering_aux_file((resources + "/decay_half16.egev2").c_str(), false), "restore Decay bank");
            require(load_eval_move_ordering_mid_file((resources + "/offset0_16.egev2").c_str(), false), "restore Offset for parallel check");
            eval_move_ordering_auto_enabled = true;
            clear_search_tables();
            Search parallel(&end, mpc, true, false);
            const auto parallel_result = first_nega_scout_legal(&parallel, -64, 64, 12, true, {}, end.get_legal(), tim(), &searching);
            require(parallel_result.first == expected.first, "parallel automatic endgame agrees with serial configuration");
        }
        std::cout << "PASS policy_cases=10 manual_endgame_comparisons=3\n";
#else
        Search search(&board, MPC_100_LEVEL, false, false);
        eval_move_ordering_auto_enabled = true;
        search.configure_move_ordering(40, true);
        require(!search_uses_separate_ordering_evaluation(&search), "disabled builds retain ordinary ordering");
        require(mid_evaluate_for_search(&search) == mid_evaluate_diff(&search), "disabled builds retain ordinary leaves");
        std::cout << "PASS disabled_policy\n";
#endif
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
