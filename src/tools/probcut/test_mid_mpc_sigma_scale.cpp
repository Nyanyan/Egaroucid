/* Regression tests for the selectivity-specific midgame MPC sigma scale. */

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../engine/engine_all.hpp"

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[noreturn]] void fail_table_value(
    const char *table,
    int level,
    int n_discs,
    int depth1,
    int depth2,
    int actual,
    int expected
) {
    throw std::runtime_error(
        std::string(table) + " mismatch at level=" + std::to_string(level) +
        " n_discs=" + std::to_string(n_discs) +
        " depth1=" + std::to_string(depth1) +
        " depth2=" + std::to_string(depth2) +
        ": expected " + std::to_string(expected) +
        ", got " + std::to_string(actual)
    );
}

bool initialize_engine() {
    thread_pool.resize(0);
    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    endsearch_init();
#if USE_MPC_PRE_CALCULATION
    mpc_init();
#endif
    move_ordering_init();
    if (!hash_resize(DEFAULT_HASH_LEVEL, 20, "./", false)) {
        return false;
    }
    stability_init();
    return evaluate_init(
        "./resources/eval.egev2",
        "./resources/eval_move_ordering_end.egev",
        false
    );
}

void test_scale_boundary_position() {
    // Validation position 137 guards the low-selectivity safety boundary:
    // scale 0.98 selects g5 and loses three discs at the depth-16
    // 100%-selectivity reference. Production keeps the original 1.0 margin.
    Board board;
    require(
        board.from_str(
            "----------------XXXXX----XXOOO--OXXOOO--OOOOOO--O-OO------O----- X"
        ),
        "boundary board parses"
    );
    require(HW2 - board.n_discs() == 38, "boundary board has 38 empty squares");

    transposition_table.init();
    global_searching = true;
    bool searching = true;
    Search search(&board, MPC_74_LEVEL, false, false);
    const auto result = first_nega_scout_legal(
        &search,
        -SCORE_MAX,
        SCORE_MAX,
        16,
        false,
        std::vector<Clog_result>(),
        board.get_legal(),
        tim(),
        &searching
    );
    require(searching && global_searching, "boundary search completes");
    require(idx_to_coord(result.second) == "f7", "boundary search keeps f7");
    require(result.first == 5, "boundary search keeps the depth-16 value");
}

void test_high_selectivity_search_paths() {
    Board board;
    require(
        board.from_str(
            "--OXXO--OXXXXO---OXXOOO--OOXXOOO-OOOXX----OX-------------------- X"
        ),
        "high-selectivity board parses"
    );
    require(
        HW2 - board.n_discs() == 34,
        "high-selectivity board has 34 empty squares"
    );

    constexpr uint_fast8_t levels[] = {
        MPC_98_LEVEL,
        MPC_99_LEVEL,
        MPC_999_LEVEL
    };
    for (const uint_fast8_t level : levels) {
        transposition_table.init();
        global_searching = true;
        bool searching = true;
        Search search(&board, level, false, false);
        const auto result = first_nega_scout_legal(
            &search,
            -SCORE_MAX,
            SCORE_MAX,
            16,
            false,
            std::vector<Clog_result>(),
            board.get_legal(),
            tim(),
            &searching
        );
        require(searching && global_searching, "high-selectivity search completes");
        require(idx_to_coord(result.second) == "g5", "high-selectivity best move");
        require(result.first == 10, "high-selectivity depth-16 value");
    }
}

} // namespace

int main() {
    require(MID_MPC_SIGMA_SCALE[MPC_74_LEVEL] == 1.0, "74% scale");
    require(MID_MPC_SIGMA_SCALE[MPC_88_LEVEL] == 1.0, "88% scale");
    require(MID_MPC_SIGMA_SCALE[MPC_93_LEVEL] == 1.0, "93% scale");
    require(MID_MPC_SIGMA_SCALE[MPC_98_LEVEL] == 0.90, "98% scale");
    require(MID_MPC_SIGMA_SCALE[MPC_99_LEVEL] == 0.85, "99% scale");
    require(MID_MPC_SIGMA_SCALE[MPC_999_LEVEL] == 0.85, "99.9% scale");
    require(MID_MPC_SIGMA_SCALE[MPC_100_LEVEL] == 1.0, "100% scale");

#if USE_MPC_PRE_CALCULATION
    mpc_init();
#endif

    for (int n_discs = 0; n_discs <= HW2; ++n_discs) {
        for (int shallow_depth = 0; shallow_depth < HW2 - 3; ++shallow_depth) {
            const double end_sigma = probcut_sigma_end(n_discs, shallow_depth);
            require(end_sigma > 0.0, "end sigma stays positive");
            int previous_end_error = -1;
            for (int level = 0; level < N_SELECTIVITY_LEVEL; ++level) {
                const int expected_end = std::ceil(
                    MPC_ERROR_SCALE * SELECTIVITY_MPCT[level] * end_sigma
                );
                const int actual_end = probcut_error(level, end_sigma);
                if (actual_end != expected_end) {
                    fail_table_value(
                        "end helper", level, n_discs, shallow_depth, -1,
                        actual_end, expected_end
                    );
                }
                require(actual_end >= previous_end_error, "end margins are monotone");
                previous_end_error = actual_end;
#if USE_MPC_PRE_CALCULATION
                if (mpc_error_end[level][n_discs][shallow_depth] != expected_end) {
                    fail_table_value(
                        "precomputed end", level, n_discs, shallow_depth, -1,
                        mpc_error_end[level][n_discs][shallow_depth], expected_end
                    );
                }
#endif
            }

            for (int deep_depth = 0; deep_depth < HW2 - 3; ++deep_depth) {
                const double mid_sigma = probcut_sigma(
                    n_discs, shallow_depth, deep_depth
                );
                require(mid_sigma > 0.0, "mid sigma stays positive");
                int previous_mid_error = -1;
                for (int level = 0; level < N_SELECTIVITY_LEVEL; ++level) {
                    const int expected_mid = std::ceil(
                        MPC_ERROR_SCALE * SELECTIVITY_MPCT[level] *
                        MID_MPC_SIGMA_SCALE[level] * mid_sigma
                    );
                    const int actual_mid = probcut_error_mid(level, mid_sigma);
                    if (actual_mid != expected_mid) {
                        fail_table_value(
                            "mid helper", level, n_discs, shallow_depth,
                            deep_depth, actual_mid, expected_mid
                        );
                    }
                    require(actual_mid >= previous_mid_error, "mid margins are monotone");
                    previous_mid_error = actual_mid;
#if USE_MPC_PRE_CALCULATION
                    if (
                        mpc_error[level][n_discs][shallow_depth][deep_depth] !=
                        expected_mid
                    ) {
                        fail_table_value(
                            "precomputed mid", level, n_discs, shallow_depth,
                            deep_depth,
                            mpc_error[level][n_discs][shallow_depth][deep_depth],
                            expected_mid
                        );
                    }
#endif
                    if (level == MPC_100_LEVEL) {
                        require(
                            actual_mid == probcut_error(level, mid_sigma),
                            "100% mid margin stays unchanged"
                        );
                    }
                }
            }
        }
    }

    require(initialize_engine(), "engine initialization");
    test_scale_boundary_position();
    test_high_selectivity_search_paths();

    std::cout << "Midgame MPC sigma scale tests passed\n";
    return 0;
}
