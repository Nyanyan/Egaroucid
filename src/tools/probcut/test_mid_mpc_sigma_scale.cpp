/* Regression tests for phase-specific MPC probabilities and z thresholds. */

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../engine/engine_all.hpp"

namespace {

constexpr double LEGACY_BASE_Z[N_SELECTIVITY_LEVEL] = {
    1.13, 1.55, 1.81, 2.32, 2.57, 3.29, 9.99
};
constexpr double LEGACY_MID_SCALE[N_SELECTIVITY_LEVEL] = {
    1.0, 1.0, 1.0, 0.90, 0.85, 0.85, 1.0
};

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(double actual, double expected, double tolerance, const char *message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            std::string(message) + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
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

void test_error_helpers_preserve_legacy_behavior() {
    for (int n_discs = 0; n_discs <= HW2; ++n_discs) {
        for (int depth = 0; depth < HW2 - 3; ++depth) {
            for (int level = 0; level < N_SELECTIVITY_LEVEL; ++level) {
                const int legacy_end_static = std::ceil(
                    MPC_ERROR_SCALE * LEGACY_BASE_Z[level] *
                    probcut_sigma_end(n_discs, 0)
                );
                const int legacy_mid_static = std::ceil(
                    MPC_ERROR_SCALE * LEGACY_BASE_Z[level] *
                    LEGACY_MID_SCALE[level] *
                    probcut_sigma(n_discs, 0, depth)
                );
                if (mpc_static_error<true>(level, n_discs, depth) != legacy_end_static) {
                    fail_table_value(
                        "end static helper", level, n_discs, 0, depth,
                        mpc_static_error<true>(level, n_discs, depth),
                        legacy_end_static
                    );
                }
                if (mpc_static_error<false>(level, n_discs, depth) != legacy_mid_static) {
                    fail_table_value(
                        "mid static helper", level, n_discs, 0, depth,
                        mpc_static_error<false>(level, n_discs, depth),
                        legacy_mid_static
                    );
                }

                for (int search_depth = 0; search_depth < HW2 - 3; ++search_depth) {
                    int actual_end_search = -1;
                    int actual_end_eval = -1;
                    int actual_mid_search = -1;
                    int actual_mid_eval = -1;
                    mpc_search_errors<true>(
                        level, n_discs, search_depth, depth,
                        &actual_end_search, &actual_end_eval
                    );
                    mpc_search_errors<false>(
                        level, n_discs, search_depth, depth,
                        &actual_mid_search, &actual_mid_eval
                    );

                    const double end_sigma_search =
                        probcut_sigma_end(n_discs, search_depth);
                    const double end_sigma_zero = probcut_sigma_end(n_discs, 0);
                    const double raw_mid_sigma_search =
                        probcut_sigma(n_discs, search_depth, depth);
                    const double mid_sigma_search = LEGACY_MID_SCALE[level] *
                        raw_mid_sigma_search;
                    const double mid_sigma_zero = LEGACY_MID_SCALE[level] *
                        probcut_sigma(n_discs, 0, depth);
                    const int legacy_end_search = std::ceil(
                        MPC_ERROR_SCALE * LEGACY_BASE_Z[level] * end_sigma_search
                    );
#if USE_MPC_PRE_CALCULATION
                    const int legacy_mid_search = std::ceil(
                        MPC_ERROR_SCALE * LEGACY_BASE_Z[level] *
                        LEGACY_MID_SCALE[level] * raw_mid_sigma_search
                    );
                    const int legacy_end_zero = std::ceil(
                        MPC_ERROR_SCALE * LEGACY_BASE_Z[level] * end_sigma_zero
                    );
                    const int legacy_mid_zero = std::ceil(
                        MPC_ERROR_SCALE * LEGACY_BASE_Z[level] *
                        LEGACY_MID_SCALE[level] *
                        probcut_sigma(n_discs, 0, depth)
                    );
                    const int legacy_end_eval =
                        (legacy_end_zero + legacy_end_search + 1) / 2;
                    const int legacy_mid_eval =
                        (legacy_mid_zero + legacy_mid_search + 1) / 2;
#else
                    const int legacy_mid_search = std::ceil(
                        MPC_ERROR_SCALE * LEGACY_BASE_Z[level] * mid_sigma_search
                    );
                    const int legacy_end_eval = std::ceil(
                        MPC_ERROR_SCALE * LEGACY_BASE_Z[level] * 0.5 *
                        (end_sigma_zero + end_sigma_search)
                    );
                    const int legacy_mid_eval = std::ceil(
                        MPC_ERROR_SCALE * LEGACY_BASE_Z[level] * 0.5 *
                        (mid_sigma_zero + mid_sigma_search)
                    );
#endif
                    if (actual_end_search != legacy_end_search) {
                        fail_table_value(
                            "end search helper", level, n_discs, search_depth,
                            depth, actual_end_search, legacy_end_search
                        );
                    }
                    if (actual_mid_search != legacy_mid_search) {
                        fail_table_value(
                            "mid search helper", level, n_discs, search_depth,
                            depth, actual_mid_search, legacy_mid_search
                        );
                    }
                    if (actual_end_eval != legacy_end_eval) {
                        fail_table_value(
                            "end eval helper", level, n_discs, search_depth,
                            depth, actual_end_eval, legacy_end_eval
                        );
                    }
                    if (actual_mid_eval != legacy_mid_eval) {
                        fail_table_value(
                            "mid eval helper", level, n_discs, search_depth,
                            depth, actual_mid_eval, legacy_mid_eval
                        );
                    }
                }
            }
        }
    }
}

} // namespace

int main() {
    for (int level = 0; level < N_SELECTIVITY_LEVEL; ++level) {
        require_near(
            MPC_SELECTIVITY_Z_MID[level],
            LEGACY_BASE_Z[level] * LEGACY_MID_SCALE[level],
            0.0,
            "midgame z preserves the measured search threshold"
        );
        require_near(
            MPC_SELECTIVITY_Z_END[level],
            LEGACY_BASE_Z[level],
            0.0,
            "endgame z preserves the original search threshold"
        );
        require_near(
            SELECTIVITY_MPCT[level], MPC_SELECTIVITY_Z_END[level], 0.0,
            "legacy z table name remains an endgame alias"
        );
        require_near(
            mpc_selectivity_z(level, false),
            MPC_SELECTIVITY_Z_MID[level],
            0.0,
            "midgame z selector"
        );
        require_near(
            mpc_selectivity_z(level, true),
            MPC_SELECTIVITY_Z_END[level],
            0.0,
            "endgame z selector"
        );

        const double expected_mid_probability = 100.0 * std::erf(
            MPC_SELECTIVITY_Z_MID[level] / std::sqrt(2.0)
        );
        const double expected_end_probability = 100.0 * std::erf(
            MPC_SELECTIVITY_Z_END[level] / std::sqrt(2.0)
        );
        require_near(
            MPC_SELECTIVITY_PERCENTAGE_MID[level],
            expected_mid_probability,
            1.0e-12,
            "midgame probability matches its z threshold"
        );
        require_near(
            MPC_SELECTIVITY_PERCENTAGE_END[level],
            expected_end_probability,
            1.0e-12,
            "endgame probability matches its z threshold"
        );
        require_near(
            mpc_selectivity_percentage(level, false),
            MPC_SELECTIVITY_PERCENTAGE_MID[level],
            0.0,
            "midgame probability selector"
        );
        require_near(
            mpc_selectivity_percentage(level, true),
            MPC_SELECTIVITY_PERCENTAGE_END[level],
            0.0,
            "endgame probability selector"
        );
    }

    require(
        MPC_SELECTIVITY_PERCENTAGE_MID[MPC_98_LEVEL] <
            MPC_SELECTIVITY_PERCENTAGE_END[MPC_98_LEVEL],
        "high-selectivity midgame and endgame probabilities are distinct"
    );
    constexpr double legacy_labels[N_SELECTIVITY_LEVEL] = {
        74, 88, 93, 98, 99, 99.9, 100
    };
    for (int level = 0; level < N_SELECTIVITY_LEVEL; ++level) {
        require_near(
            SELECTIVITY_PERCENTAGE[level], legacy_labels[level], 0.0,
            "legacy selectivity labels stay unchanged"
        );
    }

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
                    MPC_ERROR_SCALE * LEGACY_BASE_Z[level] * end_sigma
                );
                const int actual_end = probcut_error_end(level, end_sigma);
                if (actual_end != expected_end) {
                    fail_table_value(
                        "end helper", level, n_discs, shallow_depth, -1,
                        actual_end, expected_end
                    );
                }
                require(
                    probcut_error(level, end_sigma) == actual_end,
                    "legacy end error helper remains compatible"
                );
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
                        MPC_ERROR_SCALE * LEGACY_BASE_Z[level] *
                        LEGACY_MID_SCALE[level] * mid_sigma
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
                            actual_mid == probcut_error_end(level, mid_sigma),
                            "100% mid margin stays unchanged"
                        );
                    }
                }
            }
        }
    }

    test_error_helpers_preserve_legacy_behavior();

    require(initialize_engine(), "engine initialization");
    test_scale_boundary_position();
    test_high_selectivity_search_paths();

    std::cout << "Phase-specific MPC probability tests passed\n";
    return 0;
}
