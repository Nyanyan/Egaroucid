/*
    Egaroucid Project

    Focused regression tests for GGS tournament time allocation.

    Example:
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \
            -DIS_GGS_TOURNAMENT test_ggs_time_management.cpp \
            -o test_ggs_time_management.exe
*/

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
// ai.hpp includes time_management.hpp after declaring the search entry points;
// include it in this order because the two engine headers are mutually linked.
#include "../../engine/ai.hpp"

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_equal(uint64_t actual, uint64_t expected, const std::string &message) {
    if (actual != expected) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void require_near(double actual, double expected, const std::string &message) {
    if (std::abs(actual - expected) > 1.0e-9) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void test_pair_boost_phase_scale() {
    require_near(time_management_ggs_pair_boost_phase_scale(14), 0.60, "disc 14 scale");
    require_near(time_management_ggs_pair_boost_phase_scale(16), 0.60, "disc 16 scale");
    require_near(time_management_ggs_pair_boost_phase_scale(17), 0.65, "disc 17 scale");
    require_near(time_management_ggs_pair_boost_phase_scale(20), 0.80, "disc 20 scale");
    require_near(time_management_ggs_pair_boost_phase_scale(23), 0.95, "disc 23 scale");
    require_near(time_management_ggs_pair_boost_phase_scale(24), 1.00, "disc 24 scale");
    require_near(time_management_ggs_pair_boost_phase_scale(40), 1.00, "late scale");
}

void test_extra_time_is_not_budgeted_for_normal_search() {
    require_near(
        TIME_MANAGEMENT_GGS_REMAINING_MOVES_EXTRA,
        0.0,
        "GGS extra time must not be added to the normal-search budget"
    );
}

void test_cap_is_continuous_at_reserve() {
    constexpr double remaining_moves = 17.0;
    constexpr uint64_t reserve = 17400ULL;
    constexpr uint64_t requested = 10000ULL;

    const uint64_t below = time_management_ggs_cap_time_limit(
        requested, reserve - 1ULL, remaining_moves
    );
    const uint64_t at = time_management_ggs_cap_time_limit(
        requested, reserve, remaining_moves
    );
    const uint64_t above = time_management_ggs_cap_time_limit(
        requested, reserve + 1ULL, remaining_moves
    );
    require(at >= below, "cap decreased at the reserve boundary");
    require(above >= at, "cap decreased immediately above the reserve boundary");
    require(above - below <= 2ULL, "reserve boundary must be continuous");
    require_equal(at, 4350ULL, "reserve boundary retains the low-time cap");

    uint64_t previous = time_management_ggs_cap_time_limit(
        requested, 10000ULL, remaining_moves
    );
    for (uint64_t remaining = 10001ULL; remaining <= 60000ULL; ++remaining) {
        const uint64_t allocation = time_management_ggs_cap_time_limit(
            requested, remaining, remaining_moves
        );
        require(allocation >= previous, "cap must be monotonic in remaining time");
        require(allocation <= requested, "cap exceeded the requested time");
        previous = allocation;
    }
}

void test_early_endgame_ramp() {
    constexpr uint64_t selected = 1482ULL;
    constexpr double ramp_start = 20000.0;
    constexpr double full = 32000.0;
    constexpr double leave = 12000.0;
    constexpr double maximum = 12000.0;

    require_equal(
        time_management_ggs_ramped_endgame_force_time(
            selected, 20000ULL, ramp_start, full, leave, maximum
        ),
        selected,
        "ramp start keeps ordinary allocation"
    );
    require_equal(
        time_management_ggs_ramped_endgame_force_time(
            selected, 24500ULL, ramp_start, full, leave, maximum
        ),
        4687ULL,
        "24.5 second regression allocation"
    );
    require_equal(
        time_management_ggs_ramped_endgame_force_time(
            selected, 32000ULL, ramp_start, full, leave, maximum
        ),
        12000ULL,
        "full ramp reaches the existing force cap"
    );
    require_equal(
        time_management_ggs_ramped_endgame_force_time(
            13000ULL, 24500ULL, ramp_start, full, leave, maximum
        ),
        13000ULL,
        "ramp never reduces a larger ordinary allocation"
    );

    const uint64_t just_below_25 = time_management_ggs_ramped_endgame_force_time(
        selected, 24999ULL, ramp_start, full, leave, maximum
    );
    const uint64_t at_25 = time_management_ggs_ramped_endgame_force_time(
        selected, 25000ULL, ramp_start, full, leave, maximum
    );
    require(at_25 >= just_below_25, "ramp must be monotonic at 25 seconds");
    require(at_25 - just_below_25 <= 2ULL, "25 second boundary must be continuous");

    for (uint64_t remaining = 20001ULL; remaining <= 50000ULL; remaining += 137ULL) {
        const uint64_t allocation = time_management_ggs_ramped_endgame_force_time(
            selected, remaining, ramp_start, full, leave, maximum
        );
        require(allocation <= maximum, "early ramp exceeded force cap");
        require(allocation <= remaining - (uint64_t)leave, "early ramp consumed safety reserve");
    }
}

void test_late_endgame_ramp() {
    const uint64_t allocation = time_management_ggs_ramped_endgame_force_time(
        1482ULL,
        24500ULL,
        20000.0,
        32000.0,
        9000.0,
        20000.0
    );
    require_equal(allocation, 5812ULL, "late ramp at 24.5 seconds");
    require(allocation <= 24500ULL - 9000ULL, "late ramp consumed safety reserve");
}

void test_match_boundary_revalidation_gate() {
    Board board;
    const std::string board_text =
        std::string(11, 'X') + std::string(10, 'O') + std::string(43, '-') + " X";
    require(board.from_str(board_text), "failed to construct 43-empty gate board");

    AI_Time_Limit_Match_Context context;
    context.has_pair_result = true;
    context.pair_value = -16;
    context.real_remaining_time_msec = 25000ULL;
    require(ai_tl_ggs_match_revalidation_gate(board, 700ULL, &context), "lower time gate");
    require(ai_tl_ggs_match_revalidation_gate(board, 1800ULL, &context), "upper time gate");
    require(!ai_tl_ggs_match_revalidation_gate(board, 699ULL, &context), "below time gate");
    require(!ai_tl_ggs_match_revalidation_gate(board, 1801ULL, &context), "above time gate");
    context.real_remaining_time_msec = 24999ULL;
    require(!ai_tl_ggs_match_revalidation_gate(board, 1000ULL, &context), "remaining-time gate");
    require(!ai_tl_ggs_match_revalidation_gate(board, 1000ULL, nullptr), "null context gate");
}

void test_match_boundary_classification() {
    require(ai_tl_ggs_crosses_match_boundary(-16, 14, 16), "loss-to-draw crossing");
    require(ai_tl_ggs_crosses_match_boundary(-16, 16, 17), "draw-to-win crossing");
    require(!ai_tl_ggs_crosses_match_boundary(-16, 14, 15), "same-outcome candidates");
    require(
        ai_tl_ggs_match_revalidation_should_switch(-16, 15, 16) ==
            (AI_TL_GGS_MATCH_REVALIDATE_SWITCH_MARGIN == 1),
        "one-disc loss-to-draw switch must follow the configured margin"
    );
    require(
        ai_tl_ggs_match_revalidation_should_switch(-16, 14, 16),
        "two-disc loss-to-draw candidate must switch"
    );
    require(ai_tl_ggs_match_boundary_precheck(-16, 14), "near-boundary precheck");
    require(!ai_tl_ggs_match_boundary_precheck(-16, 10), "far-boundary precheck");
}

void test_match_boundary_reserve_trigger() {
    AI_TL_Iteration_Diagnostics diagnostics;
    diagnostics.enable_match_revalidation = true;
    diagnostics.pair_value = -16;
    ai_tl_record_recent_policy(&diagnostics, 1);
    ai_tl_record_recent_policy(&diagnostics, 2);
    ai_tl_record_recent_policy(&diagnostics, 1);
    ai_tl_record_recent_policy(&diagnostics, 2);

    Search_result result;
    result.policy = 2;
    result.value = 14;
    result.depth = 26;
    result.probability = 74;
    require(
        ai_tl_ggs_should_hold_match_reserve(&diagnostics, result, 26, 88),
        "oscillating/incomplete boundary reserve"
    );
    require(diagnostics.policy_oscillation, "policy oscillation was not recorded");
    require(diagnostics.next_selectivity_incomplete, "incomplete selectivity was not recorded");

    AI_TL_Iteration_Diagnostics stable;
    stable.enable_match_revalidation = true;
    stable.pair_value = -16;
    for (int i = 0; i < 4; ++i) {
        ai_tl_record_recent_policy(&stable, 2);
    }
    require(
        !ai_tl_ggs_should_hold_match_reserve(&stable, result, 27, 74),
        "stable next-depth search should release reserve"
    );
}

void test_policy_verify_timeout_fallback() {
    require(
        ai_tl_ggs_should_use_main_on_verify_timeout(
            true, false, 74, 74, false, false, SCORE_UNDEFINED, SCORE_UNDEFINED
        ),
        "first endgame iteration should replace a midgame result"
    );
    require(
        ai_tl_ggs_should_use_main_on_verify_timeout(
            true, true, 88, 74, false, false, SCORE_UNDEFINED, SCORE_UNDEFINED
        ),
        "higher-selectivity endgame iteration should replace the previous result"
    );
    require(
        ai_tl_ggs_should_use_main_on_verify_timeout(
            true, true, 88, 88, true, false, SCORE_UNDEFINED, -6
        ),
        "completed new full-window search should be usable"
    );
    require(
        !ai_tl_ggs_should_use_main_on_verify_timeout(
            true, true, 88, 88, true, true, -5, -6
        ),
        "previous fail-high must prevent the new-move fallback"
    );
    require(
        !ai_tl_ggs_should_use_main_on_verify_timeout(
            false, false, 88, 74, true, false, SCORE_UNDEFINED, -6
        ),
        "midgame timeout must not use the endgame fallback"
    );

    require(
        ai_tl_ggs_can_keep_previous_on_verify_timeout(8000ULL, 0, 1, 35),
        "short-time fallback should not depend on evaluation sign"
    );
    require(
        ai_tl_ggs_can_keep_previous_on_verify_timeout(20000ULL, 88, 34, 35),
        "strong adjacent iteration may be kept"
    );
    require(
        !ai_tl_ggs_can_keep_previous_on_verify_timeout(20000ULL, 74, 27, 35),
        "weak stale iteration must not be kept"
    );
}

void test_end_boundary_reserve() {
    require_equal(
        ai_time_limit_ggs_end_boundary_reserve_time(35, true, 5000ULL, 31000ULL),
        9500ULL,
        "ambiguous 35-empty boundary gets a ramped reserve"
    );
    require_equal(
        ai_time_limit_ggs_end_boundary_reserve_time(35, false, 5000ULL, 40000ULL),
        5000ULL,
        "quiet boundary does not get a reserve"
    );
    require_equal(
        ai_time_limit_ggs_end_boundary_reserve_time(39, true, 5000ULL, 40000ULL),
        5000ULL,
        "reserve is limited to the end-search boundary"
    );
    require_equal(
        ai_time_limit_ggs_end_boundary_reserve_time(35, true, 10000ULL, 25000ULL),
        10000ULL,
        "reserve never reduces the existing allocation"
    );
}

void test_candidate_stage2_selection() {
    std::vector<Ponder_elem> moves(6);
    const double values[] = {4.0, 2.0, 0.5, -0.1, -3.0, -8.0};
    for (int i = 0; i < 6; ++i) {
        moves[i].value = values[i];
        moves[i].count = 1;
    }
    require_equal(
        (uint64_t)ai_get_values_stage2_candidate_count(moves),
        3ULL,
        "stage two keeps the top two and all moves within four discs"
    );
    moves[3].value = 0.0;
    require_equal(
        (uint64_t)ai_get_values_stage2_candidate_count(moves),
        4ULL,
        "stage two accepts at most four close candidates"
    );
    moves[1].value = -10.0;
    moves[2].value = -11.0;
    moves[3].value = -12.0;
    moves[4].value = -13.0;
    moves[5].value = -14.0;
    require_equal(
        (uint64_t)ai_get_values_stage2_candidate_count(moves),
        2ULL,
        "stage two always rechecks at least the top two"
    );
}

} // namespace

int main() {
    try {
        test_cap_is_continuous_at_reserve();
        test_pair_boost_phase_scale();
        test_extra_time_is_not_budgeted_for_normal_search();
        test_early_endgame_ramp();
        test_late_endgame_ramp();
        test_match_boundary_revalidation_gate();
        test_match_boundary_classification();
        test_match_boundary_reserve_trigger();
        test_policy_verify_timeout_fallback();
        test_end_boundary_reserve();
        test_candidate_stage2_selection();
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "GGS time-management tests passed" << std::endl;
    return 0;
}
