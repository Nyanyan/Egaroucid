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

} // namespace

int main() {
    try {
        test_cap_is_continuous_at_reserve();
        test_pair_boost_phase_scale();
        test_early_endgame_ramp();
        test_late_endgame_ramp();
        test_match_boundary_revalidation_gate();
        test_match_boundary_classification();
        test_match_boundary_reserve_trigger();
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "GGS time-management tests passed" << std::endl;
    return 0;
}
