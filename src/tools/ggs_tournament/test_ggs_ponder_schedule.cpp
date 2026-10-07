/*
    Egaroucid Project

    Regression tests for the main-search iteration order that Ponder follows
    (ai_ponder_main_schedule). The 3-minute expectations were checked against
    every main search at 36-50 empties in the 2026-10-05/06 GGS search logs.

    Example:
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \
            -DINCLUDE_GGS -DIS_GGS_TOURNAMENT -DIS_GGS_3MIN test_ggs_ponder_schedule.cpp \
            -o test_ggs_ponder_schedule.exe -lws2_32
    Pass --print to list the schedules for 36-60 empties.
*/

#include <iostream>
#include <stdexcept>
#include <string>
#include "../../engine/engine_all.hpp"
#include "../../console/console_all.hpp"

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string iteration_text(const AI_Ponder_Iteration &iteration) {
    return std::string(iteration.is_end_search ? "end" : "mid") + " " + std::to_string(iteration.depth) + "@" + probability_to_str(SELECTIVITY_PERCENTAGE[iteration.mpc_level]);
}

std::string schedule_text(int n_empties) {
    std::string text;
    for (const AI_Ponder_Iteration &iteration: ai_ponder_main_schedule(n_empties)) {
        if (!text.empty()) {
            text += ", ";
        }
        text += iteration_text(iteration);
    }
    return text;
}

// mid 1@100 ... 29@88, shared by every schedule at 37 or more empties.
std::string midgame_prefix_through(int last_depth, bool last_88) {
    std::string text = "mid 1@100, mid 4@100, mid 7@100, mid 10@100, mid 13@100";
    for (int depth = 16; depth <= last_depth; ++depth) {
        text += ", mid " + std::to_string(depth) + "@74";
        if (depth >= 26 && (depth < last_depth || last_88)) {
            text += ", mid " + std::to_string(depth) + "@88";
        }
    }
    return text;
}

std::string verified_midgame(int first_depth, int last_depth) {
    std::string text;
    for (int depth = first_depth; depth <= last_depth; ++depth) {
        text += ", mid " + std::to_string(depth) + "@74, mid " + std::to_string(depth) + "@88, mid " + std::to_string(depth) + "@93";
    }
    return text;
}

std::string endgame(int n_empties, const std::string &last_probability) {
    const std::string probabilities[] = {"74", "88", "93", "98", "99", "99.9", "100"};
    std::string text;
    for (const std::string &probability: probabilities) {
        text += ", end " + std::to_string(n_empties) + "@" + probability;
        if (probability == last_probability) {
            break;
        }
    }
    return text;
}

void require_schedule(int n_empties, const std::string &expected) {
    const std::string actual = schedule_text(n_empties);
    if (actual != expected) {
        throw std::runtime_error(
            "schedule at " + std::to_string(n_empties) + " empties\n  expected: " + expected + "\n  actual:   " + actual
        );
    }
}

void test_3min_schedules() {
#if IS_GGS_3MIN
    require_schedule(50, midgame_prefix_through(29, true) + verified_midgame(30, 41) + ", mid 42@74" + endgame(50, "88"));
    require_schedule(46, midgame_prefix_through(29, true) + verified_midgame(30, 37) + ", mid 38@74" + endgame(46, "93"));
    // 44-38 empties switch to the endgame search after depth 29 at 88%.
    require_schedule(44, midgame_prefix_through(29, true) + endgame(44, "98"));
    require_schedule(40, midgame_prefix_through(29, true) + endgame(40, "99.9"));
    require_schedule(38, midgame_prefix_through(29, true) + endgame(38, "99.9"));
    require_schedule(37, midgame_prefix_through(29, false) + endgame(37, "99.9"));
    require_schedule(36, midgame_prefix_through(28, false) + endgame(36, "99.9"));
#endif
}

void test_schedule_ends_at_the_endgame_limit() {
    for (int n_empties = PONDER_MAIN_SCHEDULE_MIN_N_EMPTY; n_empties <= HW2 - 4; ++n_empties) {
        const std::vector<AI_Ponder_Iteration> schedule = ai_ponder_main_schedule(n_empties);
        const AI_Ponder_Iteration &last = schedule.back();
        require(last.is_end_search && last.depth == n_empties, "schedule ends with an endgame search at " + std::to_string(n_empties));
        require(last.mpc_level == ai_endgame_max_mpc_level(n_empties), "schedule ends at the endgame MPC limit at " + std::to_string(n_empties));
        // ai_ponder treats a child as finished once its endgame MPC cannot rise.
        require(!ai_endgame_mpc_level_allowed(n_empties, last.mpc_level + 1), "no further endgame iteration at " + std::to_string(n_empties));
    }
}

void test_which_children_use_the_schedule() {
    // 14 discs: 50 empties, black to move.
    const Board early = convert_board_from_str("------------------OOXX----XOX-----XOOX----XXX------------------- X").first;
    require(ai_ponder_uses_main_schedule(early) == PONDER_USE_MAIN_SCHEDULE, "50-empty children follow the switch");
    // 62 discs: 2 empties, never on the schedule.
    const Board late = convert_board_from_str("XXXXXXXX-XXOOOXOOOXOXXOOOOOXOOXOOXOOOOXOOOXOOOXOO-XXXXOOOOOOOOOO X").first;
    require(!ai_ponder_uses_main_schedule(late), "late children keep the level order");
#if IS_GGS_3MIN
    require(PONDER_USE_MAIN_SCHEDULE == (EGAROUCID_PONDER_MAIN_SCHEDULE == 1), "3-minute builds follow the macro");
#elif !IS_GGS_TOURNAMENT
    require(!PONDER_USE_MAIN_SCHEDULE, "non-tournament builds keep the level order");
#endif
}

}

int main(int argc, char **argv) {
    try {
        bit_init();
        mobility_init();
        flip_init();
        if (argc >= 2 && std::string(argv[1]) == "--print") {
            for (int n_empties = PONDER_MAIN_SCHEDULE_MIN_N_EMPTY; n_empties <= HW2 - 4; ++n_empties) {
                int stage = 0;
                for (const AI_Ponder_Iteration &iteration: ai_ponder_main_schedule(n_empties)) {
                    std::cout << n_empties << " " << ++stage << " " << iteration_text(iteration) << std::endl;
                }
            }
            return 0;
        }
        test_3min_schedules();
        test_schedule_ends_at_the_endgame_limit();
        test_which_children_use_the_schedule();
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "PASS" << std::endl;
    return 0;
}
