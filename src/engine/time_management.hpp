/*
    Egaroucid Project

    @file time_management.hpp
        Time management system
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "ai.hpp"

#if IS_GGS_TOURNAMENT
constexpr int TIME_MANAGEMENT_INITIAL_N_EMPTIES = 50; // 64 - 14 (s8r14)
#endif

#if IS_GGS_TOURNAMENT
    #define TIME_MANAGEMENT_REMAINING_TIME_OFFSET 180 // ms / move
    #define TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE 2200 // ms
    #ifndef TIME_MANAGEMENT_GGS_FIRST_MOVE_COE
        #define TIME_MANAGEMENT_GGS_FIRST_MOVE_COE 0.14
    #endif
    #ifndef TIME_MANAGEMENT_GGS_FIRST_REPLY_MOVE_COE
        #define TIME_MANAGEMENT_GGS_FIRST_REPLY_MOVE_COE 0.14
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PHASE_TIME_COE_45_OR_MORE
        #define TIME_MANAGEMENT_GGS_PHASE_TIME_COE_45_OR_MORE 2.05
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PHASE_TIME_COE_34_OR_MORE
        #define TIME_MANAGEMENT_GGS_PHASE_TIME_COE_34_OR_MORE 2.25
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PHASE_TIME_COE_22_OR_MORE
        #define TIME_MANAGEMENT_GGS_PHASE_TIME_COE_22_OR_MORE 2.35
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PHASE_TIME_COE_LATE
        #define TIME_MANAGEMENT_GGS_PHASE_TIME_COE_LATE 2.55
    #endif
    #ifndef TIME_MANAGEMENT_GGS_REMAINING_MOVES_EXTRA
        #define TIME_MANAGEMENT_GGS_REMAINING_MOVES_EXTRA 0.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_COMPLETE_DEPTH_TIME_COE
        #define TIME_MANAGEMENT_GGS_COMPLETE_DEPTH_TIME_COE 0.60
    #endif
    #ifndef TIME_MANAGEMENT_GGS_COMPLETE_USE_TIME_COE
        #define TIME_MANAGEMENT_GGS_COMPLETE_USE_TIME_COE 0.72
    #endif
    #ifndef TIME_MANAGEMENT_GGS_COMPLETE_DEPTH_CAP_32_OR_MORE
        #define TIME_MANAGEMENT_GGS_COMPLETE_DEPTH_CAP_32_OR_MORE 22000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_COMPLETE_USE_CAP_32_OR_MORE
        #define TIME_MANAGEMENT_GGS_COMPLETE_USE_CAP_32_OR_MORE 28000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_COMPLETE_DEPTH_CAP_30_OR_MORE
        #define TIME_MANAGEMENT_GGS_COMPLETE_DEPTH_CAP_30_OR_MORE 15000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_COMPLETE_USE_CAP_30_OR_MORE
        #define TIME_MANAGEMENT_GGS_COMPLETE_USE_CAP_30_OR_MORE 30000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_ENDGAME_DEPTH_TIME_COE
        #define TIME_MANAGEMENT_GGS_ENDGAME_DEPTH_TIME_COE 0.10
    #endif
    #ifndef TIME_MANAGEMENT_GGS_ENDGAME_USE_TIME_COE
        #define TIME_MANAGEMENT_GGS_ENDGAME_USE_TIME_COE 0.19
    #endif
    #ifndef TIME_MANAGEMENT_GGS_TARGET_SAFE_REMAINING_MSEC
        #define TIME_MANAGEMENT_GGS_TARGET_SAFE_REMAINING_MSEC 3000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_SHORTAGE_SOFT_MSEC_PER_MOVE
        #define TIME_MANAGEMENT_GGS_SHORTAGE_SOFT_MSEC_PER_MOVE 2600.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_SHORTAGE_HARD_MSEC_PER_MOVE
        #define TIME_MANAGEMENT_GGS_SHORTAGE_HARD_MSEC_PER_MOVE 900.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_SHORTAGE_MIN_SCALE
        #define TIME_MANAGEMENT_GGS_SHORTAGE_MIN_SCALE 0.35
    #endif
    #ifndef TIME_MANAGEMENT_GGS_MAX_RESERVE_MSEC_PER_MOVE
        #define TIME_MANAGEMENT_GGS_MAX_RESERVE_MSEC_PER_MOVE 700.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_MAX_SURPLUS_USE_COE
        #define TIME_MANAGEMENT_GGS_MAX_SURPLUS_USE_COE 0.88
    #endif
    #ifndef TIME_MANAGEMENT_GGS_LOW_TIME_MAX_USE_COE
        #define TIME_MANAGEMENT_GGS_LOW_TIME_MAX_USE_COE 0.25
    #endif
    #ifndef TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_MIN_N_EMPTY
        #define TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_MIN_N_EMPTY 32
    #endif
    #ifndef TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_MAX_N_EMPTY
        #define TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_MAX_N_EMPTY 36
    #endif
    #ifndef TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_RAMP_START_REMAINING
        #define TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_RAMP_START_REMAINING 20000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_FULL_REMAINING
        #define TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_FULL_REMAINING 32000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_LEAVE_MSEC
        #define TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_LEAVE_MSEC 12000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_MAX_TIME
        #define TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_MAX_TIME 12000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_MIN_N_EMPTY
        #define TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_MIN_N_EMPTY 30
    #endif
    #ifndef TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_MAX_N_EMPTY
        #define TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_MAX_N_EMPTY 31
    #endif
    #ifndef TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_RAMP_START_REMAINING
        #define TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_RAMP_START_REMAINING 20000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_FULL_REMAINING
        #define TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_FULL_REMAINING 32000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_LEAVE_MSEC
        #define TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_LEAVE_MSEC 9000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_MAX_TIME
        #define TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_MAX_TIME 20000.0
    #endif
    // Phase plan for s8r14 games. Each own move before the late endgame gets a
    // share of the distributable clock proportional to its phase weight. The
    // weights are the intended search time per move (ms) in a 1:00 game.
    // Without trust mode GGS charged about 0.4 s of network lag per own move
    // beyond the search. Trust mode charges the client-measured time instead,
    // so the overhead is only local processing. In the 2026-09-26 tournament
    // the own moves at 28 empties or less searched at most 1.9 s in total.
    #ifndef TIME_MANAGEMENT_GGS_PLAN_ENABLED
        #define TIME_MANAGEMENT_GGS_PLAN_ENABLED true
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PLAN_MOVE_OVERHEAD_MSEC
        #define TIME_MANAGEMENT_GGS_PLAN_MOVE_OVERHEAD_MSEC 100.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PLAN_LATE_RESERVE_MSEC
        #define TIME_MANAGEMENT_GGS_PLAN_LATE_RESERVE_MSEC 3500.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PLAN_MIDGAME_MIN_N_EMPTY
        #define TIME_MANAGEMENT_GGS_PLAN_MIDGAME_MIN_N_EMPTY 37
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PLAN_MIDGAME_MAX_N_EMPTY
        #define TIME_MANAGEMENT_GGS_PLAN_MIDGAME_MAX_N_EMPTY 46
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PLAN_EARLY_ENDGAME_MIN_N_EMPTY
        #define TIME_MANAGEMENT_GGS_PLAN_EARLY_ENDGAME_MIN_N_EMPTY 35
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PLAN_PRE_COMPLETE_MIN_N_EMPTY
        #define TIME_MANAGEMENT_GGS_PLAN_PRE_COMPLETE_MIN_N_EMPTY 29
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PLAN_FIRST_MOVE_WEIGHT
        #define TIME_MANAGEMENT_GGS_PLAN_FIRST_MOVE_WEIGHT 6600.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PLAN_OPENING_WEIGHT
        #define TIME_MANAGEMENT_GGS_PLAN_OPENING_WEIGHT 4400.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PLAN_MIDGAME_WEIGHT
        #define TIME_MANAGEMENT_GGS_PLAN_MIDGAME_WEIGHT 4000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PLAN_EARLY_ENDGAME_WEIGHT
        // The 36/35 searches also fill the TT for 34-31: with weight 4000 in
        // the 2026-10-06 3:00 games, exact results at 32 empties fell from 24%
        // to 0%, so both profiles keep 7000.
        #define TIME_MANAGEMENT_GGS_PLAN_EARLY_ENDGAME_WEIGHT 7000.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PLAN_PRE_COMPLETE_WEIGHT
        #define TIME_MANAGEMENT_GGS_PLAN_PRE_COMPLETE_WEIGHT 2500.0
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PLAN_32_TO_31_WEIGHT
        #define TIME_MANAGEMENT_GGS_PLAN_32_TO_31_WEIGHT TIME_MANAGEMENT_GGS_PLAN_PRE_COMPLETE_WEIGHT
    #endif
    #ifndef TIME_MANAGEMENT_GGS_PLAN_30_TO_29_WEIGHT
        #if IS_GGS_3MIN
            // The last planned own move (30 or 29 empties) gets the whole rest
            // of the plan, so its weight only decides how much the earlier
            // moves leave for it. In the 2026-10-16 3:00 games (10 threads)
            // these moves used 1.5 s / 0.4 s of a 15.5 s / 16.5 s mean share,
            // because the exact search or Ponder had mostly solved them. A
            // clock replay with 1000 moved 11 s per pair to the earlier moves
            // and cut 21 of the 274 moves at 30/29 by 4.4 s on average.
            #define TIME_MANAGEMENT_GGS_PLAN_30_TO_29_WEIGHT 1000.0
        #else
            #define TIME_MANAGEMENT_GGS_PLAN_30_TO_29_WEIGHT TIME_MANAGEMENT_GGS_PLAN_32_TO_31_WEIGHT
        #endif
    #endif
#else
    #define TIME_MANAGEMENT_REMAINING_TIME_OFFSET 300 // ms / move
    #define TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE 2000 // ms
#endif
// #define TIME_MANAGEMENT_REMAINING_MOVES_OFFSET 15 // 15 * 2 = 30 moves
#define TIME_MANAGEMENT_N_MOVES_COE_30_OR_MORE 1.2 // 1.4
#define TIME_MANAGEMENT_N_MOVES_COE_40_OR_MORE_ADDITIONAL 0.1 // 0.5 // additional search
#define TIME_MANAGEMENT_N_MOVES_COE_30_OR_MORE_NOTIME 1.2
#define TIME_MANAGEMENT_ADDITIONAL_TIME_COE_BASE 1.8
#define TIME_MANAGEMENT_ADDITIONAL_TIME_COE_ADD 1.8
//#define TIME_MANAGEMENT_N_MOVES_COE_ADDITIONAL_TIME 0.97

Search_result ai(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log);

#if IS_GGS_TOURNAMENT
inline double time_management_ggs_shortage_scale(uint64_t remaining_time_msec, double remaining_moves) {
    remaining_moves = std::max(1.0, remaining_moves);
    const double soft_line = TIME_MANAGEMENT_GGS_TARGET_SAFE_REMAINING_MSEC + TIME_MANAGEMENT_GGS_SHORTAGE_SOFT_MSEC_PER_MOVE * remaining_moves;
    const double hard_line = TIME_MANAGEMENT_GGS_TARGET_SAFE_REMAINING_MSEC + TIME_MANAGEMENT_GGS_SHORTAGE_HARD_MSEC_PER_MOVE * remaining_moves;
    if ((double)remaining_time_msec >= soft_line) {
        return 1.0;
    }
    if ((double)remaining_time_msec <= hard_line) {
        return TIME_MANAGEMENT_GGS_SHORTAGE_MIN_SCALE;
    }
    const double t = ((double)remaining_time_msec - hard_line) / (soft_line - hard_line);
    return TIME_MANAGEMENT_GGS_SHORTAGE_MIN_SCALE + (1.0 - TIME_MANAGEMENT_GGS_SHORTAGE_MIN_SCALE) * t;
}

inline uint64_t time_management_ggs_cap_time_limit(uint64_t time_limit, uint64_t remaining_time_msec, double remaining_moves) {
    remaining_moves = std::max(1.0, remaining_moves);
    const double reserve = TIME_MANAGEMENT_GGS_TARGET_SAFE_REMAINING_MSEC + TIME_MANAGEMENT_GGS_MAX_RESERVE_MSEC_PER_MOVE * remaining_moves;
    // Keep the cap continuous when the clock crosses the reserve line.  The
    // previous branch changed from 25% of the clock to almost zero immediately
    // above the line, which could turn a multi-second midgame allocation into
    // a few milliseconds merely because one more millisecond was available.
    const double low_time_cap = (double)remaining_time_msec * TIME_MANAGEMENT_GGS_LOW_TIME_MAX_USE_COE;
    const double surplus_cap = std::max(
        0.0,
        ((double)remaining_time_msec - reserve) * TIME_MANAGEMENT_GGS_MAX_SURPLUS_USE_COE
    );
    const double max_use_time = std::max(low_time_cap, surplus_cap);
    return std::max<uint64_t>(1ULL, std::min<uint64_t>(time_limit, (uint64_t)std::max(1.0, max_use_time)));
}

enum class Time_management_ggs_phase {
    opening,        // 47 empties or more: the first two own moves of s8r14
    midgame,        // 46-37
    early_endgame,  // 36-35: the one long selective-endgame search
    pre_complete,   // 34-29: the moves just before the complete search
    late            // 28 or less: covered by the late reserve
};

inline Time_management_ggs_phase time_management_ggs_phase(int n_empties) {
    if (n_empties > TIME_MANAGEMENT_GGS_PLAN_MIDGAME_MAX_N_EMPTY) {
        return Time_management_ggs_phase::opening;
    }
    if (n_empties >= TIME_MANAGEMENT_GGS_PLAN_MIDGAME_MIN_N_EMPTY) {
        return Time_management_ggs_phase::midgame;
    }
    if (n_empties >= TIME_MANAGEMENT_GGS_PLAN_EARLY_ENDGAME_MIN_N_EMPTY) {
        return Time_management_ggs_phase::early_endgame;
    }
    if (n_empties >= TIME_MANAGEMENT_GGS_PLAN_PRE_COMPLETE_MIN_N_EMPTY) {
        return Time_management_ggs_phase::pre_complete;
    }
    return Time_management_ggs_phase::late;
}

inline double time_management_ggs_plan_weight(int n_empties) {
    switch (time_management_ggs_phase(n_empties)) {
        case Time_management_ggs_phase::opening:
            return n_empties >= TIME_MANAGEMENT_INITIAL_N_EMPTIES - 1 ?
                TIME_MANAGEMENT_GGS_PLAN_FIRST_MOVE_WEIGHT :
                TIME_MANAGEMENT_GGS_PLAN_OPENING_WEIGHT;
        case Time_management_ggs_phase::midgame:
            return TIME_MANAGEMENT_GGS_PLAN_MIDGAME_WEIGHT;
        case Time_management_ggs_phase::early_endgame:
            return TIME_MANAGEMENT_GGS_PLAN_EARLY_ENDGAME_WEIGHT;
        case Time_management_ggs_phase::pre_complete:
            if (n_empties <= 30) {
                return TIME_MANAGEMENT_GGS_PLAN_30_TO_29_WEIGHT;
            }
            return n_empties <= 32 ? TIME_MANAGEMENT_GGS_PLAN_32_TO_31_WEIGHT :
                TIME_MANAGEMENT_GGS_PLAN_PRE_COMPLETE_WEIGHT;
        case Time_management_ggs_phase::late:
            break;
    }
    return 0.0;
}

struct Time_management_ggs_plan {
    Time_management_ggs_phase phase;
    uint64_t share_msec; // 0 when the clock is below the planned reserves

    Time_management_ggs_plan()
        : phase(Time_management_ggs_phase::late), share_msec(0ULL) {}
};

// The share of this move in the time left after the late reserve and the
// per-move GGS overhead of the planned moves. Own moves are assumed to come
// every two empties; a pass only shifts the plan until the next move.
inline Time_management_ggs_plan time_management_ggs_make_plan(int n_empties, uint64_t remaining_time_msec) {
    Time_management_ggs_plan plan;
    plan.phase = time_management_ggs_phase(n_empties);
    if (!TIME_MANAGEMENT_GGS_PLAN_ENABLED || plan.phase == Time_management_ggs_phase::late) {
        return plan;
    }
    double total_weight = 0.0;
    int n_planned_moves = 0;
    for (int e = n_empties; e >= TIME_MANAGEMENT_GGS_PLAN_PRE_COMPLETE_MIN_N_EMPTY; e -= 2) {
        total_weight += time_management_ggs_plan_weight(e);
        ++n_planned_moves;
    }
    const double distributable =
        (double)remaining_time_msec -
        TIME_MANAGEMENT_GGS_PLAN_LATE_RESERVE_MSEC -
        TIME_MANAGEMENT_GGS_PLAN_MOVE_OVERHEAD_MSEC * n_planned_moves;
    if (distributable <= 0.0 || total_weight <= 0.0) {
        return plan;
    }
    plan.share_msec = (uint64_t)(distributable * time_management_ggs_plan_weight(n_empties) / total_weight);
    return plan;
}

inline const char *time_management_ggs_phase_name(Time_management_ggs_phase phase) {
    switch (phase) {
        case Time_management_ggs_phase::opening: return "opening";
        case Time_management_ggs_phase::midgame: return "midgame";
        case Time_management_ggs_phase::early_endgame: return "early-endgame";
        case Time_management_ggs_phase::pre_complete: return "pre-complete";
        case Time_management_ggs_phase::late: return "late";
    }
    return "unknown";
}

// The opening keeps its own allocation below the share so that the ambiguity
// boost still decides how much of the share a position gets. Later planned
// phases search for the whole share.
inline uint64_t time_management_ggs_apply_plan(const Time_management_ggs_plan &plan, uint64_t time_limit, bool show_log) {
    if (plan.share_msec == 0ULL || plan.phase == Time_management_ggs_phase::late) {
        return time_limit;
    }
    const uint64_t planned = plan.phase == Time_management_ggs_phase::opening ?
        std::min(time_limit, plan.share_msec) :
        std::max(time_limit, plan.share_msec);
    if (show_log) {
        std::cerr << "ggs plan " << time_management_ggs_phase_name(plan.phase)
                  << " share " << plan.share_msec
                  << " tl " << time_limit << " -> " << planned << std::endl;
    }
    return planned;
}

// Upper bound of the whole move (probe and boosts included) that keeps the
// planned time of the later phases; 0 means no bound.
inline uint64_t time_management_ggs_plan_move_cap(const Time_management_ggs_plan &plan, uint64_t planned_time_limit) {
    if (plan.share_msec == 0ULL || plan.phase == Time_management_ggs_phase::late) {
        return 0ULL;
    }
    return std::max(planned_time_limit, plan.share_msec);
}

inline uint64_t time_management_ggs_ramped_endgame_force_time(
    uint64_t selected_time,
    uint64_t remaining_time_msec,
    double ramp_start_remaining_msec,
    double full_remaining_msec,
    double leave_msec,
    double max_force_time_msec
) {
    const double remaining = (double)remaining_time_msec;
    if (
        remaining <= ramp_start_remaining_msec ||
        remaining <= leave_msec ||
        full_remaining_msec <= ramp_start_remaining_msec
    ) {
        return selected_time;
    }
    const double ramp = std::clamp(
        (remaining - ramp_start_remaining_msec) /
            (full_remaining_msec - ramp_start_remaining_msec),
        0.0,
        1.0
    );
    const double available = remaining - leave_msec;
    const uint64_t ramped_force = (uint64_t)std::max(
        0.0,
        std::min(max_force_time_msec, available * ramp)
    );
    return std::max(selected_time, ramped_force);
}
#endif

uint64_t calc_time_limit_ply(const Board board, uint64_t remaining_time_msec, bool show_log) {
    int n_empties = HW2 - board.n_discs();
    double remaining_moves = (double)(n_empties + 1) / 2.0;
    uint64_t remaining_time_msec_margin = remaining_time_msec;
    if (remaining_time_msec > TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves + TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE) {
        remaining_time_msec_margin -= TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves + TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE;
#if IS_GGS_TOURNAMENT
        // At or below the offset the whole clock is the margin, so subtracting
        // the full offset just above it made the margin collapse: at 28 empties
        // 4 s left gave a 0.27 s search but 5 s left gave 0.012 s. Keep the
        // margin at the offset until the subtraction catches up, so the limit
        // never shrinks when more time is left.
        remaining_time_msec_margin = std::max<uint64_t>(
            remaining_time_msec_margin,
            (uint64_t)(TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves + TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE)
        );
#endif
    } else {
        if (show_log) {
            std::cerr << "don't have enough time! remaining " << remaining_time_msec_margin << std::endl;
        }
    }

#if IS_GGS_TOURNAMENT
    const double ggs_time_scale = time_management_ggs_shortage_scale(remaining_time_msec, remaining_moves);
    const Time_management_ggs_plan ggs_plan = time_management_ggs_make_plan(n_empties, remaining_time_msec);
#endif

#if IS_GGS_TOURNAMENT
    // First own moves around the s8r14 setup. The second player otherwise gets
    // only a normal midgame slice for their first response.
    if (n_empties == TIME_MANAGEMENT_INITIAL_N_EMPTIES) {
        if (show_log) {
            std::cerr << "first move time limit" << std::endl;
        }
        return time_management_ggs_apply_plan(
            ggs_plan,
            time_management_ggs_cap_time_limit(remaining_time_msec_margin * TIME_MANAGEMENT_GGS_FIRST_MOVE_COE * ggs_time_scale, remaining_time_msec, remaining_moves),
            show_log
        );
    }
    if (n_empties == TIME_MANAGEMENT_INITIAL_N_EMPTIES - 1) {
        if (show_log) {
            std::cerr << "first reply move time limit" << std::endl;
        }
        return time_management_ggs_apply_plan(
            ggs_plan,
            time_management_ggs_cap_time_limit(remaining_time_msec_margin * TIME_MANAGEMENT_GGS_FIRST_REPLY_MOVE_COE * ggs_time_scale, remaining_time_msec, remaining_moves),
            show_log
        );
    }
#endif

    // try complete search
    // Nodes(depth) = a * exp(b * depth)
    constexpr double complete_const_a = 0.60; //2.1747;
    constexpr double complete_const_b = 0.75;
    constexpr double complete_nps = 7.0e8;
#if IS_GGS_TOURNAMENT
    double complete_depth_time = (double)remaining_time_msec_margin * TIME_MANAGEMENT_GGS_COMPLETE_DEPTH_TIME_COE * ggs_time_scale;
    double complete_use_time = (double)remaining_time_msec_margin * TIME_MANAGEMENT_GGS_COMPLETE_USE_TIME_COE * ggs_time_scale;
    if (n_empties >= 32) {
        complete_depth_time = std::min(complete_depth_time, TIME_MANAGEMENT_GGS_COMPLETE_DEPTH_CAP_32_OR_MORE);
        complete_use_time = std::min(complete_use_time, TIME_MANAGEMENT_GGS_COMPLETE_USE_CAP_32_OR_MORE);
    } else if (n_empties >= 30) {
        complete_depth_time = std::min(complete_depth_time, TIME_MANAGEMENT_GGS_COMPLETE_DEPTH_CAP_30_OR_MORE);
        complete_use_time = std::min(complete_use_time, TIME_MANAGEMENT_GGS_COMPLETE_USE_CAP_30_OR_MORE);
    }
#else
    double complete_use_time = (double)remaining_time_msec_margin * 0.9;
    double complete_depth_time = complete_use_time;
#endif
    double complete_search_depth = log(complete_depth_time / 1000.0 * complete_nps / complete_const_a) / complete_const_b;

    // try endgame search
    // Nodes(depth) = a * exp(b * depth)
    constexpr double endgame_const_a = 0.05; //1.8654;
    constexpr double endgame_const_b = 0.62;
    constexpr double endgame_nps = 3.5e8;
#if IS_GGS_TOURNAMENT
    double endgame_depth_time = (double)remaining_time_msec_margin * TIME_MANAGEMENT_GGS_ENDGAME_DEPTH_TIME_COE * ggs_time_scale;
    double endgame_use_time = (double)remaining_time_msec_margin * TIME_MANAGEMENT_GGS_ENDGAME_USE_TIME_COE * ggs_time_scale;
#else
    double endgame_use_time = (double)remaining_time_msec_margin * 0.15;
    double endgame_depth_time = endgame_use_time;
#endif
    double endgame_search_depth = log(endgame_depth_time / 1000.0 * endgame_nps / endgame_const_a) / endgame_const_b;

    if (show_log) {
        std::cerr << "complete search depth " << complete_search_depth << " endgame search depth " << endgame_search_depth << " n_empties " << n_empties << std::endl;
    }

    // midgame search time
#if IS_GGS_TOURNAMENT
    double phase_time_coe = TIME_MANAGEMENT_GGS_PHASE_TIME_COE_LATE;
    if (n_empties >= 45) {
        phase_time_coe = TIME_MANAGEMENT_GGS_PHASE_TIME_COE_45_OR_MORE;
    } else if (n_empties >= 34) {
        phase_time_coe = TIME_MANAGEMENT_GGS_PHASE_TIME_COE_34_OR_MORE;
    } else if (n_empties >= 22) {
        phase_time_coe = TIME_MANAGEMENT_GGS_PHASE_TIME_COE_22_OR_MORE;
    }
    double remaining_moves_proc = std::max(3.0, remaining_moves + TIME_MANAGEMENT_GGS_REMAINING_MOVES_EXTRA);
    uint64_t midgame_use_time = std::max<uint64_t>(1ULL, (uint64_t)(phase_time_coe * remaining_time_msec_margin / remaining_moves_proc * ggs_time_scale));
#else
    double remaining_moves_proc = 0;
    if (remaining_time_msec_margin < remaining_time_msec) {
        if (remaining_moves >= 30 / 2) { // 30 or more
            remaining_moves_proc += (remaining_moves - (30 / 2)) * TIME_MANAGEMENT_N_MOVES_COE_30_OR_MORE;
        }
        if (remaining_moves >= 40 / 2) { // 40 or more
            remaining_moves_proc += (remaining_moves - (40 / 2)) * TIME_MANAGEMENT_N_MOVES_COE_40_OR_MORE_ADDITIONAL;
        }
    } else {
        remaining_moves_proc += (remaining_moves - (26 / 2)) * TIME_MANAGEMENT_N_MOVES_COE_30_OR_MORE_NOTIME;
    }
    remaining_moves_proc = std::max(2.0, remaining_moves_proc); // at least 2 moves
    uint64_t midgame_use_time = std::max<uint64_t>(1ULL, (uint64_t)(remaining_time_msec_margin / remaining_moves_proc));
#endif

    uint64_t selected_time;
    if (n_empties <= complete_search_depth) {
        if (show_log) {
            std::cerr << "try complete search tl max(" << complete_use_time << ", " << midgame_use_time << ")" << std::endl;
        }
        selected_time = std::max((uint64_t)complete_use_time, midgame_use_time);
    } else if (n_empties <= endgame_search_depth) {
        if (show_log) {
            std::cerr << "try endgame search tl max(" << endgame_use_time << ", " << midgame_use_time << ")" << std::endl;
        }
        selected_time = std::max((uint64_t)endgame_use_time, midgame_use_time);
    } else {
        selected_time = midgame_use_time;
    }
#if IS_GGS_TOURNAMENT
    // The phase plan sizes the 36-29 moves itself, so the endgame ramps are
    // only used without it.
    uint64_t endgame_forced_time = 0ULL;
    if (
        !TIME_MANAGEMENT_GGS_PLAN_ENABLED &&
        TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_MIN_N_EMPTY <= n_empties &&
        n_empties <= TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_MAX_N_EMPTY &&
        n_empties <= endgame_search_depth &&
        (double)remaining_time_msec > TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_RAMP_START_REMAINING
    ) {
        endgame_forced_time = time_management_ggs_ramped_endgame_force_time(
            selected_time,
            remaining_time_msec,
            TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_RAMP_START_REMAINING,
            TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_FULL_REMAINING,
            TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_LEAVE_MSEC,
            TIME_MANAGEMENT_GGS_EARLY_ENDGAME_FORCE_MAX_TIME
        );
        if (endgame_forced_time > selected_time) {
            if (show_log) {
                std::cerr << "ggs early endgame ramp tl " << selected_time << " -> " << endgame_forced_time
                          << " remaining " << remaining_time_msec << std::endl;
            }
            selected_time = endgame_forced_time;
        } else {
            endgame_forced_time = 0ULL;
        }
    }
    if (
        !TIME_MANAGEMENT_GGS_PLAN_ENABLED &&
        TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_MIN_N_EMPTY <= n_empties &&
        n_empties <= TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_MAX_N_EMPTY &&
        n_empties <= endgame_search_depth &&
        (double)remaining_time_msec > TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_RAMP_START_REMAINING
    ) {
        endgame_forced_time = time_management_ggs_ramped_endgame_force_time(
            selected_time,
            remaining_time_msec,
            TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_RAMP_START_REMAINING,
            TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_FULL_REMAINING,
            TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_LEAVE_MSEC,
            TIME_MANAGEMENT_GGS_LATE_ENDGAME_FORCE_MAX_TIME
        );
        if (endgame_forced_time > selected_time) {
            if (show_log) {
                std::cerr << "ggs late endgame ramp tl " << selected_time << " -> " << endgame_forced_time
                          << " remaining " << remaining_time_msec << std::endl;
            }
            selected_time = endgame_forced_time;
        } else {
            endgame_forced_time = 0ULL;
        }
    }
    const uint64_t capped_time = time_management_ggs_cap_time_limit(selected_time, remaining_time_msec, remaining_moves);
    if (endgame_forced_time > capped_time) {
        if (show_log) {
            std::cerr << "ggs endgame force bypass cap " << capped_time << " -> " << endgame_forced_time << std::endl;
        }
        return endgame_forced_time;
    }
    if (show_log && (ggs_time_scale < 0.999 || capped_time < selected_time)) {
        std::cerr << "ggs time adjust scale " << ggs_time_scale << " capped " << capped_time << " from " << selected_time << std::endl;
    }
    return time_management_ggs_apply_plan(ggs_plan, capped_time, show_log);
#else
    return selected_time;
#endif
}

uint64_t calc_time_limit_ply_MCTS(const Board board, uint64_t remaining_time_msec, bool show_log) {
    int n_empties = HW2 - board.n_discs();
    double remaining_moves = (double)(n_empties + 1) / 2.0;
    uint64_t remaining_time_msec_margin = remaining_time_msec;
    if (remaining_time_msec > TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves + TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE) {
        remaining_time_msec_margin -= TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves + TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE;
    } else {
        if (show_log) {
            std::cerr << "don't have enough time! remaining " << remaining_time_msec_margin << std::endl;
        }
    }

// #if IS_GGS_TOURNAMENT
//     // first move
//     if (n_empties == TIME_MANAGEMENT_INITIAL_N_EMPTIES || n_empties == TIME_MANAGEMENT_INITIAL_N_EMPTIES - 1) {
//         if (show_log) {
//             std::cerr << "first move time limit" << std::endl;
//         }
//         return remaining_time_msec_margin * 0.4;
//     }
// #endif

    // try complete search
    // Nodes(depth) = a * exp(b * depth)
    constexpr double complete_const_a = 0.60; //2.1747;
    constexpr double complete_const_b = 0.75;
    constexpr double complete_nps = 7.0e8;
    double complete_use_time = (double)remaining_time_msec_margin * 0.9;
    double complete_search_depth = log(complete_use_time / 1000.0 * complete_nps / complete_const_a) / complete_const_b;

    // try endgame search
    // Nodes(depth) = a * exp(b * depth)
    constexpr double endgame_const_a = 0.05; //1.8654;
    constexpr double endgame_const_b = 0.62;
    constexpr double endgame_nps = 3.5e8;
    double endgame_use_time = (double)remaining_time_msec_margin * 0.15;
    double endgame_search_depth = log(endgame_use_time / 1000.0 * endgame_nps / endgame_const_a) / endgame_const_b;

    if (show_log) {
        std::cerr << "complete search depth " << complete_search_depth << " endgame search depth " << endgame_search_depth << " n_empties " << n_empties << std::endl;
    }

    // midgame search time
    double remaining_moves_proc = 0;
    if (remaining_moves > 30 / 2) {
        remaining_moves_proc += remaining_moves - 30 / 2;
    }
    remaining_moves_proc = std::max(2.0, remaining_moves_proc); // at least 2 moves
    double coe = 1.0;
    if (n_empties >= 35) {
        coe += 1.5 * (double)(n_empties - 35) / (60.0 - 35.0);
    }
    std::cerr << "n_empties " << n_empties << " coe " << coe << std::endl;
    uint64_t midgame_use_time = std::max<uint64_t>(1ULL, (uint64_t)(coe * remaining_time_msec_margin / remaining_moves_proc));

    if (n_empties <= complete_search_depth) {
        if (show_log) {
            std::cerr << "try complete search tl max(" << complete_use_time << ", " << midgame_use_time << ")" << std::endl;
        }
        return std::max((uint64_t)complete_use_time, midgame_use_time);
    }
    if (n_empties <= endgame_search_depth) {
        if (show_log) {
            std::cerr << "try endgame search tl max(" << endgame_use_time << ", " << midgame_use_time << ")" << std::endl;
        }
        return std::max((uint64_t)endgame_use_time, midgame_use_time);
    }
    return midgame_use_time;
}

uint64_t request_more_time(Board board, uint64_t remaining_time_msec, uint64_t time_limit, bool show_log) {
    int n_empties = HW2 - board.n_discs();
    double remaining_moves = (double)(n_empties + 1) / 2.0;
    uint64_t remaining_time_msec_margin = remaining_time_msec;
    if (remaining_time_msec > TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves + TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE) {
        remaining_time_msec_margin -= TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves + TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE;
    } else {
        remaining_time_msec_margin = 1;
    }
    if (show_log) {
        std::cerr << "requesting more time remaining " << remaining_time_msec << " remaining_margin " << remaining_time_msec_margin << " tl before " << time_limit << std::endl;
    }
    if (remaining_time_msec_margin > time_limit && remaining_time_msec_margin > 40000ULL) {
        // int remaining_moves_proc = std::max(2, (int)round((remaining_moves - TIME_MANAGEMENT_REMAINING_MOVES_OFFSET) * TIME_MANAGEMENT_N_MOVES_COE_ADDITIONAL_TIME)); // at least 2 moves
        int remaining_moves_proc = 0;
        if (remaining_moves >= 30 / 2) { // 30 or more
            remaining_moves_proc += std::round((remaining_moves - (30 / 2)) * TIME_MANAGEMENT_N_MOVES_COE_30_OR_MORE);
        }
        if (remaining_moves >= 40 / 2) { // 40 or more
            remaining_moves_proc += std::round((remaining_moves - (40 / 2)) * TIME_MANAGEMENT_N_MOVES_COE_40_OR_MORE_ADDITIONAL);
        }
        remaining_moves_proc = std::max(2, remaining_moves_proc); // at least 2 moves
        double coe = TIME_MANAGEMENT_ADDITIONAL_TIME_COE_BASE;
        if (remaining_moves >= 40 / 2) { // 40 or more
            coe += TIME_MANAGEMENT_ADDITIONAL_TIME_COE_ADD * (double)(remaining_moves - (40.0 / 2.0)) / (60.0 / 2.0 - 40.0 / 2.0);
        }
        if (show_log) {
            std::cerr << "time request coe " << coe << std::endl;
        }
        uint64_t additional_time = (remaining_time_msec_margin - time_limit) / remaining_moves_proc * coe;
        additional_time = std::min(additional_time, remaining_time_msec_margin / 2);
        if (show_log) {
            std::cerr << "additional time " << additional_time << std::endl;
        }
        time_limit += additional_time;
    }
    if (show_log) {
        std::cerr << "more time requested: new time limit " << time_limit << std::endl;
    }
    return time_limit;
}


void selfplay(Board board, int level) {
    Search_result result;
    Flip flip;
    while (board.check_pass()) {
        result = ai(board, level, true, 0, false, false);
        calc_flip(&flip, &board, result.policy);
        board.move_board(&flip);
    }
}


void time_management_selfplay(Board board, bool show_log, uint64_t use_legal, uint64_t time_limit) {
    uint64_t start_time = tim();
    if (show_log) {
        std::cerr << "self play time " << time_limit << " ms" << std::endl;
    }
    std::vector<Flip> move_list(pop_count_ull(use_legal));
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&use_legal); use_legal; cell = next_bit(&use_legal)) {
        calc_flip(&move_list[idx], &board, cell);
        ++idx;
    }
    int select_idx = 0;
    uint64_t count = 0;
    while (tim() - start_time < time_limit) {
        board.move_board(&move_list[select_idx]);
            selfplay(board, 11);
        board.undo_board(&move_list[select_idx]);
        ++count;
        ++select_idx;
        select_idx %= move_list.size();
    }
    if (show_log) {
        std::cerr << "self play count " << count << std::endl;
    }
}
