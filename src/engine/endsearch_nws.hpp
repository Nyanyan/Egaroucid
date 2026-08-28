/*
    Egaroucid Project

    @file endsearch_nws.hpp
        Search near endgame with NWS (Null Window Search)
        last2/3/4_nws imported from Edax AVX, (C) 1998 - 2018 Richard Delorme, 2014 - 23 Toshihiko Okuhara
    @date 2021-2026
    @author Takuto Yamana
    @author Toshihiko Okuhara
    @license GPL-3.0-or-later
*/

#pragma once
#include <atomic>
#include <iostream>
#include <vector>
#include <functional>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "move_ordering.hpp"
#include "multi_probcut.hpp"
#include "transposition_table.hpp"
#include "util.hpp"
#include "stability_cutoff.hpp"
#include "endsearch_common.hpp"
#include "parallel.hpp"
#include "ybwc.hpp"
#if USE_SIMD
#include "endsearch_nws_last_simd.hpp"
#else
#include "endsearch_nws_last_generic.hpp"
#endif

constexpr int LOCAL_TT_SIZE = 2048;
constexpr int LOCAL_TT_SIZE_BIT = 11;
static_assert(
    (END_NWS_CANCELLATION_POLL_MASK & (END_NWS_CANCELLATION_POLL_MASK + 1)) == 0,
    "END_NWS_CANCELLATION_POLL_MASK must be one less than a power of two"
);

inline bool end_nws_cancellation_requested(Search *search, const std::vector<bool*> &searchings) {
#if USE_END_NWS_CANCELLATION_POLLING
    return
        (search->n_nodes & END_NWS_CANCELLATION_POLL_MASK) == 0ULL &&
        !is_searching(searchings);
#else
    (void)search;
    (void)searchings;
    return false;
#endif
}

/*
    @brief Get a final score with few empties (NWS)

    Only with parity-based ordering.
    imported from search_shallow of Edax AVX, (C) 1998 - 2018 Richard Delorme, 2014 - 23 Toshihiko Okuhara

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @param skipped              already passed?
    @return the final score
*/
int nega_alpha_end_fast_nws(Search *search, int alpha, const bool skipped, const std::vector<bool*> &searchings) {
    if (end_nws_cancellation_requested(search, searchings)) {
        return SCORE_UNDEFINED;
    }
    ++search->n_nodes;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[search->n_discs];
#endif
#if USE_END_SC
    if (!skipped) {
        int stab_res = stability_cut_nws(search, alpha);
        if (stab_res != SCORE_UNDEFINED) {
            return stab_res;
        }
    }
#endif
    uint64_t legal = search->board.get_legal();
    if (legal == 0ULL) {
        if (skipped) {
            return end_evaluate(&search->board);
        }
        search->pass_noeval();
            const int child_value = nega_alpha_end_fast_nws(search, -alpha - 1, true, searchings);
        search->pass_noeval();
        return child_value == SCORE_UNDEFINED ? SCORE_UNDEFINED : -child_value;
    }

    Board board0;
    search->board.copy(&board0);
    int v = -SCORE_INF;
    int g;
    Flip flip;
    uint_fast8_t cell;
    uint64_t prioritymoves = legal;
#if USE_END_PO
    prioritymoves &= empty1_bb(search->board.player, search->board.opponent);
    if (prioritymoves == 0) {
        prioritymoves = legal;
    }
#endif
    
    if (search->n_discs == 59) {     // transfer to lastN, no longer uses n_discs, parity
        do {
            legal ^= prioritymoves;
            for (cell = first_bit(&prioritymoves); prioritymoves; cell = next_bit(&prioritymoves)) {
                calc_flip(&flip, &board0, cell);
                board0.move_copy(&flip, &search->board);
                g = last4_nws(search, alpha);
                if (alpha < g) {
                    board0.copy(&search->board);
                    return g;
                }
                if (v < g) {
                    v = g;
                }
            }
        } while ((prioritymoves = legal));
    } else {
        ++search->n_discs;  // for next depth
        do {
            legal ^= prioritymoves;
            for (cell = first_bit(&prioritymoves); prioritymoves; cell = next_bit(&prioritymoves)) {
                calc_flip(&flip, &board0, cell);
                board0.move_copy(&flip, &search->board);
                const int child_value = nega_alpha_end_fast_nws(search, -alpha - 1, false, searchings);
                if (child_value == SCORE_UNDEFINED) {
                    --search->n_discs;
                    board0.copy(&search->board);
                    return SCORE_UNDEFINED;
                }
                g = -child_value;
                if (alpha < g) {
                    --search->n_discs;
                    board0.copy(&search->board);
                    return g;
                }
                if (v < g) {
                    v = g;
                }
            }
        } while ((prioritymoves = legal));
        --search->n_discs;
    }
    board0.copy(&search->board);
    return v;
}

struct LocalTTEntry {
    uint64_t player;
    uint64_t opponent;
    int lower;
    int upper;

    bool cmp(Board *board) {
        return board->player == player && board->opponent == opponent;
    }

    void set_score(Board *board, int l, int u) {
        player = board->player;
        opponent = board->opponent;
        lower = l;
        upper = u;
    }

};

static thread_local LocalTTEntry lttable[MID_TO_END_DEPTH - END_FAST_DEPTH][LOCAL_TT_SIZE];

#if USE_LOCAL_TT_STATISTICS
struct LocalTTStats {
    std::atomic<uint64_t> probes{0};
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> lower_cuts{0};
    std::atomic<uint64_t> upper_cuts{0};
    std::atomic<uint64_t> lower_stores{0};
    std::atomic<uint64_t> upper_stores{0};
    std::atomic<uint64_t> overwrites{0};
    std::atomic<uint64_t> occupied_misses{0};
};

inline LocalTTStats local_tt_stats[MID_TO_END_DEPTH - END_FAST_DEPTH];

inline uint32_t local_tt_depth_index(uint32_t n_discs) {
    return HW2 - n_discs - END_FAST_DEPTH;
}

inline bool local_tt_is_occupied(const LocalTTEntry *tt) {
    return tt->player != 0ULL || tt->opponent != 0ULL || tt->lower != 0 || tt->upper != 0;
}

inline void local_tt_record_probe(uint32_t n_discs, LocalTTEntry *tt, Board *board) {
    const uint32_t idx = local_tt_depth_index(n_discs);
    local_tt_stats[idx].probes.fetch_add(1, std::memory_order_relaxed);
    if (local_tt_is_occupied(tt) && !tt->cmp(board)) {
        local_tt_stats[idx].occupied_misses.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void local_tt_record_hit(uint32_t n_discs) {
    local_tt_stats[local_tt_depth_index(n_discs)].hits.fetch_add(1, std::memory_order_relaxed);
}

inline void local_tt_record_lower_cut(uint32_t n_discs) {
    local_tt_stats[local_tt_depth_index(n_discs)].lower_cuts.fetch_add(1, std::memory_order_relaxed);
}

inline void local_tt_record_upper_cut(uint32_t n_discs) {
    local_tt_stats[local_tt_depth_index(n_discs)].upper_cuts.fetch_add(1, std::memory_order_relaxed);
}

inline void local_tt_record_store(uint32_t n_discs, LocalTTEntry *tt, Board *board, const bool lower_store) {
    const uint32_t idx = local_tt_depth_index(n_discs);
    if (lower_store) {
        local_tt_stats[idx].lower_stores.fetch_add(1, std::memory_order_relaxed);
    } else {
        local_tt_stats[idx].upper_stores.fetch_add(1, std::memory_order_relaxed);
    }
    if (local_tt_is_occupied(tt) && !tt->cmp(board)) {
        local_tt_stats[idx].overwrites.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void local_tt_stats_print() {
    uint64_t total_probes = 0;
    uint64_t total_hits = 0;
    uint64_t total_lower_cuts = 0;
    uint64_t total_upper_cuts = 0;
    uint64_t total_lower_stores = 0;
    uint64_t total_upper_stores = 0;
    uint64_t total_overwrites = 0;
    uint64_t total_occupied_misses = 0;
    std::cout << "LOCAL_TT_STATS_BEGIN\n";
    std::cout << "empties\tprobes\thits\tlower_cuts\tupper_cuts\tlower_stores\tupper_stores\toverwrites\toccupied_misses\n";
    for (int i = 0; i < MID_TO_END_DEPTH - END_FAST_DEPTH; ++i) {
        const uint64_t probes = local_tt_stats[i].probes.load(std::memory_order_relaxed);
        const uint64_t hits = local_tt_stats[i].hits.load(std::memory_order_relaxed);
        const uint64_t lower_cuts = local_tt_stats[i].lower_cuts.load(std::memory_order_relaxed);
        const uint64_t upper_cuts = local_tt_stats[i].upper_cuts.load(std::memory_order_relaxed);
        const uint64_t lower_stores = local_tt_stats[i].lower_stores.load(std::memory_order_relaxed);
        const uint64_t upper_stores = local_tt_stats[i].upper_stores.load(std::memory_order_relaxed);
        const uint64_t overwrites = local_tt_stats[i].overwrites.load(std::memory_order_relaxed);
        const uint64_t occupied_misses = local_tt_stats[i].occupied_misses.load(std::memory_order_relaxed);
        if (probes || lower_stores || upper_stores) {
            std::cout << (i + END_FAST_DEPTH) << '\t'
                << probes << '\t'
                << hits << '\t'
                << lower_cuts << '\t'
                << upper_cuts << '\t'
                << lower_stores << '\t'
                << upper_stores << '\t'
                << overwrites << '\t'
                << occupied_misses << '\n';
        }
        total_probes += probes;
        total_hits += hits;
        total_lower_cuts += lower_cuts;
        total_upper_cuts += upper_cuts;
        total_lower_stores += lower_stores;
        total_upper_stores += upper_stores;
        total_overwrites += overwrites;
        total_occupied_misses += occupied_misses;
    }
    std::cout << "total\t"
        << total_probes << '\t'
        << total_hits << '\t'
        << total_lower_cuts << '\t'
        << total_upper_cuts << '\t'
        << total_lower_stores << '\t'
        << total_upper_stores << '\t'
        << total_overwrites << '\t'
        << total_occupied_misses << '\n';
    std::cout << "LOCAL_TT_STATS_END\n";
}
#endif

inline uint32_t hash_bb(Board *board) {
#if USE_SIMD && USE_CRC32C_HASH_LTT
    uint64_t res = _mm_crc32_u64(0, board->player);
    res = _mm_crc32_u64(res, board->opponent);
    return res & (LOCAL_TT_SIZE - 1);
#else
    return ((board->player * 0x9dda1c54cfe6b6e9ull) ^ (board->opponent * 0xa2e6c0300831e05aull)) >> (HW2 - LOCAL_TT_SIZE_BIT);
#endif
}

inline LocalTTEntry *get_ltt(Board *board, uint32_t n_discs) {
    return lttable[HW2 - n_discs - END_FAST_DEPTH] + hash_bb(board);
}

/*
    @brief Get a final score with some empties (NWS)

    Search with move ordering for endgame and transposition tables.

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @param skipped              already passed?
    @param legal                for use of previously calculated legal bitboard
    @return the final score
*/
int nega_alpha_end_simple_nws(Search *search, int alpha, const bool skipped, uint64_t legal, const std::vector<bool*> &searchings) {
    if (end_nws_cancellation_requested(search, searchings)) {
        return SCORE_UNDEFINED;
    }
    if (search->n_discs >= HW2 - END_FAST_DEPTH) {
        return nega_alpha_end_fast_nws(search, alpha, skipped, searchings);
    }
    ++search->n_nodes;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[search->n_discs];
#endif
#if USE_END_SC
    if (!skipped) {
        int stab_res = stability_cut_nws(search, alpha);
        if (stab_res != SCORE_UNDEFINED) {
            return stab_res;
        }
    }
#endif
    if (legal == LEGAL_UNDEFINED) {
        legal = search->board.get_legal();
    }
    int v = -SCORE_INF;
    if (legal == 0ULL) {
        if (skipped) {
            return end_evaluate(&search->board);
        }
        search->pass_noeval();
            const int child_value = nega_alpha_end_simple_nws(search, -alpha - 1, true, LEGAL_UNDEFINED, searchings);
        search->pass_noeval();
        return child_value == SCORE_UNDEFINED ? SCORE_UNDEFINED : -child_value;
    }
    const int canput = pop_count_ull(legal);
    const uint32_t child_n_discs = search->n_discs + 1;
    LocalTTEntry *const child_lttable =
        lttable[HW2 - child_n_discs - END_FAST_DEPTH];
    Flip_value move_list[END_SIMPLE_DEPTH];
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip_value(&move_list[idx], &search->board, cell);
        if (move_list[idx].flip.flip == search->board.opponent) {
            return SCORE_MAX;
        }
        ++idx;
    }
    int g;
    // move ordering
    uint64_t done = 0;
    if (canput > 1) {
        for (int i = 0; i < canput; ++i) {
            Flip_value *flip_value = move_list + i;
            flip_value->value = 0;
            if (search->parity & cell_div4[flip_value->flip.pos]) {
                flip_value->value += W_END_NWS_SIMPLE_PARITY;
            }
            search->move_noeval(&flip_value->flip);
                Board nboard = search->board;
                // static_eval is unused in endgame move ordering.  Carry the
                // already-computed local-TT index with the move through sort.
                flip_value->static_eval = hash_bb(&nboard);
                LocalTTEntry *tt = child_lttable + flip_value->static_eval;
#if USE_LOCAL_TT_STATISTICS
                local_tt_record_probe(child_n_discs, tt, &nboard);
#endif
                if (tt->cmp(&nboard)) {
#if USE_LOCAL_TT_STATISTICS
                    local_tt_record_hit(child_n_discs);
#endif
                    if (alpha < tt->lower) {
#if USE_LOCAL_TT_STATISTICS
                        local_tt_record_lower_cut(child_n_discs);
#endif
                        v = tt->lower;
                        search->undo_noeval(&flip_value->flip);
                        return v;
                    }
                    if (tt->upper <= alpha) {
#if USE_LOCAL_TT_STATISTICS
                        local_tt_record_upper_cut(child_n_discs);
#endif
                        if (v < tt->upper) {
                            v = tt->upper;
                        }
                        done |= (1ULL << flip_value->flip.pos);
                        search->undo_noeval(&flip_value->flip);
                        continue;
                    }
                }
                flip_value->n_legal = search->board.get_legal();
                int nm = get_n_moves_cornerX2(flip_value->n_legal);
                if (nm <= 1) {
                    const int child_value = nega_alpha_end_simple_nws(search, -alpha - 1, false, flip_value->n_legal, searchings);
                    search->undo_noeval(&flip_value->flip);
                    if (child_value == SCORE_UNDEFINED) {
                        return SCORE_UNDEFINED;
                    }
                    g = -child_value;
                    if (v < g) {
                        v = g;
                        if (alpha < v) {
#if USE_LOCAL_TT_STATISTICS
                            local_tt_record_store(child_n_discs, tt, &nboard, true);
#endif
                            tt->set_score(&nboard, v, 64);
                            return v;
                        }
                    }
#if USE_LOCAL_TT_STATISTICS
                    local_tt_record_store(child_n_discs, tt, &nboard, false);
#endif
                    tt->set_score(&nboard, -64, g);
                    done |= (1ULL << flip_value->flip.pos);
                    continue;
                }
                flip_value->value += (MO_OFFSET_L_PM - nm) * W_END_NWS_SIMPLE_MOBILITY;
            search->undo_noeval(&flip_value->flip);
        }
    }

    for (int move_idx = 0; move_idx < canput; ++move_idx) {
        swap_next_best_move(move_list, move_idx, canput);
        if ((1ULL << move_list[move_idx].flip.pos) & done) {
            continue;
        }
        search->move_noeval(&move_list[move_idx].flip);
            Board nboard = search->board;
            const uint32_t ltt_index = canput > 1
                ? move_list[move_idx].static_eval
                : hash_bb(&nboard);
            LocalTTEntry *tt = child_lttable + ltt_index;
            const int child_value = nega_alpha_end_simple_nws(search, -alpha - 1, false, move_list[move_idx].n_legal, searchings);
        search->undo_noeval(&move_list[move_idx].flip);
        if (child_value == SCORE_UNDEFINED) {
            return SCORE_UNDEFINED;
        }
        g = -child_value;
        if (v < g) {
            v = g;
            if (alpha < v) {
#if USE_LOCAL_TT_STATISTICS
                local_tt_record_store(child_n_discs, tt, &nboard, true);
#endif
                tt->set_score(&nboard, v, 64);
                break;
            }
        }
#if USE_LOCAL_TT_STATISTICS
        local_tt_record_store(child_n_discs, tt, &nboard, false);
#endif
        tt->set_score(&nboard, -64, g);
    }
    return v;
}

/*
    @brief Get a final score with some empties (NWS)

    Search with move ordering for endgame and transposition tables.

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @param skipped              already passed?
    @param legal                for use of previously calculated legal bitboard
    @return the final score
*/
int nega_alpha_end_nws(Search *search, int alpha, const bool skipped, uint64_t legal, const std::vector<bool*> &searchings) {
    if (end_nws_cancellation_requested(search, searchings)) {
        return SCORE_UNDEFINED;
    }
    if (search->n_discs >= HW2 - END_SIMPLE_DEPTH) {
        return nega_alpha_end_simple_nws(search, alpha, skipped, legal, searchings);
    }
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    #if USE_END_SC
        if (!skipped) {
            int stab_res = stability_cut_nws(search, alpha);
            if (stab_res != SCORE_UNDEFINED) {
                return stab_res;
            }
        }
    #endif
    if (legal == LEGAL_UNDEFINED) {
        legal = search->board.get_legal();
    }
    int v = -SCORE_INF;
    if (legal == 0ULL) {
        if (skipped) {
            return end_evaluate(&search->board);
        }
        search->pass_endsearch();
            const int child_value = nega_alpha_end_nws(search, -alpha - 1, true, LEGAL_UNDEFINED, searchings);
        search->pass_endsearch();
        return child_value == SCORE_UNDEFINED ? SCORE_UNDEFINED : -child_value;
    }
    int g;
    const int canput = pop_count_ull(legal);
    const uint32_t child_n_discs = search->n_discs + 1;
    LocalTTEntry *const child_lttable =
        lttable[HW2 - child_n_discs - END_FAST_DEPTH];
    Flip_value move_list[MID_TO_END_DEPTH];
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip_value(&move_list[idx], &search->board, cell);
        if (move_list[idx].flip.flip == search->board.opponent) {
            return SCORE_MAX;
        }
        ++idx;
    }
    uint64_t done = 0;
    if (canput > 1) {
        for (int i = 0; i < canput; ++i) {
            Flip_value *flip_value = move_list + i;
            flip_value->value = 0;
            search->move_endsearch(&flip_value->flip);
                Board nboard = search->board;
                // static_eval is unused in endgame move ordering.  Carry the
                // already-computed local-TT index with the move through sort.
                flip_value->static_eval = hash_bb(&nboard);
                LocalTTEntry *tt = child_lttable + flip_value->static_eval;
#if USE_LOCAL_TT_STATISTICS
                local_tt_record_probe(child_n_discs, tt, &nboard);
#endif
                if (tt->cmp(&nboard)) {
#if USE_LOCAL_TT_STATISTICS
                    local_tt_record_hit(child_n_discs);
#endif
                    if (alpha < tt->lower) {
#if USE_LOCAL_TT_STATISTICS
                        local_tt_record_lower_cut(child_n_discs);
#endif
                        v = tt->lower;
                        search->undo_endsearch(&flip_value->flip);
                        return v;
                    }
                    if (tt->upper <= alpha) {
#if USE_LOCAL_TT_STATISTICS
                        local_tt_record_upper_cut(child_n_discs);
#endif
                        if (v < tt->upper) {
                            v = tt->upper;
                        }
                        done |= (1ULL << flip_value->flip.pos);
                        search->undo_endsearch(&flip_value->flip);
                        continue;
                    }
                }
                flip_value->n_legal = search->board.get_legal();
                int nm = get_n_moves_cornerX2(flip_value->n_legal);
                if (nm <= 1) {
                    const int child_value = nega_alpha_end_nws(search, -alpha - 1, false, flip_value->n_legal, searchings);
                    search->undo_endsearch(&flip_value->flip);
                    if (child_value == SCORE_UNDEFINED) {
                        return SCORE_UNDEFINED;
                    }
                    g = -child_value;
                    if (v < g) {
                        v = g;
                        if (alpha < v) {
#if USE_LOCAL_TT_STATISTICS
                            local_tt_record_store(child_n_discs, tt, &nboard, true);
#endif
                            tt->set_score(&nboard, v, 64);
                            return v;
                        }
                    }
#if USE_LOCAL_TT_STATISTICS
                    local_tt_record_store(child_n_discs, tt, &nboard, false);
#endif
                    tt->set_score(&nboard, -64, g);
                    done |= (1ULL << flip_value->flip.pos);
                    continue;
                }
                flip_value->value += (MO_OFFSET_L_PM - nm) * W_END_NWS_MOBILITY;
                flip_value->value += (MO_OFFSET_L_PM - mid_evaluate_move_ordering_end(search)) * W_END_NWS_VALUE;
            search->undo_endsearch(&flip_value->flip);
        }
    }
    for (int move_idx = 0; move_idx < canput; ++move_idx) {
        swap_next_best_move(move_list, move_idx, canput);
        if ((1ULL << move_list[move_idx].flip.pos) & done) {
            continue;
        }
        search->move_endsearch(&move_list[move_idx].flip);
            Board nboard = search->board;
            const uint32_t ltt_index = canput > 1
                ? move_list[move_idx].static_eval
                : hash_bb(&nboard);
            LocalTTEntry *tt = child_lttable + ltt_index;
            const int child_value = nega_alpha_end_nws(search, -alpha - 1, false, move_list[move_idx].n_legal, searchings);
        search->undo_endsearch(&move_list[move_idx].flip);
        if (child_value == SCORE_UNDEFINED) {
            return SCORE_UNDEFINED;
        }
        g = -child_value;
        if (v < g) {
            v = g;
            if (alpha < v) {
#if USE_LOCAL_TT_STATISTICS
                local_tt_record_store(child_n_discs, tt, &nboard, true);
#endif
                tt->set_score(&nboard, v, 64);
                break;
            }
        }
#if USE_LOCAL_TT_STATISTICS
        local_tt_record_store(child_n_discs, tt, &nboard, false);
#endif
        tt->set_score(&nboard, -64, g);
    }
    return v;
}
