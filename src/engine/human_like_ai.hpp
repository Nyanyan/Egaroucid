/*
    Egaroucid Project

    @file human_like_ai.hpp
        Human-like AI Algorithms
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include <random>
#include "level.hpp"
#include "setting.hpp"
#include "util.hpp"
#include "search.hpp"
#include "evaluate.hpp"
#include "move_ordering.hpp"
#include "transposition_table.hpp"

std::random_device seed_gen;
std::mt19937 engine(seed_gen());
std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFFFFFFFFULL);

void noise_flip(Flip *flip, int depth) {
    uint64_t mask = flip->flip;
    int n_masked = pop_count_ull(flip->flip);
    for (int i = 0; i < depth + 2; ++i) { // deeper -> more errors
        uint64_t n_mask = flip->flip & dist(engine);
        int n_n_masked = pop_count_ull(n_mask);
        if (n_n_masked <= n_masked) {
            mask = n_mask;
            n_masked = n_n_masked;
        }
    }
    flip->flip ^= mask; // off some bits
}

int nega_alpha_human_like(Search *search, int alpha, int beta, int depth, bool skipped, bool is_end_search, bool *searching) {
    if (!(*searching) || !global_searching) {
        return SCORE_UNDEFINED;
    }
    ++search->n_nodes;
    if (depth == 0) {
        if (is_end_search) {
            return end_evaluate(&search->board);
        } else {
            return mid_evaluate_diff(search);
        }
    }
    uint64_t legal = search->board.get_legal();
    if (legal == 0) {
        if (skipped) {
            return end_evaluate(&search->board);
        }
        search->pass();
            int v = -nega_alpha_human_like(search, -beta, -alpha, depth, true, is_end_search, searching);
        search->pass();
        return v;
    }
    int canput = pop_count_ull(legal);
    std::vector<Flip_value> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&move_list[idx].flip, &search->board, cell);
        noise_flip(&move_list[idx].flip, depth);
        ++idx;
    }
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
    move_list_evaluate(search, move_list, moves, depth, alpha, beta, searching);
    if (beta <= alpha) {
        return alpha;
    }
    int g, v = -SCORE_INF;
    for (int move_idx = 0; move_idx < canput && *searching; ++move_idx) {
        swap_next_best_move(move_list, move_idx, canput);
        search->move(&move_list[move_idx].flip);
            g = -nega_alpha_human_like(search, -beta, -alpha, depth - 1, false, is_end_search, searching);
        search->undo(&move_list[move_idx].flip);
        if (v < g) {
            v = g;
            if (alpha < v) {
                if (beta <= v) {
                    break;
                }
                alpha = v;
            }
        }
    }
    return v;
}

Search_result nega_alpha_human_like_root(Search *search, int alpha, int beta, int depth, bool is_end_search, bool *searching) {
    Search_result res;
    if (!(*searching) || !global_searching || depth <= 0) {
        res.value = SCORE_UNDEFINED;
        res.policy = MOVE_NOMOVE;
        return res;
    }
    uint64_t legal = search->board.get_legal();
    if (legal == 0) {
        res.value = SCORE_UNDEFINED;
        res.policy = MOVE_NOMOVE;
        return res;
    }
    int canput = pop_count_ull(legal);
    std::vector<Flip_value> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&move_list[idx].flip, &search->board, cell);
        //noise_flip(&move_list[idx].flip, depth);
        ++idx;
    }
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
    move_list_evaluate(search, move_list, moves, depth, alpha, beta, searching);
    int g;
    res.value = -SCORE_INF;
    res.policy = MOVE_NOMOVE;
    for (int move_idx = 0; move_idx < canput && *searching; ++move_idx) {
        swap_next_best_move(move_list, move_idx, canput);
        search->move(&move_list[move_idx].flip);
            g = -nega_alpha_human_like(search, -beta, -alpha, depth - 1, false, is_end_search, searching);
        search->undo(&move_list[move_idx].flip);
        std::cerr << "human like ai move " << move_idx + 1 << "/" << canput << " value " << g << " policy " << idx_to_coord(move_list[move_idx].flip.pos) << " window " << "[" << alpha << "," << beta << "] " << std::endl;
        if (res.value < g) {
            res.value = g;
            res.policy = move_list[move_idx].flip.pos;
            if (alpha < res.value) {
                if (beta <= res.value) {
                    break;
                }
                alpha = res.value;
            }
        }
    }
    return res;
}

Search_result human_like_ai(Board board, int level, bool show_log) {
    int value_sign = 1;
    Search_result res;
    if (board.get_legal() == 0ULL) {
        board.pass();
        if (board.get_legal() == 0ULL) {
            res.policy = 64;
            res.value = -board.score_player();
            res.depth = 0;
            res.nps = 0;
            res.is_end_search = true;
            res.probability = 100;
            return res;
        } else {
            value_sign = -1;
        }
    }
    int depth;
    bool is_mid_search;
    uint_fast8_t mpc_level;
    get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
    Search search(&board, mpc_level, false, false);
    bool searching = true;
    uint64_t strt = tim();
    res = nega_alpha_human_like_root(&search, -SCORE_MAX, SCORE_MAX, depth, !is_mid_search, &searching);
    res.value *= value_sign;
    res.time = tim() - strt;
    res.nps = calc_nps(res.nodes, res.time);
    res.depth = depth;
    res.is_end_search = !is_mid_search;
    res.probability = 100;
    res.clog_nodes = 0;
    res.clog_time = 0;
    if (show_log) {
        std::cerr << "human like ai value " << res.value << " policy " << idx_to_coord(res.policy) << std::endl;
    }
    return res;
}