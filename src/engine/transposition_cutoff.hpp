/*
    Egaroucid Project

    @file transposition_cutoff.hpp
        Transposition Cutoff + Enhanced Transposition Cutoff
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/
#pragma once
#include "setting.hpp"
#include "search.hpp"
#include "transposition_table.hpp"
#include "move_ordering.hpp"

constexpr int MID_ETC_DEPTH = 16;
constexpr int MID_ETC_DEPTH_NWS = 16;

inline bool transposition_cutoff(Search *search, const uint32_t hash_code, int depth, int *alpha, int *beta, int *v, uint_fast8_t moves[]) {
    int lower = -SCORE_MAX, upper = SCORE_MAX;
    transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
    if (upper == lower || upper <= *alpha) {
        *v = upper;
        return true;
    }
    if (*beta <= lower) {
        *v = lower;
        return true;
    }
    if (*alpha < lower) {
        *alpha = lower;
    }
    if(upper < *beta) {
        *beta = upper;
    }
    return false;
}

inline bool transposition_cutoff_bestmove(Search *search, const uint32_t hash_code, int depth, int *alpha, int *beta, int *v, int *best_move) {
    int lower = -SCORE_MAX, upper = SCORE_MAX;
    uint_fast8_t moves[N_TRANSPOSITION_MOVES];
    transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
    if (upper == lower || upper <= *alpha) {
        *v = upper;
        *best_move = moves[0];
        return true;
    }
    if (*beta <= lower) {
        *v = lower;
        *best_move = moves[0];
        return true;
    }
    if (*alpha < lower) {
        *alpha = lower;
    }
    if(upper < *beta) {
        *beta = upper;
    }
    return false;
}

inline bool transposition_cutoff_nws(Search *search, const uint32_t hash_code, int depth, int alpha, int *v, uint_fast8_t moves[]) {
    int lower = -SCORE_MAX, upper = SCORE_MAX;
    transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
    if (upper == lower || upper <= alpha) {
        *v = upper;
        return true;
    }
    if (alpha < lower) {
        *v = lower;
        return true;
    }
    return false;
}

inline bool transposition_cutoff_nws(Search *search, const uint32_t hash_code, int depth, int alpha, int *v) {
    int lower = -SCORE_MAX, upper = SCORE_MAX;
    transposition_table.get_bounds(search, hash_code, depth, &lower, &upper);
    if (upper == lower || upper <= alpha) {
        *v = upper;
        return true;
    }
    if (alpha < lower) {
        *v = lower;
        return true;
    }
    return false;
}

inline bool transposition_cutoff_nws_bestmove(Search *search, const uint32_t hash_code, int depth, int alpha, int *v, int *best_move) {
    int lower = -SCORE_MAX, upper = SCORE_MAX;
    uint_fast8_t moves[N_TRANSPOSITION_MOVES];
    transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
    if (upper == lower || upper <= alpha) {
        *v = upper;
        *best_move = moves[0];
        return true;
    }
    if (alpha < lower) {
        *v = lower;
        *best_move = moves[0];
        return true;
    }
    return false;
}



/*
    @brief Enhanced Transposition Cutoff (ETC)

    @param hash_level_failed    hash level used when failed
    @param hash_level           new hash level
    @return hash resized?
*/
inline bool etc(Search *search, std::vector<Flip_value> &move_list, int depth, int *alpha, int *beta, int *v, int *n_etc_done) {
    *n_etc_done = 0;
    int l, u, n_beta = *alpha;
    for (Flip_value &flip_value: move_list) {
        l = -SCORE_MAX;
        u = SCORE_MAX;
        search->move(&flip_value.flip);
            if (transposition_table.has_node_any_level_get_bounds(search, search->board.hash(), depth - 1, &l, &u)) {
                flip_value.value = W_TT_BONUS;
            }
        search->undo(&flip_value.flip);
        if (*beta <= -u) { // alpha < beta <= -u <= -l
            *v = -u;
            return true; // fail high
        } else if (*alpha <= -u && -u < *beta) { // alpha <= -u <= beta <= -l or alpha <= -u <= -l <= beta
            *alpha = -u; // update alpha (alpha <= -u)
            *v = -u;
            if (-l <= *v || u == l) { // better move already found or this move is already done
                flip_value.flip.flip = 0ULL; // make this move invalid
                flip_value.value = -INF;
                ++(*n_etc_done);
            }
        } else if (-l <= *alpha) { // -u <= -l <= alpha < beta
            *v = std::max(*v, -l); // this move is worse than alpha
            flip_value.flip.flip = 0ULL; // make this move invalid
            flip_value.value = -INF;
            ++(*n_etc_done);
        }
    }
    return false;
}

/*
    @brief Enhanced Transposition Cutoff (ETC)

    @param hash_level_failed    hash level used when failed
    @param hash_level           new hash level
    @return hash resized?
*/
inline bool etc(Search *search, Flip_value move_list[], int canput, int depth, int *alpha, int *beta, int *v, int *n_etc_done) {
    *n_etc_done = 0;
    int l, u, n_beta = *alpha;
    for (int i = 0; i < canput; ++i) {
        l = -SCORE_MAX;
        u = SCORE_MAX;
        search->move(&move_list[i].flip);
            if (transposition_table.has_node_any_level_get_bounds(search, search->board.hash(), depth - 1, &l, &u)) {
                move_list[i].value = W_TT_BONUS;
            }
        search->undo(&move_list[i].flip);
        if (*beta <= -u) { // alpha < beta <= -u <= -l
            *v = -u;
            return true; // fail high
        } else if (*alpha <= -u && -u < *beta) { // alpha <= -u <= beta <= -l or alpha <= -u <= -l <= beta
            *alpha = -u; // update alpha (alpha <= -u)
            *v = -u;
            if (-l <= *v || u == l) { // better move already found or this move is already done
                move_list[i].flip.flip = 0ULL; // make this move invalid
                move_list[i].value = -INF;
                ++(*n_etc_done);
            }
        } else if (-l <= *alpha) { // -u <= -l <= alpha < beta
            *v = std::max(*v, -l); // this move is worse than alpha
            move_list[i].flip.flip = 0ULL; // make this move invalid
            move_list[i].value = -INF;
            ++(*n_etc_done);
        }
    }
    return false;
}

/*
    @brief Enhanced Transposition Cutoff (ETC) for NWS

    @param hash_level_failed    hash level used when failed
    @param hash_level           new hash level
    @return hash resized?
*/
inline bool etc_nws(Search *search, std::vector<Flip_value> &move_list, int depth, int alpha, int *v, int *n_etc_done) {
    *n_etc_done = 0;
    int l, u;
    for (Flip_value &flip_value: move_list) {
        l = -SCORE_MAX;
        u = SCORE_MAX;
        search->move(&flip_value.flip);
            if (transposition_table.has_node_any_level_get_bounds(search, search->board.hash(), depth - 1, &l, &u)) {
                flip_value.value = W_NWS_TT_BONUS;
            }
        search->undo(&flip_value.flip);
        if (alpha < -u) { // fail high at parent node
            *v = -u;
            return true;
        }
        if (-alpha <= l) { // fail high at child node
            if (*v < -l)
                *v = -l;
            flip_value.flip.flip = 0ULL; // make this move invalid
            flip_value.value = -INF;
            ++(*n_etc_done);
        }
    }
    return false;
}


/*
    @brief Enhanced Transposition Cutoff (ETC) for NWS

    @param hash_level_failed    hash level used when failed
    @param hash_level           new hash level
    @return hash resized?
*/
inline bool etc_nws(Search *search, Flip_value move_list[], int canput, int depth, int alpha, int *v, int *n_etc_done) {
    *n_etc_done = 0;
    int l, u;
    for (int i = 0; i < canput; ++i) {
        l = -SCORE_MAX;
        u = SCORE_MAX;
        search->move(&move_list[i].flip);
            if (transposition_table.has_node_any_level_get_bounds(search, search->board.hash(), depth - 1, &l, &u)) {
                move_list[i].value = W_NWS_TT_BONUS;
            }
        search->undo(&move_list[i].flip);
        if (alpha < -u) { // fail high at parent node
            *v = -u;
            return true;
        }
        if (-alpha <= l) { // fail high at child node
            if (*v < -l)
                *v = -l;
            move_list[i].flip.flip = 0ULL; // make this move invalid
            move_list[i].value = -INF;
            ++(*n_etc_done);
        }
    }
    return false;
}