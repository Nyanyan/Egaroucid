/*
    Egaroucid Project

    @file etc.hpp
        Enhanced Transposition Cutoff
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/
#pragma once
#include "setting.hpp"
#include "search.hpp"
#include "transposition_table.hpp"
#include "move_ordering.hpp"

/*
    @brief Enhanced Transposition Cutoff (ETC)

    @param hash_level_failed    hash level used when failed
    @param hash_level           new hash level
    @return hash resized?
*/
inline bool etc(Search *search, std::vector<Flip_value> &move_list, int depth, int *alpha, int *beta, int *v, int *etc_done_idx){
    *etc_done_idx = 0;
    int l, u, n_beta = *alpha;
    for (Flip_value &flip_value: move_list){
        l = -SCORE_MAX;
        u = SCORE_MAX;
        search->move(&flip_value.flip);
            transposition_table.get(search, search->board.hash(), depth - 1, &l, &u);
        search->undo(&flip_value.flip);
        if (*beta <= -u){ // fail high at parent node
            *v = -u;
            return true;
        }
        if (-l <= *alpha){ // fail high at child node
            if (*v < -l)
                *v = -l;
            flip_value.flip.flip = 0ULL; // make this move invalid
            ++(*etc_done_idx);
        } else if (*alpha < -u && -u < *beta){ // child window is [-beta, u]
            if (*v < -u)
                *v = -u;
            *alpha = -u;
            if (u == l){
                flip_value.flip.flip = 0ULL; // make this move invalid
                ++(*etc_done_idx);
            }
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
inline bool etc_nws(Search *search, std::vector<Flip_value> &move_list, int depth, int alpha, int *v, int *etc_done_idx){
    *etc_done_idx = 0;
    int l, u;
    for (Flip_value &flip_value: move_list){
        l = -SCORE_MAX;
        u = SCORE_MAX;
        search->move(&flip_value.flip);
            transposition_table.get(search, search->board.hash(), depth - 1, &l, &u);
        search->undo(&flip_value.flip);
        if (alpha < -u){ // fail high at parent node
            *v = -u;
            return true;
        }
        if (-alpha <= l){ // fail high at child node
            if (*v < -l)
                *v = -l;
            flip_value.flip.flip = 0ULL; // make this move invalid
            ++(*etc_done_idx);
        }
    }
    return false;
}
