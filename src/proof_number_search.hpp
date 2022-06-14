// refer to https://github.com/eukaryo/algorithm-study/blob/master/DFPN_single_file.cpp

#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <future>
#include <unordered_map>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#include "endsearch.hpp"
#include "move_ordering.hpp"
#include "probcut.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
    #include "ybwc.hpp"
#endif
#if USE_LOG
    #include "log.hpp"
#endif
#include "util.hpp"
#include "book.hpp"
#include "midsearch.hpp"

using namespace std;

#define INF_DF_PN 0x0100000000000000ULL
#define DF_PN_TO_NWS_DEPTH 20

struct Entry_df_pn{
    uint64_t proof;
    uint64_t disproof;
    //int lower;
    //int upper;
};

bool operator==(const Board& a, const Board& b){
    return a.player == b.player && a.opponent == b.opponent;
}

struct Hash_df_pn{
    size_t operator()(Board item) const{
        return item.hash();
    }
};

unordered_map<Board, Entry_df_pn, Hash_df_pn> table_df_pn;

bool get_table_df_pn(const Board &board, const int score_threshold, uint64_t *proof, uint64_t *disproof){
    if (table_df_pn.find(board) == table_df_pn.end()){
        *proof = 1;
        *disproof = 1;
        return false;
    }
    Entry_df_pn entry = table_df_pn[board];
    /*
    if (score_threshold < entry.lower){
        *proof = 0;
        *disproof = INF_DF_PN;
        return true;
    }
    if (entry.upper < score_threshold){
        *proof = INF_DF_PN;
        *disproof = 0;
        return true;
    }
    */
    *proof = entry.proof;
    *disproof = entry.disproof;
    return true;
}

void set_table_df_pn(const Board &board, const int score_threshold, const uint64_t proof, const uint64_t disproof){
    if (table_df_pn.find(board) == table_df_pn.end()){
        Entry_df_pn entry;
        /*
        entry.upper = INF;
        entry.lower = -INF;
        if (proof == INF_DF_PN)
            entry.upper = score_threshold;
        else if (disproof == INF_DF_PN)
            entry.lower = score_threshold + 1;
        */
        entry.disproof = disproof;
        entry.proof = proof;
        table_df_pn[board] = entry;
        return;
    }
    /*
    if (proof == INF_DF_PN)
        table_df_pn[board].upper = score_threshold;
    else if (disproof == INF_DF_PN)
        table_df_pn[board].lower = score_threshold;
    */
    table_df_pn[board].proof = proof;
    table_df_pn[board].disproof = disproof;
}

inline uint64_t add_df_pn(const uint64_t a, const uint64_t b){
    return min(a + b, INF_DF_PN);
}

inline uint64_t sub_df_pn(const uint64_t a, const uint64_t b){
    return (a == INF_DF_PN) ? a : (a - b);
}

inline uint64_t epsilon_trick(const uint64_t a){
    return min(a + 1 + (a >> 2), INF_DF_PN);
}

void df_pn_mid(Search *search, const uint64_t proof_number_threshold, const uint64_t disproof_number_threshold, const int score_threshold, bool passed, const int mul){
    ++search->n_nodes;
    uint64_t proof_number, disproof_number;
    /*
    if (HW2 - search->board.n <= DF_PN_TO_NWS_DEPTH){
        const bool searching = true;
        calc_features(search);
        const int score = nega_alpha_ordering(search, score_threshold - 1, score_threshold, HW2 - search->board.n, passed, LEGAL_UNDEFINED, true, &searching);
        if (score_threshold <= score){
            proof_number = 0;
            disproof_number = INF_DF_PN;
        } else{
            proof_number = INF_DF_PN;
            disproof_number = 0;
        }
        set_table_df_pn(search->board, score_threshold, proof_number, disproof_number);
        return;
    }
    */
    uint64_t legal = search->board.get_legal();
    if (legal == 0ULL){
        if (passed || search->board.n == HW2){
            const int score = mul * search->board.score_player();
            if (mul == 1){
                if (score_threshold <= score){
                    proof_number = 0;
                    disproof_number = INF_DF_PN;
                } else{
                    proof_number = INF_DF_PN;
                    disproof_number = 0;
                }
            } else{
                if (score_threshold > score){
                    proof_number = 0;
                    disproof_number = INF_DF_PN;
                } else{
                    proof_number = INF_DF_PN;
                    disproof_number = 0;
                }
            }
            set_table_df_pn(search->board, score_threshold, proof_number, disproof_number);
        } else{
            uint64_t p, d;
            search->board.pass();
                df_pn_mid(search, disproof_number_threshold, proof_number_threshold, -score_threshold + 1, true, mul);
                get_table_df_pn(search->board, -score_threshold + 1, &p, &d);
            search->board.pass();
            set_table_df_pn(search->board, score_threshold, d, p);
        }
        return;
    }
    const int canput = pop_count_ull(legal);
    vector<Flip> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
        calc_flip(&move_list[idx++], &search->board, cell);
    uint64_t min_disproof_children, sum_proof_children, phi_child, delta_2, delta_child, phi, delta, max_num, num, n_proof_number, n_disproof_number;
    int child_move;
    bool select_child_flag;
    Flip n_flip;
    while (true){
        min_disproof_children = INF_DF_PN;
        sum_proof_children = 0;
        phi_child = INF_DF_PN;
        delta_2 = 0;
        delta_child = INF_DF_PN;
        child_move = 0;
        max_num = 0;
        num = 0;
        select_child_flag = false;
        for (const Flip &flip: move_list){
            phi = 1;
            delta = 1;
            search->board.move(&flip);
                get_table_df_pn(search->board, -score_threshold + 1, &phi, &delta);
            search->board.undo(&flip);
            max_num = max(max_num, phi);
            min_disproof_children = min(min_disproof_children, delta);
            sum_proof_children = add_df_pn(sum_proof_children, phi);
            if (!select_child_flag){
                if (delta < delta_child){
                    delta_2 = delta_child;
                    phi_child = phi;
                    delta_child = delta;
                    child_move = flip.pos;
                } else if (delta < delta_2)
                    delta_2 = delta;
                if (phi == INF_DF_PN)
                    select_child_flag = true;
            }
        }
        //sum_proof_children = add_df_pn(max_num, canput - 1);
        // search parent node
        if (proof_number_threshold <= min_disproof_children || disproof_number_threshold <= sum_proof_children){
            set_table_df_pn(search->board, score_threshold, min_disproof_children, sum_proof_children);
            break;
        }
        // search child node
        calc_flip(&n_flip, &search->board, child_move);
        n_proof_number = sub_df_pn(add_df_pn(disproof_number_threshold, phi_child), sum_proof_children);
        n_disproof_number = min(proof_number_threshold, epsilon_trick(delta_2));
        search->board.move(&n_flip);
            df_pn_mid(search, n_proof_number, n_disproof_number, -score_threshold + 1, false, mul);
        search->board.undo(&n_flip);
    }
}

int df_pn_fail_high(Search *search, int alpha, int beta){
    table_df_pn.clear();
    df_pn_mid(search, INF_DF_PN - 1, INF_DF_PN - 1, beta, false, 1);
    cerr << table_df_pn.size() << endl;

    uint64_t phi = 0, delta = 0;
    get_table_df_pn(search->board, beta, &phi, &delta);
    if (delta == INF_DF_PN)
        return beta;
    else if (phi == INF_DF_PN)
        return alpha;
    return SCORE_UNDEFINED;
}

int df_pn_fail_low(Search *search, int alpha, int beta){
    table_df_pn.clear();
    df_pn_mid(search, INF_DF_PN - 1, INF_DF_PN - 1, -alpha, false, -1);
    cerr << table_df_pn.size() << endl;

    uint64_t phi = 0, delta = 0;
    get_table_df_pn(search->board, -alpha, &phi, &delta);
    cerr << phi << " " << delta << endl;
    if (delta == INF_DF_PN)
        return beta;
    else if (phi == INF_DF_PN)
        return alpha;
    return SCORE_UNDEFINED;
}

