#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "book.hpp"
#include "evaluate.hpp"
#include "midsearch.hpp"
#include "util.hpp"

#define human_sense_value_weight1 0.01
#define human_sense_value_weight2 1.0

struct Human_value{
    int moves;
    int prospects;
    double stability_black;
    double stability_white;
};

void calc_human_value(Board *b, int depth, bool passed, bool is_black, int search_depth, Search *search, pair<double, double> &res){
    if (!global_searching)
        return;
    int val;
    double v;
    if (depth == 0){
        val = -book.get(b);
        if (val == INF)
            val = value_to_score_int(nega_scout(search, -SCORE_MAX, SCORE_MAX, search_depth, passed, LEGAL_UNDEFINED, false));
        v = human_sense_value_weight2 * exp(-human_sense_value_weight1 * val * val);
        if (is_black)
            res.first += v;
        else
            res.second += v;
        return;
    }
	uint64_t legal = b->get_legal();
    if (!legal){
        if (passed){
            val = -book.get(b);
            if (val == INF)
                val = value_to_score_int(nega_scout(search, -SCORE_MAX, SCORE_MAX, search_depth, passed, LEGAL_UNDEFINED, false));
            v = human_sense_value_weight2 * exp(-human_sense_value_weight1 * val * val);
            if (is_black)
                res.first += v;
            else
                res.second += v;
        } else{
            b->pass();
                calc_human_value(b, depth, true, !is_black, search_depth, search, res);
            b->pass();
        }
        return;
    }
	Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, b, cell);
        b->move(&flip);
            calc_human_value(b, depth - 1, false, !is_black, search_depth, search, res);
            val = value_to_score_int(nega_scout(search, -SCORE_MAX, SCORE_MAX, search_depth, passed, LEGAL_UNDEFINED, false));
        b->undo(&flip);
        v = human_sense_value_weight2 * exp(-human_sense_value_weight1 * val * val);
        if (is_black)
            res.first += v;
        else
            res.second += v;
    }
    return;
}

void calc_all_human_value(Board b, int depth, Human_value res[], int search_depth) {
    uint64_t legal = b.get_legal();
	Flip flip;
    pair<double, double> searched_res = make_pair(0.0, 0.0);
    Search search;
    search.mpct = 1.5;
    search.use_mpc = true;
    search.n_nodes = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &b, cell);
        searched_res.first = 0.0;
        searched_res.second = 0.0;
        b.move(&flip);
            calc_human_value(&b, depth - 1, false, b.p == BLACK, search_depth, &search, searched_res);
        b.undo(&flip);
        res[cell].moves = b.n - 4;
        res[cell].stability_black = searched_res.first;
        res[cell].stability_white = searched_res.second;
        cerr << idx_to_coord(cell) << " " << searched_res.first << " " << searched_res.second << endl;
    }
}

void update_human_value_stone_values(Human_value res[], uint64_t legal, const int stone_values[]){
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
        res[cell].prospects = stone_values[cell];
}