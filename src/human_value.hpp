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

struct Human_value{
    int moves;
    int prospect;
    double stability_black;
    double stability_white;
};

inline int get_human_sense_raw_value(Board *b, Search *search, int search_depth, bool passed){
    int val = -book.get(b);
    if (val == INF){
        search->board = b->copy();
        val = value_to_score_int(nega_alpha_ordering_nomemo(search, -SCORE_MAX, SCORE_MAX, search_depth, passed, LEGAL_UNDEFINED));
    }
    return val;
}

void calc_human_value_stability(Board *b, int depth, bool passed, int search_depth, Search *search, double res[], int searched_times[]){
    if (!global_searching || depth == 0)
        return;
	uint64_t legal = b->get_legal();
    if (!legal){
        if (!passed){
            b->pass();
                calc_human_value_stability(b, depth, true, search_depth, search, res, searched_times);
            b->pass();
        }
        return;
    }
	Flip flip;
    const int canput = pop_count_ull(legal);
    vector<int> values;
    int val, max_val = -INF;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, b, cell);
        b->move(&flip);
            calc_human_value_stability(b, depth - 1, false, search_depth, search, res, searched_times);
            val = -get_human_sense_raw_value(b, search, search_depth, passed);
        b->undo(&flip);
        max_val = max(max_val, val);
        values.emplace_back(val);
    }
    double v = 0.0;
    if (max_val >= 0){
        for (const int &value: values)
            v += max(0.0, (double)value);
        v /= canput;
    } else{
        for (const int &value: values)
            v += (double)(value + HW2);
        v /= canput;
    }
    res[b->p == WHITE] += v;
    ++searched_times[b->p == WHITE];
}

void calc_all_human_value(Board b, int depth, Human_value res[], int search_depth) {
    uint64_t legal = b.get_legal();
	Flip flip;
    double values[2];
    int searched_times[2];
    Search search;
    search.mpct = 1.5;
    search.use_mpc = true;
    search.n_nodes = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &b, cell);
        res[cell].moves = b.n - 4;
        values[0] = 0.0;
        values[1] = 0.0;
        searched_times[0] = 0;
        searched_times[1] = 0;
        b.move(&flip);
            calc_human_value_stability(&b, depth - 1, false, search_depth, &search, values, searched_times);
        b.undo(&flip);
        if (searched_times[0])
            res[cell].stability_black = values[0] / searched_times[0];
        if (searched_times[1])
            res[cell].stability_white = values[1] / searched_times[1];
        cerr << idx_to_coord(cell) << " " << res[cell].stability_black << " " << res[cell].stability_white << endl;
    }
}

void update_human_value_stone_values(Human_value res[], uint64_t legal, const int stone_values[]){
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
        res[cell].prospect = stone_values[cell];
}

Human_value calc_human_value(Board b, int depth, int search_depth, int calculated_value){
    double values[2] = {0.0, 0.0};
    int searched_times[2] = {0, 0};
    Search search;
    search.mpct = 1.5;
    search.use_mpc = true;
    search.n_nodes = 0;
    calc_human_value_stability(&b, depth - 1, false, search_depth, &search, values, searched_times);
    Human_value res;
    res.moves = b.n - 4;
    res.prospect = calculated_value;
    if (searched_times[0])
        res.stability_black = values[0] / searched_times[0];
    if (searched_times[1])
        res.stability_white = values[1] / searched_times[1];
    cerr << "human sense values " << res.stability_black << " " << res.stability_white << endl;
    return res;
}