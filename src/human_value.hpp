#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "book.hpp"
#include "evaluate.hpp"
#if USE_CUDA
	#include "cuda_midsearch.hpp"
#else
	#include "midsearch.hpp"
#endif
#include "util.hpp"

#define human_sense_a 0.78125
#define human_sense_b 0.02

struct Human_value{
    int moves;
    double stability_black;
    double stability_white;
};

inline int get_human_sense_raw_value(Board *b, Search *search, int search_depth, bool passed){
    int val = -book.get(b);
    if (val == INF){
        bool searching = true;
        search->board = b->copy();
        calc_features(search);
        val = value_to_score_int(nega_alpha_ordering_nomemo(search, -SCORE_MAX, SCORE_MAX, search_depth, passed, LEGAL_UNDEFINED, &searching));
    }
    return val;
}

inline double calc_human_sense_value(int v, int v_max){
    return human_sense_a * (double)(v + 64) * exp(-human_sense_b * (v_max - v));
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
    int v_max = -INF, v;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, b, cell);
        b->move(&flip);
            calc_human_value_stability(b, depth - 1, false, search_depth, search, res, searched_times);
            v = -get_human_sense_raw_value(b, search, search_depth, passed);
        b->undo(&flip);
        values.emplace_back(v);
        v_max = max(v_max, v);
    }
    double val = 0.0;
    for (const int &value: values)
        val += calc_human_sense_value(value, v_max);
    val /= canput;
    res[b->p == WHITE] += val;
    ++searched_times[b->p == WHITE];
}

void calc_all_human_value(Board b, int depth, Human_value res[], int search_depth) {
    uint64_t legal = b.get_legal();
	Flip flip;
    double values[2];
    int searched_times[2];
    int val;
    Search search;
    search.mpct = 1.5;
    search.use_mpc = true;
    search.n_nodes = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &b, cell);
        res[cell].moves = b.n - 4;
        val = get_human_sense_raw_value(&b, &search, search_depth, false) * (b.p ? -1 : 1);
        values[0] = (double)(val + HW2) * 99.99 / (HW2 * 2);
        values[1] = (double)(-val + HW2) * 99.99 / (HW2 * 2);
        searched_times[0] = 1;
        searched_times[1] = 1;
        b.move(&flip);
            calc_human_value_stability(&b, depth - 1, false, search_depth, &search, values, searched_times);
        b.undo(&flip);
        res[cell].stability_black = values[0] / searched_times[0];
        res[cell].stability_white = values[1] / searched_times[1];
        //cerr << idx_to_coord(cell) << " " << res[cell].stability_black << " " << res[cell].stability_white << endl;
    }
}

Human_value calc_human_value(Board b, int depth, int search_depth){
    Search search;
    search.mpct = 1.5;
    search.use_mpc = true;
    search.n_nodes = 0;
    int val = get_human_sense_raw_value(&b, &search, search_depth, false) * (b.p ? -1 : 1);
    double values[2] = {(double)(val + HW2) * 99.99 / (HW2 * 2), (double)(-val + HW2) * 99.99 / (HW2 * 2)};
    int searched_times[2] = {1, 1};
    calc_human_value_stability(&b, depth - 1, false, search_depth, &search, values, searched_times);
    Human_value res;
    res.moves = b.n - 4;
    res.stability_black = values[0] / searched_times[0];
    res.stability_white = values[1] / searched_times[1];
    cerr << "human sense values " << res.stability_black << " " << res.stability_white << endl;
    return res;
}