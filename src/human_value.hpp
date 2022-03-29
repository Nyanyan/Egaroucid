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

#define human_sense_value_weight 0.01

struct Human_value{
    int moves;
    int prospects;
    double stability_black;
    double stability_white;
};

pair<double, double> calc_human_value(Board *b, int depth, bool passed, bool is_black){
    if (!global_searching)
        return make_pair(0.0, 0.0);
    int val;
    double v;
    if (depth == 0){
        val = -book.get(b);
        if (val == INF)
            val = mid_evaluate(b);
        v = exp(-human_sense_value_weight * val * val);
        if (is_black)
            return make_pair(v, 0.0);
        else
            return make_pair(0.0, v);
    }
    pair<double, double> res = make_pair(0.0, 0.0);
	uint64_t legal = b->get_legal();
    if (!legal){
        if (passed){
            val = -book.get(b);
            if (val == INF)
                val = mid_evaluate(b);
            v = exp(-human_sense_value_weight * val * val);
            if (is_black)
                return make_pair(v, 0.0);
            else
                return make_pair(0.0, v);
        } else{
            b->pass();
                res = -calc_human_value(b, depth, true, !is_black);
            b->pass();
            return res;
        }
    }
	Flip flip;
    pair<double, double> next_res;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, b, cell);
        b->move(&flip);
            next_res = calc_human_value(b, depth - 1, false, !is_black);
        b->undo(&flip);
        res.first += next_res.first;
        res.second += next_res.second;
    }
    return res;
}

void calc_all_human_value(Board b, int depth, Human_value res[], uint64_t legal, const int stone_values[]) {
	Flip flip;
    pair<double, double> searched_res;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &b, cell);
        b.move(&flip);
            searched_res = calc_human_value(&b, depth - 1, false, !is_black);
        b.undo(&flip);
        res[cell].moves = b.n - 4;
        res[cell].stability_black = searched_res.first;
        res[cell].stability_white = searched_res.second;
        res[cell].prospects = stone_values[cell];
    }
}
