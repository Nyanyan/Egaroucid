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

double calc_human_value(board *b, int depth, bool passed, double a){
	if (!global_searching)
		return 0.0;
    if (depth == 0){
		int val = -book.get(b);
		if (val == inf){
			val = mid_evaluate(b);
			//unsigned long long n_nodes = 0;
			//val = nega_alpha(b, false, 3, -hw2, hw2, &n_nodes);
		}
        double v = tanh(a * (double)val / (double)hw2);
		return v;
		//return val;
    }
    double res = 0.0;
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (passed){
            int val = -book.get(b);
			if (val == inf){
				val = mid_evaluate(b);
				//unsigned long long n_nodes = 0;
				//val = nega_alpha(b, false, 3, -hw2, hw2, &n_nodes);
			}
			double v = tanh(a * (double)val / (double)hw2);
			return v;
			//return val;
        }
        b->p = 1 - b->p;
        res = -calc_human_value(b, depth, true, a);
        b->p = 1 - b->p;
        return res;
    }
    mobility mob;
	int cell;
	for (cell = 0; cell < hw2; ++cell) {
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            b->move(&mob);
            //res = max(res, -calc_human_value(b, depth - 1, false, a));
			res += -calc_human_value(b, depth - 1, false, a);
            b->undo(&mob);
        }
    }
	//return res; // / (double)pop_count_ull(legal);
	return res / (double)pop_count_ull(legal);
}

void calc_all_human_value(board b, int depth, double a, int res[]) {
	unsigned long long legal = b.mobility_ull();
	int canput = pop_count_ull(legal);
	double double_res[hw2];
	mobility mob;
	int cell;
	for (cell = 0; cell < hw2; ++cell) {
		if (1 & (legal >> cell)) {
			calc_flip(&mob, &b, cell);
			b.move(&mob);
			double_res[cell] = -calc_human_value(&b, depth, false, a);
			b.undo(&mob);
			//cerr << cell << " " << double_res[cell] << endl;
		}
	}
	for (cell = 0; cell < hw2; ++cell) {
		if (1 & (legal >> cell)) {
			res[cell] = min(99, (int)((double_res[cell] + 1.0) * 50.0));
		}
	}
}
