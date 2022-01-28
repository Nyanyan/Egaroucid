#pragma once
#include <iostream>
#include "level.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "book.hpp"

struct cell_value {
	int value;
	int depth;
};

search_result ai(board b, int level, int book_error){
    search_result res;
    book_value book_result = book.get_random(&b, book_error);
    cerr << book_result.policy << " " << book_result.policy << endl;
    if (book_result.policy != -1){
        cerr << "BOOK " << book_result.policy << endl;
        res.policy = book_result.policy;
        res.value = book_result.value;
        res.depth = -1;
        res.nps = 0;
        return res;
    }
    int depth1, depth2;
    bool use_mpc;
    double mpct;
    get_level(level, b.n - 4, &depth1, &depth2, &use_mpc, &mpct);
    cerr << "level status " << level << " " << hw2 - b.n << " " << depth1 << " " << depth2 << " " << use_mpc << " " << mpct << endl;
    if (b.n >= hw2 - depth2 - 1)
        res = endsearch(b, tim(), use_mpc, mpct);
    else
        res = midsearch(b, tim(), depth1, use_mpc, mpct);
    return res;
}
