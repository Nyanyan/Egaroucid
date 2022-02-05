#pragma once
#include <iostream>
#include "level.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "book.hpp"

search_result ai(board b, int level, int book_error){
    search_result res;
    book_value book_result = book.get_random(&b, book_error);
    if (book_result.policy != -1){
        cerr << "BOOK " << book_result.policy << " " << book_result.value << endl;
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
    cerr << "level status " << level << " " << b.n - 3 << " " << depth1 << " " << depth2 << " " << use_mpc << " " << mpct << endl;
    if (b.n >= hw2 - depth2)
        res = endsearch(b, tim(), use_mpc, mpct);
    else
        res = midsearch(b, tim(), depth1, use_mpc, mpct);
    return res;
}

int ai_value_nomemo(board b, int level){
    int depth1, depth2;
    bool use_mpc;
    double mpct;
    int res;
    get_level(level, b.n - 3, &depth1, &depth2, &use_mpc, &mpct);
    cerr << "level status " << level << " " << b.n - 3 << " " << depth1 << " " << depth2 << " " << use_mpc << " " << mpct << endl;
    if (b.n >= hw2 - depth2)
        res = endsearch_value_nomemo(b, tim(), use_mpc, mpct).value;
    else
        res = midsearch_value_nomemo(b, tim(), depth1 + 1, use_mpc, mpct).value;
    return res;
}

int ai_value_memo(board b, int level){
    int depth1, depth2;
    bool use_mpc;
    double mpct;
    int res;
    get_level(level, b.n - 3, &depth1, &depth2, &use_mpc, &mpct);
    cerr << "level status " << level << " " << b.n - 3 << " " << depth1 << " " << depth2 << " " << use_mpc << " " << mpct << endl;
    if (b.n >= hw2 - depth2)
        res = endsearch_value_memo(b, tim(), use_mpc, mpct).value;
    else
        res = midsearch_value_memo(b, tim(), depth1 + 1, use_mpc, mpct).value;
    return res;
}