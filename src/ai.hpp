#pragma once
#include <iostream>
#include <future>
#include "level.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "book.hpp"

#define search_final_define 100
#define search_book_define -1

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
    //cerr << "level status " << level << " " << b.n - 3 << " " << depth1 << " " << depth2 << " " << use_mpc << " " << mpct << endl;
    if (b.n >= hw2 - depth2)
        res = endsearch_value_nomemo(b, tim(), use_mpc, mpct).value;
    else
        res = midsearch_value_nomemo(b, tim(), depth1 + 1, use_mpc, mpct).value;
    return res;
}

int ai_value_memo(board b, int level, int pre_calc_value){
    int depth1, depth2;
    bool use_mpc;
    double mpct;
    int res;
    get_level(level, b.n - 3, &depth1, &depth2, &use_mpc, &mpct);
    //cerr << "level status " << level << " " << b.n - 3 << " " << depth1 << " " << depth2 << " " << use_mpc << " " << mpct << endl;
    if (b.n >= hw2 - depth2)
        res = endsearch_value_memo(b, tim(), use_mpc, mpct, pre_calc_value).value;
    else
        res = midsearch_value_memo(b, tim(), depth1 + 1, use_mpc, mpct).value;
    return res;
}

void ai_hint(board b, int level, int res[], int info[], unsigned long long legal){
    mobility mob;
    board nb;
    future<int> val_future[hw2];
    int pre_calc_values[hw2];
    unsigned long long searched_nodes = 0;
    int depth1, depth2;
    bool use_mpc;
    double mpct;
    get_level(level, b.n - 4, &depth1, &depth2, &use_mpc, &mpct);
    transpose_table.init_now();
    for (int i = 0; i < hw2; ++i){
        if (1 & (legal >> i)){
            calc_flip(&mob, &b, i);
            b.move_copy(&mob, &nb);
            pre_calc_values[i] = -mtd(&nb, false, depth1 / 2, -hw2, hw2, true, 0.35, &searched_nodes);
        }
    }
    swap(transpose_table.now, transpose_table.prev);
    transpose_table.init_now();
    for (int i = 0; i < hw2; ++i){
        if (1 & (legal >> i)){
            calc_flip(&mob, &b, i);
            b.move_copy(&mob, &nb);
            res[i] = book.get(&nb);
            if (res[i] == -inf){
                val_future[i] = async(launch::async, ai_value_memo, nb, level, pre_calc_values[i]);
                if (b.n >= hw2 - depth2 && !use_mpc)
                    info[i] = search_final_define;
                else
                    info[i] = level;
            } else
                info[i] = search_book_define;
        }
    }
    for (int i = 0; i < hw2; ++i){
        if (1 & (legal >> i)){
            if (res[i] == -inf)
                res[i] = -val_future[i].get();
        }
    }
}