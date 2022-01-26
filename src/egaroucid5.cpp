#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "book.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#include "book.hpp"
#include "ai.hpp"
#include "human_value.hpp"

inline void init(){
    board_init();
    search_init();
    transpose_table_init();
    evaluate_init();
    #if !MPC_MODE && !EVAL_MODE && !BOOK_MODE && USE_BOOK
        book_init();
    #endif
    #if !MPC_MODE && !EVAL_MODE && !BOOK_MODE
        human_value_init();
    #endif
}

inline void input_board(board *b, int ai_player){
    int i, j;
    unsigned long long bk = 0, wt = 0;
    char elem;
    b->p = ai_player;
    b->n = 0;
    b->parity = 0;
    vacant_lst.clear();
    for (i = 0; i < hw; ++i){
        string raw_board;
        cin >> raw_board;
        cin.ignore();
        cerr << raw_board << endl;
        for (j = 0; j < hw; ++j){
            elem = raw_board[j];
            if (elem != '.'){
                bk |= (unsigned long long)(elem == '0') << (i * hw + j);
                wt |= (unsigned long long)(elem == '1') << (i * hw + j);
                ++b->n;
            } else{
                vacant_lst.push_back(i * hw + j);
                b->parity ^= cell_div4[i * hw + j];
            }
        }
    }
    if (b->n < hw2_m1)
        sort(vacant_lst.begin(), vacant_lst.end(), cmp_vacant);
    for (i = 0; i < b_idx_num; ++i){
        b->b[i] = n_line - 1;
        for (j = 0; j < idx_n_cell[i]; ++j){
            if (1 & (bk >> global_place[i][j]))
                b->b[i] -= pow3[hw_m1 - j] * 2;
            else if (1 & (wt >> global_place[i][j]))
                b->b[i] -= pow3[hw_m1 - j];
        }
    }
}

inline double calc_result_value(int v){
    return v;
    //return (double)round((double)v * hw2 / sc_w * 100) / 100.0;
}

inline void print_result(int policy, int value){
    cout << policy / hw << " " << policy % hw << " " << calc_result_value(value) << endl;
}

inline void print_result(search_result result){
    cout << result.policy / hw << " " << result.policy % hw << " " << calc_result_value(result.value) << endl;
}

int main(){
    init();
    board b;
    #if !MPC_MODE && !EVAL_MODE
        search_result result;
        int level = 30, book_error = 0;
    #endif
    #if USE_MULTI_THREAD
        thread_pool.resize(16);
    #endif
    int ai_player;
    //cin >> ai_player;
    //cin >> depth;
    //cin >> end_depth;
    while (true){
        #if MPC_MODE
            cin >> ai_player;
            int max_depth;
            cin >> max_depth;
            input_board(&b, ai_player);
            transpose_table.init_now();
            transpose_table.init_prev();
            bool use_mpc = max_depth >= 11 ? true : false;
            double use_mpct = 1.0;
            cout << mtd(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct) << endl;
        #elif EVAL_MODE
            cin >> ai_player;
            input_board(&b, ai_player);
            cout << calc_canput(&b, ai_player) << " " << calc_surround(&b, black) << " " << calc_surround(&b, white) << endl;
        #else
            cin >> ai_player;
            input_board(&b, ai_player);
            cerr << b.p << endl;
            cerr << b.n << " " << mid_evaluate(&b) << endl;
            //search_human(b, tim(), depth, 7);
            result = ai(b, level, book_error);
            print_result(result);
        #endif
    }
    return 0;
}