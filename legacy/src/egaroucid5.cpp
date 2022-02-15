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
//#include "human_value.hpp"

inline void init(){
    mobility_init();
    transpose_table_init();
    evaluate_init();
    #if !MPC_MODE && !EVAL_MODE && !BOOK_MODE && USE_BOOK
        book_init();
    #endif
    #if !MPC_MODE && !EVAL_MODE && !BOOK_MODE
        //human_value_init();
    #endif
}

inline vector<int> input_board(board *b, int ai_player){
    int i;
    char elem;
    int arr[hw2];
    vector<int> vacant_lst;
    for (i = 0; i < hw2; ++i){
        cin >> elem;
        if (elem == '.'){
            arr[i] = vacant;
            vacant_lst.emplace_back(hw2_m1 - i);
        } else
            arr[i] = (int)elem - (int)'0';
    }
    b->translate_from_arr(arr, ai_player);
    if (vacant_lst.size() >= 2)
        sort(vacant_lst.begin(), vacant_lst.end(), cmp_vacant);
    return vacant_lst;
}

inline double calc_result_value(int v){
    return v;
    //return (double)round((double)v * hw2 / sc_w * 100) / 100.0;
}

inline void print_result(int policy, int value){
    cout << (hw_m1 - policy / hw) << " " << (hw_m1 - policy % hw) << " " << calc_result_value(value) << endl;
}

inline void print_result(search_result result){
    cout << (hw_m1 - result.policy / hw) << " " << (hw_m1 - result.policy % hw) << " " << calc_result_value(result.value) << endl;
}

int main(){
    init();
    board b;
    vector<int> vacant_lst;
    #if !MPC_MODE && !EVAL_MODE
        search_result result;
        int level = 30, book_error = 0;
    #endif
    #if USE_MULTI_THREAD
        thread_pool.resize(8);
        cerr << "use " << thread_pool.size() << " threads" << endl;
    #endif
    int ai_player;
    //cin >> ai_player;
    //cin >> depth;
    //cin >> end_depth;
    while (true){
        #if MPC_MODE
            cin >> ai_player;
            int max_depth;
            //cin >> max_depth;
            vacant_lst = input_board(&b, ai_player);
            transpose_table.init_now();
            transpose_table.init_prev();
            bool use_mpc = true; //max_depth >= 11 ? true : false;
            double use_mpct = 1.0;
            unsigned long long searched_nodes = 0;
            //cout << mtd(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct, &searched_nodes, vacant_lst) << endl;
            cout << mid_evaluate(&b) << endl;
            /*
            double use_mpct = 0.3;
            max_depth = hw2 - b.n;
            cerr << "start! depth " << max_depth << endl;
            int g = mtd_final(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct, mid_evaluate(&b), &searched_nodes);
            cerr << g << endl;
            cout << g << endl;
            */
        #elif EVAL_MODE
            cin >> ai_player;
            input_board(&b, ai_player);
            cout << calc_canput(&b, ai_player) << " " << calc_surround(&b, black) << " " << calc_surround(&b, white) << endl;
        #else
            cin >> ai_player;
            vacant_lst = input_board(&b, ai_player);
            //cerr << b.p << endl;
            //cerr << b.n << " " << mid_evaluate(&b) << endl;
            //search_human(b, tim(), depth, 7);
            result = ai(b, level, book_error, vacant_lst);
            print_result(result);
        #endif
    }
    return 0;
}