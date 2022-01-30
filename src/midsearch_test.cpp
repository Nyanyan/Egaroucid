#include <iostream>
#include <algorithm>
#include "setting.hpp"
#include "common.hpp"
#include "mobility.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "midsearch.hpp"

inline void input_board(board *b, int ai_player){
    int i;
    char elem;
    int arr[hw2];
    vacant_lst.clear();
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
        sort(vacant_lst.begin(), vacant_lst.end()); // , cmp_vacant
}

int main(){
    mobility_init();
    board_init();
    evaluate_init();
    transpose_table_init();
    #if USE_MULTI_THREAD
        thread_pool.resize(8);
    #endif
    board b;
    int ai_player;
    while (true){
        cin >> ai_player;
        input_board(&b, ai_player);
        b.print();
        search_result res = midsearch(b, tim(), 15, false, 0.0);
        cerr << res.policy << " " << res.value << endl;
    }
    return 0;
}