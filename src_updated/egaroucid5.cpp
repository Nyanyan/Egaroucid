#include <iostream>
#include <algorithm>
#include "setting.hpp"
#include "common.hpp"
#include "mobility.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "midsearch.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
#endif

inline vector<int> input_board(Board *b, int ai_player){
    int i;
    char elem;
    int arr[HW2];
    vector<int> vacant_lst;
    for (i = 0; i < HW2; ++i){
        cin >> elem;
        if (elem == '.'){
            arr[i] = VACANT;
            vacant_lst.emplace_back(HW2_M1 - i);
        } else
            arr[i] = (int)elem - (int)'0';
    }
    b->translate_from_arr(arr, ai_player);
    if (vacant_lst.size() >= 2)
        sort(vacant_lst.begin(), vacant_lst.end(), cmp_vacant);
    return vacant_lst;
}

int main(){
    cerr << "start!" << endl;
    #if USE_MULTI_THREAD
        thread_pool.resize(8);
    #endif
    mobility_init();
    evaluate_init();
    parent_transpose_table.init();
    child_transpose_table.init();
    cerr << "initialized" << endl;
    Board b;
    int ai_player;
    while (true){
        cin >> ai_player;
        vector<int> vacant_lst = input_board(&b, ai_player);
        b.print();
        #if USE_LOG
            set_timer();
        #endif
        Search_result res = tree_search(b, 60, false, 0.0, vacant_lst);
        cerr << res.policy << " " << res.value << endl;
        #if USE_LOG
            return 0;
        #endif
    }
    return 0;
}