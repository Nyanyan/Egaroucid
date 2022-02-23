#include <iostream>
#include <algorithm>
#include "setting.hpp"
#include "common.hpp"
#include "mobility.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "level.hpp"
#include "book.hpp"
#include "ai.hpp"
#include "umigame.hpp"
#include "human_value.hpp"
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

inline void print_result(int policy, int value){
    cout << (HW_M1 - policy / HW) << " " << (HW_M1 - policy % HW) << " " << value << endl;
}

inline void print_result(Search_result result){
    cout << (HW_M1 - result.policy / HW) << " " << (HW_M1 - result.policy % HW) << " " << result.value << endl;
}

int main(){
    cerr << "start!" << endl;
    #if USE_MULTI_THREAD
        thread_pool.resize(8);
    #endif
    board_init();
    mobility_init();
    evaluate_init();
    #if USE_BOOK
        book_init();
    #endif
    //move_ordering_init();
    parent_transpose_table.init();
    child_transpose_table.init();
    cerr << "initialized" << endl;
    Board b;
    int ai_player;
    const int level = 21;
    const int book_error = 0;
    #if MPC_MODE
        int depth;
        Search search;
        while (true){
            cin >> ai_player;
            search.vacant_list = input_board(&search.board, ai_player);
            cin >> depth;
            search.n_nodes = 0;
            search.skipped = false;
            search.tt_child_idx = 0;
            search.tt_parent_idx = 0;
            cin >> search.use_mpc;
            cin >> search.mpct;
            cout << nega_alpha_ordering_nomemo(&search, -HW2, HW2, depth) << endl;
        }
    #elif EVAL_TEST_MODE
        int depth;
        int best_move;
        unsigned long long legal;
        int score, best_score;
        Search search;
        Mobility mob;
        while (true){
            //cin >> depth;
            search.vacant_list = input_board(&search.board, BLACK);
            search.n_nodes = 0;
            search.skipped = false;
            search.tt_child_idx = 0;
            search.tt_parent_idx = 0;
            search.use_mpc = false;
            search.mpct = 1000000.0;
            best_move = -1;
            legal = search.board.mobility_ull();
            best_score = -INF;
            for (const int &cell: search.vacant_list){
                if (1 & (legal >> cell)){
                    calc_flip(&mob, &search.board, cell);
                    search.board.move(&mob);
                        score = -mid_evaluate(&search.board);
                    search.board.undo(&mob);
                    if (score > best_score){
                        best_score = score;
                        best_move = cell;
                    }
                }
            }
            cout << HW2_M1 - best_move << endl;
        }
    #else
        while (true){
            #if MOVE_ORDERING_ADJUST
                move_ordering_init();
            #endif
            cin >> ai_player;
            vector<int> vacant_lst = input_board(&b, ai_player);
            b.print();
            #if USE_LOG
                set_timer();
            #endif
            Search_result res = ai(b, level, book_error, vacant_lst);
            #if USE_LOG
                return 0;
            #endif
            print_result(res);
            //return 0;
        }
    #endif
    return 0;
}