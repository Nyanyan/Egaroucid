/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

#include <iostream>
#include "ai.hpp"

inline void init(){
    board_init();
    mobility_init();
    stability_init();
    parent_transpose_table.first_init();
    child_transpose_table.first_init();
    evaluate_init("resources/eval.egev");
    book.init("resources/book.egbk");
}

inline int input_board(Board *bd, const int *arr, const int ai_player){
    int i, j;
    uint64_t b = 0ULL, w = 0ULL;
    int elem;
    int n_stones = 0;
    for (i = 0; i < hw; ++i){
        for (j = 0; j < hw; ++j){
            elem = arr[i * hw + j];
            if (elem != -1){
                b |= (uint64_t)(elem == 0) << (i * hw + j);
                w |= (uint64_t)(elem == 1) << (i * hw + j);
                ++n_stones;
            }
        }
    }
    if (ai_player == 0){
        bd->player = b;
        bd->opponent = w;
    } else{
        bd->player = w;
        bd->opponent = b;
    }
    return n_stones;
}

inline double calc_result_value(int v){
    return (double)v;
}

inline void print_result(int policy, int value){
    cout << policy / hw << " " << policy % hw << " " << calc_result_value(value) << endl;
}

inline void print_result(search_result result){
    cout << result.policy / hw << " " << result.policy % hw << " " << calc_result_value(result.value) << endl;
}

inline int output_coord(int policy, int raw_val){
    return 1000 * policy + 100 + raw_val;
}

extern "C" int initialize_ai(){
    cout << "initializing AI" << endl;
    init();
    cout << "AI iniitialized" << endl;
    return 0;
}

extern "C" int ai_js(int *arr_board, int level, int ai_player){
    cout << "start AI" << endl;
    int i, n_stones, policy;
    Board b;
    Search_result result;
    cout << endl;
    n_stones = input_board(&b, arr_board);
    b.print();
    cout << "ply " << n_stones - 3 << endl;
    result = ai_search(b, level, true, false, true);
    cout << "searched policy " << result.policy << " value " << result.value << " nps " << result.nps << endl;
    int res = output_coord(result.policy, result.value);
    cout << "res " << res << endl;
    return res;
}

extern "C" void calc_value(int *arr_board, int *res, int level, int ai_player){
    ai_player = 1 - ai_player;
    int i, n_stones, policy;
    board b;
    search_result result;
    n_stones = input_board(&b, arr_board);
    b.print();
    b.n = n_stones;
    b.p = ai_player;
    cout << n_stones - 4 << "moves" << endl;
    int tmp_res[hw2];
    vector<int> moves;
    uint64_t legal = b.mobility_ull();
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            moves.emplace_back(cell);
        }
    }
    for (i = 0; i < hw2; ++i)
        tmp_res[i] = -1;
    board nb;
    mobility mob;
    int depth1, depth2;
    bool use_mpc;
    double mpct;
    get_level(level, b.n - 3, &depth1, &depth2, &use_mpc, &mpct);
    uint64_t searched_nodes = 0;
    if (b.n >= hw2 - depth2 - 1){
        transpose_table.init_now();
        int g;
        for (const int &policy: moves){
            calc_flip(&mob, &b, policy);
            b.move_copy(&mob, &nb);
            g = -mtd(&nb, false, depth2 / 2, -hw2, hw2, use_mpc, mpct, &searched_nodes);
            tmp_res[policy] = -mtd_final(&nb, false, depth2, -hw2, hw2, use_mpc, mpct, g, &searched_nodes);
        }
    } else{
        transpose_table.init_now();
        for (int depth = min(3, max(0, depth1 - 4)); depth <= depth1; ++depth){
            for (const int &policy: moves){
                calc_flip(&mob, &b, policy);
                b.move_copy(&mob, &nb);
                tmp_res[policy] += -mtd(&nb, false, depth2, -hw2, hw2, use_mpc, mpct, &searched_nodes);
                tmp_res[policy] /= 2;
            }
        }
        swap(transpose_table.now, transpose_table.prev);
    }
    for (i = 0; i < hw2; ++i)
        res[10 + i] = max(-64, min(64, tmp_res[i]));
    for (int y = 0; y < hw; ++y){
        for (int x = 0; x < hw; ++x)
            cout << tmp_res[y * hw + x] << " ";
        cout << endl;
    }
    ai_player = 1 - ai_player;
}
