#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "book.hpp"
#include "evaluate.hpp"
#include "search_common.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "book.hpp"

inline void init(){
    board_init();
    search_init();
    evaluate_init();
    book_init();
}

inline int input_board(int board[]){
    int i, j;
    unsigned long long b = 0, w = 0;
    char elem;
    int n_stones = 0;
    vacant_lst.clear();
    for (i = 0; i < hw; ++i){
        string raw_board;
        cin >> raw_board; cin.ignore();
        cerr << raw_board << endl;
        for (j = 0; j < hw; ++j){
            elem = raw_board[j];
            if (elem != '.'){
                b |= (unsigned long long)(elem == '0') << (i * hw + j);
                w |= (unsigned long long)(elem == '1') << (i * hw + j);
                ++n_stones;
            } else{
                vacant_lst.push_back(i * hw + j);
            }
        }
    }
    if (n_stones < hw2_m1)
        sort(vacant_lst.begin(), vacant_lst.end(), cmp_vacant);
    for (i = 0; i < b_idx_num; ++i){
        board[i] = n_line - 1;
        for (j = 0; j < idx_n_cell[i]; ++j){
            if (1 & (b >> global_place[i][j]))
                board[i] -= pow3[hw_m1 - j] * 2;
            else if (1 & (w >> global_place[i][j]))
                board[i] -= pow3[hw_m1 - j];
        }
    }
    return n_stones;
}

inline double calc_result_value(int v){
    return (double)v * hw2 / sc_w;
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
    search_result result;
    const int first_moves[4] = {19, 26, 37, 44};
    int depth, end_depth, policy, ai_player;
    //cin >> ai_player;
    //cin >> depth;
    //cin >> end_depth;
    depth = 10;
    end_depth = 20;
    while (true){
        cin >> ai_player;
        //int d;
        //cin >> d;
        b.p = ai_player;
        b.n = input_board(b.b);
        /*
        transpose_table.init_now();
        transpose_table.init_prev();
        cout << nega_scout(&b, false, d, -sc_w, sc_w) << endl;
        continue;
        */
        cerr << b.n << endl;
        if (b.n == 4){
            policy = first_moves[myrandrange(0, 4)];
            print_result(policy, 0);
            continue;
        }
        if (b.n < book_stones){
            policy = book.get(&b);
            cerr << "book policy " << policy << endl;
            if (policy != -1){
                b = b.move(policy);
                ++b.n;
                result = midsearch(b, tim(), 10);
                print_result(policy, -result.value);
                continue;
            }
        }
        if (b.n >= hw2 - end_depth)
            result = endsearch(b, tim());
        else
            result = midsearch(b, tim(), depth);
        print_result(result);
    }
    return 0;
}