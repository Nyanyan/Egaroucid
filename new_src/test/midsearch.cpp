#include <iostream>
#include "./../midsearch.hpp"

using namespace std;

Board input_board(){
    Board res;
    char elem;
    int player;
    cin >> player;
    res.player = 0;
    res.opponent = 0;
    for (int i = 0; i < HW2; ++i){
        cin >> elem;
        if (elem == '0'){
            if (player == BLACK)
                res.player |= 1ULL << (HW2_M1 - i);
            else
                res.opponent |= 1ULL << (HW2_M1 - i);
        } else if (elem == '1'){
            if (player == WHITE)
                res.player |= 1ULL << (HW2_M1 - i);
            else
                res.opponent |= 1ULL << (HW2_M1 - i);
        }
    }
    res.p = player;
    res.parity = 0;
    res.n = HW2;
    uint64_t empties = ~(res.player | res.opponent);
    for (int i = 0; i < HW2; ++i){
        if (1 & (empties >> i)){
            res.parity ^= cell_div4[i];
            --res.n;
        }
    }
    res.print();
    return res;
}

int main(){
    bit_init();
    flip_init();
    board_init();
    evaluate_init();
    parent_transpose_table.first_init();
    child_transpose_table.first_init();
    Search search;
    int pre_search_value;
    int depth;

    while (true){
        search.board = input_board();
        search.n_nodes = 0;
        cerr << "0 move " << mid_evaluate(&search.board) << endl;
        //cerr << nega_alpha_eval1(&search, -HW2, HW2, false) << endl;
        uint64_t strt = tim();

        depth = HW2 - search.board.n;

        parent_transpose_table.init();
        child_transpose_table.ready_next_search();
        search.tt_child_idx = child_transpose_table.now_idx();
        search.mpct = 1.3;
        search.use_mpc = true;
        pre_search_value = nega_scout(&search, -HW2, HW2, depth, false, true);
        cerr << pre_search_value << endl;

        parent_transpose_table.init();
        child_transpose_table.ready_next_search();
        search.tt_child_idx = child_transpose_table.now_idx();
        search.use_mpc = false;
        cerr << nega_scout(&search, pre_search_value - 1, pre_search_value + 1, depth, false, true) << endl;
        cerr << search.n_nodes << " " << (tim() - strt) << " " << search.n_nodes * 1000 / (tim() - strt) << endl;
    }

    return 0;
}