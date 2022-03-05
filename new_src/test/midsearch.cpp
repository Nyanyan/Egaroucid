#include <iostream>
#include <string>
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

string idx_to_coord(int idx){
    int y = HW_M1 - idx / HW;
    int x = HW_M1 - idx % HW;
    const string x_coord = "abcdefgh";
    return x_coord[x] + to_string(y + 1);
}

int main(){
    bit_init();
    flip_init();
    board_init();
    evaluate_init();
    parent_transpose_table.first_init();
    child_transpose_table.first_init();
    thread_pool.resize(16);
    Search search;
    int depth, alpha, beta, g;

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
        g = nega_scout(&search, -HW2, HW2, depth, false, true) / 2 * 2;
        cerr << "[-64,64] " << g << " " << idx_to_coord(child_transpose_table.get_now(&search.board, search.board.hash() & TRANSPOSE_TABLE_MASK)) << endl;

        parent_transpose_table.init();
        child_transpose_table.ready_next_search();
        search.tt_child_idx = child_transpose_table.now_idx();
        alpha = -INF;
        beta = -INF;
        while (g <= alpha || beta <= g){
            alpha = g - 1;
            beta = g + 1;
            search.use_mpc = false;
            g = nega_scout(&search, alpha, beta, depth, false, true);
            cerr << "[" << alpha << "," << beta << "] " << g << " " << idx_to_coord(child_transpose_table.get_now(&search.board, search.board.hash() & TRANSPOSE_TABLE_MASK)) << endl;
        }
        cerr << search.n_nodes << " " << (tim() - strt) << " " << search.n_nodes * 1000 / (tim() - strt) << endl;
    }

    return 0;
}