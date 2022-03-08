#include <iostream>
#include <string>
#include "./../midsearch.hpp"
#include "./../util.hpp"

using namespace std;


int main(){
    bit_init();
    flip_init();
    board_init();
    evaluate_init();
    parent_transpose_table.first_init();
    child_transpose_table.first_init();
    #if USE_MULTI_THREAD
        thread_pool.resize(16);
    #endif
    Search search;
    int depth, alpha, beta, g;

    while (true){
        search.board = input_board();
        search.n_nodes = 0;
        cerr << "0 move " << mid_evaluate(&search.board) << endl;
        //cerr << nega_alpha_eval1(&search, -HW2, HW2, false) << endl;

        depth = HW2 - search.board.n;
        child_transpose_table.init();

        parent_transpose_table.init();
        uint64_t strt = tim(), strt2 = tim(), search_time = 0ULL;
        search.mpct = 0.8;
        search.use_mpc = true;
        g = nega_scout(&search, -HW2, HW2, depth / 2, false, LEGAL_UNDEFINED, false);
        cerr << "presearch d=" << depth / 2 << " t=" << search.mpct << " [-64,64] " << g << " " << idx_to_coord(child_transpose_table.get(&search.board, search.board.hash() & TRANSPOSE_TABLE_MASK)) << endl;
        search_time += tim() - strt2;

        parent_transpose_table.init();
        strt2 = tim();
        search.mpct = 1.5;
        search.use_mpc = true;
        g = nega_scout(&search, -HW2, HW2, depth, false, LEGAL_UNDEFINED, true) / 2 * 2;
        cerr << "presearch d=" << depth << " t=" << search.mpct << " [-64,64] " << g << " " << idx_to_coord(child_transpose_table.get(&search.board, search.board.hash() & TRANSPOSE_TABLE_MASK)) << endl;
        search_time += tim() - strt2;

        if (depth >= 22){
            parent_transpose_table.init();
            strt2 = tim();
            search.mpct = 2.0;
            search.use_mpc = true;
            alpha = max(-HW2, g - 3);
            beta = min(HW2, g + 3);
            g = nega_scout(&search, alpha, beta, depth, false, LEGAL_UNDEFINED, true) / 2 * 2;
            cerr << "presearch d=" << depth << " t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(child_transpose_table.get(&search.board, search.board.hash() & TRANSPOSE_TABLE_MASK)) << endl;
            search_time += tim() - strt2;

            if (depth >= 24){
                parent_transpose_table.init();
                strt2 = tim();
                search.mpct = 2.2;
                search.use_mpc = true;
                alpha = max(-HW2, g - 1);
                beta = min(HW2, g + 1);
                g = nega_scout(&search, alpha, beta, depth, false, LEGAL_UNDEFINED, true) / 2 * 2;
                cerr << "presearch d=" << depth << " t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(child_transpose_table.get(&search.board, search.board.hash() & TRANSPOSE_TABLE_MASK)) << endl;
                search_time += tim() - strt2;
            }
        }

        cerr << search.n_nodes * 1000 / search_time << endl;

        parent_transpose_table.init();
        strt2 = tim();
        alpha = -INF;
        beta = -INF;
        while (g <= alpha || beta <= g){
            alpha = max(-HW2, g - 1);
            beta = min(HW2, g + 1);
            search.use_mpc = false;
            g = nega_scout(&search, alpha, beta, depth, false, LEGAL_UNDEFINED, true);
            cerr << "[" << alpha << "," << beta << "] " << g << " " << idx_to_coord(child_transpose_table.get(&search.board, search.board.hash() & TRANSPOSE_TABLE_MASK)) << endl;
            if (alpha == -HW2 && g == -HW2)
                break;
            if (beta == HW2 && g == HW2)
                break;
        }
        search_time += tim() - strt2;

        cerr << "depth " << depth << " value " << g << " policy " << idx_to_coord(child_transpose_table.get(&search.board, search.board.hash() & TRANSPOSE_TABLE_MASK)) << " nodes " << search.n_nodes << " whole time " << (tim() - strt) << " search time " << search_time << " nps " << search.n_nodes * 1000 / max(1ULL, search_time) << endl;
        cout << "depth " << depth << " value " << g << " policy " << idx_to_coord(child_transpose_table.get(&search.board, search.board.hash() & TRANSPOSE_TABLE_MASK)) << " nodes " << search.n_nodes << " whole time " << (tim() - strt) << " search time " << search_time << " nps " << search.n_nodes * 1000 / max(1ULL, search_time) << endl;
    }

    return 0;
}