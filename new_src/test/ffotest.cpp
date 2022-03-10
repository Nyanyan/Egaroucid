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
    pair<int, int> result;

    while (true){
        search.board = input_board();
        /*
        cerr << search.board.player << endl;
        cerr << search.board.opponent << endl;
        cerr << (int)search.board.n << endl;
        cerr << (int)search.board.p << endl;
        cerr << (int)search.board.parity << endl;
        */
        search.n_nodes = 0;
        cerr << "0 move " << mid_evaluate(&search.board) << endl;
        //cerr << nega_alpha_eval1(&search, -HW2, HW2, false) << endl;

        depth = HW2 - search.board.n;
        child_transpose_table.init();

        uint64_t strt, strt2, search_time = 0ULL;

        parent_transpose_table.init();
        strt = tim();
        strt2 = tim();
        search.mpct = 0.6;
        search.use_mpc = true;
        result = first_nega_scout(&search, -HW2, HW2, depth / 3, false, false);
        g = result.first;
        cerr << "presearch d=" << depth / 3 << " t=" << search.mpct << " [-64,64] " << g << " " << idx_to_coord(result.second) << endl;
        search_time += tim() - strt2;

        if (depth >= 24){
            parent_transpose_table.init();
            strt2 = tim();
            search.mpct = 1.2;
            //search.mpct = 0.0;
            search.use_mpc = true;
            result = first_nega_scout(&search, -HW2, HW2, depth, false, true);
            g = result.first;
            cerr << "presearch d=" << depth << " t=" << search.mpct << " [-64,64] " << g << " " << idx_to_coord(result.second) << endl;
            search_time += tim() - strt2;

            if (depth >= 26){
                parent_transpose_table.init();
                strt2 = tim();
                search.mpct = 1.7;
                search.use_mpc = true;
                alpha = max(-HW2, g - 3);
                beta = min(HW2, g + 3);
                result = first_nega_scout(&search, alpha, beta, depth, false, true);
                g = result.first;
                cerr << "presearch d=" << depth << " t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(result.second) << endl;
                search_time += tim() - strt2;

                if (depth >= 28){
                    parent_transpose_table.init();
                    strt2 = tim();
                    search.mpct = 2.3;
                    search.use_mpc = true;
                    alpha = max(-HW2, g - 2);
                    beta = min(HW2, g + 2);
                    result = first_nega_scout(&search, alpha, beta, depth, false, true);
                    g = result.first;
                    cerr << "presearch d=" << depth << " t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(result.second) << endl;
                    search_time += tim() - strt2;
                }
            }
        }

        cerr << "presearch n_nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / search_time << endl;

        parent_transpose_table.init();
        search.use_mpc = false;
        strt2 = tim();
        alpha = -INF;
        beta = -INF;
        while (g <= alpha || beta <= g){
            if (g % 2){
                alpha = max(-HW2, g - 2);
                beta = min(HW2, g + 2);
            } else{
                alpha = max(-HW2, g - 1);
                beta = min(HW2, g + 1);
            }
            result = first_nega_scout(&search, alpha, beta, depth, false, true);
            g = result.first;
            cerr << "mainsearch d=" << depth << " t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(result.second) << endl;
            if (alpha == -HW2 && g == -HW2)
                break;
            if (beta == HW2 && g == HW2)
                break;
        }
        search_time += tim() - strt2;

        cerr << "depth " << depth << " value " << g << " policy " << idx_to_coord(result.second) << " nodes " << search.n_nodes << " whole time " << (tim() - strt) << " search time " << search_time << " nps " << search.n_nodes * 1000 / max(1ULL, search_time) << endl;
        cout << "depth " << depth << " value " << g << " policy " << idx_to_coord(result.second) << " nodes " << search.n_nodes << " whole time " << (tim() - strt) << " search time " << search_time << " nps " << search.n_nodes * 1000 / max(1ULL, search_time) << endl;
    }

    return 0;
}