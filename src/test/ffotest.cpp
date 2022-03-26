#include <iostream>
#include <string>
#include "./../ai.hpp"
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
        thread_pool.resize(8);
    #endif
    Board b;
    Search_result result;
    uint64_t strt, elapsed;

    while (true){
        b = input_board();
        strt = tim();
        result = ai(b, 21, false, 0);
        elapsed = tim() - strt;
        cerr << "depth " << (HW2 - b.n) << " value " << result.value << " policy " << idx_to_coord(result.policy) << " nodes " << result.nodes << " time " << elapsed << " nps " << result.nodes * 1000 / max(1ULL, elapsed) << endl;
        cout << "depth " << (HW2 - b.n) << " value " << result.value << " policy " << idx_to_coord(result.policy) << " nodes " << result.nodes << " time " << elapsed << " nps " << result.nodes * 1000 / max(1ULL, elapsed) << endl;
    }

    return 0;
}