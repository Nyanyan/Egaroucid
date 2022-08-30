#include <iostream>
#include <string>
#include "ai.hpp"
#include "util.hpp"

using namespace std;


int main() {
    thread_pool.resize(8);
    bit_init();
    board_init();
    evaluate_init("resources/eval.egev");
    parent_transpose_table.first_init();
    child_transpose_table.first_init();
    bak_parent_transpose_table.first_init();
    bak_child_transpose_table.first_init();
    Board board;
    Search_result res;
    while (true) {
        board = input_board();
        res = ai(board, 60, false, 0, true);
        cout << "depth " << res.depth << " value " << res.value << " policy " << idx_to_coord(res.policy) << " nodes " << res.nodes << " time " << res.time << " nps " << res.nps << endl;
    }

    return 0;
}