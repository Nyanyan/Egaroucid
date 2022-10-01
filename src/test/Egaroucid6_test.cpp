#include <iostream>
#include "./../ai.hpp"

using namespace std;
/*
int main(int argc, char* argv[]) {
    if (argc >= 2)
        thread_pool.resize(atoi(argv[1]));
    else
        thread_pool.resize(15);
    bit_init();
    board_init();
    stability_init();
    evaluate_init("resources/eval.egev");
    //book_init();
    parent_transpose_table.first_init();
    child_transpose_table.first_init();
    int level = 60;
    constexpr int book_error = 0;
    Board board;
    Search_result result;
    int estimated_score = 0;
    //Search search;
    //search.mpct = 10000;
    //search.use_mpc = false;
    while (true) {
        board = input_board();
        //board.player = 592232182158198784;
        //board.opponent = 131354417640484;
        //cerr << board.player << endl;
        //cerr << board.opponent << endl;
        
        //search.board = board;

        //calc_features(&search);

        //cerr << "value depth 0 " << mid_evaluate_diff(&search) << endl;

        result = ai(board, level, true, true, true);
        cout << "depth " << (HW2 - pop_count_ull(board.player | board.opponent)) << " value " << result.value << " policy " << idx_to_coord(result.policy) << " nodes " << result.nodes << " time " << result.time << " nps " << result.nps << endl;
        //return 0;
    }

    return 0;
}
*/

int main(int argc, char* argv[]) {
    thread_pool.resize(10);
    bit_init();
    board_init();
    stability_init();
    evaluate_init("resources/eval.egev");
    //book_init();
    parent_transpose_table.first_init();
    child_transpose_table.first_init();
    int level = 10;
    if (argc >= 2)
        level = atoi(argv[1]);
    cerr << "level " << level << endl;
    Board board;
    Search_result result;
    while (true) {
        board = input_board();
        result = ai(board, level, true, true, true);
        cout << result.value << " " << idx_to_coord(result.policy) << endl;
    }

    return 0;
}
