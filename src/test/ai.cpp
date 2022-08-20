#include <iostream>
#include "./../ai.hpp"

using namespace std;

int main(int argc, char* argv[]) {
#if USE_MULTI_THREAD
    thread_pool.resize(16);
#endif
    bit_init();
    flip_init();
    board_init();
    evaluate_init("resources/eval.egev");
    //evaluate_init(argv[2]);
    book_init("resources/book.egbk");
    parent_transpose_table.first_init();
    child_transpose_table.first_init();
    int level = 21;
    if (argc >= 2)
        level = atoi(argv[1]);
    Board board;
    Search_result result;
    while (true) {
        board = input_board();
        result = ai(board, level, true, 0, true);
        cout << result.value << " " << idx_to_coord(result.policy) << endl;
    }

    return 0;
}