#include <iostream>
#include "./../ai.hpp"

using namespace std;

int main(int argc, char *argv[]){
    bit_init();
    flip_init();
    board_init();
    evaluate_init();
    //book_init();
    parent_transpose_table.first_init();
    child_transpose_table.first_init();
    #if USE_MULTI_THREAD
        thread_pool.resize(8);
    #endif
    int level = 5;
    if (argc >= 2)
        level = atoi(argv[1]);
    constexpr int book_error = 0;
    Board board;
    Search_result result;
    while (true){
        board = input_board();
        result = ai(board, level, true, book_error);
        cout << result.value << " " << idx_to_coord(result.policy) << endl;
    }

    return 0;
}