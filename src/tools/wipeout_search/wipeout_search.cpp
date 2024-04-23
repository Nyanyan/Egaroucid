#include "./../../engine/board.hpp"
#include "./../../engine/util.hpp"

void wipeout_init(){
    bit_init();
    mobility_init();
    flip_init();
}

void wipeout_search(Board *board, int depth, std::vector<int> &transcript){
    if (depth == 0){
        if (board->is_end() && (board->player == 0 || board->opponent == 0)){
            for (int &elem: transcript){
                std::cout << idx_to_coord(elem);
            }
            std::cout << std::endl;
            //board->print();
        }
        return;
    }
    uint64_t legal = board->get_legal();
    bool passed = false;
    if (legal == 0){
        passed = true;
        board->pass();
        legal = board->get_legal();
        if (legal == 0){
            board->pass();
            return;
        }
    }
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, board, cell);
        board->move_board(&flip);
        transcript.emplace_back(cell);
            wipeout_search(board, depth - 1, transcript);
        transcript.pop_back();
        board->undo_board(&flip);
    }
    if (passed){
        board->pass();
    }
}

int main(){
    wipeout_init();
    Board board;
    board.reset();
    Flip flip;
    calc_flip(&flip, &board, 26); // f5
    board.move_board(&flip);
    std::vector<int> transcript;
    transcript.emplace_back(26);
    for (int depth = 1; depth <= 30; ++depth){
        std::cout << "depth: " << depth << std::endl;
        wipeout_search(&board, depth - 1, transcript);
    }
    std::cerr << "done" << std::endl;
}