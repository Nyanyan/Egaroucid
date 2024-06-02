#include <iostream>
#include <string>
#include "engine/engine_all.hpp"

void board_to_line_init(std::string book_file){
    thread_pool.resize(32);
    bit_init();
    mobility_init();
    flip_init();
    book_init(book_file, true);
}

void board_to_line_print_transcript(std::vector<int> &transcript){
    for (int &move: transcript){
        std::cout << idx_to_coord(move);
    }
    std::cout << std::endl;
}

void board_to_line(Board board, const int depth, const int error_per_move, int remaining_error, std::vector<int> &transcript){
    if (board.n_discs() >= depth){
        board_to_line_print_transcript(transcript);
        return;
    }
    bool move_found = false;
    uint64_t legal = board.get_legal();
    Flip flip;
    Book_elem parent_elem = book.get(board);
    for (uint8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
        transcript.emplace_back(cell);
        bool passed = false;
        bool is_end = false;
        if (board.get_legal() == 0){
            passed = true;
            board.pass();
            if (board.get_legal() == 0){
                passed = false;
                is_end = true;
                board.pass();
            }
        }

            if (!is_end){
                if (book.contain(&board)){
                    Book_elem book_elem = book.get(board);
                    int val = -book_elem.value;
                    if (passed){
                        val *= -1;
                    }
                    int error = parent_elem.value - val;
                    int n_remaining_error = remaining_error - error;
                    if (error <= error_per_move && n_remaining_error >= 0){
                        move_found = true;
                        board_to_line(board, depth, error_per_move, n_remaining_error, transcript);
                    }
                }
            }

        if (passed){
            board.pass();
        }
        transcript.pop_back();
        board.undo_board(&flip);
    }
    if (!move_found){
        board_to_line_print_transcript(transcript);
    }
}

int main(int argc, char* argv[]){
    if (argc < 6){
        std::cerr << "input [book_file] [depth] [error_per_move] [error_sum] [transcript]" << std::endl;
        return 1;
    }
    std::string book_file = std::string(argv[1]);
    int depth = atoi(argv[2]);
    int error_per_move = atoi(argv[3]);
    int error_sum = atoi(argv[4]);
    std::string init_transcript = argv[5];
    board_to_line_init(book_file);
    std::vector<int> transcript;
    Board board;
    board.reset();
    Flip flip;
    for (int i = 0; i < init_transcript.size(); i += 2){
        int x = init_transcript[i] - 'a';
        int y = init_transcript[i + 1] - '1';
        int coord = 63 - (y * 8 + x);
        calc_flip(&flip, &board, coord);
        board.move_board(&flip);
        if (board.get_legal() == 0){
            board.pass();
        }
        transcript.emplace_back(coord);
    }
    board_to_line(board, depth, error_per_move, error_sum, transcript);
    return 0;
}