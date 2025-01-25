#include <iostream>
#include <string>
#include "../../engine/engine_all.hpp"

void board_to_line_init(std::string book_file){
    thread_pool.resize(32);
    bit_init();
    mobility_init();
    flip_init();
    book_init(book_file, false);
}

void board_to_line_print_transcript(std::vector<int> &transcript){
    for (int &move: transcript){
        std::cout << idx_to_coord(move);
    }
    std::cout << std::endl;
}

void board_to_line(Board board, int player, const int depth, const int error_per_moves[], int remaining_errors[], std::vector<int> &transcript){
    if (board.n_discs() >= depth + 4){
        board_to_line_print_transcript(transcript);
        return;
    }
    int before_player = player;
    bool move_found = false;
    uint64_t legal = board.get_legal();
    Flip flip;
    Book_elem parent_elem = book.get(board);
    for (uint8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
        player ^= 1;
        transcript.emplace_back(cell);
        bool passed = false;
        bool is_end = false;
        if (board.get_legal() == 0){
            passed = true;
            board.pass();
            player ^= 1;
            if (board.get_legal() == 0){
                passed = false;
                is_end = true;
                board.pass();
                player ^= 1;
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
                int n_remaining_error = remaining_errors[before_player] - error;
                if (error <= error_per_moves[before_player] && n_remaining_error >= 0){
                    move_found = true;
                }
                if (!book_elem.seen){
                    //book.flag_book_elem(board);
                    if (error <= error_per_moves[before_player] && n_remaining_error >= 0){
                        remaining_errors[before_player] -= error;
                            board_to_line(board, player, depth, error_per_moves, remaining_errors, transcript);
                        remaining_errors[before_player] += error;
                    }
                }
            }
        }

        if (passed){
            board.pass();
        }
        transcript.pop_back();
        board.undo_board(&flip);
        player ^= 1;
    }
    if (!move_found){
        board_to_line_print_transcript(transcript);
    }
}

int main(int argc, char* argv[]){
    if (argc < 8){
        std::cerr << "input [book_file] [depth] [error_per_move_black] [error_sum_black] [error_per_move_white] [error_sum_white] [init_transcript]" << std::endl;
        return 1;
    }
    std::string book_file = std::string(argv[1]);
    int depth = atoi(argv[2]);
    int error_per_move_black = atoi(argv[3]);
    int error_sum_black = atoi(argv[4]);
    int error_per_move_white = atoi(argv[5]);
    int error_sum_white = atoi(argv[6]);
    std::string init_transcript = argv[7];
    board_to_line_init(book_file);
    std::vector<int> transcript;
    Board board;
    board.reset();
    int player = BLACK;
    Flip flip;
    for (int i = 0; i < init_transcript.size(); i += 2){
        int x = init_transcript[i] - 'a';
        int y = init_transcript[i + 1] - '1';
        int coord = 63 - (y * 8 + x);
        calc_flip(&flip, &board, coord);
        board.move_board(&flip);
        player ^= 1;
        if (board.get_legal() == 0){
            board.pass();
            player ^= 1;
        }
        transcript.emplace_back(coord);
    }
    int error_per_moves[2] = {error_per_move_black, error_per_move_white};
    int error_sums[2] = {error_sum_black, error_sum_white};
    board_to_line(board, player, depth, error_per_moves, error_sums, transcript);
    return 0;
}