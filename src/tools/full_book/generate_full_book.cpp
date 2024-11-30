#include <iostream>
#include <string>
#include "../../engine/engine_all.hpp"

void full_book_init(){
    thread_pool.resize(32);
    bit_init();
    mobility_init();
    flip_init();
    book_init("data/empty_book.egbk3", true);
}

void board_to_line_print_transcript(std::vector<int> &transcript){
    for (int &move: transcript){
        std::cout << idx_to_coord(move);
    }
    std::cout << std::endl;
}

void board_to_line(Board board, const int depth, const int error_per_move, int remaining_error, std::vector<int> &transcript){
    if (board.n_discs() >= depth + 4){
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
                    }
                    if (!book_elem.seen){
                        //book.flag_book_elem(board);
                        if (error <= error_per_move && n_remaining_error >= 0){
                            board_to_line(board, depth, error_per_move, n_remaining_error, transcript);
                        }
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

std::unordered_map<Board, int, Book_hash> data;

void load_data(std::string data_dir) {
    std::stringstream ss;
    std::string file_name, line;
    std::string board_str, val_str;
    int val;
    Board board;
    for (int file_idx = 0; file_idx < 1000000; ++file_idx) {
        ss.clear();
        ss << std::setfill('0') << std::setw(7) << file_idx << ".txt";
        file_name = ss.str();
        std::ifstream ifs(file_name);
        if (!ifs) {
            break;
        }
        int n_data_file = 0;
        while (getline(ifs, line)) {
            board_str = line.substr(0, 66);
            val_str = line.substr(67);
            val = stoi(val_str);
            
            ++n_data_file;
        }
        std::cerr << file_name << std::endl;
    }
}

int main(int argc, char* argv[]){
    if (argc < 4){
        std::cerr << "input [depth] [level] [data_dir]" << std::endl;
        return 1;
    }
    int depth = atoi(argv[1]);
    int level = atoi(argv[2]);
    std::string data_dir = std::string(argv[3]);
    full_book_init();
    load_data(data_dir);
    Board board;
    board.reset();
    generate_full_book(board, depth, level);
}
