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

std::vector<std::pair<Board, int>> import_transcript(std::string transcript, int depth, int score_threshold) {
    bool passed = false;
    int x, y, coord;
    uint64_t legal;
    Board board;
    board.reset();
    int player = BLACK;
    std::vector<std::pair<Board, int>> res;
    Flip flip;
    for (int i = 0; i < (int)transcript.size(); i += 2) {
        if (is_pass_like_str(transcript.substr(i, 2)) && passed) {
            continue;
        }
        x = (int)(transcript[i] | 0x20) - (int)'a';
        y = (int)transcript[i + 1] - (int)'1';
        y = HW_M1 - y;
        x = HW_M1 - x;
        coord = y * HW + x;
        calc_flip(&flip, &board, coord);
        if (flip.flip == 0) {
            std::cerr << "illegal " << transcript[i] << transcript[i + 1] << std::endl;
            board.print();
        }
        board.move_board(&flip);
        player ^= 1;
        passed = false;
        if (board.get_legal() == 0ULL) {
            board.pass();
            player ^= 1;
            passed = true;
            if (board.get_legal() == 0ULL) {
                board.pass();
                player ^= 1;
            }
        }
        if (board.n_discs() <= depth + 4) {
            res.emplace_back(std::make_pair(board, player));
        }
    }
    //board.print();
    if (!board.is_end()) {
        res.clear();
        return res;
    }
    int score = board.score_player();
    if (abs(score) > score_threshold) {
        res.clear();
        return res;
    }
    for (std::pair<Board, int> &elem: res) {
        if (player == elem.second) {
            elem.second = score;
        } else {
            elem.second = -score;
        }
    }
    return res;
}

int main(int argc, char* argv[]){
    if (argc < 5){
        std::cerr << "input [transcript_file] [depth] [level] [score_threshold]" << std::endl;
        return 1;
    }

    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    endsearch_init();
    mpc_init();
    move_ordering_init();
    stability_init();

    book_hash_init_rand();

    std::string transcript_file = std::string(argv[1]);
    int depth = atoi(argv[2]);
    int level = atoi(argv[3]);
    int score_threshold = atoi(argv[4]);
    book.delete_all();
    std::ifstream ifs(transcript_file);
    std::string line;
    uint64_t t = 0;
    while (std::getline(ifs, line)) {
        std::vector<std::pair<Board, int>> boards = import_transcript(line, depth, score_threshold);
        for (std::pair<Board, int> &elem: boards) {
            book.change(&elem.first, elem.second, level);
        }
        ++t;
    }
    std::cerr << t << " " << book.size() << std::endl;
    book.fix(false);
    book.save_egbk3("output/book.egbk3", level);
    return 0;
}