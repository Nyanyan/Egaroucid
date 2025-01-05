#include <iostream>
#include <string>
#include "../../engine/engine_all.hpp"

void full_book_init(){
    thread_pool.resize(32);
    bit_init();
    mobility_init();
    flip_init();
    book_hash_init_rand();
    book.delete_all();
}

std::unordered_map<Board, int, Book_hash> data;

void load_data(std::string data_dir) {
    std::string file_name, line;
    std::string board_str, val_str;
    int val;
    Board board;
    std::cerr << "loading data" << std::endl;
    for (int file_idx = 0; file_idx < 1000000; ++file_idx) {
        if (file_idx % 100 == 99) {
            std::cerr << file_idx << std::endl;
        }
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(7) << file_idx << ".txt";
        file_name = ss.str();
        
        std::ifstream ifs(data_dir + "/" + file_name);
        if (!ifs) {
            break;
        }
        int n_data_file = 0;
        while (getline(ifs, line)) {
            board_str = line.substr(0, 66);
            board.from_str(board_str);
            board = get_representative_board(board);
            val_str = line.substr(67);
            val = stoi(val_str);
            data[board] = val;
            ++n_data_file;
        }
        //std::cerr << file_name << " " << n_data_file << " data found" << std::endl;
    }
    std::cerr << data.size() << " data found" << std::endl;
}

void generate_full_book(Board board, int depth, int level, bool passed) {
    if (board.n_discs() > 4) {
        if (book.contain(&board)) { // already searched
            return;
        }
    }
    //board.print();
    //std::cerr << std::endl;
    Book_elem book_elem;
    book_elem.level = level;
    uint64_t legal = board.get_legal();
    if (legal == 0) { // pass or game over
        if (passed) { // game over
            //book_elem.value = board.score_player();
            //book_elem.level = MAX_LEVEL;
            //book.reg(&board, book_elem);
            board.pass();
            book_elem.value = board.score_player();
            book_elem.level = MAX_LEVEL;
            book.reg(&board, book_elem);
            return;
        } else {
            board.pass();
                generate_full_book(board, depth, level, true);
            board.pass();
            return;
        }
    }
    if (depth == 0) { // leaf
        Board unique_board = get_representative_board(board);
        if (data.find(unique_board) == data.end()) { // no data found
            std::cerr << "[ERROR] " << board.to_str() << std::endl;
        } else {
            book_elem.value = data[unique_board];
            book.reg(&board, book_elem);
        }
        return;
    }
    book_elem.value = -64;
    book.reg(&board, book_elem); // register data (no value)
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
            generate_full_book(board, depth - 1, level, false);
        board.undo_board(&flip);
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
    generate_full_book(board, depth, level, false);
    std::cerr << "generated" << std::endl;
    book.fix();
    std::cerr << "fixed" << std::endl;
    book.save_egbk3("data/book.egbk3", "data/book.egbk3.bak");
    std::cerr << "saved" << std::endl;
}