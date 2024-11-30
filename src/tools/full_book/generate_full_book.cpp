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
            board.from_str(board_str);
            val_str = line.substr(67);
            val = stoi(val_str);
            data[board] = val;
            ++n_data_file;
        }
        std::cerr << file_name << " " << n_data_file << " data found" << std::endl;
    }
    std::cerr << data.size() << " data found" << std::endl;
}

void generate_full_book(Board board, int depth, int level) {
    if (depth == 0) {
        if ()
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
    book.negamax_book();
}
