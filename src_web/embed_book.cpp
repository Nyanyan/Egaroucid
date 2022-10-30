#include <iostream>
#include <string>
#include <fstream>
#include "bookrw.hpp"

using namespace std;

#define BOOK_DEPTH 20

Book new_book;

void create_min_book(Board board, int player, const int ai_player){
    if (board.n_discs() > 4 + BOOK_DEPTH)
        return;
    int val = book.get(&board);
    if (val == -INF)
        return;
    new_book.reg(board, val);
    uint64_t legal = board.get_legal();
    if (legal == 0ULL){
        board.pass();
        player ^= 1;
        legal = board.get_legal();
        if (legal == 0ULL)
            return;
    }
    vector<uint_fast8_t> moves;
    Flip flip;
    if (player == ai_player){
        int best_val = -INF, val;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &board, cell);
            board.move_board(&flip);
                val = book.get(&board);
            board.undo_board(&flip);
            if (-64 <= val && val <= 64){
                if (best_val < val){
                    best_val = val;
                    moves.clear();
                }
                if (best_val == val)
                    moves.emplace_back(cell);
            }
        }
    } else{
        int val;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &board, cell);
            board.move_board(&flip);
                val = book.get(&board);
            board.undo_board(&flip);
            if (-64 <= val && val <= 64)
                moves.emplace_back(cell);
        }
    }
    player ^= 1;
    for (uint_fast8_t cell: moves){
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
            create_min_book(board, player, ai_player);
        board.undo_board(&flip);
    }
}

int main(int argc, char *argv[]){
    board_init();
    mobility_init();

    book_init(string(argv[1]));
    new_book.init();
    Board board;
    board.reset();
    new_book.reg(&board, 0);
    Flip flip;
    calc_flip(&flip, &board, 26);
    board.move_board(&flip);
    board.print();
    for (int player = 0; player < 2; ++player){
        create_min_book(board, WHITE, player);
        cerr << "min book size " << new_book.get_n_book() << endl;
    }
    //new_book.save_bin(string(argv[2]), "resources/bak.egbk");

    ifstream ifs("head.txt");
    string line;
    while (getline(ifs, line)){
        cout << line << endl;
    }

    cout << "#define N_EMBED_BOOK " << new_book.get_n_book() << endl;
    cout << "struct Embed_book{\n    uint64_t player;\n    uint64_t opponent;\n    int_fast8_t value;\n};\nEmbed_book embed_book[N_EMBED_BOOK] = {" << endl;
    new_book.save_cout();
    cout << "};\n";

    return 0;
}