#include <iostream>
#include "./../setting.hpp"
#include "./../bit.hpp"
#include "./../board.hpp"
#include "./../util.hpp"
#include "./../flip_variation/stone.hpp"
#include "./../flip_variation/flipping.hpp"

using namespace std;

int main(){
    bit_init();
	flip_init();
	board_init();
    Board board;
    board = input_board();

    Flip flip;
    calc_flip(&flip, &board, 52);
    cerr << is_flip_2_stones(&board, &flip) << endl;

    board.move(&flip);
    board.print();

    return 0;
}