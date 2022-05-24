#include <iostream>
#include "./../setting.hpp"
#include "./../bit.hpp"
#include "./../board.hpp"
#include "./../util.hpp"
#include "./../flip_variation/stone.hpp"

using namespace std;

int main(){
    Board board;
    /*
    board.p = 0;
    board.player = 9264090048841921536ULL;
    board.opponent = 4480051588954062904ULL;
    */
    board = input_board();
    board.print();
    
    uint64_t outside = calc_outside_stones(&board);
    bit_print_board(outside);
    cerr << endl;

    uint64_t player_wall = calc_wall_stones(outside & board.player);
    bit_print_board(player_wall);

    player_wall = calc_wall_stones(outside & board.opponent);
    bit_print_board(player_wall);

    return 0;
}