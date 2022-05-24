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
    Flip flip;
    board = input_board();

    uint64_t outside = calc_outside_stones(&board);
    uint64_t empties = ~(board.player | board.opponent);
    uint64_t face_stones = calc_face_stones(outside, empties);
    uint64_t end_stones = calc_end_stones(outside, empties);
    uint64_t bound_stones = calc_opponent_bound_stones(&board, outside);

    cerr << "outside" << endl;
    //bit_print_board(outside);
    cerr << "face_stones" << endl;
    //bit_print_board(face_stones);
    cerr << "end_stones" << endl;
    //bit_print_board(end_stones);
    cerr << "bound_stones" << endl;
    bit_print_board(bound_stones);


    calc_flip(&flip, &board, 52);
    cerr << is_thrust(&flip, face_stones) << is_turn_over(&flip, face_stones) << is_cut(&flip, outside) << is_cover(&flip, end_stones, face_stones) << is_pull(&flip, end_stones) << is_stab(&flip, end_stones, face_stones) << endl;

    calc_flip(&flip, &board, 53);
    cerr << is_thrust(&flip, face_stones) << is_turn_over(&flip, face_stones) << is_cut(&flip, outside) << is_cover(&flip, end_stones, face_stones) << is_pull(&flip, end_stones) << is_stab(&flip, end_stones, face_stones) << endl;

    calc_flip(&flip, &board, 25);
    cerr << is_thrust(&flip, face_stones) << is_turn_over(&flip, face_stones) << is_cut(&flip, outside) << is_cover(&flip, end_stones, face_stones) << is_pull(&flip, end_stones) << is_stab(&flip, end_stones, face_stones) << endl;

    calc_flip(&flip, &board, 38);
    cerr << is_thrust(&flip, face_stones) << is_turn_over(&flip, face_stones) << is_cut(&flip, outside) << is_cover(&flip, end_stones, face_stones) << is_pull(&flip, end_stones) << is_stab(&flip, end_stones, face_stones) << endl;

    calc_flip(&flip, &board, 30);
    cerr << is_thrust(&flip, face_stones) << is_turn_over(&flip, face_stones) << is_cut(&flip, outside) << is_cover(&flip, end_stones, face_stones) << is_pull(&flip, end_stones) << is_stab(&flip, end_stones, face_stones) << endl;

    calc_flip(&flip, &board, 22);
    cerr << is_thrust(&flip, face_stones) << is_turn_over(&flip, face_stones) << is_cut(&flip, outside) << is_cover(&flip, end_stones, face_stones) << is_pull(&flip, end_stones) << is_stab(&flip, end_stones, face_stones) << endl;

    return 0;
}