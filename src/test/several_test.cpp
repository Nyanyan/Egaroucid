#include <iostream>
#include "./../setting.hpp"
#include "./../bit.hpp"
#include "./../board.hpp"
#include "./../util.hpp"

using namespace std;

inline uint64_t or_dismiss_1(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t d){
    return (a | b | c) & (b | c | d) & (d | a | b);
}

inline uint64_t calc_legal_flip_inside(const uint64_t player, const uint64_t opponent){
    const uint64_t empty = ~(player | opponent);
    uint64_t masked = empty & 0x7E7E7E7E7E7E7E7EULL;
    const uint64_t shift1 = (masked << 1) | (masked >> 1);
    masked = empty & 0x00FFFFFFFFFFFF00ULL;
    const uint64_t shift8 = (masked << HW) | (masked >> HW);
    masked = empty & 0x007E7E7E7E7E7E00ULL;
    const uint64_t shift7 = (masked << HW_M1) | (masked >> HW_M1);
    const uint64_t shift9 = (masked << HW_P1) | (masked >> HW_P1);
    uint64_t outside_stones_mask = shift1 | shift8 | shift7 | shift9;
    outside_stones_mask &= or_dismiss_1(shift1, shift8, shift7, shift9);
    if ((opponent & ~outside_stones_mask) == 0ULL)
        return 0ULL;
    uint64_t legal = calc_legal(player, opponent & ~outside_stones_mask) & empty;
    if (legal == 0ULL)
        return 0ULL;
    return legal & ~calc_legal(player, opponent & outside_stones_mask);
}

int main(){
    Board board = input_board();
    cerr << "full board:" << endl;
    board.print();

    uint64_t legal_p = calc_legal_flip_inside(board.player, board.opponent);
    uint64_t legal_o = calc_legal_flip_inside(board.opponent, board.player);
    
    board.player = legal_p;
    board.opponent = legal_o;
    cerr << "legals of flipping inside:" << endl;
    board.print();

    return 0;
}