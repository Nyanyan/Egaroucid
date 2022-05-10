#include <iostream>
#include "./../setting.hpp"
#include "./../bit.hpp"
#include "./../board.hpp"
#include "./../util.hpp"

using namespace std;

int main(){
    Board board = input_board();
    cerr << "full board:" << endl;
    board.print();

    uint64_t player = board.player;
    uint64_t opponent = board.opponent;

    const uint64_t empty = ~(player | opponent);
    uint64_t masked = empty & 0x007E7E7E7E7E7E00ULL;
    uint64_t outside_stones_mask = (masked << HW_M1) | (masked >> HW_M1);
    uint64_t shifted = (masked << HW_P1) | (masked >> HW_P1);
    uint64_t dismiss_tmp = outside_stones_mask ^ shifted;
    uint64_t dismiss = dismiss_tmp;
    outside_stones_mask |= shifted;
    masked = empty & 0x7E7E7E7E7E7E7E7EULL;
    shifted = (masked << 1) | (masked >> 1);
    outside_stones_mask |= shifted;
    dismiss_tmp ^= shifted;
    dismiss &= dismiss_tmp;
    masked = empty & 0x00FFFFFFFFFFFF00ULL;
    shifted = (masked << HW) | (masked >> HW);
    outside_stones_mask |= shifted;
    dismiss_tmp ^= shifted;
    dismiss &= dismiss_tmp;
    outside_stones_mask &= ~dismiss;
    
    board.player = player;
    board.opponent = opponent & ~outside_stones_mask;
    cerr << "inside:" << endl;
    board.print();

    board.player = player;
    board.opponent = opponent & outside_stones_mask;
    cerr << "outside:" << endl;
    board.print();

    return 0;
}