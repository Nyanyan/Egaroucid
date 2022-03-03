#include <iostream>
#include "./../evaluate.hpp"

using namespace std;

Board input_board(){
    Board res;
    char elem;
    cin >> res.p;
    res.player = 0;
    res.opponent = 0;
    res.parity = 0;
    res.n = HW2;
    for (int i = 0; i < HW2; ++i){
        cin >> elem;
        if (elem == '0'){
            if (res.p == BLACK)
                res.player |= 1ULL << (HW2_M1 - i);
            else
                res.opponent |= 1ULL << (HW2_M1 - i);
        } else if (elem == '1'){
            if (res.p == WHITE)
                res.player |= 1ULL << (HW2_M1 - i);
            else
                res.opponent |= 1ULL << (HW2_M1 - i);
        }
    }
    uint64_t empties = ~(res.player | res.opponent);
    for (int i = 0; i < HW2; ++i){
        if (1 & (empties >> i)){
            res.parity ^= cell_div4[i];
            --res.n;
        }
    }
    res.print();
    return res;
}

int main(){
    bit_init();
    flip_init();
    board_init();
    evaluate_init();
    Board board;

    while (true){
        board = input_board();
        cerr << mid_evaluate(&board) << endl;
    }

    return 0;
}