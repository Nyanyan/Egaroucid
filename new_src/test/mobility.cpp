#include <iostream>
#include "./../mobility.hpp"

int main(){
    line_to_board_init();
    uint64_t p, o;
    uint8_t player;
    cin >> player;
    input_board(&p, &o);
    cerr << endl;
    print_board(p, o);
    cerr << endl;
    p = rotate_45(p);
    o = rotate_45(o);
    print_board(p, o);
    return 0;
    uint64_t strt = tim();
    //for (uint32_t i = 0; i < 1000000; ++i)
    uint64_t mobility = calc_mobility(p, o);
    bit_print_board(mobility);
    cerr << tim() - strt << endl;
    return 0;
}