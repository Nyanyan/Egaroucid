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
    uint64_t strt, mobility;
    strt = tim();
    for (uint32_t i = 0; i < 10000000; ++i)
        mobility = calc_mobility(p, o);
    cerr << tim() - strt << endl;

    strt = tim();
    for (uint32_t i = 0; i < 10000000; ++i)
        mobility = get_mobility(p, o);
    cerr << tim() - strt << endl;
    
    return 0;
}