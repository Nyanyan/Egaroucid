#include <iostream>
#include "./../mobility.hpp"

int main(){
    line_to_board_init();
    uint64_t p, o;
    uint8_t player;
    //cin >> player;
    //input_board(&p, &o);
    //cerr << endl;
    //print_board(p, o);
    uint64_t strt, mobility;
    strt = tim();
    for (uint32_t i = 0; i < 1000000000; ++i){
        p = myrand_ull();
        o = myrand_ull() & (~p);
        mobility = calc_mobility(p, o);
    }
    //bit_print_board(mobility);
    //cerr << (mobility == get_mobility(p, o)) << endl;
    cerr << tim() - strt << endl;

    strt = tim();
    for (uint32_t i = 0; i < 1000000000; ++i){
        p = myrand_ull();
        o = myrand_ull() & (~p);
        mobility = get_mobility(p, o);
    }
    cerr << tim() - strt << endl;
    //bit_print_board(mobility);

    return 0;
}