#include <iostream>
#include "./../flip.hpp"
#include "./../mobility.hpp"

int main(){
    bit_init();
    flip_init();
    uint64_t p, o;
    /*
    uint8_t player;
    cin >> player;
    input_board(&p, &o);
    cerr << endl;
    print_board(p, o);
    uint64_t mobility = calc_mobility(p, o);
    Flip flip;
    for (uint32_t i = 0; i < HW2; ++i){
        if (1 & (mobility >> i)){
            flip.calc_flip(p, o, i);
            cerr << i << endl;
            bit_print_board(flip.flip);
            cerr << endl;
        }
    }
    */


    uint64_t strt, mobility;
    strt = tim();
    uint32_t i, j;
    Flip flip;
    for (i = 0; i < 10000000; ++i){
        p = myrand_ull();
        o = myrand_ull() & (~p);
        mobility = calc_mobility(p, o);
        for (j = 0; j < HW2; ++j){
            if (1 & (mobility >> j)){
                flip.calc_flip(p, o, j);
            }
        }
    }
    cerr << tim() - strt << endl;


    return 0;
}