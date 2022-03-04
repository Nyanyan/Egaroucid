#include <iostream>
#include "./../bit.hpp"

int main(){
    uint64_t x;
    x = 376240289238932486;
    bit_print_board(x);
    bit_print_board(rotate_90(x));
    bit_print_board(rotate_270(x));
    bit_print_board(x);
    for (uint_fast8_t cell = first_bit(&x); x; cell = next_bit(&x))
        cerr << (int)cell << endl;
    return 0;
}