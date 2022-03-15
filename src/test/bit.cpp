#include <iostream>
#include "./../bit.hpp"

int main(){
    uint64_t x;
    x = 376240289238932486;
    bit_print_board(x);

    bit_print_board(x & (0b0000000000000010000001000000100000010000001000000100000010000001ULL << 5));
    uint8_t y = join_d7_line(x, 5);
    bit_print_uchar(y);
    cerr << endl;
    bit_print_board(split_d7_line(y, 5));


    bit_print_board(rotate_90(x));
    bit_print_board(rotate_270(x));
    bit_print_board(x);
    x = 0b11000000'00000000'00000000'00000000'00000000'00000000'00000000'00000001;
    for (uint_fast8_t cell = first_bit(&x); x; cell = next_bit(&x))
        cerr << (int)cell << endl;
    return 0;
}