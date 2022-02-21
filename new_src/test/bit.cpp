#include <iostream>
#include "./../bit.hpp"

int main(){
    uint64_t x;
    x = 376240289238932486;
    bit_print_board(x);
    bit_print_board(rotate_90(x));
    bit_print_board(rotate_270(x));
    return 0;
}