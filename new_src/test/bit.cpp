#include <iostream>
#include "./../bit.hpp"

int main(){
    uint64_t x;
    cin >> x;
    bit_print_board(x);
    bit_print_board(white_line_mirror(x));
    bit_print_board(black_line_mirror(x));
    return 0;
}