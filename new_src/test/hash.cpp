#include <iostream>
#include "./../board.hpp"

#define N_TESTCASES 1000000

#define HASH_SIZE (1U << 25)
#define HASH_MASK ((1U << 25) - 1)

uint32_t result[HASH_SIZE];

int main(){
    bit_init();
    flip_init();
    board_init();
    for (uint32_t i = 0; i < HASH_SIZE; ++i)
        result[i] = 0;
    Board board;
    for (uint32_t i = 0; i < N_TESTCASES; ++i){
        board.player = myrand_ull();
        board.opponent = myrand_ull() & (~board.player);
        ++result[board.hash() & HASH_MASK];
    }
    uint32_t hash_conf = 0;
    for (uint32_t i = 0; i < HASH_SIZE; ++i){
        if (result[i] > 1)
            hash_conf += result[i] - 1;
    }
    cerr << hash_conf << " " << (double)hash_conf / N_TESTCASES << endl;

    return 0;
}