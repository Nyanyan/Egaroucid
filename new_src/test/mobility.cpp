#include <iostream>
#include "./../mobility.hpp"

uint64_t get_mobility(const uint64_t P, const uint64_t O){
    __m256i	PP, mOO, MM, flip_l, flip_r, pre_l, pre_r, shift2;
    __m128i	M;
    const __m256i shift1897 = _mm256_set_epi64x(7, 9, 8, 1);
    const __m256i mflipH = _mm256_set_epi64x(0x7E7E7E7E7E7E7E7E, 0x7E7E7E7E7E7E7E7E, -1, 0x7E7E7E7E7E7E7E7E);
    PP = _mm256_broadcastq_epi64(_mm_cvtsi64_si128(P));
    mOO = _mm256_and_si256(_mm256_broadcastq_epi64(_mm_cvtsi64_si128(O)), mflipH);
    flip_l = _mm256_and_si256(mOO, _mm256_sllv_epi64(PP, shift1897));
    flip_r = _mm256_and_si256(mOO, _mm256_srlv_epi64(PP, shift1897));
    flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(mOO, _mm256_sllv_epi64(flip_l, shift1897)));
    flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(mOO, _mm256_srlv_epi64(flip_r, shift1897)));
    pre_l = _mm256_and_si256(mOO, _mm256_sllv_epi64(mOO, shift1897));
    pre_r = _mm256_srlv_epi64(pre_l, shift1897);
    shift2 = _mm256_add_epi64(shift1897, shift1897);
    flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)));
    flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)));
    flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)));
    flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)));
    MM = _mm256_sllv_epi64(flip_l, shift1897);
    MM = _mm256_or_si256(MM, _mm256_srlv_epi64(flip_r, shift1897));
    M = _mm_or_si128(_mm256_castsi256_si128(MM), _mm256_extracti128_si256(MM, 1));
    M = _mm_or_si128(M, _mm_unpackhi_epi64(M, M));
    return _mm_cvtsi128_si64(M) & ~(P | O);
}

#define N_TESTCASES 10000000

uint64_t testcases[N_TESTCASES][2];
uint64_t test_results[N_TESTCASES][2];

int main(){
    bit_init();
    uint8_t player;
    uint64_t p, o;
    cin >> player;
    input_board(&p, &o);
    cerr << endl;
    print_board(p, o);
    bit_print_board(calc_legal(p, o));
    return 0;

    uint64_t strt;
    for (uint32_t i = 0; i < N_TESTCASES; ++i){
        testcases[i][0] = myrand_ull();
        testcases[i][1] = myrand_ull() & (~testcases[i][0]);
    }
    cerr << "start!" << endl;
    strt = tim();
    for (volatile uint32_t t = 0; t < 10; ++t){
        for (volatile uint32_t i = 0; i < N_TESTCASES; ++i){
            test_results[i][0] = calc_legal(testcases[i][0], testcases[i][1]);
        }
    }
    //bit_print_board(mobility);
    //cerr << (mobility == get_mobility(p, o)) << endl;
    cerr << tim() - strt << endl;

    strt = tim();
    for (volatile uint32_t t = 0; t < 10; ++t){
        for (volatile uint32_t i = 0; i < N_TESTCASES; ++i){
            test_results[i][1] = get_mobility(testcases[i][0], testcases[i][1]);
        }
    }
    cerr << tim() - strt << endl;
    //bit_print_board(mobility);
    for (uint32_t i = 0; i < N_TESTCASES; ++i){
        if (test_results[i][0] != test_results[i][1])
            cerr << "a";
    }

    return 0;
}