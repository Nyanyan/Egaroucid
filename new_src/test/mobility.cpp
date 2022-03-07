#include <iostream>
#include "./../mobility.hpp"


#ifdef _MSC_VER
	#define	mirror_v(x)	_byteswap_uint64(x)
#else
	#define	mirror_v(x)	__builtin_bswap64(x)
#endif

inline unsigned long long get_mobility(const unsigned long long P, const unsigned long long O){
    unsigned long long moves, mO, flip1, pre1, flip8, pre8;
    __m128i	PP, mOO, MM, flip, pre;
    mO = O & 0x7e7e7e7e7e7e7e7eULL;
    PP  = _mm_set_epi64x(mirror_v(P), P);
    mOO = _mm_set_epi64x(mirror_v(mO), mO);
    flip = _mm_and_si128(mOO, _mm_slli_epi64(PP, 7));				            flip1  = mO & (P << 1);		    flip8  = O & (P << 8);
    flip = _mm_or_si128(flip, _mm_and_si128(mOO, _mm_slli_epi64(flip, 7)));		flip1 |= mO & (flip1 << 1);	    flip8 |= O & (flip8 << 8);
    pre  = _mm_and_si128(mOO, _mm_slli_epi64(mOO, 7));				            pre1   = mO & (mO << 1);	    pre8   = O & (O << 8);
    flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 14)));	flip1 |= pre1 & (flip1 << 2);	flip8 |= pre8 & (flip8 << 16);
    flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 14)));	flip1 |= pre1 & (flip1 << 2);	flip8 |= pre8 & (flip8 << 16);
    MM = _mm_slli_epi64(flip, 7);							                    moves = flip1 << 1;		        moves |= flip8 << 8;
    flip = _mm_and_si128(mOO, _mm_slli_epi64(PP, 9));				            flip1  = mO & (P >> 1);		    flip8  = O & (P >> 8);
    flip = _mm_or_si128(flip, _mm_and_si128(mOO, _mm_slli_epi64(flip, 9)));		flip1 |= mO & (flip1 >> 1);	    flip8 |= O & (flip8 >> 8);
    pre = _mm_and_si128(mOO, _mm_slli_epi64(mOO, 9));				            pre1 >>= 1;			            pre8 >>= 8;
    flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 18)));	flip1 |= pre1 & (flip1 >> 2);	flip8 |= pre8 & (flip8 >> 16);
    flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 18)));	flip1 |= pre1 & (flip1 >> 2);	flip8 |= pre8 & (flip8 >> 16);
    MM = _mm_or_si128(MM, _mm_slli_epi64(flip, 9));					            moves |= flip1 >> 1;		    moves |= flip8 >> 8;
    moves |= _mm_cvtsi128_si64(MM) | mirror_v(_mm_cvtsi128_si64(_mm_unpackhi_epi64(MM, MM)));
    return moves & ~(P|O);
}

#define N_TESTCASES 10000000

uint64_t testcases[N_TESTCASES][2];
uint64_t test_results[N_TESTCASES][2];

int main(){
    bit_init();
    //uint8_t player;
    //cin >> player;
    //input_board(&p, &o);
    //cerr << endl;
    //print_board(p, o);
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