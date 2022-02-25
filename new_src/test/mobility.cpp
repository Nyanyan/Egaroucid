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