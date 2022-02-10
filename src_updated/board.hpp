#pragma once
#include <iostream>
#include <emmintrin.h>
#include "mobility.hpp"

using namespace std;

#define P171 17ULL
#define P172 289ULL
#define P173 4913ULL
#define P174 83521ULL
#define P175 1419857ULL
#define P176 24137569ULL
#define P177 410338673ULL
#define P191 19ULL
#define P192 361ULL
#define P193 6859ULL
#define P194 130321ULL
#define P195 2476099ULL
#define P196 47045881ULL
#define P197 893871739ULL

constexpr int cell_div4[HW2] = {
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8
};

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

class Board {
    public:
        unsigned long long b;
        unsigned long long w;
        int p;
        int policy;
        int v;
        int n;
        int parity;

    public:
        bool operator<(const Board& another) const {
            return v > another.v;
        }

        inline Board copy(){
            Board res;
            res.b = b;
            res.w = w;
            res.p = p;
            res.policy = policy;
            res.v = v;
            res.n = n;
            res.parity = parity;
            return res;
        }

        inline void copy(Board *res){
            res->b = b;
            res->w = w;
            res->p = p;
            res->policy = policy;
            res->v = v;
            res->n = n;
            res->parity = parity;
        }

        inline unsigned long long hash(){
            /*
            return
                (b * 3) ^ 
                ((b >> 16) * P171) ^ 
                ((b >> 32) * P173) ^ 
                ((b >> 48) * P175) ^ 
                (w * 5) ^ 
                ((w >> 16) * P191) ^ 
                ((w >> 32) * P193) ^ 
                ((w >> 48) * P195);
            */
            return
                p ^ 
                (b * 3) ^ ((b >> 8) * 7) ^ 
                ((b >> 16) * P171) ^ 
                ((b >> 24) * P172) ^ 
                ((b >> 32) * P173) ^ 
                ((b >> 40) * P174) ^ 
                ((b >> 48) * P175) ^ 
                ((b >> 56) * P176) ^ 
                (w * 5) ^ ((w >> 8) * 11) ^ 
                ((w >> 16) * P191) ^ 
                ((w >> 24) * P192) ^ 
                ((w >> 32) * P193) ^ 
                ((w >> 40) * P194) ^ 
                ((w >> 48) * P195) ^ 
                ((w >> 56) * P196);
            
            /*
            unsigned long long res = 0;
            for (int i = 0; i < HW2; ++i){
                if (1 & (b >> i))
                    res ^= hash_rand[0][i];
                if (1 & (w >> i))
                    res ^= hash_rand[1][i];
            }
            return res;
            */
        }

        inline unsigned long long hash_player(){
            if (p == BLACK)
                return hash();
            /*
            return
                (w * 3) ^ 
                ((w >> 16) * P171) ^ 
                ((w >> 32) * P173) ^ 
                ((w >> 48) * P175) ^ 
                (b * 5) ^ 
                ((b >> 16) * P191) ^ 
                ((b >> 32) * P193) ^ 
                ((b >> 48) * P195);
            */
            return
                (w * 3) ^ ((w >> 8) * 7) ^ 
                ((w >> 16) * P171) ^ 
                ((w >> 24) * P172) ^ 
                ((w >> 32) * P173) ^ 
                ((w >> 40) * P174) ^ 
                ((w >> 48) * P175) ^ 
                ((w >> 56) * P176) ^ 
                (b * 5) ^ ((b >> 8) * 11) ^ 
                ((b >> 16) * P191) ^ 
                ((b >> 24) * P192) ^ 
                ((b >> 32) * P193) ^ 
                ((b >> 40) * P194) ^ 
                ((b >> 48) * P195) ^ 
                ((b >> 56) * P196);
        }

        inline void WHITE_mirror(){
            b = white_line(b);
            w = white_line(w);
            if (policy != -1)
                policy = (policy % HW) * HW + (policy / HW);
        }

        inline void BLACK_mirror(){
            b = black_line(b);
            w = black_line(w);
            if (policy != -1)
                policy = (HW_M1 - policy % HW) * HW + (HW_M1 - policy / HW);
        }

        inline void vertical_mirror(){
            b = mirror_v(b);
            w = mirror_v(w);
            if (policy != -1)
                policy = (HW_M1 - policy / HW) * HW + (HW_M1 - policy % HW);
        }

        inline void print() {
            for (int i = HW2_M1; i >= 0; --i){
                if (1 & (b >> i))
                    cerr << "X ";
                else if (1 & (w >> i))
                    cerr << "O ";
                else
                    cerr << ". ";
                if (i % HW == 0)
                    cerr << endl;
            }
        }

        inline unsigned long long mobility_ull(){
            unsigned long long res;
            if (p == BLACK)
                res = get_mobility(b, w);
            else
                res = get_mobility(w, b);
            return res;
        }

        inline void full_stability(unsigned long long *h, unsigned long long *v, unsigned long long *d7, unsigned long long *d9){
            const unsigned long long stones = (b | w);
            *h = full_stability_h(stones);
            *v = full_stability_v(stones);
            full_stability_d(stones, d7, d9);
        }

        inline void move(const Mobility *mob) {
            if (p == BLACK){
                b ^= mob->flip;
                w &= ~b;
                b |= 1ULL << mob->pos;
            } else{
                w ^= mob->flip;
                b &= ~w;
                w |= 1ULL << mob->pos;
            }
            p = 1 - p;
            ++n;
            policy = mob->pos;
            parity ^= cell_div4[mob->pos];
        }

        inline void move_copy(const Mobility *mob, Board *res) {
            if (p == BLACK){
                res->b = b ^ mob->flip;
                res->w = w & (~res->b);
                res->b |= 1ULL << mob->pos;
            } else{
                res->w = w ^ mob->flip;
                res->b = b & (~res->w);
                res->w |= 1ULL << mob->pos;
            }
            res->p = 1 - p;
            res->n = n + 1;
            res->policy = mob->pos;
            res->parity = parity ^ cell_div4[mob->pos];
        }

        inline Board move_copy(const Mobility *mob) {
            Board res;
            move_copy(mob, &res);
            return res;
        }

        inline void undo(const Mobility *mob){
            p = 1 - p;
            --n;
            policy = -1;
            parity ^= cell_div4[mob->pos];
            if (p == BLACK){
                b &= ~(1ULL << mob->pos);
                b ^= mob->flip;
                w |= mob->flip;
            } else{
                w &= ~(1ULL << mob->pos);
                w ^= mob->flip;
                b |= mob->flip;
            }
        }

        inline void translate_to_arr(int res[]) {
            for (int i = 0; i < HW2; ++i){
                if (1 & (b >> i))
                    res[HW2_M1 - i] = BLACK;
                else if (1 & (w >> i))
                    res[HW2_M1 - i] = WHITE;
                else
                    res[HW2_M1 - i] = VACANT;
            }
        }

        inline void translate_from_arr(const int arr[], int player) {
            int i;
            b = 0;
            w = 0;
            n = HW2;
            parity = 0;
            for (i = 0; i < HW2; ++i) {
                if (arr[HW2_M1 - i] == BLACK)
                    b |= 1ULL << i;
                else if (arr[HW2_M1 - i] == WHITE)
                    w |= 1ULL << i;
                else{
                    --n;
                    parity ^= cell_div4[i];
                }
            }
            p = player;
            policy = -1;
        }

        inline void translate_from_ull(const unsigned long long bk, const unsigned long long wt, int player) {
            if (bk & wt)
                cerr << "both on same square" << endl;
            b = bk;
            w = wt;
            n = HW2;
            parity = 0;
            for (int i = 0; i < HW2; ++i) {
                if ((1 & (bk >> i)) == 0 && (1 & (wt >> i)) == 0){
                    --n;
                    parity ^= cell_div4[i];
                }
            }
            p = player;
            policy = -1;
        }

        inline int score(int player){
            int b_score = pop_count_ull(b), w_score = pop_count_ull(w);
            int BLACK_score = b_score - w_score, VACANT_score = HW2 - b_score - w_score;
            if (BLACK_score > 0)
                BLACK_score += VACANT_score;
            else if (BLACK_score < 0)
                BLACK_score -= VACANT_score;
            return (player ? -1 : 1) * BLACK_score;
        }

        inline int score(){
            return score(p);
        }

        inline int count_player(){
            if (p == BLACK)
                return pop_count_ull(b);
            return pop_count_ull(w);
        }

        inline int count_opponent(){
            if (p == WHITE)
                return pop_count_ull(b);
            return pop_count_ull(w);
        }

        inline int raw_count(){
            if (p == BLACK)
                return pop_count_ull(b);
            return pop_count_ull(w);
        }

        inline void board_canput(int canput_arr[], const unsigned long long mobility_BLACK, const unsigned long long mobility_WHITE){
            for (int i = 0; i < HW2; ++i){
                canput_arr[i] = 0;
                if (1 & (mobility_BLACK >> i))
                    ++canput_arr[i];
                if (1 & (mobility_WHITE >> i))
                    canput_arr[i] += 2;
            }
        }

        inline void board_canput(int canput_arr[]){
            const unsigned long long mobility_BLACK = get_mobility(b, w);
            const unsigned long long mobility_WHITE = get_mobility(w, b);
            board_canput(canput_arr, mobility_BLACK, mobility_WHITE);
        }

        inline void check_player(){
            bool passed = (mobility_ull() == 0);
            if (passed){
                p = 1 - p;
                passed = (mobility_ull() == 0);
                if (passed)
                    p = VACANT;
            }
        }

        inline void reset(){
            constexpr int first_board[HW2] = {
                VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,
                VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,
                VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,
                VACANT,VACANT,VACANT,WHITE,BLACK,VACANT,VACANT,VACANT,
                VACANT,VACANT,VACANT,BLACK,WHITE,VACANT,VACANT,VACANT,
                VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,
                VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,
                VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,VACANT,VACANT
            };
            translate_from_arr(first_board, BLACK);
        }

        inline int phase(){
            return min(N_PHASES - 1, (n - 4) / PHASE_N_STONES);
        }
    
    private:
        inline unsigned long long full_stability_h(unsigned long long full){
            full &= full >> 1;
            full &= full >> 2;
            full &= full >> 4;
            return (full & 0x0101010101010101) * 0xff;
        }

        inline unsigned long long full_stability_v(unsigned long long full){
            full &= (full >> 8) | (full << 56);
            full &= (full >> 16) | (full << 48);
            full &= (full >> 32) | (full << 32);
            return full;
        }

        inline void full_stability_d(unsigned long long full, unsigned long long *full_d7, unsigned long long *full_d9){
            static const unsigned long long edge = 0xff818181818181ff;
            static const unsigned long long e7[] = {
                0xffff030303030303, 0xc0c0c0c0c0c0ffff, 0xffffffff0f0f0f0f, 0xf0f0f0f0ffffffff };
            static const unsigned long long e9[] = {
                0xffffc0c0c0c0c0c0, 0x030303030303ffff, 0x0f0f0f0ff0f0f0f0 };
            unsigned long long l7, r7, l9, r9;
            l7 = r7 = full;
            l7 &= edge | (l7 >> 7);		r7 &= edge | (r7 << 7);
            l7 &= e7[0] | (l7 >> 14);	r7 &= e7[1] | (r7 << 14);
            l7 &= e7[2] | (l7 >> 28);	r7 &= e7[3] | (r7 << 28);
            *full_d7 = l7 & r7;

            l9 = r9 = full;
            l9 &= edge | (l9 >> 9);		r9 &= edge | (r9 << 9);
            l9 &= e9[0] | (l9 >> 18);	r9 &= e9[1] | (r9 << 18);
            *full_d9 = l9 & r9 & (e9[2] | (l9 >> 36) | (r9 << 36));
        }
};

inline void calc_flip(Mobility *mob, Board *b, const int policy){
    if (b->p == BLACK)
        mob->calc_flip(b->b, b->w, policy);
    else
        mob->calc_flip(b->w, b->b, policy);
}

inline Mobility calc_flip(Board *b, const int policy){
    Mobility mob;
    if (b->p == BLACK)
        mob.calc_flip(b->b, b->w, policy);
    else
        mob.calc_flip(b->w, b->b, policy);
    return mob;
}
