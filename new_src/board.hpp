#pragma once
#include <iostream>
#include "common.hpp"
#include "mobility.hpp"
#include "flip.hpp"

using namespace std;

#define LEGAL_UNDEFINED 0x0000001818000000ULL

uint32_t hash_rand_player[4][65536];
uint32_t hash_rand_opponent[4][65536];

inline uint64_t full_stability_h(uint64_t full);
inline uint64_t full_stability_v(uint64_t full);
inline void full_stability_d(uint64_t full, uint64_t *full_d7, uint64_t *full_d9);

class Board {
    public:
        uint64_t player;
        uint64_t opponent;
        uint_fast8_t p;
        uint_fast8_t n;
        uint_fast8_t parity;

    public:
        inline Board copy(){
            Board res;
            res.player = player;
            res.opponent = opponent;
            res.p = p;
            res.n = n;
            res.parity = parity;
            return res;
        }

        inline void copy(Board *res){
            res->player = player;
            res->opponent = opponent;
            res->p = p;
            res->n = n;
            res->parity = parity;
        }

        inline uint32_t hash(){
            return 
                hash_rand_player[0][0xFFFF & player] ^ 
                hash_rand_player[1][0xFFFF & (player >> 16)] ^ 
                hash_rand_player[2][0xFFFF & (player >> 32)] ^ 
                hash_rand_player[3][0xFFFF & (player >> 48)] ^ 
                hash_rand_opponent[0][0xFFFF & opponent] ^ 
                hash_rand_opponent[1][0xFFFF & (opponent >> 16)] ^ 
                hash_rand_opponent[2][0xFFFF & (opponent >> 32)] ^ 
                hash_rand_opponent[3][0xFFFF & (opponent >> 48)];
        }

        inline void board_white_line_mirror(){
            player = white_line_mirror(player);
            opponent = white_line_mirror(opponent);
        }

        inline void board_black_line_mirror(){
            player = black_line_mirror(player);
            opponent = black_line_mirror(opponent);
        }

        inline void board_vertical_mirror(){
            player = vertical_mirror(player);
            opponent = vertical_mirror(opponent);
        }

        inline void board_horizontal_mirror(){
            player = horizontal_mirror(player);
            opponent = horizontal_mirror(opponent);
        }

        inline void board_rotate_90(){
            player = rotate_90(player);
            opponent = rotate_90(opponent);
        }

        inline void board_rotate_270(){
            player = rotate_270(player);
            opponent = rotate_270(opponent);
        }

        inline void board_rotate_180(){
            player = rotate_180(player);
            opponent = rotate_180(opponent);
        }

        inline void print() {
            for (int i = HW2_M1; i >= 0; --i){
                if (1 & (player >> i))
                    cerr << "X ";
                else if (1 & (opponent >> i))
                    cerr << "O ";
                else
                    cerr << ". ";
                if (i % HW == 0)
                    cerr << endl;
            }
        }

        inline uint64_t get_legal(){
            return calc_legal(player, opponent);
        }

        inline bool check_player(){
            bool passed = (get_legal() == 0);
            if (passed){
                p = 1 - p;
                passed = (get_legal() == 0);
            }
            return passed;
        }

        inline void full_stability(uint64_t *h, uint64_t *v, uint64_t *d7, uint64_t *d9){
            const uint64_t stones = (player | opponent);
            *h = full_stability_h(stones);
            *v = full_stability_v(stones);
            full_stability_d(stones, d7, d9);
        }

        inline void move(const Flip *flip) {
            player ^= flip->flip;
            opponent ^= flip->flip;
            player ^= 1ULL << flip->pos;
            swap(player, opponent);
            p = 1 - p;
            ++n;
            parity ^= cell_div4[flip->pos];
        }

        inline void move_copy(const Flip *flip, Board *res) {
            res->opponent = player ^ flip->flip;
            res->player = opponent ^ flip->flip;
            res->opponent ^= 1ULL << flip->pos;
            res->p = 1 - p;
            res->n = n + 1;
            res->parity = parity ^ cell_div4[flip->pos];
        }

        inline Board move_copy(const Flip *flip) {
            Board res;
            move_copy(flip, &res);
            return res;
        }

        inline void pass(){
            swap(player, opponent);
            p = 1 - p;
        }

        inline void undo(const Flip *flip){
            p = 1 - p;
            --n;
            parity ^= cell_div4[flip->pos];
            swap(player, opponent);
            player ^= 1ULL << flip->pos;
            player ^= flip->flip;
            opponent ^= flip->flip;
        }

        inline void translate_to_arr_player(uint_fast8_t res[]) {
            for (int i = 0; i < HW2; ++i){
                if (1 & (player >> i))
                    res[HW2_M1 - i] = 0;
                else if (1 & (opponent >> i))
                    res[HW2_M1 - i] = 1;
                else
                    res[HW2_M1 - i] = 2;
            }
        }

        inline void translate_to_arr(int res[]) {
            if (p == BLACK){
                for (int i = 0; i < HW2; ++i){
                    if (1 & (player >> i))
                        res[HW2_M1 - i] = BLACK;
                    else if (1 & (opponent >> i))
                        res[HW2_M1 - i] = WHITE;
                    else
                        res[HW2_M1 - i] = VACANT;
                }
            } else{
                for (int i = 0; i < HW2; ++i){
                    if (1 & (opponent >> i))
                        res[HW2_M1 - i] = BLACK;
                    else if (1 & (player >> i))
                        res[HW2_M1 - i] = WHITE;
                    else
                        res[HW2_M1 - i] = VACANT;
                }
            }
        }

        inline void translate_from_arr(const int arr[], int player_idx) {
            int i;
            player = 0;
            opponent = 0;
            n = HW2;
            parity = 0;
            if (player_idx == BLACK){
                for (i = 0; i < HW2; ++i) {
                    if (arr[HW2_M1 - i] == BLACK)
                        player |= 1ULL << i;
                    else if (arr[HW2_M1 - i] == WHITE)
                        opponent |= 1ULL << i;
                    else{
                        --n;
                        parity ^= cell_div4[i];
                    }
                }
            } else{
                for (i = 0; i < HW2; ++i) {
                    if (arr[HW2_M1 - i] == BLACK)
                        opponent |= 1ULL << i;
                    else if (arr[HW2_M1 - i] == WHITE)
                        player |= 1ULL << i;
                    else{
                        --n;
                        parity ^= cell_div4[i];
                    }
                }
            }
            p = player;
        }

        inline void translate_from_ull(const uint64_t pl, const uint64_t op, int player) {
            if (pl & op)
                cerr << "both on same square" << endl;
            player = pl;
            opponent = op;
            n = HW2;
            parity = 0;
            for (int i = 0; i < HW2; ++i) {
                if ((1 & (pl >> i)) == 0 && (1 & (op >> i)) == 0){
                    --n;
                    parity ^= cell_div4[i];
                }
            }
            p = player;
        }

        inline int score_player(){
            int p_score = pop_count_ull(player), o_score = pop_count_ull(opponent);
            int score = p_score - o_score, vacant_score = HW2 - p_score - o_score;
            if (score > 0)
                score += vacant_score;
            else if (score < 0)
                score -= vacant_score;
            return score;
        }

        inline int score_opponent(){
            int p_score = pop_count_ull(player), o_score = pop_count_ull(opponent);
            int score = o_score - p_score, vacant_score = HW2 - o_score - p_score;
            if (score > 0)
                score += vacant_score;
            else if (score < 0)
                score -= vacant_score;
            return score;
        }

        inline int count_player(){
            return pop_count_ull(player);
        }

        inline int count_opponent(){
            return pop_count_ull(opponent);
        }

        inline void board_canput(int canput_arr[], const uint64_t mobility_player, const uint64_t mobility_opponent){
            for (int i = 0; i < HW2; ++i){
                canput_arr[i] = 0;
                if (1 & (mobility_player >> i))
                    ++canput_arr[i];
                if (1 & (mobility_opponent >> i))
                    canput_arr[i] += 2;
            }
        }

        inline void board_canput(int canput_arr[]){
            const uint64_t mobility_player = calc_legal(player, opponent);
            const uint64_t mobility_opponent = calc_legal(player, opponent);
            board_canput(canput_arr, mobility_player, mobility_opponent);
        }

        inline bool check_pass(){
            bool passed = (get_legal() == 0);
            if (passed){
                pass();
                passed = (get_legal() == 0);
                if (passed)
                    return false;
            }
            return true;
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
};

inline uint64_t full_stability_h(uint64_t full){
    full &= full >> 1;
    full &= full >> 2;
    full &= full >> 4;
    return (full & 0x0101010101010101) * 0xff;
}

inline uint64_t full_stability_v(uint64_t full){
    full &= (full >> 8) | (full << 56);
    full &= (full >> 16) | (full << 48);
    full &= (full >> 32) | (full << 32);
    return full;
}

inline void full_stability_d(uint64_t full, uint64_t *full_d7, uint64_t *full_d9){
    static const uint64_t edge = 0xFF818181818181FF;
    static const uint64_t e7[4] = {
        0xFFFF030303030303, 0xC0C0C0C0C0C0FFFF, 0xFFFFFFFF0F0F0F0F, 0xF0F0F0F0FFFFFFFF};
    static const uint64_t e9[3] = {
        0xFFFFC0C0C0C0C0C0, 0x030303030303FFFF, 0x0F0F0F0FF0F0F0F0};
    uint64_t l7, r7, l9, r9;
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

inline void full_stability(uint64_t player, uint64_t opponent, uint64_t *h, uint64_t *v, uint64_t *d7, uint64_t *d9){
    const uint64_t stones = (player | opponent);
    *h = full_stability_h(stones);
    *v = full_stability_v(stones);
    full_stability_d(stones, d7, d9);
}

void board_init(){
    int i, j;
    for (i = 0; i < 4; ++i){
        for (j = 0; j < 65536; ++j){
            hash_rand_player[i][j] = 0;
            while (pop_count_uint(hash_rand_player[i][j]) < 8)
                hash_rand_player[i][j] = myrand_uint(); //(uint32_t)(rotate_180(myrand_ull()) >> 32);
            hash_rand_opponent[i][j] = 0;
            while (pop_count_uint(hash_rand_opponent[i][j]) < 8)
                hash_rand_opponent[i][j] = myrand_uint(); //(uint32_t)(rotate_180(myrand_ull()) >> 32);
        }
    }
    cerr << "board initialized" << endl;
}

inline void calc_flip(Flip *flip, Board *b, const int policy){
    flip->calc_flip(b->player, b->opponent, policy);
    flip->n_legal = LEGAL_UNDEFINED;
}

inline Flip calc_flip(Board *b, const int policy){
    Flip flip;
    flip.calc_flip(b->player, b->opponent, policy);
    flip.n_legal = LEGAL_UNDEFINED;
    return flip;
}
