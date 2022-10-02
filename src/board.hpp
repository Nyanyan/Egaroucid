/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include "common.hpp"
#include "mobility.hpp"
#include "flip.hpp"
#include "last_flip.hpp"
using namespace std;

uint32_t hash_rand_player[4][65536];
uint32_t hash_rand_opponent[4][65536];

class Board {
    public:
        uint64_t player;
        uint64_t opponent;

    public:
        inline Board copy(){
            Board res;
            res.player = player;
            res.opponent = opponent;
            return res;
        }

        inline void copy(Board *res){
            res->player = player;
            res->opponent = opponent;
        }

        inline uint32_t hash(){
            const uint16_t *p = (uint16_t*)&player;
            const uint16_t *o = (uint16_t*)&opponent;
            return 
                hash_rand_player[0][p[0]] ^ 
                hash_rand_player[1][p[1]] ^ 
                hash_rand_player[2][p[2]] ^ 
                hash_rand_player[3][p[3]] ^ 
                hash_rand_opponent[0][o[0]] ^ 
                hash_rand_opponent[1][o[1]] ^ 
                hash_rand_opponent[2][o[2]] ^ 
                hash_rand_opponent[3][o[3]];
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

        inline void print() const{
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

        inline void move_board(const Flip *flip) {
            player ^= flip->flip;
            opponent ^= flip->flip;
            player ^= 1ULL << flip->pos;
            swap(player, opponent);
        }

        inline void move_copy(const Flip *flip, Board *res) {
            res->opponent = player ^ flip->flip;
            res->player = opponent ^ flip->flip;
            res->opponent ^= 1ULL << flip->pos;
        }

        inline Board move_copy(const Flip *flip) {
            Board res;
            move_copy(flip, &res);
            return res;
        }

        inline void pass(){
            swap(player, opponent);
        }

        inline void undo_board(const Flip *flip){
            swap(player, opponent);
            player ^= 1ULL << flip->pos;
            player ^= flip->flip;
            opponent ^= flip->flip;
        }

        inline void translate_to_arr_player(uint_fast8_t res[]) {
            for (int i = 0; i < HW2; ++i)
                res[HW2_M1 - i] = 2 - (1 & (player >> i)) * 2 - (1 & (opponent >> i));
        }

        inline void translate_to_arr_player(int res[]) {
            for (int i = 0; i < HW2; ++i)
                res[HW2_M1 - i] = 2 - (1 & (player >> i)) * 2 - (1 & (opponent >> i));
        }

        inline void translate_to_arr(int res[], int p) {
            if (p == 0){
                for (int i = 0; i < HW2; ++i)
                    res[HW2_M1 - i] = 2 - (1 & (player >> i)) * 2 - (1 & (opponent >> i));
            } else{
                for (int i = 0; i < HW2; ++i)
                    res[HW2_M1 - i] = 2 - (1 & (player >> i)) - (1 & (opponent >> i)) * 2;
            }
        }

        inline void translate_to_arr(uint_fast8_t res[], int p) {
            if (p == 0){
                for (int i = 0; i < HW2; ++i)
                    res[HW2_M1 - i] = 2 - (1 & (player >> i)) * 2 - (1 & (opponent >> i));
            } else{
                for (int i = 0; i < HW2; ++i)
                    res[HW2_M1 - i] = 2 - (1 & (player >> i)) - (1 & (opponent >> i)) * 2;
            }
        }

        inline void translate_to_arr_player_rev(uint_fast8_t res[]) {
            for (int i = 0; i < HW2; ++i)
                res[i] = 2 - (1 & (player >> i)) * 2 - (1 & (opponent >> i));
        }

        inline void translate_from_arr(const int arr[], int player_idx) {
            int i;
            player = 0;
            opponent = 0;
            if (player_idx == BLACK){
                for (i = 0; i < HW2; ++i) {
                    if (arr[HW2_M1 - i] == BLACK)
                        player |= 1ULL << i;
                    else if (arr[HW2_M1 - i] == WHITE)
                        opponent |= 1ULL << i;
                }
            } else{
                for (i = 0; i < HW2; ++i) {
                    if (arr[HW2_M1 - i] == BLACK)
                        opponent |= 1ULL << i;
                    else if (arr[HW2_M1 - i] == WHITE)
                        player |= 1ULL << i;
                }
            }
        }

        inline void translate_from_ull(const uint64_t pl, const uint64_t op) {
            if (pl & op)
                cerr << "both on same square" << endl;
            player = pl;
            opponent = op;
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

        inline int count_player() const{
            return pop_count_ull(player);
        }

        inline int count_opponent() const{
            return pop_count_ull(opponent);
        }

        inline int n_discs() const{
            return pop_count_ull(player | opponent);
        }

        inline bool check_player(){
            bool passed = (get_legal() == 0);
            if (passed){
                pass();
                passed = (get_legal() == 0);
                if (passed)
                    pass();
            }
            return passed;
        }

        inline bool is_end(){
            return (calc_legal(player, opponent) == 0ULL) && (calc_legal(opponent, player) == 0ULL);
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
            player = 0x0000000810000000ULL;
            opponent = 0x0000001008000000ULL;
        }

        inline int phase_slow(){
            int n_discs = pop_count_ull(player | opponent);
            return min(N_PHASES - 1, (n_discs - 4) / PHASE_N_STONES);
        }
};


bool operator==(const Board& a, const Board& b){
    return a.player == b.player && a.opponent == b.opponent;
}

bool operator!=(const Board& a, const Board& b){
    return a.player != b.player || a.opponent != b.opponent;
}

struct Board_hash {
    size_t operator()(Board board) const{
        return board.hash();
    }
};

void board_init(){
    int i, j;
    for (i = 0; i < 4; ++i){
        for (j = 0; j < 65536; ++j){
            hash_rand_player[i][j] = 0;
            while (pop_count_uint(hash_rand_player[i][j]) < 4)
                hash_rand_player[i][j] = myrand_uint_rev(); //(uint32_t)(rotate_180(myrand_ull()) >> 32);
            hash_rand_opponent[i][j] = 0;
            while (pop_count_uint(hash_rand_opponent[i][j]) < 4)
                hash_rand_opponent[i][j] = myrand_uint_rev(); //(uint32_t)(rotate_180(myrand_ull()) >> 32);
        }
    }
    cerr << "board initialized" << endl;
}

inline void calc_flip(Flip *flip, Board *board, uint_fast8_t place){
    flip->calc_flip(board->player, board->opponent, place);
}