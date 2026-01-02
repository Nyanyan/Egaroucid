/*
    Egaroucid Project

    @file board.hpp
        Board class
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include "common.hpp"
#include "mobility.hpp"
#include "flip.hpp"
#include "last_flip.hpp"
#include "hash.hpp"

#if USE_CRC32C_HASH
uint32_t global_hash_bit_mask = (1U << DEFAULT_HASH_LEVEL) - 1;
#endif

/*
    @brief Board class

    Bitboard manipulation
*/
class Board {
    public:
        /*
            @brief player / opponent definition

            @param player               a player to put with this board
            @param opponent             a player to put with this board
        */
        uint64_t player;
        uint64_t opponent;

    public:
        Board(uint64_t player_, uint64_t opponent_)
            : player(player_), opponent(opponent_) {}
        
        Board(std::string board_str) {
            from_str(board_str);
        }
    
        Board() {}
        /*
            @brief copy this board

            @return board class
        */
        inline Board copy() const {
            Board res(player, opponent);
            return res;
        }

        /*
            @brief copy this board

            @param res                  board class
        */
        inline void copy(Board *res) const {
            res->player = player;
            res->opponent = opponent;
        }

        /*
            @brief calculate hash code

            @return hash code of this board
        */
#if USE_CRC32C_HASH
        // CRC32C Hash idea from http://www.amy.hi-ho.ne.jp/okuhara/edaxopt.htm#crc32hash
        inline uint32_t hash() const {
            uint64_t res = _mm_crc32_u64(0, player);
            res = _mm_crc32_u64(res, opponent);
            return global_hash_bit_mask & res;
        }
#else
        inline uint32_t hash() const {
            const uint16_t *p = (uint16_t*)&player;     const uint16_t *o = (uint16_t*)&opponent;
            uint32_t h1 = hash_rand_player[0][p[0]];    uint32_t h2 = hash_rand_opponent[0][o[0]];
            h1 ^= hash_rand_player[1][p[1]];            h2 ^= hash_rand_opponent[1][o[1]];
            h1 ^= hash_rand_player[2][p[2]];            h2 ^= hash_rand_opponent[2][o[2]];
            h1 ^= hash_rand_player[3][p[3]];            h2 ^= hash_rand_opponent[3][o[3]];
            return h1 ^ h2;
        }
#endif

        /*
            @brief mirroring in white line
        */
        inline void board_white_line_mirror() {
            player = white_line_mirror(player);
            opponent = white_line_mirror(opponent);
        }

        /*
            @brief mirroring in black line
        */
        inline void board_black_line_mirror() {
            player = black_line_mirror(player);
            opponent = black_line_mirror(opponent);
        }

        /*
            @brief mirroring in vertical
        */
        inline void board_vertical_mirror() {
            player = vertical_mirror(player);
            opponent = vertical_mirror(opponent);
        }

        /*
            @brief mirroring in horizontal
        */
        inline void board_horizontal_mirror() {
            player = horizontal_mirror(player);
            opponent = horizontal_mirror(opponent);
        }

        /*
            @brief rotate 90 degrees counter clockwise
        */
        inline void board_rotate_90() {
            player = rotate_90(player);
            opponent = rotate_90(opponent);
        }

        /*
            @brief rotate 270 degrees counter clockwise
        */
        inline void board_rotate_270() {
            player = rotate_270(player);
            opponent = rotate_270(opponent);
        }

        /*
            @brief rotate 180 degrees
        */
        inline void board_rotate_180() {
            player = rotate_180(player);
            opponent = rotate_180(opponent);
        }

        /*
            @brief mirroring in white line
        */
        inline Board get_white_line_mirror() {
            Board res(white_line_mirror(player), white_line_mirror(opponent));
            return res;
        }

        /*
            @brief mirroring in black line
        */
        inline Board get_black_line_mirror() {
            Board res(black_line_mirror(player), black_line_mirror(opponent));
            return res;
        }

        /*
            @brief mirroring in vertical
        */
        inline Board get_vertical_mirror() {
            Board res(vertical_mirror(player), vertical_mirror(opponent));
            return res;
        }

        /*
            @brief mirroring in horizontal
        */
        inline Board get_horizontal_mirror() {
            Board res(horizontal_mirror(player), horizontal_mirror(opponent));
            return res;
        }

        /*
            @brief print the board
        */
        inline void print() const{
            for (int i = HW2_M1; i >= 0; --i) {
                if (1 & (player >> i))
                    std::cerr << "X ";
                else if (1 & (opponent >> i))
                    std::cerr << "O ";
                else
                    std::cerr << ". ";
                if (i % HW == 0)
                    std::cerr << std::endl;
            }
        }

        /*
            @brief get legal moves in bitboard

            @return a bitboard with bits of legal cells set
        */
        inline uint64_t get_legal() const {
            return calc_legal(player, opponent);
        }

        /*
            @brief move the board

            @param flip                 Flip structure
        */
        inline void move_board(const Flip *flip) {
            player ^= flip->flip;
            opponent ^= flip->flip;
            player ^= 1ULL << flip->pos;
            std::swap(player, opponent);
        }

        /*
            @brief move the board

            @param flip                 Flip structure
            @param res                  board to store result
        */
        inline void move_copy(const Flip *flip, Board *res) {
            res->opponent = player ^ flip->flip;
            res->player = opponent ^ flip->flip;
            res->opponent ^= 1ULL << flip->pos;
        }

        /*
            @brief move the board

            @param flip                 Flip structure
            @return board class
        */
        inline Board move_copy(const Flip *flip) {
            Board res;
            move_copy(flip, &res);
            return res;
        }

        /*
            @brief pass

            swap player and opponent
        */
        inline void pass() {
            std::swap(player, opponent);
        }

        /*
            @brief undo previout move

            @param flip                 Flip structure
        */
        inline void undo_board(const Flip *flip) {
            std::swap(player, opponent);
            player ^= 1ULL << flip->pos;
            player ^= flip->flip;
            opponent ^= flip->flip;
        }

        /*
            @brief convert a bitboard to an array

            @param res                  array to store result
        */
        inline void translate_to_arr_player(uint_fast8_t res[]) {
            for (int i = 0; i < HW2; ++i)
                res[HW2_M1 - i] = 2 - (1 & (player >> i)) * 2 - (1 & (opponent >> i));
        }

        /*
            @brief convert a bitboard to an array

            @param res                  array to store result
        */
        inline void translate_to_arr_player(int res[]) {
            for (int i = 0; i < HW2; ++i)
                res[HW2_M1 - i] = 2 - (1 & (player >> i)) * 2 - (1 & (opponent >> i));
        }

        /*
            @brief convert a bitboard to an array

            @param res                  array to store result
            @param p                    player (0: black, 1: white)
        */
        inline void translate_to_arr(int res[], int p) {
            if (p == 0) {
                for (int i = 0; i < HW2; ++i)
                    res[HW2_M1 - i] = 2 - (1 & (player >> i)) * 2 - (1 & (opponent >> i));
            } else{
                for (int i = 0; i < HW2; ++i)
                    res[HW2_M1 - i] = 2 - (1 & (player >> i)) - (1 & (opponent >> i)) * 2;
            }
        }

        /*
            @brief convert a bitboard to an array

            @param res                  array to store result
            @param p                    player (0: black, 1: white)
        */
        inline void translate_to_arr(uint_fast8_t res[], int p) {
            if (p == 0) {
                for (int i = 0; i < HW2; ++i)
                    res[HW2_M1 - i] = 2 - (1 & (player >> i)) * 2 - (1 & (opponent >> i));
            } else{
                for (int i = 0; i < HW2; ++i)
                    res[HW2_M1 - i] = 2 - (1 & (player >> i)) - (1 & (opponent >> i)) * 2;
            }
        }

        /*
            @brief convert a bitboard to an array in reversed

            @param res                  array to store result
        */
        inline void translate_to_arr_player_rev(uint_fast8_t res[]) {
            for (int i = 0; i < HW2; ++i)
                res[i] = 2 - (1 & (player >> i)) * 2 - (1 & (opponent >> i));
        }

        /*
            @brief convert a bitboard to an array in reversed

            @param res                  array to store result
        */
        inline void translate_to_arr_player_rev(int res[]) {
            for (int i = 0; i < HW2; ++i)
                res[i] = 2 - (1 & (player >> i)) * 2 - (1 & (opponent >> i));
        }

        /*
            @brief convert an array to a bitboard

            @param arr                  array representing a board
            @param player_idx           player (0: black, 1: white)
        */
        inline void translate_from_arr(const int arr[], int player_idx) {
            int i;
            player = 0;
            opponent = 0;
            if (player_idx == BLACK) {
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

        /*
            @brief convert bitboards to a bitboard

            @param pl                   a bitboard representing player
            @param op                   a bitboard representing opponent
        */
        inline void translate_from_ull(const uint64_t pl, const uint64_t op) {
            if (pl & op)
                std::cerr << "both on same square" << std::endl;
            player = pl;
            opponent = op;
        }

        /*
            @brief calculate the score of player

            @return score of player
        */
        inline int score_player() {
            int e = pop_count_ull(~(player | opponent));
            int score = pop_count_ull(player) * 2 + e;
            score += (((score >> 6) & 1) + (((score + HW2_M1) >> 7) & 1) - 1) * e;
            return score - HW2;
        }

        /*
            @brief calculate the score of opponent

            @return score of opponent
        */
        inline int score_opponent() {
            int e = pop_count_ull(~(player | opponent));
            int score = pop_count_ull(opponent) * 2 + e;
            score += (((score >> 6) & 1) + (((score + HW2_M1) >> 7) & 1) - 1) * e;
            return score - HW2;
        }

        /*
            @brief calculate the number of discs of player

            @return the number of discs of player
        */
        inline int count_player() const{
            return pop_count_ull(player);
        }

        /*
            @brief calculate the number of discs of opponent

            @return the number of discs of opponent
        */
        inline int count_opponent() const{
            return pop_count_ull(opponent);
        }

        /*
            @brief calculate the number of discs of both players

            @return the number of discs of both players
        */
        inline int n_discs() const{
            return pop_count_ull(player | opponent);
        }

        /*
            @brief check player

            @return pass found?
        */
        inline bool check_player() {
            bool passed = (get_legal() == 0);
            if (passed) {
                pass();
                passed = (get_legal() == 0);
                if (passed)
                    pass();
            }
            return passed;
        }

        /*
            @brief check game over

            @return game over?
        */
        inline bool is_end() {
            return (calc_legal(player, opponent) == 0ULL) && (calc_legal(opponent, player) == 0ULL);
        }

        /*
            @brief check pass

            @return game not over?
        */
        inline bool check_pass() {
            bool passed = (get_legal() == 0);
            if (passed) {
                pass();
                passed = (get_legal() == 0);
                if (passed) {
                    return false;
                }
            }
            return true;
        }

        /*
            @brief reset board
        */
        inline void reset() {
            player = 0x0000000810000000ULL;
            opponent = 0x0000001008000000ULL;
        }

        /*
            @brief calculate phase

            This function is slow.

            @return the phase of the board
        */
        /*
        inline int phase_slow() {
            int n_discs = pop_count_ull(player | opponent);
            return std::min(N_PHASES - 1, (n_discs - 4) / PHASE_N_STONES);
        }
        */

        inline std::string to_str() const {
            std::string res;
            uint64_t cell_bit = 1ULL << (HW2 - 1);
            for (int i = 0; i < HW2; ++i) {
                if (player & cell_bit) {
                    res += "X";
                } else if (opponent & cell_bit) {
                    res += "O";
                } else {
                    res += "-";
                }
                cell_bit >>= 1;
            }
            res += " X";
            return res;
        }

        inline std::string to_str(int player_to_move) const {
            std::string res;
            uint64_t cell_bit = 1ULL << (HW2 - 1);
            for (int i = 0; i < HW2; ++i) {
                if (player & cell_bit) {
                    if (player_to_move == BLACK) {
                        res += "X";
                    } else {
                        res += "O";
                    }
                } else if (opponent & cell_bit) {
                    if (player_to_move == BLACK) {
                        res += "O";
                    } else {
                        res += "X";
                    }
                } else {
                    res += "-";
                }
                cell_bit >>= 1;
            }
            if (player_to_move == BLACK) {
                res += " X";
            } else {
                res += " O";
            }
            return res;
        }

        inline bool from_str(std::string board_str) { // returns OK: true NG: false
            board_str.erase(std::remove_if(board_str.begin(), board_str.end(), ::isspace), board_str.end());
            if (board_str.length() != HW2 + 1) {
                std::cerr << "[ERROR] invalid argument got length " << board_str.length() << " expected " << HW2 + 1 << std::endl;
                return false;
            }
            player = 0ULL;
            opponent = 0ULL;
            for (int i = 0; i < HW2; ++i) {
                if (is_black_like_char(board_str[i])) {
                    player |= 1ULL << (HW2_M1 - i);
                } else if (is_white_like_char(board_str[i])) {
                    opponent |= 1ULL << (HW2_M1 - i);
                }
            }
            if (is_white_like_char(board_str[HW2])) {
                std::swap(player, opponent);
            } else if (!is_black_like_char(board_str[HW2])) {
                std::cerr << "[ERROR] invalid player argument" << std::endl;
                return false;
            }
            return true;
        }
};

bool operator==(const Board& a, const Board& b) {
    return a.player == b.player && a.opponent == b.opponent;
}

bool operator!=(const Board& a, const Board& b) {
    return a.player != b.player || a.opponent != b.opponent;
}

/*
    @brief calculate flip

    @param flip                 flip structure to store result
    @param board                board class
    @param place                cell to put disc
    @return flip.flip
*/
inline uint64_t calc_flip(Flip *flip, Board *board, uint_fast8_t place) {
    return flip->calc_flip(board->player, board->opponent, place);
}