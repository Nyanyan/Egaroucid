/*
    Egaroucid Project

    @file umigame.hpp
        Calculate Minimum Memorization Number a.k.a. Umigame's value
        Umigame's value is published here: https://umigamep.github.io/BookAdviser/
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <unordered_map>
#include "common.hpp"
#include "board.hpp"
#include "book.hpp"
#include "util.hpp"

/*
    @brief Constants for Umigame's value
*/
constexpr int UMIGAME_UNDEFINED = -1;


/*
    @brief Result of umigame value search 

    @param b                            black player's umigame value
    @param w                            white player's umigame value
*/
struct Umigame_result {
    int b;
    int w;

    Umigame_result()
        : b(UMIGAME_UNDEFINED), w(UMIGAME_UNDEFINED) {}

    Umigame_result operator+(const Umigame_result& other) {
        Umigame_result res;
        res.b = b + other.b;
        res.w = w + other.w;
        return res;
    }
};

class Umigame {
    private:
        std::mutex mtx;
        std::unordered_map<Board, Umigame_result, Book_hash> umigame;

    public:
        void calculate(Board *board, int player, int depth) {
            umigame_search(board, player, depth);
        }

        void delete_all() {
            std::lock_guard<std::mutex> lock(mtx);
            umigame.clear();
        }

        /*
            @brief get registered umigame's value

            @param b                    board
            @return registered umigame's value (if not registered, returns default value)
        */
        Umigame_result get_umigame(Board *b) {
            std::lock_guard<std::mutex> lock(mtx);
            Board unique_board = representative_board(b->copy());
            return get_oneumigame(unique_board);
        }

    private:
        /*
            @brief Result of umigame value search 

            @param b                            board to solve
            @param player                       the player of this board
            @return Umigame's value
        */
        Umigame_result umigame_search(Board *b, int player, int depth) {
            Umigame_result umigame_res;
			if (!global_searching)
                return umigame_res;
			if (!book.contain(b) || b->n_discs() >= depth + 4) {
				umigame_res.b = 1;
                umigame_res.w = 1;
                return umigame_res;
            }
            umigame_res = get_umigame(b);
            if (umigame_res.b != UMIGAME_UNDEFINED)
                return umigame_res;
            int max_val = -INF;
            if (b->get_legal() == 0ULL) {
                player ^= 1;
                b->pass();
            }
            Flip flip;
            Book_elem book_elem = book.get(b);
            std::vector<int> best_moves = book.get_all_best_moves(b);
            //b->print();
            if (best_moves.size() == 0) {
                umigame_res.b = 1;
                umigame_res.w = 1;
				reg(b, umigame_res);
                return umigame_res;
            }
            std::vector<Board> boards;
            for (uint_fast8_t cell: best_moves) {
                calc_flip(&flip, b, cell);
                boards.emplace_back(b->move_copy(&flip));
            }
            if (player == BLACK) {
                umigame_res.b = INF;
                umigame_res.w = 0;
                for (Board &nnb : boards) {
                    Umigame_result nres = umigame_search(&nnb, player ^ 1, depth);
                    umigame_res.b = std::min(umigame_res.b, nres.b);
                    umigame_res.w += nres.w;
                }
            } else {
                umigame_res.b = 0;
                umigame_res.w = INF;
                for (Board &nnb : boards) {
                    Umigame_result nres = umigame_search(&nnb, player ^ 1, depth);
                    umigame_res.w = std::min(umigame_res.w, nres.w);
                    umigame_res.b += nres.b;
                }
            }
            if (global_searching) {
                reg(b, umigame_res);
            }
            return umigame_res;
        }

        inline void reg(Board *b, Umigame_result val) {
            Board unique_board = representative_board(b->copy());
            std::lock_guard<std::mutex> lock(mtx);
            umigame[unique_board] = val;
        }

        /*
            @brief get registered umigame's value

            @param b                    a board to find
            @return registered umigame's value (if not registered, returns defalut)
        */
        inline Umigame_result get_oneumigame(Board b) {
            Umigame_result res;
            res.b = UMIGAME_UNDEFINED;
            res.w = UMIGAME_UNDEFINED;
            if (umigame.find(b) != umigame.end()) {
                res = umigame[b];
            }
            return res;
        }

        inline bool contain(Board b) {
            Board unique_board = representative_board(b);
            return umigame.find(unique_board) != umigame.end();
        }
};

Umigame umigame;

/*
    @brief Calculate Umigame's value

    @param b                            board to solve
    @param player                       the player of this board
    @return Umigame's value in Umigame_result structure
*/
Umigame_result calculate_umigame(Board *b, int player, int depth) {
    Umigame_result res = umigame.get_umigame(b);
    if (res.b == UMIGAME_UNDEFINED) {
        umigame.calculate(b, player, depth);
        res = umigame.get_umigame(b);
    }
    return res;
}