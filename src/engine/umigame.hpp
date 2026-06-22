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

struct Umigame_condition {
    int max_move_loss;
    int black_max_loss;
    int white_max_loss;

    Umigame_condition()
        : max_move_loss(0), black_max_loss(HW2), white_max_loss(HW2) {}

    Umigame_condition(int max_move_loss, int black_max_loss, int white_max_loss)
        : max_move_loss(max_move_loss), black_max_loss(black_max_loss), white_max_loss(white_max_loss) {}
};

inline bool operator==(const Umigame_condition& lhs, const Umigame_condition& rhs) {
    return lhs.max_move_loss == rhs.max_move_loss
        && lhs.black_max_loss == rhs.black_max_loss
        && lhs.white_max_loss == rhs.white_max_loss;
}

inline bool operator!=(const Umigame_condition& lhs, const Umigame_condition& rhs) {
    return !(lhs == rhs);
}

struct Umigame_black_score_interval {
    int lower;
    int upper;
    bool valid;

    Umigame_black_score_interval()
        : lower(-INF), upper(INF), valid(true) {}

    Umigame_black_score_interval(int lower, int upper)
        : lower(lower), upper(upper), valid(lower <= upper) {}
};

inline Umigame_black_score_interval make_umigame_black_score_interval(const Umigame_condition& condition) {
    return Umigame_black_score_interval(-condition.black_max_loss, condition.white_max_loss);
}

inline bool is_within_umigame_black_score_interval(int score_black, const Umigame_black_score_interval& interval) {
    return interval.valid && interval.lower <= score_black && score_black <= interval.upper;
}


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
        int generation;
        bool cache_context_initialized;
        int cache_depth;
        Umigame_condition cache_condition;

    public:
        Umigame()
            : generation(0),
              cache_context_initialized(false),
              cache_depth(0),
              cache_condition() {}

        void calculate(Board *board, int player, int depth, Umigame_condition condition) {
            int search_generation = ensure_cache_context(depth, condition);
            Umigame_black_score_interval interval = make_umigame_black_score_interval(condition);
            if (!interval.valid) {
                return;
            }
            umigame_search(board, player, depth, condition, interval, search_generation);
        }

        void delete_all() {
            std::lock_guard<std::mutex> lock(mtx);
            umigame.clear();
            ++generation;
        }

        int ensure_cache_context(int depth, const Umigame_condition& condition) {
            std::lock_guard<std::mutex> lock(mtx);
            if (!cache_context_initialized || cache_depth != depth || cache_condition != condition) {
                umigame.clear();
                ++generation;
                cache_context_initialized = true;
                cache_depth = depth;
                cache_condition = condition;
            }
            return generation;
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
        Umigame_result umigame_search(Board *b, int player, int depth, const Umigame_condition& condition, const Umigame_black_score_interval& interval, int search_generation) {
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
            if (b->get_legal() == 0ULL) {
                player ^= 1;
                b->pass();
            }
            Flip flip;
            std::vector<Book_value> moves = book.get_all_moves_within_child_loss(b, condition.max_move_loss);
            std::vector<int> policies;
            for (const Book_value& move: moves) {
                int score_black = player == BLACK ? move.value : -move.value;
                if (is_within_umigame_black_score_interval(score_black, interval)) {
                    policies.emplace_back(move.policy);
                }
            }
            //b->print();
            if (policies.size() == 0) {
                umigame_res.b = 1;
                umigame_res.w = 1;
				reg(b, umigame_res, search_generation);
                return umigame_res;
            }
            std::vector<Board> boards;
            for (uint_fast8_t cell: policies) {
                calc_flip(&flip, b, cell);
                boards.emplace_back(b->move_copy(&flip));
            }
            if (player == BLACK) {
                umigame_res.b = INF;
                umigame_res.w = 0;
                for (Board &nnb : boards) {
                    Umigame_result nres = umigame_search(&nnb, player ^ 1, depth, condition, interval, search_generation);
                    umigame_res.b = std::min(umigame_res.b, nres.b);
                    umigame_res.w += nres.w;
                }
            } else {
                umigame_res.b = 0;
                umigame_res.w = INF;
                for (Board &nnb : boards) {
                    Umigame_result nres = umigame_search(&nnb, player ^ 1, depth, condition, interval, search_generation);
                    umigame_res.w = std::min(umigame_res.w, nres.w);
                    umigame_res.b += nres.b;
                }
            }
            if (global_searching) {
                reg(b, umigame_res, search_generation);
            }
            return umigame_res;
        }

        inline void reg(Board *b, Umigame_result val, int search_generation) {
            Board unique_board = representative_board(b->copy());
            std::lock_guard<std::mutex> lock(mtx);
            if (generation != search_generation) {
                return;
            }
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
Umigame_result calculate_umigame(Board *b, int player, int depth, Umigame_condition condition = Umigame_condition()) {
    umigame.ensure_cache_context(depth, condition);
    Umigame_result res = umigame.get_umigame(b);
    if (res.b == UMIGAME_UNDEFINED) {
        umigame.calculate(b, player, depth, condition);
        res = umigame.get_umigame(b);
    }
    return res;
}

Umigame_result calculate_umigame(Board *b, int player, int depth, int max_move_loss) {
    return calculate_umigame(b, player, depth, Umigame_condition(max_move_loss, HW2, HW2));
}
