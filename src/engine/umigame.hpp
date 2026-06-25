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
    @brief Recursive search condition for Umigame's value

    The default condition preserves the original best-move-only Umigame search.
    GUI callers can enable generalized Umigame search by passing a black-score
    range; that range is applied to every internal child branch.
*/
struct Umigame_condition {
    bool use_score_range;
    int score_min;
    int score_max;
    int integration_error;

    Umigame_condition()
        : use_score_range(false),
          score_min(-HW2),
          score_max(HW2),
          integration_error(0) {}

    Umigame_condition(int score_min, int score_max)
        : Umigame_condition(score_min, score_max, 0) {}

    Umigame_condition(int score_min, int score_max, int integration_error)
        : use_score_range(true),
          score_min(std::min(score_min, score_max)),
          score_max(std::max(score_min, score_max)),
          integration_error(std::max(0, integration_error)) {}

    bool accepts_black_score(int score_black) const {
        return !use_score_range || (score_min <= score_black && score_black <= score_max);
    }

    bool uses_original_best_moves() const {
        return integration_error == 0 &&
            (!use_score_range || (score_min <= -HW2 && HW2 <= score_max));
    }

    bool filters_score_range() const {
        return use_score_range && (score_min > -HW2 || score_max < HW2);
    }
};

inline bool operator==(const Umigame_condition& lhs, const Umigame_condition& rhs) {
    return lhs.use_score_range == rhs.use_score_range &&
        lhs.score_min == rhs.score_min &&
        lhs.score_max == rhs.score_max &&
        lhs.integration_error == rhs.integration_error;
}

inline bool operator!=(const Umigame_condition& lhs, const Umigame_condition& rhs) {
    return !(lhs == rhs);
}

inline int umigame_book_value_to_black_score(int value, int player) {
    return player == BLACK ? value : -value;
}

struct Umigame_cache_key {
    Board board;
    int remaining_error;

    bool operator==(const Umigame_cache_key& other) const {
        return board == other.board && remaining_error == other.remaining_error;
    }
};

struct Umigame_cache_hash {
    size_t operator()(const Umigame_cache_key& key) const {
        size_t seed = Book_hash()(key.board);
        seed ^= static_cast<size_t>(key.remaining_error + HW2) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        return seed;
    }
};


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
        std::unordered_map<Umigame_cache_key, Umigame_result, Umigame_cache_hash> umigame;
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

        Umigame_result calculate(Board *board, int player, int depth, Umigame_condition condition) {
            int search_generation = ensure_cache_context(depth, condition);
            return calculate(board, player, depth, condition, search_generation);
        }

        Umigame_result calculate(Board *board, int player, int depth, Umigame_condition condition, int search_generation) {
            return umigame_search(board, player, depth, condition, condition.integration_error, search_generation);
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
            return get_oneumigame(unique_board, 0);
        }

        Umigame_result get_umigame(Board *b, int remaining_error, int search_generation) {
            std::lock_guard<std::mutex> lock(mtx);
            if (generation != search_generation) {
                return Umigame_result();
            }
            Board unique_board = representative_board(b->copy());
            return get_oneumigame(unique_board, remaining_error);
        }

    private:
        /*
            @brief Result of umigame value search 

            @param b                            board to solve
            @param player                       the player of this board
            @return Umigame's value
        */
        Umigame_result umigame_search(Board *b, int player, int depth, const Umigame_condition& condition, int remaining_error, int search_generation) {
            Umigame_result umigame_res;
			if (!global_searching)
                return umigame_res;
			if (!book.contain(b) || b->n_discs() >= depth + 4) {
				umigame_res.b = 1;
                umigame_res.w = 1;
                return umigame_res;
            }
            umigame_res = get_umigame(b, remaining_error, search_generation);
            if (umigame_res.b != UMIGAME_UNDEFINED)
                return umigame_res;
            if (b->get_legal() == 0ULL) {
                player ^= 1;
                b->pass();
            }
            Flip flip;
            std::vector<std::pair<int, int>> candidate_moves = get_candidate_moves(b, player, condition, remaining_error);
            //b->print();
            if (candidate_moves.size() == 0) {
                umigame_res.b = 1;
                umigame_res.w = 1;
				reg(b, umigame_res, remaining_error, search_generation);
                return umigame_res;
            }
            std::vector<Board> boards;
            std::vector<int> remaining_errors;
            for (const std::pair<int, int>& candidate_move: candidate_moves) {
                int cell = candidate_move.first;
                calc_flip(&flip, b, cell);
                boards.emplace_back(b->move_copy(&flip));
                remaining_errors.emplace_back(candidate_move.second);
            }
            if (player == BLACK) {
                umigame_res.b = INF;
                umigame_res.w = 0;
                for (size_t i = 0; i < boards.size(); ++i) {
                    Umigame_result nres = umigame_search(&boards[i], player ^ 1, depth, condition, remaining_errors[i], search_generation);
                    umigame_res.b = std::min(umigame_res.b, nres.b);
                    umigame_res.w += nres.w;
                }
            } else {
                umigame_res.b = 0;
                umigame_res.w = INF;
                for (size_t i = 0; i < boards.size(); ++i) {
                    Umigame_result nres = umigame_search(&boards[i], player ^ 1, depth, condition, remaining_errors[i], search_generation);
                    umigame_res.w = std::min(umigame_res.w, nres.w);
                    umigame_res.b += nres.b;
                }
            }
            if (global_searching) {
                reg(b, umigame_res, remaining_error, search_generation);
            }
            return umigame_res;
        }

        inline std::vector<std::pair<int, int>> get_candidate_moves(Board *b, int player, const Umigame_condition& condition, int remaining_error) {
            std::vector<std::pair<int, int>> policies;
            if (remaining_error == 0) {
                for (int policy: book.get_all_best_moves(b)) {
                    if (!condition.uses_original_best_moves() && condition.filters_score_range()) {
                        int value;
                        if (!get_registered_move_value(b, policy, &value) ||
                            !condition.accepts_black_score(umigame_book_value_to_black_score(value, player))) {
                            continue;
                        }
                    }
                    policies.emplace_back(policy, remaining_error);
                }
                return policies;
            }
            std::vector<Book_value> moves = get_registered_moves_with_value(b);
            int best_value = -INF;
            for (const Book_value& move: moves) {
                best_value = std::max(best_value, move.value);
            }
            for (const Book_value& move: moves) {
                int local_loss = best_value - move.value;
                if (local_loss <= remaining_error &&
                    condition.accepts_black_score(umigame_book_value_to_black_score(move.value, player))) {
                    policies.emplace_back(move.policy, remaining_error - local_loss);
                }
            }
            return policies;
        }

        inline bool get_registered_move_value(Board *b, int policy, int *value) {
            Flip flip;
            calc_flip(&flip, b, policy);
            b->move_board(&flip);

            bool found = false;
            if (b->is_end()) {
                if (book.contain(b)) {
                    *value = -book.get(b).value;
                    found = true;
                } else {
                    b->pass();
                    if (book.contain(b)) {
                        *value = book.get(b).value;
                        found = true;
                    }
                    b->pass();
                }
            } else if (b->get_legal() == 0ULL) {
                b->pass();
                if (book.contain(b)) {
                    *value = book.get(b).value;
                    found = true;
                }
                b->pass();
            } else if (book.contain(b)) {
                *value = -book.get(b).value;
                found = true;
            }

            b->undo_board(&flip);
            return found;
        }

        inline std::vector<Book_value> get_registered_moves_with_value(Board *b) {
            std::vector<Book_value> moves;
            uint64_t legal = b->get_legal();
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                int value;
                if (get_registered_move_value(b, static_cast<int>(cell), &value)) {
                    Book_value move;
                    move.policy = static_cast<int>(cell);
                    move.value = value;
                    moves.emplace_back(move);
                }
            }
            return moves;
        }

        inline void reg(Board *b, Umigame_result val, int remaining_error, int search_generation) {
            Board unique_board = representative_board(b->copy());
            std::lock_guard<std::mutex> lock(mtx);
            if (generation != search_generation) {
                return;
            }
            umigame[Umigame_cache_key{unique_board, remaining_error}] = val;
        }

        /*
            @brief get registered umigame's value

            @param b                    a board to find
            @return registered umigame's value (if not registered, returns defalut)
        */
        inline Umigame_result get_oneumigame(Board b, int remaining_error) {
            Umigame_result res;
            res.b = UMIGAME_UNDEFINED;
            res.w = UMIGAME_UNDEFINED;
            Umigame_cache_key key{b, remaining_error};
            if (umigame.find(key) != umigame.end()) {
                res = umigame[key];
            }
            return res;
        }

        inline bool contain(Board b) {
            Board unique_board = representative_board(b);
            return umigame.find(Umigame_cache_key{unique_board, 0}) != umigame.end();
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
    int search_generation = umigame.ensure_cache_context(depth, condition);
    Umigame_result res = umigame.get_umigame(b, condition.integration_error, search_generation);
    if (res.b == UMIGAME_UNDEFINED) {
        res = umigame.calculate(b, player, depth, condition, search_generation);
    }
    return res;
}
