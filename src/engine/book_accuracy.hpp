/*
    Egaroucid Project

    @file book_accuracy.hpp
        Calculate book accuracy level
        This function is inspired by Uenon Edax: https://uenon1.com/archives/14099254.html
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

constexpr int BOOK_ACCURACY_LEVEL_UNDEFINED = -127;
constexpr int N_BOOK_ACCURACY_LEVEL = 6;
constexpr int BOOK_ACCURACY_A_SHIFT = 5;
// [-2, 2] range
constexpr int BOOK_ACCURACY_LEVEL_AA = (0 - BOOK_ACCURACY_A_SHIFT);
constexpr int BOOK_ACCURACY_LEVEL_AB = (1 - BOOK_ACCURACY_A_SHIFT);
constexpr int BOOK_ACCURACY_LEVEL_AC = (2 - BOOK_ACCURACY_A_SHIFT);
constexpr int BOOK_ACCURACY_LEVEL_AD = (3 - BOOK_ACCURACY_A_SHIFT);
constexpr int BOOK_ACCURACY_LEVEL_AE = (4 - BOOK_ACCURACY_A_SHIFT);
constexpr int BOOK_ACCURACY_LEVEL_AF = (5 - BOOK_ACCURACY_A_SHIFT);
// [-1, 1] range
constexpr int BOOK_ACCURACY_LEVEL_A = 0; // A: all lines calculated with perfect search line
constexpr int BOOK_ACCURACY_LEVEL_B = 1; // B: at least one perfect search line found and other all lines calculated with endgame search line
constexpr int BOOK_ACCURACY_LEVEL_C = 2; // C: all lines calculated with endgame search
constexpr int BOOK_ACCURACY_LEVEL_D = 3; // D: at least one perfect search line found
constexpr int BOOK_ACCURACY_LEVEL_E = 4; // E: at least one endgame search line found
constexpr int BOOK_ACCURACY_LEVEL_F = 5; // F: other

class Book_accuracy {
    private:
        std::mutex mtx;
        std::unordered_map<Board, int, Book_hash> book_accuracy[2]; // 0 for A-F, 1 for AA-AF
    
    public:
        void calculate(Board *board) {
            if (book.contain(board)) {
                int accuracy = book_accuracy_search(board->copy(), false);
                if (accuracy == BOOK_ACCURACY_LEVEL_A) {
                    book_accuracy_search(board->copy(), true);
                }
            }
        }

        void delete_all() {
            std::lock_guard<std::mutex> lock(mtx);
            for (int i = 0; i < 2; ++i) {
                book_accuracy[i].clear();
            }
        }

        int get(Board *board) {
            int res = get_raw(board, false);
            if (res == BOOK_ACCURACY_LEVEL_A) {
                int res2 = get_raw(board, true);
                if (res2 != BOOK_ACCURACY_LEVEL_UNDEFINED) {
                    res = res2 - BOOK_ACCURACY_A_SHIFT;
                } else{
                    res = BOOK_ACCURACY_LEVEL_UNDEFINED;
                }
            }
            return res;
        }

    private:
        int book_accuracy_search(Board board, bool is_high_level) {
            if (!global_searching) {
                return BOOK_ACCURACY_LEVEL_UNDEFINED;
            }
            int res = get_raw(&board, is_high_level);
            if (res != BOOK_ACCURACY_LEVEL_UNDEFINED) {
                return res;
            }
            if (board.get_legal() == 0ULL) {
                board.pass();
            }
            Book_elem book_elem = book.get(board);
            std::vector<Book_value> links = book.get_all_moves_with_value(&board);
            if (links.size() == 0) {
                int complete_depth = 60, endgame_depth = 60;
                if (book_elem.level < N_LEVEL) {
                    complete_depth = get_level_complete_depth(book_elem.level);
                    endgame_depth = get_level_endsearch_depth(book_elem.level);
                }
                int res = BOOK_ACCURACY_LEVEL_F;
                if (complete_depth >= HW2 - board.n_discs()) {
                    res = BOOK_ACCURACY_LEVEL_A;
                } else if (endgame_depth >= HW2 - board.n_discs()) {
                    res = BOOK_ACCURACY_LEVEL_C;
                }
                reg(board, res, is_high_level);
                return res;
            }
            int best_score = -INF;
            for (Book_value &link: links) {
                best_score = std::max(best_score, link.value);
            }
            bool is_end = true;
            uint32_t identifier = 0;
            Flip flip;
            int accept_loss = 1;
            if (is_high_level) {
                accept_loss = 2;
            }
            for (Book_value &link: links) {
                if (link.value >= best_score - accept_loss) {
                    calc_flip(&flip, &board, link.policy);
                    board.move_board(&flip);
                        int child_book_acc = book_accuracy_search(board, is_high_level);
                    board.undo_board(&flip);
                    if (child_book_acc == BOOK_ACCURACY_LEVEL_UNDEFINED) {
                        return BOOK_ACCURACY_LEVEL_UNDEFINED;
                    }
                    for (int i = 0; i < N_BOOK_ACCURACY_LEVEL; ++i) {
                        identifier |= (child_book_acc == i) << i;
                    }
                }
            }
            if (book_elem.leaf.value >= best_score - accept_loss) {
                int complete_depth = 60, endgame_depth = 60;
                if (book_elem.leaf.level < N_LEVEL) {
                    complete_depth = get_level_complete_depth(book_elem.leaf.level);
                    endgame_depth = get_level_endsearch_depth(book_elem.leaf.level);
                }
                if (HW2 - (board.n_discs() + 1) <= complete_depth) {
                    identifier |= 1 << BOOK_ACCURACY_LEVEL_A;
                } else if (HW2 - (board.n_discs() + 1) <= endgame_depth) {
                    identifier |= 1 << BOOK_ACCURACY_LEVEL_C;
                }
            }
            //    COMPLETE  ENDGAME MIDGAME
            // A:        1        0       0
            // B:        1        1       0
            // C:        0        1       0
            // D:        1        -       1
            // E:        0        1       1
            // F:        0        0       1
            // identifier bit : FEDCBA
            // if A:      RES = A or B or D
            // else if B: RES = B or D
            // else if C: RES = C or D
            // else:      RES = best_level
            res = BOOK_ACCURACY_LEVEL_F;
            if (identifier & (1 << BOOK_ACCURACY_LEVEL_A)) { // A found
                if ((identifier & ~((1 << BOOK_ACCURACY_LEVEL_B) - 1)) == 0) { // B-F not found -> RES = A
                    res = BOOK_ACCURACY_LEVEL_A;
                } else if ((identifier & ~((1 << BOOK_ACCURACY_LEVEL_D) - 1)) == 0) { // D-F not found (and B or C found) -> RES = B
                    res = BOOK_ACCURACY_LEVEL_B;
                } else { // B and C not found, D-F found -> RES = D
                    res = BOOK_ACCURACY_LEVEL_D;
                }
            } else if (identifier & (1 << BOOK_ACCURACY_LEVEL_B)) { // B found (A not found)
                if ((identifier & ~((1 << BOOK_ACCURACY_LEVEL_D) - 1)) == 0) { // D-F not found -> RES = B
                    res = BOOK_ACCURACY_LEVEL_B;
                } else { // D-F found -> RES = D
                    res = BOOK_ACCURACY_LEVEL_D;
                }
            } else if (identifier & (1 << BOOK_ACCURACY_LEVEL_C)) { // C found (A and B not found)
                if ((identifier & ~((1 << BOOK_ACCURACY_LEVEL_D) - 1)) == 0) { // D-F not found -> RES = C
                    res = BOOK_ACCURACY_LEVEL_C;
                } else { // D-F found -> RES = D
                    res = BOOK_ACCURACY_LEVEL_D;
                }
            } else { // D-F found (A-C not found)
                uint32_t lsb = (identifier >> 3); // bit: FED
                lsb = pop_count_uint(~lsb & (lsb - 1)); // least bit is D: 0 E: 1 F: 2
                res = BOOK_ACCURACY_LEVEL_D + lsb; // best accuracy level
            }
            reg(board, res, is_high_level);
            return res;
        }

        inline void reg(Board b, int val, bool is_high_level) {
            Board unique_board = get_representative_board(b);
            std::lock_guard<std::mutex> lock(mtx);
            book_accuracy[is_high_level][unique_board] = val;
        }

        inline int get_representive(Board b, bool is_high_level) {
            int res = BOOK_ACCURACY_LEVEL_UNDEFINED;
            if (book_accuracy[is_high_level].find(b) != book_accuracy[is_high_level].end())
                res = book_accuracy[is_high_level][b];
            return res;
        }

        inline bool contain(Board b, bool is_high_level) {
            Board unique_board = get_representative_board(b);
            return book_accuracy[is_high_level].find(unique_board) != book_accuracy[is_high_level].end();
        }

        inline void first_update_representative_board(Board *res, Board *sym) {
            uint64_t vp = vertical_mirror(sym->player);
            uint64_t vo = vertical_mirror(sym->opponent);
            if (res->player > vp || (res->player == vp && res->opponent > vo)) {
                res->player = vp;
                res->opponent = vo;
            }
        }

        inline void update_representative_board(Board *res, Board *sym) {
            if (res->player > sym->player || (res->player == sym->player && res->opponent > sym->opponent))
                sym->copy(res);
            uint64_t vp = vertical_mirror(sym->player);
            uint64_t vo = vertical_mirror(sym->opponent);
            if (res->player > vp || (res->player == vp && res->opponent > vo)) {
                res->player = vp;
                res->opponent = vo;
            }
        }

        inline Board get_representative_board(Board b) {
            Board res = b;
            first_update_representative_board(&res, &b);
            b.board_black_line_mirror();
            update_representative_board(&res, &b);
            b.board_horizontal_mirror();
            update_representative_board(&res, &b);
            b.board_white_line_mirror();
            update_representative_board(&res, &b);
            return res;
        }

        int get_raw(Board *board, bool is_high_level) {
            std::lock_guard<std::mutex> lock(mtx);
            Board unique_board = get_representative_board(board->copy());
            return get_representive(unique_board, is_high_level);
        }
};

Book_accuracy book_accuracy;

int calculate_book_accuracy(Board *b) {
    int res = book_accuracy.get(b);
    if (res == BOOK_ACCURACY_LEVEL_UNDEFINED) {
        book_accuracy.calculate(b);
        res = book_accuracy.get(b);
    }
    return res;
}