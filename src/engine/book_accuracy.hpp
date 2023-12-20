/*
    Egaroucid Project

    @file book_accuracy.hpp
        Calculate book accuracy level
        This function is inspired by Uenon Edax: https://uenon1.com/archives/14099254.html
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <unordered_map>
#include "common.hpp"
#include "board.hpp"
#include "book.hpp"

#define BOOK_ACCURACY_LEVEL_UNDEFINED -1
#define BOOK_ACCURACY_LEVEL_A 0 // A: all lines calculated with perfect search line
#define BOOK_ACCURACY_LEVEL_B 1 // B: at least one perfect search line found and other all lines calculated with endgame search line
#define BOOK_ACCURACY_LEVEL_C 2 // C: all lines calculated with endgame search
#define BOOK_ACCURACY_LEVEL_D 3 // D: at least one perfect search line found
#define BOOK_ACCURACY_LEVEL_E 4 // E: at least one endgame search line found
#define BOOK_ACCURACY_LEVEL_F 5 // F: other
#define N_BOOK_ACCURACY_LEVEL 6

class Book_accuracy{
    private:
        std::mutex mtx;
        std::unordered_map<Board, int, Book_hash> book_accuracy;
    
    public:
        void calculate(Board *board){
            if (book.contain(board))
                book_accuracy_search(board->copy());
        }

        void delete_all(){
            std::lock_guard<std::mutex> lock(mtx);
            book_accuracy.clear();
        }

        int get(Board *board){
            std::lock_guard<std::mutex> lock(mtx);
            Board unique_board = get_representative_board(board->copy());
            return get_representive(unique_board);
        }

    private:
        int book_accuracy_search(Board board){
            if (!global_searching)
                return BOOK_ACCURACY_LEVEL_UNDEFINED;
            int res = get(&board);
            if (res != BOOK_ACCURACY_LEVEL_UNDEFINED)
                return res;
            if (board.get_legal() == 0ULL)
                board.pass();
            Book_elem book_elem = book.get(board);
            std::vector<Book_value> links = book.get_all_moves_with_value(&board);
            if (links.size() == 0){
                int complete_depth = 60, endgame_depth = 60;
                if (book_elem.level < N_LEVEL){
                    complete_depth = get_level_complete_depth(book_elem.level);
                    endgame_depth = get_level_endgame_depth(book_elem.level);
                }
                int res = BOOK_ACCURACY_LEVEL_F;
                if (complete_depth >= HW2 - board.n_discs())
                    res = BOOK_ACCURACY_LEVEL_A;
                else if (endgame_depth >= HW2 - board.n_discs())
                    res = BOOK_ACCURACY_LEVEL_C;
                reg(board, res);
                return res;
            }
            int best_score = -INF;
            for (Book_value &link: links)
                best_score = std::max(best_score, link.value);
            bool is_end = true;
            int identifier = 0;
            Flip flip;
            for (Book_value &link: links){
                if (link.value >= best_score - 1){
                    calc_flip(&flip, &board, link.policy);
                    board.move_board(&flip);
                        int child_book_acc = book_accuracy_search(board);
                    board.undo_board(&flip);
                    if (child_book_acc == BOOK_ACCURACY_LEVEL_UNDEFINED)
                        return BOOK_ACCURACY_LEVEL_UNDEFINED;
                    for (int i = 0; i < N_BOOK_ACCURACY_LEVEL; ++i)
                        identifier |= (child_book_acc == i) << i;
                }
            }
            // identifier : FEDCBA
            // if A:      RES = A or B or D
            // else if B: RES = B or D
            // else if C: RES = C or D
            // else:      RES = best_level
            res = BOOK_ACCURACY_LEVEL_F;
            if (identifier & (1 << BOOK_ACCURACY_LEVEL_A)){ // A found
                if ((identifier & ~((1 << BOOK_ACCURACY_LEVEL_B) - 1)) == 0) // B-F not found
                    res = BOOK_ACCURACY_LEVEL_A;
                else if ((identifier & ~((1 << BOOK_ACCURACY_LEVEL_D) - 1)) == 0) // D-F not found
                    res = BOOK_ACCURACY_LEVEL_B;
                else
                    res = BOOK_ACCURACY_LEVEL_D;
            } else if (identifier & (1 << BOOK_ACCURACY_LEVEL_B)){ // B found
                if ((identifier & ~((1 << BOOK_ACCURACY_LEVEL_C) - 1)) == 0) // C-F not found
                    res = BOOK_ACCURACY_LEVEL_B;
                else 
                    res = BOOK_ACCURACY_LEVEL_D;
            } else if (identifier & (1 << BOOK_ACCURACY_LEVEL_C)){ // C found
                if ((identifier & ~((1 << BOOK_ACCURACY_LEVEL_D) - 1)) == 0) // D-F not found
                    res = BOOK_ACCURACY_LEVEL_C;
                else 
                    res = BOOK_ACCURACY_LEVEL_D;
            } else{
                int lsb = (identifier >> 3);
                lsb = pop_count_uint(~lsb & (lsb - 1));
                res = BOOK_ACCURACY_LEVEL_D + lsb; // best level
            }
            reg(board, res);
            return res;
        }

        inline void reg(Board b, int val){
            Board unique_board = get_representative_board(b);
            std::lock_guard<std::mutex> lock(mtx);
            book_accuracy[unique_board] = val;
        }

        inline int get_representive(Board b){
            int res = BOOK_ACCURACY_LEVEL_UNDEFINED;
            if (book_accuracy.find(b) != book_accuracy.end())
                res = book_accuracy[b];
            return res;
        }

        inline bool contain(Board b){
            Board unique_board = get_representative_board(b);
            return book_accuracy.find(unique_board) != book_accuracy.end();
        }

        inline void first_update_representative_board(Board *res, Board *sym){
            uint64_t vp = vertical_mirror(sym->player);
            uint64_t vo = vertical_mirror(sym->opponent);
            if (res->player > vp || (res->player == vp && res->opponent > vo)){
                res->player = vp;
                res->opponent = vo;
            }
        }

        inline void update_representative_board(Board *res, Board *sym){
            if (res->player > sym->player || (res->player == sym->player && res->opponent > sym->opponent))
                sym->copy(res);
            uint64_t vp = vertical_mirror(sym->player);
            uint64_t vo = vertical_mirror(sym->opponent);
            if (res->player > vp || (res->player == vp && res->opponent > vo)){
                res->player = vp;
                res->opponent = vo;
            }
        }

        inline Board get_representative_board(Board b){
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
};

Book_accuracy book_accuracy;

int calculate_book_accuracy(Board *b) {
    int res = book_accuracy.get(b);
    if (res == BOOK_ACCURACY_LEVEL_UNDEFINED){
        book_accuracy.calculate(b);
        res = book_accuracy.get(b);
    }
    return res;
}
