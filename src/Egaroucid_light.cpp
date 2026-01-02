/*
    Egaroucid Project

    @file Egaroucid_light.cpp
        Main file for Egaroucid Light

    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <iostream>
#include <string>
#include "web/ai.hpp"

inline void init() {
    board_init();
    mobility_init();
    stability_init();
    parent_transpose_table.first_init();
    child_transpose_table.first_init();
    evaluate_init();
    #ifndef NO_BOOK
        book_init();
    #endif
}

Board input_board_po() {
    Board board;
    board.player = 0;
    board.opponent = 0;
    std::string s;
    std::cin >> s;
    for (int i = 0; i < HW2; ++i) {
        if (s[i] == 'P')
            board.player |= 1ULL << (HW2_M1 - i);
        else if (s[i] == 'O')
            board.opponent |= 1ULL << (HW2_M1 - i);
    }
    return board;
}

int main() {
    init();
    Board board;
    #ifndef NO_BOOK
        constexpr bool use_book = true;
    #else
        constexpr bool use_book = false;
    #endif
    int level = 10;
    bool show_log = true;
    while (true) {
        board = input_board_po();
        board.print();

        // Search search;
        // search.init_board(&board);
        // calc_features(&search);
        // std::cerr << mid_evaluate_diff(&search) << std::endl;
        // uint64_t legal = board.get_legal();
        // Flip flip;
        // for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        //     calc_flip(&flip, &search.board, cell);
        //     search.move(&flip);
        //     eval_move(&search, &flip);
        //         search.board.print();
        //         std::cerr << idx_to_coord(cell) << " " << mid_evaluate_diff(&search) << std::endl;
        //     eval_undo(&search, &flip);
        //     search.undo(&flip);
        // }
        // return 0;
        
        Search_result search_result = ai(board, level, use_book, false, show_log);
        std::cout << idx_to_coord(search_result.policy) << " " << search_result.value << std::endl;
    }
    return 0;
}