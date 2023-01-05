/*
    Egaroucid Project

    @file umigame.hpp
        Calculate Minimum Memorization Number a.k.a. Umigame's value
        Umigame's value is published here: https://umigamep.github.io/BookAdviser/
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <unordered_map>
#include "common.hpp"
#include "board.hpp"
#include "book.hpp"

/*
    @brief Result of umigame value search 

    @param b                            black player's umigame value
    @param w                            white player's umigame value
*/
struct Umigame_result {
    int b;
    int w;
    Umigame_result operator+(const Umigame_result& other) {
        Umigame_result res;
        res.b = b + other.b;
        res.w = w + other.w;
        return res;
    }
};

/*
    @brief Constants for Umigame's value
*/
#define UMIGAME_SEARCH_DEPTH 100
#define UMIGAME_UNDEFINED -1

/*
    @brief Result of umigame value search 

    @param b                            board to solve
    @param depth                        remaining depth
    @param player                       the player of this board
    @param target_player                target player to calculate umigame's value
    @return Umigame's value
*/
int umigame_search(Board *b, int depth, int player, const int target_player){
    if (depth == 0)
        return 1;
    if (!global_searching)
        return 0;
    int val, max_val = -INF;
    std::vector<Board> boards;
    uint64_t legal = b->get_legal();
    if (legal == 0ULL){
        player ^= 1;
        b->pass();
        legal = b->get_legal();
    }
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, b, cell);
        b->move_board(&flip);
            val = book.get(b);
            if (val != -INF && val >= max_val) {
                if (val > max_val) {
                    boards.clear();
                    max_val = val;
                }
                boards.emplace_back(b->copy());
            }
        b->undo_board(&flip);
    }
    if (boards.size() == 0)
        return 1;
    int res;
    if (player == target_player) {
        res = INF;
        for (Board &nnb : boards)
            res = std::min(res, umigame_search(&nnb, depth - 1, player ^ 1, target_player));
    } else {
        res = 0;
        for (Board &nnb : boards)
            res += umigame_search(&nnb, depth - 1, player ^ 1, target_player);
    }
    return res;
}

/*
    @brief Calculate Umigame's value

    @param b                            board to solve
    @param player                       the player of this board
    @return Umigame's value in Umigame_result structure
*/
Umigame_result calculate_umigame(Board *b, int player) {
    Umigame_result res;
    if (book.get(b) != -INF){
        res.b = umigame_search(b, UMIGAME_SEARCH_DEPTH, player, BLACK);
        res.w = umigame_search(b, UMIGAME_SEARCH_DEPTH, player, WHITE);
    } else{
        res.b = UMIGAME_UNDEFINED;
        res.w = UMIGAME_UNDEFINED;
    }
    return res;
}