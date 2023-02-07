/*
    Egaroucid Project

    @file square.hpp
        Square structure
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include "common.hpp"
#include "const.hpp"
#include "board.hpp"

#define N_SQUARE_TYPE 10
#define EMPTY_UNDEFINED 65

struct Square{
    uint_fast8_t cell;
    uint64_t cell_bit;
    uint_fast8_t parity_id;
    Square *prev;
    Square *next;
};

constexpr uint64_t square_type_mask[N_SQUARE_TYPE] = {
    0x8100000000000081ULL, // corner
    0x0000182424180000ULL, // box edge
    0x0000240000240000ULL, // box corner
    0x2400810000810024ULL, // edge a
    0x1800008181000018ULL, // edge b
    0x0018004242001800ULL, // midedge center
    0x0024420000422400ULL, // midedge corner
    0x4281000000008142ULL, // edge c
    0x0042000000004200ULL, // x
    0x0000001818000000ULL  // center
};

inline void empty_list_init(Square empty_list[], Board *board){
    uint_fast8_t cell, square_type;
    uint64_t mask;
    uint64_t empties = ~(board->player | board->opponent);
    empty_list[0].cell = EMPTY_UNDEFINED;
    empty_list[0].cell_bit = 0;
    empty_list[0].parity_id = 0;
    empty_list[0].prev = nullptr;
    empty_list[0].next = empty_list + 1;
    int idx = 1;
    for (square_type = 0; square_type < N_SQUARE_TYPE; ++square_type){
        mask = square_type_mask[square_type] & empties;
        for (cell = first_bit(&mask); mask; cell = next_bit(&mask)){
            empty_list[idx].cell = cell;
            empty_list[idx].cell_bit = 1ULL << cell;
            empty_list[idx].parity_id = cell_div4[cell];
            empty_list[idx].prev = empty_list + idx - 1;
            empty_list[idx].next = empty_list + idx + 1;
            ++idx;
        }
    }
    empty_list[idx].cell = EMPTY_UNDEFINED;
    empty_list[idx].cell_bit = 0;
    empty_list[idx].parity_id = 0;
    empty_list[idx].prev = empty_list + idx - 1;
    empty_list[idx].next = nullptr;
}

inline void empty_list_move(Square *empty){
    empty->prev->next = empty->next;
    empty->next->prev = empty->prev;
}

inline void empty_list_undo(Square *empty){
    empty->prev->next = empty;
    empty->next->prev = empty;
}

#define foreach_square(square, empty_list, legal) for ((square) = (empty_list[0]).next; (square); (square) = (square)->next) if ((square)->cell != EMPTY_UNDEFINED && (1 & ((legal) >> (square)->cell)))

#define foreach_odd_square(square, empty_list, legal, parity) for ((square) = (empty_list[0]).next; (square); (square) = (square)->next) if ((square)->cell != EMPTY_UNDEFINED && (1 & ((legal) >> (square)->cell)) && ((square)->parity_id & (parity)))

#define foreach_even_square(square, empty_list, legal, parity) for ((square) = (empty_list[0]).next; (square); (square) = (square)->next) if ((square)->cell != EMPTY_UNDEFINED && (1 & ((legal) >> (square)->cell)) && ((square)->parity_id & (parity)) == 0)
