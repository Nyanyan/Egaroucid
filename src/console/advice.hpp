/*
    Egaroucid Project

    @file advice.hpp
        Advice for Human by Egaroucid
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "lib/json.hpp"
#include "./../engine/engine_all.hpp"
#include "board_info.hpp"

constexpr int ADVICE_VALUE_LEVEL = 21;

struct Advice_Move {
    int policy;
    int value;
    bool is_flip_inside;
    bool is_flip_inside_creation;
    bool is_edge;
    bool is_corner;
};

bool is_flip_inside(Board board, uint_fast8_t cell) {
    Flip flip;
    calc_flip(&flip, &board, cell);
    uint64_t discs = board.player | board.opponent | (1ULL << cell);
    uint64_t outside_flip =  (flip.flip & 0x7F7F7F7F7F7F7F7FULL) << 1;
                outside_flip |= (flip.flip & 0xFEFEFEFEFEFEFEFEULL) >> 1;
                outside_flip |= (flip.flip & 0x00FFFFFFFFFFFFFFULL) << HW;
                outside_flip |= (flip.flip & 0xFFFFFFFFFFFFFF00ULL) >> HW;
                outside_flip |= (flip.flip & 0x007F7F7F7F7F7F7FULL) << (HW + 1);
                outside_flip |= (flip.flip & 0xFEFEFEFEFEFEFE00ULL) >> (HW + 1);
                outside_flip |= (flip.flip & 0x00FEFEFEFEFEFEFEULL) << (HW - 1);
                outside_flip |= (flip.flip & 0x7F7F7F7F7F7F7F00ULL) >> (HW - 1);
    return (discs & outside_flip) == outside_flip;
}

bool has_flip_inside(Board board) {
    bool has_flip_inside = false;
    uint64_t legal = board.get_legal();
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        has_flip_inside |= is_flip_inside(board, cell);
    }
    return has_flip_inside;
}

void print_advice(Board_info *board_info) {
    nlohmann::json res;

    Board board = board_info->board.copy();
    res["board"] = board.to_str(board_info->player);

    uint64_t legal = board.get_legal();
    int n_legal = pop_count_ull(legal);
    res["n_legal"] = n_legal;

    std::vector<Advice_Move> good_moves;
    {
        int best_value = -100;
        uint64_t legal_cpy = legal;
        while (legal_cpy) {
            Search_result search_result = ai_legal(board, ADVICE_VALUE_LEVEL, true, 0, true, false, legal_cpy);
            if (search_result.value >= best_value - 2) {
                Advice_Move move;
                move.policy = search_result.policy;
                move.value = search_result.value;
                good_moves.emplace_back(move);
                best_value = std::max(best_value, search_result.value);
            } else {
                break;
            }
            legal_cpy ^= 1ULL << search_result.policy;
        }
        std::cerr << "good moves size " << good_moves.size() << std::endl;
        res["best_value"] = best_value;
        res["n_good_moves"] = good_moves.size();
    }

    res["has_flip_inside"] = has_flip_inside(board);

    Board op_board = board.copy();
    op_board.pass();
    res["has_op_flip_inside"] = has_flip_inside(op_board);

    {
        for (Advice_Move &move: good_moves) {
            move.is_flip_inside = is_flip_inside(board, move.policy);
        }
    }

    {
        Flip flip;
        for (Advice_Move &move: good_moves) {
            calc_flip(&flip, &board, move.policy);
            board.move_board(&flip);
            board.pass();
                move.is_flip_inside_creation = has_flip_inside(board);
            board.pass();
            board.undo_board(&flip);
        }
    }

    for (Advice_Move &move: good_moves) {
        move.is_edge = 0x7E8181818181817EULL & (1ULL << move.policy);
    }

    for (Advice_Move &move: good_moves) {
        move.is_corner = 0x8100000000000081ULL & (1ULL << move.policy);
    }

    res["good_moves"] = nlohmann::json::array();
    for (Advice_Move &move: good_moves) {
        nlohmann::json j = {
            {"move", idx_to_coord(move.policy)},
            {"value", move.value},
            {"is_flip_inside", move.is_flip_inside},
            {"is_flip_inside_creation", move.is_flip_inside_creation},
            {"is_edge", move.is_edge},
            {"is_corner", move.is_corner}
        };
        res["good_moves"].push_back(j);
    }

    std::cout << res.dump() << std::endl;
}
