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
    int n_flipped_discs;
    int n_flipped_direction;
    bool is_flip_inside;
    bool is_flip_inside_creation;
    bool is_op_flip_inside_creation;
    bool is_op_flip_inside_deletion;
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
        if ((1ULL << cell) & 0x0042000000004200ULL) { // exclude X
            continue;
        }
        has_flip_inside |= is_flip_inside(board, cell);
    }
    return has_flip_inside;
}

uint64_t get_flip_inside_places(Board board) {
    uint64_t flip_inside_places = 0;
    uint64_t legal = board.get_legal();
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        if ((1ULL << cell) & 0x0042000000004200ULL) { // exclude X
            continue;
        }
        if (is_flip_inside(board, cell)) {
            flip_inside_places |= 1ULL << cell;
        }
    }
    return flip_inside_places;
}

void print_advice(Board_info *board_info) {
    nlohmann::json res;

    Board board = board_info->board.copy();
    res["board"] = board.to_str(board_info->player);

    uint64_t legal = board.get_legal();
    int n_legal = pop_count_ull(legal);
    res["n_legal"] = n_legal;

    std::vector<Advice_Move> moves;
    {
        int best_value = -100;
        uint64_t legal_cpy = legal;
        while (legal_cpy) {
            Search_result search_result = ai_legal(board, ADVICE_VALUE_LEVEL, true, 0, true, false, legal_cpy);
            // if (search_result.value >= best_value - 2) {
                Advice_Move move;
                move.policy = search_result.policy;
                move.value = search_result.value;
                moves.emplace_back(move);
                best_value = std::max(best_value, search_result.value);
            // } else {
            //     break;
            // }
            legal_cpy ^= 1ULL << search_result.policy;
        }
        // std::cerr << "good moves size " << moves.size() << std::endl;
        res["best_value"] = best_value;
        // res["n_moves"] = moves.size();
    }

    res["has_flip_inside"] = has_flip_inside(board);

    Board op_board = board.copy();
    op_board.pass();
    res["has_op_flip_inside"] = has_flip_inside(op_board);
    uint64_t op_flip_inside_board = get_flip_inside_places(op_board);

    {
        Flip flip;
        for (Advice_Move &move: moves) {
            calc_flip(&flip, &board, move.policy);
            move.n_flipped_discs = pop_count_ull(flip.flip);
            move.n_flipped_direction = 0;
            constexpr int dy[8] = {-1, -1, -1, 0,  0,  1, 1, 1};
            constexpr int dx[8] = {-1,  0,  1, 1, -1, -1, 0, 1};
            int y = move.policy / HW;
            int x = move.policy % HW;
            for (int dr = 0; dr < 8; ++dr) {
                int ny = y + dy[dr];
                int nx = x + dx[dr];
                if (0 <= ny && ny < HW && 0 <= nx && nx < HW) {
                    int cell = ny * HW + nx;
                    if (flip.flip & (1ULL << cell)) {
                        ++move.n_flipped_direction;
                    }
                }
            }
        }
    }

    {
        for (Advice_Move &move: moves) {
            move.is_flip_inside = is_flip_inside(board, move.policy);
        }
    }

    {
        Flip flip;
        for (Advice_Move &move: moves) {
            calc_flip(&flip, &board, move.policy);
            board.move_board(&flip);
            board.pass();
                move.is_flip_inside_creation = has_flip_inside(board);
            board.pass();
            board.undo_board(&flip);
        }
    }

    {
        Flip flip;
        for (Advice_Move &move: moves) {
            calc_flip(&flip, &board, move.policy);
            board.move_board(&flip);
                uint64_t op_flip_inside_board_2 = get_flip_inside_places(board);
                move.is_op_flip_inside_creation = (~op_flip_inside_board) & op_flip_inside_board_2;
                if (res["has_op_flip_inside"]) {
                    move.is_op_flip_inside_deletion = op_flip_inside_board & (~op_flip_inside_board_2);
                } else {
                    move.is_op_flip_inside_deletion = false;
                }
            board.undo_board(&flip);
        }
    }

    for (Advice_Move &move: moves) {
        move.is_edge = 0x7E8181818181817EULL & (1ULL << move.policy);
    }

    for (Advice_Move &move: moves) {
        move.is_corner = 0x8100000000000081ULL & (1ULL << move.policy);
    }

    res["moves"] = nlohmann::json::array();
    for (Advice_Move &move: moves) {
        nlohmann::json j = {
            {"move", idx_to_coord(move.policy)},
            {"value", move.value},
            {"n_flipped_discs", move.n_flipped_discs},
            {"n_flipped_direction", move.n_flipped_direction},
            {"is_flip_inside", move.is_flip_inside},
            {"is_flip_inside_creation", move.is_flip_inside_creation},
            {"is_op_flip_inside_creation", move.is_op_flip_inside_creation},
            {"is_op_flip_inside_deletion", move.is_op_flip_inside_deletion},
            {"is_edge", move.is_edge},
            {"is_corner", move.is_corner},
        };
        res["moves"].push_back(j);
    }

    std::cout << res.dump() << std::endl;
}
