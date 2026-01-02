/*
    Egaroucid Project

    @file advice.hpp
        Advice for Human by Egaroucid
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "lib/json.hpp"
#include "./../engine/engine_all.hpp"
#include "board_info.hpp"

constexpr int ADVICE_VALUE_LEVEL = 21;

constexpr uint64_t flip_inside_exclude_mask = 0xFFC381818181C3FFULL; //0xC3C300000000C3C3ULL;

struct Advice_Move {
    int policy;
    int value;
    int n_flipped_discs;
    int n_flipped_discs_except_edge;
    int n_flipped_direction;
    Board board;
    int player;
    bool is_flip_inside;
    bool is_flip_inside_creation;
    bool is_op_flip_inside_creation;
    bool is_op_flip_inside_deletion;
    bool is_edge;
    bool is_corner;
    bool is_c;
    bool is_x;
    int next_corner_status;
    int next_a_status;
    bool is_corner_offer_avoidance;
    int offering_avoid_corner;
    bool is_corner_aiming;
    int aiming_corner;
    bool is_offer_corner;
    int offering_corner;
    bool is_mid_edge_side_flip;
    int next_op_n_legal;
    int next_pl_n_legal;
    int n_connected_empty_squares;
    bool op_canput;
    int n_increased_stable_discs;
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
        if ((1ULL << cell) & flip_inside_exclude_mask) { // exclude
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
        if ((1ULL << cell) & flip_inside_exclude_mask) { // exclude
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
    
    res["n_discs"] = board.n_discs();

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
    uint64_t flip_inside_board = get_flip_inside_places(board);

    Board op_board = board.copy();
    op_board.pass();
    res["has_op_flip_inside"] = has_flip_inside(op_board);
    uint64_t op_flip_inside_board = get_flip_inside_places(op_board);
    
    uint64_t op_legal = op_board.get_legal();

    {
        Flip flip;
        for (Advice_Move &move: moves) {
            calc_flip(&flip, &board, move.policy);
            move.n_flipped_discs = pop_count_ull(flip.flip);
            move.n_flipped_discs_except_edge = pop_count_ull(flip.flip & 0x007E7E7E7E7E7E00ULL);
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
            board.move_board(&flip);
                move.board = board;
                move.player = board_info->player ^ 1;
                move.next_op_n_legal = pop_count_ull(board.get_legal());
                board.pass();
                    move.next_pl_n_legal = pop_count_ull(board.get_legal());
                board.pass();
            board.undo_board(&flip);
        }
    }

    {
        for (Advice_Move &move: moves) {
            if ((1ULL << move.policy) & flip_inside_exclude_mask) { // exclude
                move.is_flip_inside = false;
            } else {
                move.is_flip_inside = is_flip_inside(board, move.policy);
            }
        }
    }

    {
        Flip flip;
        for (Advice_Move &move: moves) {
            calc_flip(&flip, &board, move.policy);
            board.move_board(&flip);
            board.pass();
                uint64_t next_flip_insides = get_flip_inside_places(board);
                move.is_flip_inside_creation = next_flip_insides & (~flip_inside_board);
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
    {
        bool has_edge_move = false;
        for (Advice_Move &move: moves) {
            has_edge_move |= move.is_edge;
        }
        res["has_edge_move"] = has_edge_move;
    }


    for (Advice_Move &move: moves) {
        move.is_corner = 0x8100000000000081ULL & (1ULL << move.policy);
    }
    {
        bool has_corner_move = false;
        for (Advice_Move &move: moves) {
            has_corner_move |= move.is_corner;
        }
        res["has_corner_move"] = has_corner_move;
    }

    for (Advice_Move &move: moves) {
        const int next_corner[HW2] = {
            -1,  0, -1, -1, -1, -1,  7, -1, 
             0,  0, -1, -1, -1, -1,  7,  7, 
            -1, -1, -1, -1, -1, -1, -1, -1, 
            -1, -1, -1, -1, -1, -1, -1, -1, 
            -1, -1, -1, -1, -1, -1, -1, -1, 
            -1, -1, -1, -1, -1, -1, -1, -1, 
            56, 56, -1, -1, -1, -1, 63, 63, 
            -1, 56, -1, -1, -1, -1, 63, -1
        };
        /*
         0,  1,  2,  3,  4,  5,  6,  7, 
         8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 
        56, 57, 58, 59, 60, 61, 62, 63
        */
        const int next_a[HW2] = {
            -1,  2, -1, -1, -1, -1,  5, -1, 
            16, -1, -1, -1, -1, -1, -1, 23, 
            -1, -1, -1, -1, -1, -1, -1, -1, 
            -1, -1, -1, -1, -1, -1, -1, -1, 
            -1, -1, -1, -1, -1, -1, -1, -1, 
            -1, -1, -1, -1, -1, -1, -1, -1, 
            40, -1, -1, -1, -1, -1, -1, 47, 
            -1, 58, -1, -1, -1, -1, 61, -1
        };
        move.is_c = (1ULL << move.policy) & 0x4281000000008142ULL;
        move.is_x = (1ULL << move.policy) & 0x0042000000004200ULL;
        move.next_corner_status = -1;
        move.next_a_status = -1;
        if (move.is_c | move.is_x) {
            if (board.player & (1ULL << next_corner[move.policy])) {
                move.next_corner_status = 0;
            } else if (board.opponent & (1ULL << next_corner[move.policy])){
                move.next_corner_status = 1;
            }

            if (move.is_c) {
                if (board.player & (1ULL << next_a[move.policy])) {
                    move.next_a_status = 0;
                } else if (board.opponent & (1ULL << next_a[move.policy])){
                    move.next_a_status = 1;
                }
            }
        }
    }

    for (Advice_Move &move: moves) {
        uint64_t empties = ~(board.player | board.opponent);
        uint64_t bit = 1ULL << move.policy;
        for (int i = 0; i < HW2; ++i) {
            uint64_t n_bit = bit;
            n_bit |= ((bit & 0x7F7F7F7F7F7F7F7FULL) << 1) & empties;
            n_bit |= ((bit & 0xFEFEFEFEFEFEFEFEULL) >> 1) & empties;
            n_bit |= ((bit & 0x00FFFFFFFFFFFFFFULL) << HW) & empties;
            n_bit |= ((bit & 0xFFFFFFFFFFFFFF00ULL) >> HW) & empties;
            n_bit |= ((bit & 0x00FEFEFEFEFEFEFEULL) << HW_M1) & empties;
            n_bit |= ((bit & 0x7F7F7F7F7F7F7F00ULL) >> HW_M1) & empties;
            n_bit |= ((bit & 0x007F7F7F7F7F7F7FULL) << HW_P1) & empties;
            n_bit |= ((bit & 0xFEFEFEFEFEFEFE00ULL) >> HW_P1) & empties;
            if (n_bit == bit) {
                break;
            }
            bit = n_bit;
        }
        move.n_connected_empty_squares = pop_count_ull(bit);
    }

    for (Advice_Move &move: moves) {
        Flip flip;
        calc_flip(&flip, &board, move.policy);
        board.move_board(&flip);
            move.is_offer_corner = false;
            move.offering_corner = -1;
            uint64_t offering_corner = board.get_legal() & ~op_legal & 0x8100000000000081ULL;
            if (offering_corner) {
                move.is_offer_corner = true;
                move.offering_corner = first_bit(&offering_corner);
            }
        board.undo_board(&flip);
    }

    for (Advice_Move &move: moves) {
        Flip flip;
        calc_flip(&flip, &board, move.policy);
        move.is_mid_edge_side_flip = ((1ULL << move.policy) & 0x003C424242423C00ULL) && (flip.flip & 0x003C424242423C00ULL);
    }

    for (Advice_Move &move: moves) {
        Flip flip;
        calc_flip(&flip, &board, move.policy);
        board.move_board(&flip);
            move.is_corner_offer_avoidance = false;
            move.offering_avoid_corner = -1;
            uint64_t offering_avoid_corner_bit = ~board.get_legal() & op_legal & 0x8100000000000081ULL;
            if (offering_avoid_corner_bit) {
                move.is_corner_offer_avoidance = true;
                move.offering_avoid_corner = first_bit(&offering_avoid_corner_bit);
            }
        board.undo_board(&flip);
    }

    for (Advice_Move &move: moves) {
        Flip flip;
        calc_flip(&flip, &board, move.policy);
        board.move_board(&flip);
        board.pass();
            move.is_corner_aiming = false;
            move.aiming_corner = -1;
            uint64_t aiming_corner = board.get_legal() & ~legal & 0x8100000000000081ULL;
            if (aiming_corner) {
                move.is_corner_aiming = true;
                move.aiming_corner = first_bit(&aiming_corner);
            }
        board.pass();
        board.undo_board(&flip);
    }

    for (Advice_Move &move: moves) {
        move.op_canput = op_legal & (1ULL << move.policy);
    }

    int n_stable = pop_count_ull(calc_stability(board.player, board.opponent));
    for (Advice_Move &move: moves) {
        Flip flip;
        calc_flip(&flip, &board, move.policy);
        board.move_board(&flip);
            move.n_increased_stable_discs = pop_count_ull(calc_stability(board.opponent, board.player)) - n_stable;
        board.undo_board(&flip);
    }

    {
        bool has_c = false;
        bool has_x = false;
        for (Advice_Move &move: moves) {
            has_c |= move.is_c;
            has_x |= move.is_x;
        }
        res["has_c"] = has_c;
        res["has_x"] = has_x;
    }

    res["moves"] = nlohmann::json::array();
    for (Advice_Move &move: moves) {
        nlohmann::json j = {
            {"move", idx_to_coord(move.policy)},
            {"value", move.value},
            {"board", move.board.to_str(move.player)},
            {"n_flipped_discs", move.n_flipped_discs},
            {"n_flipped_discs_except_edge", move.n_flipped_discs_except_edge},
            {"n_flipped_direction", move.n_flipped_direction},
            {"is_flip_inside", move.is_flip_inside},
            {"is_flip_inside_creation", move.is_flip_inside_creation},
            {"is_op_flip_inside_creation", move.is_op_flip_inside_creation},
            {"is_op_flip_inside_deletion", move.is_op_flip_inside_deletion},
            {"is_edge", move.is_edge},
            {"is_corner", move.is_corner},
            {"is_c", move.is_c},
            {"is_x", move.is_x},
            {"next_corner_status", move.next_corner_status},
            {"next_a_status", move.next_a_status},
            {"is_corner_offer_avoidance", move.is_corner_offer_avoidance},
            {"offering_avoid_corner", idx_to_coord(move.offering_avoid_corner)},
            {"is_corner_aiming", move.is_corner_aiming},
            {"aiming_corner", idx_to_coord(move.aiming_corner)},
            {"is_offer_corner", move.is_offer_corner},
            {"offering_corner", idx_to_coord(move.offering_corner)},
            {"is_mid_edge_side_flip", move.is_mid_edge_side_flip},
            {"next_op_n_legal", move.next_op_n_legal},
            {"next_pl_n_legal", move.next_pl_n_legal},
            {"n_connected_empty_squares", move.n_connected_empty_squares},
            {"op_canput", move.op_canput},
            {"n_increased_stable_discs", move.n_increased_stable_discs},
        };
        res["moves"].push_back(j);
    }

    std::cout << res.dump() << std::endl;
}