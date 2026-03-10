/*
	Egaroucid Project

	@file line_searcher.cpp
		Search specific lines
	@date 2021-2026
	@author Takuto Yamana
	@license GPL-3.0-or-later
*/

#pragma once
#include "ai.hpp"

void search_lines(Board &board, int player, int depth, int score, int black_score_min, int black_score_max, int black_max_loss, int white_max_loss, std::vector<int> &line, int last_move_player, std::vector<int> &last_move_cells, std::unordered_map<Board, std::vector<std::pair<int, int>>, Book_hash> &ok_board_memo, int level = 21) {
    if (depth <= 0) {
        return;
    }
    if (depth == 1 && player != last_move_player) {
        return;
    }

    uint64_t legal = board.get_legal();
    bool passed = false;
    if (legal == 0ULL) {
        passed = true;
        board.pass();
        player ^= 1;
        legal = board.get_legal();
        if (legal == 0ULL) {
            board.pass();
            player ^= 1;
            return;
        }
    }

    if (depth <= 1 && player == last_move_player) {
        uint64_t last_move_bits = 0;
        for (const int &cell: last_move_cells) {
            last_move_bits |= 1ULL << cell;
        }
        if ((legal & last_move_bits) == 0) {
            return;
        }
    }

    auto memo = ok_board_memo.find(board);
    if (memo != ok_board_memo.end()) {
        std::vector<std::pair<int, int>> ok_moves = memo->second; // move, score
        for (std::pair<int, int> &m: ok_moves) {
            int cell = m.first;
            int value = m.second;
            Flip flip;
            calc_flip(&flip, &board, cell);
            board.move_board(&flip);
            int next_player = player ^ 1;
            line.emplace_back(cell);
                if (player == last_move_player && std::find(last_move_cells.begin(), last_move_cells.end(), cell) != last_move_cells.end()) {
                    std::string transcript;
                    for (const int &coord: line) {
                        transcript += idx_to_coord(coord);
                    }
                    std::cout << transcript << std::endl;
                    std::cerr << transcript << std::endl;
                } else {
                    search_lines(board, next_player, depth - 1, value, black_score_min, black_score_max, black_max_loss, white_max_loss, line, last_move_player, last_move_cells, ok_board_memo, level);
                }
            line.pop_back();
            board.undo_board(&flip);
        }
    } else {
        std::vector<std::pair<int, int>> ok_moves; // move, score
        int black_score_min_p = black_score_min;
        int black_score_max_p = black_score_max;
        if (player == BLACK) {
            black_score_min_p = std::max(black_score_min_p, score - black_max_loss);
        } else {
            black_score_max_p = std::min(black_score_max_p, -(score - white_max_loss));
        }
        if (black_score_min_p < black_score_max_p) {
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                Flip flip;
                calc_flip(&flip, &board, cell);
                board.move_board(&flip);
                int next_player = player ^ 1;
                    int alpha = black_score_min_p - 1;
                    int beta = black_score_max_p + 1;
                    if (next_player != BLACK) {
                        std::swap(alpha, beta);
                        alpha = -alpha;
                        beta = -beta;
                    }
                    // board.print();
                    // int value = ai_window_legal(board, alpha, beta, level, true, 0, true, false, board.get_legal()).value;
                    int value = ai_window(board, alpha, beta, level, true, 0, true, false).value;
                    // int value = ai(board, level, true, 0, true, false).value;
                    bool is_ok = alpha < value && value < beta;
                    if (is_ok) {
                        ok_moves.emplace_back(std::make_pair(cell, value));
                        line.emplace_back(cell);
                            if (player == last_move_player && std::find(last_move_cells.begin(), last_move_cells.end(), cell) != last_move_cells.end()) {
                                std::string transcript;
                                for (const int &coord: line) {
                                    transcript += idx_to_coord(coord);
                                }
                                std::cout << transcript << std::endl;
                                std::cerr << transcript << std::endl;
                            } else {
                                search_lines(board, next_player, depth - 1, value, black_score_min, black_score_max, black_max_loss, white_max_loss, line, last_move_player, last_move_cells, ok_board_memo, level);
                            }
                        line.pop_back();
                    }
                board.undo_board(&flip);
            }
        }
        ok_board_memo[board] = ok_moves;
    }

    

    if (passed) {
        board.pass();
        player ^= 1;
    }
}


void search_lines_demo() {
    // std::string initial_line = "f5d6c3d3c4f4f6"; // stephenson
    std::string initial_line = "f5f6e6d6e7"; // tobidashi
    // std::string initial_line = "f5d6c3d3c4f4f6g5e3f3g6e2h5c5g4g3f2e1"; // 1993
    int n_max_moves = 20;
    int search_level = 21;
    int black_score_min = -6;
    int black_score_max = 1;
    int black_max_loss = 6;
    int white_max_loss = 1;
    int last_move_player = BLACK;
    std::vector<int> last_move_cells = {get_coord_from_chars('b', '2'), get_coord_from_chars('b', '7'), get_coord_from_chars('g', '2'), get_coord_from_chars('g', '7')};

    std::vector<int> initial_line_vec;
    for (int i = 0; i < initial_line.size(); i += 2) {
        initial_line_vec.emplace_back(get_coord_from_chars(initial_line[i], initial_line[i + 1]));
    }
    Board search_board;
    search_board.reset();
    int player = BLACK;
    Flip flip;
    for (const int &coord: initial_line_vec) {
        uint64_t legal = search_board.get_legal();
        if (legal == 0ULL) {
            search_board.pass();
            player ^= 1;
            legal = search_board.get_legal();
            if (legal == 0ULL) {
                std::cerr << "[ERROR] game over before initial line is fully applied" << std::endl;
                std::exit(1);
            }
        }
        if ((legal & (1ULL << coord)) == 0ULL) {
            std::cerr << "[ERROR] illegal move in initial line: " << idx_to_coord(coord) << std::endl;
            std::exit(1);
        }
        calc_flip(&flip, &search_board, coord);
        search_board.move_board(&flip);
        player ^= 1;
    }

    int n_initial_moves = initial_line_vec.size();


    std::cout << "[search params]" << std::endl;
    std::cout << "initial_line=" << initial_line << std::endl;
    std::cout << "initial_line_moves=" << n_initial_moves << std::endl;
    std::cout << "n_max_moves=" << n_max_moves << std::endl;
    std::cout << "search_level=" << search_level << std::endl;
    std::cout << "black_score_range=[" << black_score_min << ", " << black_score_max << "]" << std::endl;
    std::cout << "black_max_loss=" << black_max_loss << std::endl;
    std::cout << "white_max_loss=" << white_max_loss << std::endl;
    std::cout << "last_move_player=" << (last_move_player == BLACK ? "BLACK" : "WHITE") << std::endl;
    std::cout << "search_start_player=" << (player == BLACK ? "BLACK" : "WHITE") << std::endl;
    std::cout << "last_move_cells=";
    for (int i = 0; i < (int)last_move_cells.size(); ++i) {
        if (i) {
            std::cout << ",";
        }
        std::cout << idx_to_coord(last_move_cells[i]);
    }
    std::cout << std::endl;

    std::unordered_map<Board, std::vector<std::pair<int, int>>, Book_hash> ok_board_memo;
    int root_score = ai(search_board, search_level, true, 0, true, false).value;
    for (int n_max_moves_itr = n_max_moves % 2; n_max_moves_itr <= n_max_moves; n_max_moves_itr += 2) {
        std::cout << "search until move " << n_max_moves_itr << std::endl;
        std::cerr << "search until move " << n_max_moves_itr << std::endl;
        uint64_t strt = tim();
        search_lines(search_board, player, n_max_moves_itr - n_initial_moves, root_score, black_score_min, black_score_max, black_max_loss, white_max_loss, initial_line_vec, last_move_player, last_move_cells, ok_board_memo, search_level);
        std::cout << "searched until move " << n_max_moves_itr << " elapsed " << tim() - strt << " ms" << " memo_size=" << ok_board_memo.size() << std::endl;
        std::cerr << "searched until move " << n_max_moves_itr << " elapsed " << tim() - strt << " ms" << " memo_size=" << ok_board_memo.size() << std::endl;
    }
    std::cout << "done!" << std::endl;
    std::cerr << "done!" << std::endl;
}