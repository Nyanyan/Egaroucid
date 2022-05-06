#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "transpose_table.hpp"
#include "search.hpp"
#include "cuda_board.hpp"

extern "C" int do_search_cuda(vector<Board_simple> &boards, int depth, bool is_end_search);

using namespace std;

inline bool ybwc_split_without_move_cuda(Search *search, const int depth, const int pv_idx, bool is_end_search, vector<Board_simple> &board_arr){
    if (pv_idx > 0 && depth <= CUDA_YBWC_SPLIT_MAX_DEPTH && is_end_search){
        int use_depth = depth;
        if (is_end_search)
            ++use_depth;
        Board_simple board;
        board.player = search->board.player;
        board.opponent = search->board.opponent;
        board_arr.emplace_back(board);
        return true;
    }
    return false;
}
