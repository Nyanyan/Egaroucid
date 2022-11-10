/*
    Egaroucid Project

    @file board_info.hpp
        Board structure of Egaroucid
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include "./../engine/engine_all.hpp"

#define INVALID_CELL -1

#define MODE_HUMAN_VS_AI 0
#define MODE_AI_VS_HUMAN 1
#define MODE_AI_VS_AI 2
#define MODE_HUMAN_VS_HUMAN 3

struct Board_info{
    Board board;
    uint_fast8_t player;
    int mode;
    std::vector<Board> boards;
    std::vector<int> players;
    int ply_vec;

    void reset(){
        board.reset();
        player = BLACK;
        mode = MODE_HUMAN_VS_HUMAN;
        boards.clear();
        players.clear();
        boards.emplace_back(board);
        players.emplace_back(player);
        ply_vec = 0;
    }

    Board_info copy(){
        Board_info res;
        res.board = board.copy();
        res.player = player;
        res.mode = mode;
        res.boards = boards;
        res.players = players;
        res.ply_vec = ply_vec;
    }
};