/*
    Egaroucid Project

    @file move_ordering.hpp
        Move ordering for each search algorithm
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <vector>
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "stability.hpp"
#include "level.hpp"

/*
    @brief if wipeout found, it must be searched first.
*/
#define W_WIPEOUT 100000000
#define W_1ST_MOVE 10000000
#define W_2ND_MOVE 1000000

/*
    @brief constants for move ordering
*/
#if TUNE_MOVE_ORDERING_MID || TUNE_MOVE_ORDERING_END
    #define N_MOVE_ORDERING_PARAM 12
    int move_ordering_param_array[N_MOVE_ORDERING_PARAM] = {
        32, 28, 256, 64, 
        16, 8, 14, 2, 
        32, 4, 32, 4
    };

    #define W_MOBILITY                  move_ordering_param_array[0]
    #define W_POTENTIAL_MOBILITY        move_ordering_param_array[1]
    #define W_VALUE                     move_ordering_param_array[2]
    #define W_VALUE_DEEP_ADDITIONAL     move_ordering_param_array[3]

    #define W_NWS_MOBILITY              move_ordering_param_array[4]
    #define W_NWS_POTENTIAL_MOBILITY    move_ordering_param_array[5]
    #define W_NWS_VALUE                 move_ordering_param_array[6]
    #define W_NWS_VALUE_DEEP_ADDITIONAL move_ordering_param_array[7]

    #define W_END_MOBILITY              move_ordering_param_array[8]
    #define W_END_PARITY                move_ordering_param_array[9]
    #define W_END_POTENTIAL_MOBILITY    move_ordering_param_array[10]
    #define W_END_VALUE                 move_ordering_param_array[11]

    #define MOVE_ORDERING_MID_PARAM_START 0
    #define MOVE_ORDERING_MID_PARAM_END 7
    #define MOVE_ORDERING_END_PARAM_START 8
    #define MOVE_ORDERING_END_PARAM_END 11
#else
    // midgame search
    //#define W_CELL_WEIGHT 7
    #define W_MOBILITY 32
    #define W_POTENTIAL_MOBILITY 28
    #define W_VALUE 256
    #define W_VALUE_DEEP_ADDITIONAL 64

    // midgame null window search
    #define W_NWS_MOBILITY 16
    #define W_NWS_POTENTIAL_MOBILITY 8
    #define W_NWS_VALUE 14
    #define W_NWS_VALUE_DEEP_ADDITIONAL 2

    // endgame search
    #define W_END_MOBILITY 32
    #define W_END_PARITY 4
    #define W_END_POTENTIAL_MOBILITY 32
    #define W_END_VALUE 4
#endif

#define MOVE_ORDERING_VALUE_OFFSET_ALPHA 10
#define MOVE_ORDERING_VALUE_OFFSET_BETA 10
#define MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA 10
#define MOVE_ORDERING_NWS_VALUE_OFFSET_BETA 3

#define MOVE_ORDERING_MPC_LEVEL MPC_88_LEVEL

/*
    @brief Flip structure with more information

    @param flip                 flip information
    @param value                the move ordering value
    @param n_legal              next legal moves as a bitboard for reusing
*/
struct Flip_value{
    Flip flip;
    int value;
    uint64_t n_legal;

    Flip_value(){
        n_legal = LEGAL_UNDEFINED;
    }

    bool operator<(const Flip_value &another) const{
        return value < another.value;
    }

    bool operator>(const Flip_value &another) const{
        return value > another.value;
    }
};

int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped, const bool *searching);
#if MID_FAST_DEPTH > 1
    int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped, const bool *searching);
#endif
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

/*
    @brief Calculate openness

    Not used for now

    @param board                board
    @param flip                 flip information
    @return openness
*/
/*
inline int calc_openness(const Board *board, const Flip *flip){
    uint64_t f = flip->flip;
    uint64_t around = 0ULL;
    for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f))
        around |= bit_around[cell];
    around &= ~flip->flip;
    return pop_count_ull(~(board->player | board->opponent | (1ULL << flip->pos)) & around);
}
*/

/*
    @brief Get number of corner mobility

    Optimized for corner mobility

    @param legal                legal moves as a bitboard
    @return number of legal moves on corners
*/
inline int get_corner_mobility(uint64_t legal){
    int res = (int)((legal & 0b10000001ULL) + ((legal >> 56) & 0b10000001ULL));
    return (res & 0b11) + (res >> 7);
}

/*
    @brief Get a weighted mobility

    @param legal                legal moves as a bitboard
    @return weighted mobility
*/
inline int get_weighted_n_moves(uint64_t legal){
    return pop_count_ull(legal) * 2 + get_corner_mobility(legal);
}

/*
    @brief Get potential mobility

    Same idea as surround in evaluation function

    @param opponent             a bitboard representing opponent
    @param empties              a bitboard representing empty squares
    @return potential mobility
*/
#if USE_SIMD
    inline int get_potential_mobility(uint64_t opponent, uint64_t empties){
        return calc_surround(opponent, empties);
    }
#else
    inline int get_potential_mobility(uint64_t opponent, uint64_t empties){
        uint64_t hmask = opponent & 0x7E7E7E7E7E7E7E7EULL;
        uint64_t vmask = opponent & 0x00FFFFFFFFFFFF00ULL;
        uint64_t hvmask = opponent & 0x007E7E7E7E7E7E00ULL;
        uint64_t res = 
            (hmask << 1) | (hmask >> 1) | 
            (vmask << HW) | (vmask >> HW) | 
            (hvmask << HW_M1) | (hvmask >> HW_M1) | 
            (hvmask << HW_P1) | (hvmask >> HW_P1);
        return pop_count_ull(empties & res);
    }
#endif

/*
    @brief Evaluate a move in midgame

    @param search               search information
    @param flip_value           flip with value
    @param alpha                alpha value to search
    @param beta                 beta value to search
    @param depth                depth to search
    @param searching            flag for terminating this search
    @return true if wipeout found else false
*/
inline void move_evaluate(Search *search, Flip_value *flip_value, int alpha, int beta, int depth, const bool *searching){
    flip_value->value = 0;
    //flip_value->value = cell_weight[flip_value->flip.pos] * W_CELL_WEIGHT;
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value -= get_weighted_n_moves(flip_value->n_legal) * W_MOBILITY;
        uint64_t empties = ~(search->board.player | search->board.opponent);
        flip_value->value -= get_potential_mobility(search->board.player, empties) * W_POTENTIAL_MOBILITY;
        switch (depth){
            case 0:
                flip_value->value -= mid_evaluate_diff(search) * W_VALUE;
                break;
            case 1:
                flip_value->value -= nega_alpha_eval1(search, alpha, beta, false, searching) * (W_VALUE + W_VALUE_DEEP_ADDITIONAL);
                break;
            default:
                #if MID_FAST_DEPTH > 1
                    if (depth <= MID_FAST_DEPTH)
                        flip_value->value -= nega_alpha(search, alpha, beta, depth, false, searching) * (W_VALUE + depth * W_VALUE_DEEP_ADDITIONAL);
                    else{
                        uint_fast8_t mpc_level = search->mpc_level;
                        search->mpc_level = MOVE_ORDERING_MPC_LEVEL;
                        flip_value->value -= nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching) * (W_VALUE + depth * W_VALUE_DEEP_ADDITIONAL);
                        search->mpc_level = mpc_level;
                    }
                #else
                    uint_fast8_t mpc_level = search->mpc_level;
                    search->mpc_level = MOVE_ORDERING_MPC_LEVEL;
                    flip_value->value -= nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching) * (W_VALUE + depth * W_VALUE_DEEP_ADDITIONAL);
                    search->mpc_level = mpc_level;
                #endif
                break;
        }
    search->undo(&flip_value->flip);
    eval_undo(search, &flip_value->flip);
}

/*
    @brief Evaluate a move in midgame for NWS

    @param search               search information
    @param flip_value           flip with value
    @param alpha                alpha value to search
    @param beta                 beta value to search
    @param depth                depth to search
    @param searching            flag for terminating this search
    @return true if wipeout found else false
*/
inline void move_evaluate_nws(Search *search, Flip_value *flip_value, int alpha, int beta, int depth, const bool *searching){
    flip_value->value = cell_weight[flip_value->flip.pos];
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value -= get_weighted_n_moves(flip_value->n_legal) * W_NWS_MOBILITY;
        flip_value->value -= get_potential_mobility(search->board.opponent, ~(search->board.player | search->board.opponent)) * W_NWS_POTENTIAL_MOBILITY;
        /*
        switch (depth){
            case 0:
                flip_value->value -= mid_evaluate_diff(search) * W_NWS_VALUE;
                break;
            case 1:
                flip_value->value -= nega_alpha_eval1(search, alpha, beta, false, searching) * (W_NWS_VALUE + W_NWS_VALUE_DEEP_ADDITIONAL);
                break;
            default:
                #if MID_FAST_DEPTH > 1
                    if (depth <= MID_FAST_DEPTH)
                        flip_value->value -= nega_alpha(search, alpha, beta, depth, false, searching) * (W_NWS_VALUE + depth * W_NWS_VALUE_DEEP_ADDITIONAL);
                    else{
                        uint_fast8_t mpc_level = search->mpc_level;
                        search->mpc_level = MOVE_ORDERING_MPC_LEVEL;
                        flip_value->value -= nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching) * (W_NWS_VALUE + depth * W_NWS_VALUE_DEEP_ADDITIONAL);
                        search->mpc_level = mpc_level;
                    }
                #else
                    uint_fast8_t mpc_level = search->mpc_level;
                    search->mpc_level = MOVE_ORDERING_MPC_LEVEL;
                    flip_value->value -= nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching) * (W_NWS_VALUE + depth * W_NWS_VALUE_DEEP_ADDITIONAL);
                    search->mpc_level = mpc_level;
                #endif
                break;
        }
        */
        if (depth == 0)
            flip_value->value -= mid_evaluate_diff(search) * W_NWS_VALUE;
        else
            flip_value->value -= nega_alpha_eval1(search, alpha, beta, false, searching) * (W_NWS_VALUE + W_NWS_VALUE_DEEP_ADDITIONAL);
    search->undo(&flip_value->flip);
    eval_undo(search, &flip_value->flip);
}




#if USE_NEGA_ALPHA_END
    /*
        @brief Evaluate a move in endgame

        @param search               search information
        @param flip_value           flip with value
        @return true if wipeout found else false
    */
    inline void move_evaluate_end(Search *search, Flip_value *flip_value){
        flip_value->value = 0; //cell_weight[flip_value->flip.pos];
        if (search->parity & cell_div4[flip_value->flip.pos])
            flip_value->value += W_END_PARITY;
        search->move(&flip_value->flip);
            flip_value->n_legal = search->board.get_legal();
            flip_value->value -= pop_count_ull(flip_value->n_legal) * W_END_MOBILITY;
        search->undo(&flip_value->flip);
    }
#endif

/*
    @brief Evaluate a move in endgame

    @param search               search information
    @param flip_value           flip with value
    @return true if wipeout found else false
*/
inline void move_evaluate_end_nws(Search *search, Flip_value *flip_value){
    flip_value->value = 0; //cell_weight[flip_value->flip.pos];
    if (search->parity & cell_div4[flip_value->flip.pos])
        flip_value->value += W_END_PARITY;
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value -= pop_count_ull(flip_value->n_legal) * W_END_MOBILITY;
    search->undo(&flip_value->flip);
}

/*
    @brief Evaluate a move in endgame with eval

    @param search               search information
    @param flip_value           flip with value
    @return true if wipeout found else false
*/
inline void move_evaluate_end_nws_eval(Search *search, Flip_value *flip_value){
    flip_value->value = 0; //cell_weight[flip_value->flip.pos];
    if (search->parity & cell_div4[flip_value->flip.pos])
        flip_value->value += W_END_PARITY;
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value -= pop_count_ull(flip_value->n_legal) * W_END_MOBILITY;
        flip_value->value -= mid_evaluate_diff(search) * W_END_VALUE;
    search->undo(&flip_value->flip);
    eval_undo(search, &flip_value->flip);
}





/*
    @brief Set the best move to the first element

    @param move_list            list of moves
    @param strt                 the first index
    @param siz                  the size of move_list
*/
inline void swap_next_best_move(std::vector<Flip_value> &move_list, const int strt, const int siz){
    if (strt == siz - 1)
        return;
    int top_idx = strt;
    int best_value = move_list[strt].value;
    for (int i = strt + 1; i < siz; ++i){
        if (best_value < move_list[i].value){
            best_value = move_list[i].value;
            top_idx = i;
        }
    }
    if (top_idx != strt)
        std::swap(move_list[strt], move_list[top_idx]);
}

/*
    @brief Set the best move to the first element

    @param move_list            list of moves
    @param strt                 the first index
    @param siz                  the size of move_list
*/
inline void swap_next_best_move(Flip_value move_list[], const int strt, const int siz){
    if (strt == siz - 1)
        return;
    int top_idx = strt;
    int best_value = move_list[strt].value;
    for (int i = strt + 1; i < siz; ++i){
        if (best_value < move_list[i].value){
            best_value = move_list[i].value;
            top_idx = i;
        }
    }
    if (top_idx != strt)
        std::swap(move_list[strt], move_list[top_idx]);
}

/*
    @brief Evaluate all legal moves for midgame

    @param search               search information
    @param move_list            list of moves
    @param moves                list of moves in transposition table
    @param depth                remaining depth
    @param alpha                alpha value
    @param beta                 beta value
    @param searching            flag for terminating this search
*/
inline void move_list_evaluate(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], int depth, int alpha, int beta, const bool *searching){
    if (move_list.size() == 1)
        return;
    int eval_alpha = -std::min(SCORE_MAX, beta + MOVE_ORDERING_VALUE_OFFSET_BETA);
    int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_VALUE_OFFSET_ALPHA);
    int eval_depth = depth >> 2;
    for (Flip_value &flip_value: move_list){
        #if USE_MID_ETC
            if (flip_value.flip.flip){
                if (flip_value.flip.pos == moves[0])
                    flip_value.value = W_1ST_MOVE;
                else if (flip_value.flip.pos == moves[1])
                    flip_value.value = W_2ND_MOVE;
                else
                    move_evaluate(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
            } else
                flip_value.value = -INF;
        #else
            if (flip_value.flip.pos == moves[0])
                flip_value.value = W_1ST_MOVE;
            else if (flip_value.flip.pos == moves[1])
                flip_value.value = W_2ND_MOVE;
            else
                move_evaluate(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
        #endif
    }
}

/*
    @brief Evaluate all legal moves for midgame

    @param search               search information
    @param move_list            list of moves
    @param depth                remaining depth
    @param alpha                alpha value
    @param beta                 beta value
    @param searching            flag for terminating this search
*/
inline void move_list_evaluate(Search *search, std::vector<Flip_value> &move_list, int depth, int alpha, int beta, const bool *searching){
    if (move_list.size() == 1)
        return;
    int eval_alpha = -std::min(SCORE_MAX, beta + MOVE_ORDERING_VALUE_OFFSET_BETA);
    int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_VALUE_OFFSET_ALPHA);
    int eval_depth = depth >> 2;
    for (Flip_value &flip_value: move_list){
        #if USE_MID_ETC
            if (flip_value.flip.flip)
                move_evaluate(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
            else
                flip_value.value = -INF;
        #else
            move_evaluate(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
        #endif
    }
}

#if USE_NEGA_ALPHA_END
    /*
        @brief Evaluate all legal moves for endgame

        @param search               search information
        @param move_list            list of moves
    */
    inline void move_list_evaluate_end(Search *search, std::vector<Flip_value> &move_list, const int canput){
        if (canput == 1)
            return;
        for (Flip_value &flip_value: move_list)
            move_evaluate_end(search, &flip_value);
    }
#endif

/*
    @brief Evaluate all legal moves for endgame

    @param search               search information
    @param move_list            list of moves
*/
inline void move_list_evaluate_end_simple_nws(Search *search, Flip_value move_list[], const int canput){
    if (canput == 1)
        return;
    for (int i = 0; i < canput; ++i)
        move_evaluate_end_nws(search, &move_list[i]);
}

/*
    @brief Evaluate all legal moves for endgame

    @param search               search information
    @param move_list            list of moves
*/
inline void move_list_evaluate_end_nws(Search *search, std::vector<Flip_value> &move_list, const int canput, uint_fast8_t moves[], bool use_eval){
    if (canput == 1)
        return;
    if (use_eval){
        for (Flip_value &flip_value: move_list){
            if (flip_value.flip.pos == moves[0])
                flip_value.value = W_1ST_MOVE;
            else if (flip_value.flip.pos == moves[1])
                flip_value.value = W_2ND_MOVE;
            else
                move_evaluate_end_nws_eval(search, &flip_value);
        }
    } else{
        for (Flip_value &flip_value: move_list){
            if (flip_value.flip.pos == moves[0])
                flip_value.value = W_1ST_MOVE;
            else if (flip_value.flip.pos == moves[1])
                flip_value.value = W_2ND_MOVE;
            else
                move_evaluate_end_nws(search, &flip_value);
        }
    }
}

/*
    @brief Evaluate all legal moves for midgame NWS

    @param search               search information
    @param move_list            list of moves
    @param moves                list of moves in transposition table
    @param depth                remaining depth
    @param alpha                alpha value (beta = alpha + 1)
    @param searching            flag for terminating this search
*/
inline void move_list_evaluate_nws(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], int depth, int alpha, const bool *searching){
    if (move_list.size() == 1)
        return;
    const int eval_alpha = -std::min(SCORE_MAX, alpha + MOVE_ORDERING_NWS_VALUE_OFFSET_BETA);
    const int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA);
    int eval_depth = depth >> 4;
    for (Flip_value &flip_value: move_list){
        #if USE_MID_ETC
            if (flip_value.flip.flip){
                if (flip_value.flip.pos == moves[0])
                    flip_value.value = W_1ST_MOVE;
                else if (flip_value.flip.pos == moves[1])
                    flip_value.value = W_2ND_MOVE;
                else
                    move_evaluate_nws(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
            } else
                flip_value.value = -INF;
        #else
            if (flip_value.flip.pos == moves[0])
                flip_value.value = W_1ST_MOVE;
            else if (flip_value.flip.pos == moves[1])
                flip_value.value = W_2ND_MOVE;
            else
                move_evaluate_nws(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
        #endif
    }
}

#if MID_FAST_NWS_DEPTH > 1
    /*
        @brief Evaluate all legal moves for midgame NWS

        @param search               search information
        @param move_list            list of moves
        @param moves                list of moves in transposition table
        @param depth                remaining depth
        @param alpha                alpha value (beta = alpha + 1)
        @param searching            flag for terminating this search
    */
    inline void move_list_evaluate_nws_fast(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], int depth, int alpha, const bool *searching){
        if (move_list.size() == 1)
            return;
        const int eval_alpha = -std::min(SCORE_MAX, alpha + MOVE_ORDERING_NWS_VALUE_OFFSET_BETA);
        const int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA);
        int eval_depth = depth >> 4;
        for (Flip_value &flip_value: move_list){
            if (flip_value.flip.pos == moves[0])
                flip_value.value = W_1ST_MOVE;
            else if (flip_value.flip.pos == moves[1])
                flip_value.value = W_2ND_MOVE;
            else
                move_evaluate_nws(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
        }
    }
#endif

#if TUNE_MOVE_ORDERING_MID || TUNE_MOVE_ORDERING_END
    std::pair<int, int> first_nega_scout(Search *search, int alpha, int beta, int predicted_value, int depth, bool is_end_search, const bool is_main_search, const std::vector<Clog_result> clogs, uint64_t strt);

    Board get_board(std::string board_str){
        board_str.erase(std::remove_if(board_str.begin(), board_str.end(), ::isspace), board_str.end());
        Board new_board;
        int player = BLACK;
        new_board.player = 0ULL;
        new_board.opponent = 0ULL;
        if (board_str.length() != HW2 + 1){
            std::cerr << "[ERROR] invalid argument" << std::endl;
            return new_board;
        }
        for (int i = 0; i < HW2; ++i){
            if (board_str[i] == 'B' || board_str[i] == 'b' || board_str[i] == 'X' || board_str[i] == 'x' || board_str[i] == '0' || board_str[i] == '*')
                new_board.player |= 1ULL << (HW2_M1 - i);
            else if (board_str[i] == 'W' || board_str[i] == 'w' || board_str[i] == 'O' || board_str[i] == 'o' || board_str[i] == '1')
                new_board.opponent |= 1ULL << (HW2_M1 - i);
        }
        if (board_str[HW2] == 'B' || board_str[HW2] == 'b' || board_str[HW2] == 'X' || board_str[HW2] == 'x' || board_str[HW2] == '0' || board_str[HW2] == '*')
            player = BLACK;
        else if (board_str[HW2] == 'W' || board_str[HW2] == 'w' || board_str[HW2] == 'O' || board_str[HW2] == 'o' || board_str[HW2] == '1')
            player = WHITE;
        else{
            std::cerr << "[ERROR] invalid player argument" << std::endl;
            return new_board;
        }
        if (player == WHITE)
            std::swap(new_board.player, new_board.opponent);
        return new_board;
    }

    uint64_t n_nodes_test(int level, std::vector<Board> testcase_arr){
        uint64_t n_nodes = 0;
        for (Board &board: testcase_arr){
            int depth;
            bool is_mid_search;
            uint_fast8_t mpc_level;
            get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
            Search search;
            search.init_board(&board);
            calc_features(&search);
            search.n_nodes = 0ULL;
            search.use_multi_thread = true;
            search.mpc_level = mpc_level;
            std::vector<Clog_result> clogs;
            //transposition_table.init();
            //board.print();
            std::pair<int, int> result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, SCORE_UNDEFINED, depth, !is_mid_search, false, clogs, tim());
            //std::cerr << result.first << " " << result.second << std::endl;
            n_nodes += search.n_nodes;
        }
        return n_nodes;
    }

    void tune_move_ordering(int level){
        std::cout << "please input testcase file" << std::endl;
        std::string file;
        std::cin >> file;
        std::ifstream ifs(file);
        if (ifs.fail()){
            std::cerr << "[ERROR] [FATAL] problem file " << file << " not found" << std::endl;
            return;
        }
        std::vector<Board> testcase_arr;
        std::string line;
        while (std::getline(ifs, line)){
            testcase_arr.emplace_back(get_board(line));
        }
        std::cerr << testcase_arr.size() << " testcases loaded" << std::endl;
        uint64_t min_n_nodes = n_nodes_test(level, testcase_arr);
        double min_percentage = 100.0;
        uint64_t first_n_nodes = min_n_nodes;
        std::cerr << "min_n_nodes " << min_n_nodes << std::endl;
        int n_updated = 0;
        int n_try = 0;
        uint64_t tl = 10ULL * 60ULL * 1000ULL; // 10 min
        uint64_t strt = tim();
        while (tim() - strt < tl){
            // update parameter randomly
            #if TUNE_MOVE_ORDERING_MID
                int idx = myrandrange(MOVE_ORDERING_MID_PARAM_START, MOVE_ORDERING_MID_PARAM_END + 1); // midgame search
            #else
                int idx = myrandrange(MOVE_ORDERING_END_PARAM_START, MOVE_ORDERING_END_PARAM_END + 1); // endgame search
            #endif
            int delta = myrandrange(-4, 5);
            while (delta == 0)
                delta = myrandrange(-4, 5);
            move_ordering_param_array[idx] += delta;
            uint64_t n_nodes = n_nodes_test(level, testcase_arr);
            double percentage = 100.0 * n_nodes / first_n_nodes;

            // simulated annealing
            constexpr double start_temp = 0.1; // percent
            constexpr double end_temp = 0.001; // percent
            double temp = start_temp + (end_temp - start_temp) * (tim() - strt) / tl;
            double prob = exp((min_percentage - percentage) / temp);
            if (prob > myrandom()){
                min_n_nodes = n_nodes;
                min_percentage = percentage;
                ++n_updated;
            } else{
                move_ordering_param_array[idx] -= delta;
            }
            ++n_try;

            std::cerr << "try " << n_try << " updated " << n_updated << " min_n_nodes " << min_n_nodes << " n_nodes " << n_nodes << " " << min_percentage << "% " << tim() - strt << " ms ";
            for (int i = 0; i < N_MOVE_ORDERING_PARAM; ++i){
                std::cerr << ", " << move_ordering_param_array[i];
            }
            std::cerr << std::endl;
        }
        std::cout << "done " << min_percentage << "% ";
        for (int i = 0; i < N_MOVE_ORDERING_PARAM; ++i){
            std::cout << ", " << move_ordering_param_array[i];
        }
        std::cout << std::endl;
    }
#endif