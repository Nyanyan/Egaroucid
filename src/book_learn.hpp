#pragma once
#include <iostream>
#include <unordered_set>
#include "evaluate.hpp"
#include "board.hpp"
#include "ai.hpp"

#define AUTO_BOOK_SAVE_TIME 60000

using namespace std;

bool cmp_flip(pair<Flip, int> &a, pair<Flip, int> &b){
    return a.second > b.second;
}


inline int book_learn_calc_value(Board board, int level, int bound){
    Search search;
    search.board = board;
    calc_features(&search);
    int g, depth;
    bool is_mid_search, searching = true;
    get_level(level, search.board.n - 4, &is_mid_search, &depth, &search.use_mpc, &search.mpct);
    if (is_mid_search){
        if (depth - 1 >= 0){
            parent_transpose_table.init();
            g = nega_scout(&search, bound - 1, bound + 1, depth - 1, false, LEGAL_UNDEFINED, false, &searching);
            parent_transpose_table.init();
            g += nega_scout(&search, bound - 1, bound + 1, depth, false, LEGAL_UNDEFINED, false, &searching);
            g /= 2;
        } else{
            parent_transpose_table.init();
            g = nega_scout(&search, bound - 1, bound + 1, depth, false, LEGAL_UNDEFINED, false, &searching);
        }
    } else{
        parent_transpose_table.init();
        g = nega_scout(&search, bound - 1, bound + 1, depth, false, LEGAL_UNDEFINED, true, &searching);
    }
    return g;
}

int book_learn_search(Board board, int level, const int book_depth, const int best_value, int error_sum, const int expected_error, const int adopt_error_sum, Board *board_copy, uint64_t *strt_tim, string book_file, string book_bak){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (error_sum > adopt_error_sum)
        return SCORE_UNDEFINED;
    if (tim() - *strt_tim > AUTO_BOOK_SAVE_TIME){
        book.save_bin(book_file, book_bak);
        *strt_tim = tim();
    }
    int g;
    if (board.n >= 4 + book_depth){
        g = -INF;
        if (-SCORE_MAX <= best_value && best_value <= SCORE_MAX)
            g = book_learn_calc_value(board, level, best_value);
        if ((g != -INF && g <= best_value) || best_value < -SCORE_MAX || SCORE_MAX < best_value){
            g = ai(board, level, true, 0).value;
            //book.reg(board, -g);
            return g;
        }
        return SCORE_UNDEFINED;
    }
    uint64_t legal = board.get_legal();
    if (legal == 0ULL){
        board.pass();
            g = -book_learn_search(board, level, book_depth, -best_value, error_sum, expected_error, adopt_error_sum, board_copy, strt_tim, book_file, book_bak);
        board.pass();
        return g;
    }
    Search_result best_move = ai(board, level, true, 0);
    if (best_move.value > best_value + expected_error && -SCORE_MAX <= best_value && best_value <= SCORE_MAX)
        return best_move.value;
    //if (-SCORE_MAX <= best_move.value && best_move.value <= SCORE_MAX)
    //    book.reg(board, -best_move.value);
    vector<int> policies;
    Flip flip;
    calc_flip(&flip, &board, (uint8_t)best_move.policy);
    board.move(&flip);
        board.copy(board_copy);
        g = -book_learn_search(board, level, book_depth, SCORE_UNDEFINED, error_sum, expected_error, adopt_error_sum, board_copy, strt_tim, book_file, book_bak);
        if (-SCORE_MAX <= g && g <= SCORE_MAX){
            best_move.value = g;
            cerr << "PV value " << best_move.value << endl;
            policies.emplace_back(best_move.policy);
        } else
            best_move.value = -INF;
    board.undo(&flip);
    legal ^= 1ULL << best_move.policy;
    bool flag;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        flag = false;
        calc_flip(&flip, &board, cell);
        board.move(&flip);
            board.copy(board_copy);
            g = book.get(&board);
            if (g < -SCORE_MAX || SCORE_MAX < g){
                if (-SCORE_MAX <= best_move.value && best_move.value <= SCORE_MAX)
                    g = -book_learn_calc_value(board, level, -best_move.value + expected_error);
                else
                    flag = true;
            }
            if (flag || (-SCORE_MAX <= g && g <= SCORE_MAX && g >= best_move.value - expected_error)){
                g = -book_learn_search(board, level, book_depth, -best_move.value, error_sum + min(0, best_move.value - g), expected_error, adopt_error_sum, board_copy, strt_tim, book_file, book_bak);
                if (-SCORE_MAX <= g && g <= SCORE_MAX){
                    if (g > best_move.value){
                        best_move.value = g;
                        policies.clear();
                        policies.emplace_back((int)cell);
                    } else if (g == best_move.value){
                        policies.emplace_back((int)cell);
                    }
                    cerr << "   value " << g << " best_move_value " << best_move.value << endl;
                }
            }
        board.undo(&flip);
        //if (flag)
        //    book.reg(board, -best_move.value);
    }
    
    if (-SCORE_MAX <= best_move.value && best_move.value <= SCORE_MAX && global_searching){
        for (int &policy: policies){
            calc_flip(&flip, &board, (uint8_t)policy);
            board.move(&flip);
                book.reg(board, best_move.value);
            board.undo(&flip);
        }
        //book.reg(board, -best_move.value);
        return best_move.value;
    }
    return SCORE_UNDEFINED;
}


/*
inline int book_learn_nws(Board board, int level, int bound, bool minus_1){
    Search search;
    search.board = board;
    calc_features(&search);
    int depth;
    bool is_mid_search, searching = true;
    get_level(level, search.board.n - 4, &is_mid_search, &depth, &search.use_mpc, &search.mpct);
    if (minus_1)
        depth = max(0, depth - 1);
    int g;
    if (is_mid_search){
        parent_transpose_table.init();
        g = nega_scout(&search, bound - 1, bound + 1, max(0, depth - 1), false, LEGAL_UNDEFINED, !is_mid_search, &searching);
        parent_transpose_table.init();
        g += nega_scout(&search, bound - 1, bound + 1, depth, false, LEGAL_UNDEFINED, !is_mid_search, &searching);
        g /= 2;
    } else{
        parent_transpose_table.init();
        g = nega_scout(&search, bound - 1, bound + 1, depth, false, LEGAL_UNDEFINED, !is_mid_search, &searching);
    }
    return g;
}

inline pair<int, int> book_learn_ai(Board board, int level){
    Search search;
    search.board = board;
    calc_features(&search);
    int g, depth;
    bool is_mid_search, searching = true;
    get_level(level, search.board.n - 4, &is_mid_search, &depth, &search.use_mpc, &search.mpct);
    if (is_mid_search){
        parent_transpose_table.init();
        g = nega_scout(&search, -SCORE_MAX, SCORE_MAX, max(0, depth - 1), false, LEGAL_UNDEFINED, !is_mid_search, &searching);
    }
    parent_transpose_table.init();
    pair<int, int> res = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, !is_mid_search, false);
    if (is_mid_search){
        res.first += g;
        res.first /= 2;
    }
    return res;
}

int book_learn_search(Board board, int level, const int book_depth, const int best_value, int error_sum, const int expected_error, const int adopt_error_sum, Board *board_copy, uint64_t *strt_tim, string book_file, string book_bak){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (error_sum > adopt_error_sum)
        return SCORE_UNDEFINED;
    if (tim() - *strt_tim > AUTO_BOOK_SAVE_TIME){
        book.save_bin(book_file, book_bak);
        *strt_tim = tim();
    }
    int g;
    if (board.n >= 4 + book_depth){
        g = -INF;
        if (-SCORE_MAX <= best_value && best_value <= SCORE_MAX)
            g = book_learn_nws(board, level, best_value, false);
        if ((g != -INF && g <= best_value) || best_value < -SCORE_MAX || SCORE_MAX < best_value){
            g = book_learn_ai(board, level).first;
            book.reg(board, -g);
            return g;
        }
        return SCORE_UNDEFINED;
    }
    uint64_t legal = board.get_legal();
    if (legal == 0ULL){
        board.pass();
            g = -book_learn_search(board, level, book_depth, -best_value, error_sum, expected_error, adopt_error_sum, board_copy, strt_tim, book_file, book_bak);
        board.pass();
        return g;
    }
    pair<int, int> best_move = book_learn_ai(board, level);
    int ai_pv_value = best_move.first;
    Flip flip;
    calc_flip(&flip, &board, (uint8_t)best_move.second);
    board.move(&flip);
        board.copy(board_copy);
        g = -book_learn_search(board, level, book_depth, SCORE_UNDEFINED, error_sum, expected_error, adopt_error_sum, board_copy, strt_tim, book_file, book_bak);
        if (-SCORE_MAX <= g && g <= SCORE_MAX){
            best_move.first = g;
            cerr << "PV value " << best_move.first << endl;
        } else
            best_move.first = -INF;
    board.undo(&flip);
    ai_pv_value = min(ai_pv_value, best_move.first);
    if (best_move.first < -SCORE_MAX || SCORE_MAX < best_move.first)
        return SCORE_UNDEFINED;
    legal ^= 1ULL << best_move.second;
    if (legal){
        vector<pair<Flip, int>> move_list;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &board, cell);
            board.move(&flip);
                board.copy(board_copy);
                g = book.get(&board);
                if (g < -SCORE_MAX || SCORE_MAX < g){
                    if (-SCORE_MAX <= ai_pv_value && ai_pv_value <= SCORE_MAX)
                        g = -book_learn_nws(board, level, -ai_pv_value + expected_error, true);
                }
                if (-SCORE_MAX <= g && g <= SCORE_MAX && g >= ai_pv_value - expected_error)
                    move_list.emplace_back(make_pair(flip, g));
            board.undo(&flip);
        }
        for (pair<Flip, int> &mov: move_list){
            board.move(&mov.first);
                g = -book_learn_search(board, level, book_depth, -best_move.first, error_sum + min(0, ai_pv_value - mov.second), expected_error, adopt_error_sum, board_copy, strt_tim, book_file, book_bak);
                if (-SCORE_MAX <= g && g <= SCORE_MAX){
                    if (g > best_move.first){
                        best_move.first = g;
                        best_move.second = (int)mov.first.pos;
                    }
                    cerr << "   value " << g << " best_move_value " << best_move.first << endl;
                }
            board.undo(&mov.first);
        }
    }
    if (-SCORE_MAX <= best_move.first && best_move.first <= SCORE_MAX && global_searching){
        book.reg(board, -best_move.first);
        return best_move.first;
    }
    return SCORE_UNDEFINED;
}
*/

inline void learn_book(Board root_board, int level, const int book_depth, const int expected_error, Board *board_copy, string book_file, string book_bak, bool *book_learning){
    uint64_t strt_tim = tim();
    cerr << "book learn started" << endl;
    const int adopt_error_sum = max(expected_error, (book_depth + 4 - root_board.n) * expected_error / 5);
    int g = book_learn_search(root_board, level, book_depth, SCORE_UNDEFINED, 0, expected_error, adopt_error_sum, board_copy, &strt_tim, book_file, book_bak);
    book.reg(root_board, -g);
    root_board.copy(board_copy);
    cerr << "book learn finished " << g << endl;
    *book_learning = false;
}
