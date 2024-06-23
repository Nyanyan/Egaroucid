/*
    Egaroucid Project

    @file ai.hpp
        Main algorithm of Egaroucid
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include <unordered_set>
#include "level.hpp"
#include "setting.hpp"
#include "midsearch.hpp"
#include "book.hpp"
#include "util.hpp"
#include "clogsearch.hpp"
#include "lazy_smp.hpp"

#define SEARCH_BOOK -1

#ifndef HINT_TYPE_BOOK
    #define HINT_TYPE_BOOK 1000
#endif

#define ENDSEARCH_PRESEARCH_OFFSET 10

Search_result iterative_deepening_search(Board board, int depth, uint_fast8_t mpc_level, bool show_log, std::vector<Clog_result> clogs, uint64_t use_legal, bool use_multi_thread){
    Search_result result;
    result.value = SCORE_UNDEFINED;
    result.nodes = 0;
    uint64_t strt = tim();
    const int max_depth = HW2 - board.n_discs();
    depth = std::min(depth, max_depth);
    int search_depth = 1;
    if (depth % 2 == 0 && depth >= 2){
        search_depth = 2;
    }
    int search_mpc_level = mpc_level;
    bool is_end_search = (depth == max_depth);
    if (is_end_search){
        search_mpc_level = MPC_74_LEVEL;
    }
    while (search_depth <= depth && search_mpc_level <= mpc_level && global_searching){
        bool search_is_end_search = false;
        if (search_depth >= max_depth){
            search_is_end_search = true;
            search_depth = max_depth;
        }
        bool is_last_search = (search_depth == depth) && (search_mpc_level == mpc_level);
        Search search;
        search.init(&board, search_mpc_level, use_multi_thread, false, !is_last_search);
        bool searching = true;
        std::pair<int, int> id_result = first_nega_scout_legal(&search, -SCORE_MAX, SCORE_MAX, result.value, search_depth, search_is_end_search, clogs, use_legal, strt, &searching);
        result.nodes += search.n_nodes;
        if (result.value != SCORE_UNDEFINED && !search_is_end_search){
            double n_value = (0.9 * result.value + 1.1 * id_result.first) / 2.0;
            result.value = round(n_value);
        } else{
            result.value = id_result.first;
        }
        result.policy = id_result.second;
        result.depth = search_depth;
        result.time = tim() - strt;
        result.nps = calc_nps(result.nodes, result.time);
        if (show_log){
            if (is_last_search){
                std::cerr << "main ";
            } else{
                std::cerr << "pre ";
            }
            if (search_is_end_search){
                std::cerr << "end ";
            } else{
                std::cerr << "mid ";
            }
            std::cerr << "depth " << result.depth << "@" << SELECTIVITY_PERCENTAGE[search_mpc_level] << "%" << " value " << result.value << " (raw " << id_result.first << ") policy " << idx_to_coord(id_result.second) << " n_nodes " << result.nodes << " time " << result.time << " NPS " << result.nps << std::endl;
        }
        if (!is_end_search || search_depth < depth - ENDSEARCH_PRESEARCH_OFFSET){
            ++search_depth;
        } else{
            if (search_depth < depth){
                search_depth = depth;
                search_mpc_level = MPC_74_LEVEL;
            } else{
                if (search_mpc_level == MPC_88_LEVEL && mpc_level > MPC_88_LEVEL){
                    search_mpc_level = mpc_level;
                } else{
                    ++search_mpc_level;
                }
            }
        }
    }
    result.is_end_search = is_end_search;
    result.probability = SELECTIVITY_PERCENTAGE[mpc_level];
    return result;
}
/*
Search_result endgame_optimized_search(Board board, int depth, uint_fast8_t mpc_level, bool show_log, std::vector<Clog_result> clogs, uint64_t use_legal, bool use_multi_thread){
    Search_result result;
    result.value = SCORE_UNDEFINED;
    result.nodes = 0;
    uint64_t strt = tim();
    int search_depth = depth * 0.3;
    if (search_depth < 1){
        search_depth = 1;
    } else if (search_depth > 5){
        search_depth = 5;
    }
    int search_mpc_level = MPC_74_LEVEL;
    while (search_depth <= depth && search_mpc_level <= mpc_level && global_searching){
        bool search_is_end_search = false;
        if (search_depth >= depth){
            search_is_end_search = true;
            search_depth = depth;
        }
        bool is_last_search = (search_depth == depth) && (search_mpc_level == mpc_level);
        Search search;
        search.init(&board, search_mpc_level, use_multi_thread, false, !is_last_search);
        bool searching = true;
        std::pair<int, int> id_result = first_nega_scout_legal(&search, -SCORE_MAX, SCORE_MAX, result.value, search_depth, search_is_end_search, clogs, use_legal, strt, &searching);
        result.nodes += search.n_nodes;
        if (result.value != SCORE_UNDEFINED && !search_is_end_search){
            double n_value = (0.9 * result.value + 1.1 * id_result.first) / 2.0;
            result.value = round(n_value);
        } else{
            result.value = id_result.first;
        }
        result.policy = id_result.second;
        result.depth = search_depth;
        result.time = tim() - strt;
        result.nps = calc_nps(result.nodes, result.time);
        if (show_log){
            if (is_last_search){
                std::cerr << "main ";
            } else{
                std::cerr << "pre ";
            }
            if (search_is_end_search){
                std::cerr << "end ";
            } else{
                std::cerr << "mid ";
            }
            std::cerr << "depth " << result.depth << "@" << SELECTIVITY_PERCENTAGE[search_mpc_level] << "%" << " value " << result.value << " (raw " << id_result.first << ") policy " << idx_to_coord(id_result.second) << " n_nodes " << result.nodes << " time " << result.time << " NPS " << result.nps << std::endl;
        }
        if (search_depth < depth - ENDSEARCH_PRESEARCH_OFFSET){
            if (search_depth < 10){
                search_depth += 3;
            } else{
                search_depth += 1;
            }
        } else{
            if (search_depth < depth){
                search_depth = depth;
                search_mpc_level = MPC_74_LEVEL;
            } else{
                if (search_mpc_level == MPC_88_LEVEL && mpc_level > MPC_88_LEVEL){
                    search_mpc_level = mpc_level;
                } else{
                    ++search_mpc_level;
                }
            }
        }
    }
    result.is_end_search = true;
    result.probability = SELECTIVITY_PERCENTAGE[mpc_level];
    return result;
}
*/

void iterative_deepening_search_hint(Board board, int depth, uint_fast8_t mpc_level, bool show_log, uint64_t use_legal, bool use_multi_thread, int n_display, double values[], int hint_types[]){
    uint64_t strt = tim();
    int search_depth = 1;
    int search_mpc_level = mpc_level;
    const int max_depth = HW2 - board.n_discs();
    depth = std::min(depth, max_depth);
    bool is_end_search = (depth == max_depth);
    if (is_end_search){
        search_mpc_level = MPC_74_LEVEL;
    }
    while (search_depth <= depth && search_mpc_level <= mpc_level && global_searching){
        bool is_last_search = (search_depth == depth) && (search_mpc_level == mpc_level);
        bool search_is_end_search = false;
        if (search_depth >= max_depth){
            search_is_end_search = true;
            search_depth = max_depth;
        }
        Search search;
        search.init(&board, search_mpc_level, use_multi_thread, false, !is_last_search);
        bool searching = true;
        int hint_type = search_depth;
        if (search_is_end_search){ // endgame & this is last search
            hint_type = SELECTIVITY_PERCENTAGE[search_mpc_level];
        }
        uint64_t use_legal_copy = use_legal;
        first_nega_scout_hint(&search, search_depth, depth, search_is_end_search, use_legal, &searching, values, hint_types, hint_type, n_display);
        if (is_last_search){
            std::cerr << "main ";
        } else{
            std::cerr << "pre ";
        }
        if (search_is_end_search){
            std::cerr << "end ";
        } else{
            std::cerr << "mid ";
        }
        std::cerr << "depth " << search_depth << "@" << SELECTIVITY_PERCENTAGE[search_mpc_level] << "%" << std::endl;
        if (show_log){
            for (int y = 0; y < HW; ++y){
                for (int x = 0; x < HW; ++x){
                    int cell = HW2_M1 - y * 8 - x;
                    if (1 & (board.get_legal() >> cell))
                        std::cerr << round(values[cell]) << " ";
                    else
                        std::cerr << "  ";
                }
                std::cerr << std::endl;
            }
        }
        // update depth
        if (!is_end_search || search_depth < depth - ENDSEARCH_PRESEARCH_OFFSET){
            ++search_depth;
        } else{
            if (search_depth < depth){
                search_depth = depth;
                search_mpc_level = MPC_74_LEVEL;
            } else{
                if (search_mpc_level == MPC_88_LEVEL && mpc_level > MPC_88_LEVEL){
                    search_mpc_level = mpc_level;
                } else{
                    ++search_mpc_level;
                }
            }
        }
    }
}

/*
    @brief Get a result of a search

    Firstly, if using MPC, execute clog search for finding special endgame.
    Then do some pre-search, and main search.

    @param board                board to solve
    @param depth                depth to search
    @param mpc_level            MPC level
    @param show_log             show log?
    @param use_multi_thread     search in multi thread?
    @return the result in Search_result structure
*/
inline Search_result tree_search_legal(Board board, int depth, uint_fast8_t mpc_level, bool show_log, uint64_t use_legal, bool use_multi_thread){
    Search_result res;
    depth = std::min(HW2 - board.n_discs(), depth);
    bool is_end_search = (HW2 - board.n_discs() == depth);
    std::vector<Clog_result> clogs;
    uint64_t clog_nodes = 0;
    uint64_t clog_time = 0;
    if (mpc_level != MPC_100_LEVEL){
        uint64_t strt = tim();
        int clog_depth = std::min(depth, CLOG_SEARCH_MAX_DEPTH);
        clogs = first_clog_search(board, &clog_nodes, clog_depth, use_legal);
        clog_time = tim() - strt;
        if (show_log){
            std::cerr << "clog search depth " << clog_depth << " time " << clog_time << " nodes " << clog_nodes << " nps " << calc_nps(clog_nodes, clog_time) << std::endl;
            for (int i = 0; i < (int)clogs.size(); ++i){
                std::cerr << "clogsearch " << i + 1 << "/" << clogs.size() << " " << idx_to_coord(clogs[i].pos) << " value " << clogs[i].val << std::endl;
            }
        }
        res.clog_nodes = clog_nodes;
        res.clog_time = clog_time;
        res.depth = clog_depth;
        res.is_end_search = clog_depth >= HW2 - board.n_discs();
        res.nodes = 0;
        res.nps = 0;
        res.probability = 100;
        res.time = clog_time;
        res.value = SCORE_UNDEFINED;
        for (int i = 0; i < (int)clogs.size(); ++i){
            if (clogs[i].val > res.value){
                res.value = clogs[i].val;
                res.policy = clogs[i].pos;
            }
        }
    }
    if (use_legal){
        res = lazy_smp(board, depth, mpc_level, show_log, clogs, use_legal, use_multi_thread);
        /*
        if (is_end_search){
            res = endgame_optimized_search(board, depth, mpc_level, show_log, clogs, use_legal, use_multi_thread);
        } else{
            res = lazy_smp(board, depth, mpc_level, show_log, clogs, use_legal, use_multi_thread);
            //res = iterative_deepening_search(board, depth, mpc_level, show_log, clogs, use_legal, use_multi_thread);
        }
        */
    }
    thread_pool.reset_unavailable();
    return res;
}

inline void tree_search_hint(Board board, int depth, uint_fast8_t mpc_level, bool use_multi_thread, bool show_log, uint64_t use_legal, int n_display, double values[], int hint_types[]){
    depth = std::min(HW2 - board.n_discs(), depth);
    bool is_end_search = (HW2 - board.n_discs() == depth);
    std::vector<Clog_result> clogs;
    uint64_t clog_nodes = 0;
    uint64_t clog_time = 0;
    if (mpc_level != MPC_100_LEVEL){
        uint64_t strt = tim();
        int clog_depth = std::min(depth, CLOG_SEARCH_MAX_DEPTH);
        clogs = first_clog_search(board, &clog_nodes, clog_depth, use_legal);
        clog_time = tim() - strt;
        for (Clog_result &clog: clogs){
            if (1 & (use_legal >> clog.pos)){
                values[clog.pos] = clog.val;
                hint_types[clog.pos] = 100;
                use_legal ^= 1ULL << clog.pos;
                --n_display;
            }
        }
        if (show_log){
            std::cerr << "clog search depth " << clog_depth << " time " << clog_time << " nodes " << clog_nodes << " nps " << calc_nps(clog_nodes, clog_time) << std::endl;
            for (int i = 0; i < (int)clogs.size(); ++i){
                std::cerr << "clogsearch " << i + 1 << "/" << clogs.size() << " " << idx_to_coord(clogs[i].pos) << " value " << clogs[i].val << std::endl;
            }
        }
    }
    if (n_display < 0){
        return;
    }
    //lazy_smp_hint(board, depth, mpc_level, show_log, use_legal, use_multi_thread, n_display, values, hint_types);
    iterative_deepening_search_hint(board, depth, mpc_level, show_log, use_legal, use_multi_thread, n_display, values, hint_types);
    thread_pool.reset_unavailable();
}

/*
    @brief Get a result of a search with book or search

    Firstly, check if the given board is in the book.
    Then search the board and get the result.

    @param board                board to solve
    @param level                level of AI
    @param use_book             use book?
	@param book_acc_level		book accuracy level
    @param use_multi_thread     search in multi thread?
    @param show_log             show log?
    @return the result in Search_result structure
*/
Search_result ai_legal(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal){
    Search_result res;
    int value_sign = 1;
    if (board.get_legal() == 0ULL){
        board.pass();
        if (board.get_legal() == 0ULL){
            res.policy = 64;
            res.value = -board.score_player();
            res.depth = 0;
            res.nps = 0;
            res.is_end_search = true;
            res.probability = 100;
            return res;
        } else{
            value_sign = -1;
        }
    }
    Book_value book_result = book.get_random_specified_moves(&board, book_acc_level, use_legal);
    if (book_result.policy != -1 && use_book){
        if (show_log)
            std::cerr << "book " << idx_to_coord(book_result.policy) << " " << book_result.value << " at book error level " << book_acc_level << std::endl;
        res.policy = book_result.policy;
        res.value = value_sign * book_result.value;
        res.depth = SEARCH_BOOK;
        res.nps = 0;
        res.is_end_search = false;
        res.probability = 100;
    } else{
        int depth;
        bool is_mid_search;
        uint_fast8_t mpc_level;
        get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
        if (show_log)
            std::cerr << "level status " << level << " " << board.n_discs() - 4 << " discs depth " << depth << "@" << SELECTIVITY_PERCENTAGE[mpc_level] << "%" << std::endl;
        res = tree_search_legal(board, depth, mpc_level, show_log, use_legal, use_multi_thread);
        res.value *= value_sign;
    }
    return res;
}

/*
    @brief Get a result of a search with book or search

    Firstly, check if the given board is in the book.
    Then search the board and get the result.

    @param board                board to solve
    @param level                level of AI
    @param use_book             use book?
	@param book_acc_level		book accuracy level
    @param use_multi_thread     search in multi thread?
    @param show_log             show log?
    @return the result in Search_result structure
*/
Search_result ai(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log){
    return ai_legal(board, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal());
}

/*
    @brief Search for analyze command

    @param board                board to solve
    @param level                level of AI
    @param use_multi_thread     search in multi thread?
    @return the result in Search_result structure
*/

Analyze_result ai_analyze(Board board, int level, bool use_multi_thread, uint_fast8_t played_move){
    int depth;
    bool is_mid_search;
    uint_fast8_t mpc_level;
    get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
    depth = std::min(HW2 - board.n_discs(), depth);
    bool is_end_search = (HW2 - board.n_discs() == depth);
    std::vector<Clog_result> clogs;
    uint64_t clog_nodes = 0;
    uint64_t clog_time = 0;
    int clog_depth = std::min(depth, CLOG_SEARCH_MAX_DEPTH);
    if (mpc_level != MPC_100_LEVEL){
        uint64_t clog_strt = tim();
        clogs = first_clog_search(board, &clog_nodes, clog_depth, board.get_legal());
        clog_time = tim() - clog_strt;
    }
    Search search;
    search.init(&board, mpc_level, use_multi_thread, false, false);
    uint64_t strt = tim();
    bool searching = true;
    return first_nega_scout_analyze(&search, -SCORE_MAX, SCORE_MAX, depth, is_end_search, clogs, clog_depth, played_move, strt, &searching);
}



Search_result ai_accept_loss(Board board, int level, int acceptable_loss){
    uint64_t strt = tim();
    Flip flip;
    int v = SCORE_UNDEFINED;
    uint64_t legal = board.get_legal();
    std::vector<std::pair<int, int>> moves;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
            int g = -ai(board, level, true, 0, true, false).value;
        board.undo_board(&flip);
        v = std::max(v, g);
        moves.emplace_back(std::make_pair(g, cell));
    }
    std::vector<std::pair<int, int>> acceptable_moves;
    for (std::pair<int, int> move: moves){
        if (move.first >= v - acceptable_loss)
            acceptable_moves.emplace_back(move);
    }
    int rnd_idx = myrandrange(0, (int)acceptable_moves.size());
    int use_policy = acceptable_moves[rnd_idx].second;
    int use_value = acceptable_moves[rnd_idx].first;
    Search_result res;
    res.depth = 1;
    res.nodes = 0;
    res.time = tim() - strt;
    res.nps = calc_nps(res.nodes, res.time);
    res.policy = use_policy;
    res.value = use_value;
    res.is_end_search = board.n_discs() == HW2 - 1;
    res.probability = SELECTIVITY_PERCENTAGE[MPC_100_LEVEL];
    return res;
}

void ai_hint(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, int n_display, double values[], int hint_types[]){
    uint64_t legal = board.get_legal();
    if (use_book){
        std::vector<Book_value> links = book.get_all_moves_with_value(&board);
        for (Book_value &link: links){
            values[link.policy] = link.value;
            hint_types[link.policy] = HINT_TYPE_BOOK;
            legal ^= 1ULL << link.policy;
            --n_display;
        }
    }
    int depth;
    bool is_mid_search;
    uint_fast8_t mpc_level;
    get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
    if (show_log)
        std::cerr << "level status " << level << " " << board.n_discs() - 4 << " discs depth " << depth << "@" << SELECTIVITY_PERCENTAGE[mpc_level] << "%" << std::endl;
    tree_search_hint(board, depth, mpc_level, use_multi_thread, show_log, legal, n_display, values, hint_types);
}
