/*
    Egaroucid Project

    @file function.hpp
        Functions about engine
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include "./../engine/engine_all.hpp"
#include "./../engine/contest_book.hpp"
#include "command.hpp"

#if USE_THREAD_MONITOR
    #include "./../engine/thread_monitor.hpp"
#endif

#define SELF_PLAY_N_TRY 1

void setboard(Board_info *board, Options *options, State *state, std::string board_str);
Search_result go_noprint(Board_info *board, Options *options, State *state);
void print_search_result_head();
void print_search_result_body(Search_result result, const Options *options, const State *state);
void go(Board_info *board, Options *options, State *state, uint64_t start_time);

void solve_problems(std::vector<std::string> arg, Options *options, State *state) {
    if (arg.size() < 1) {
        std::cerr << "[ERROR] [FATAL] please input problem file" << std::endl;
        return;
    }
    std::string file = arg[0];
    std::ifstream ifs(file);
    if (ifs.fail()) {
        std::cerr << "[ERROR] [FATAL] no problem file found" << std::endl;
        return;
    }
    std::string line;
    Board_info board;
    board.reset();
    print_search_result_head();
    Search_result total;
    total.nodes = 0;
    total.time = 0;
    #if USE_YBWC_SPLIT_STATISTICS
        ybwc_split_stats_reset();
    #endif
    while (std::getline(ifs, line)) {
        transposition_table.init();
        setboard(&board, options, state, line);
        #if USE_THREAD_MONITOR
            start_thread_monitor();
        #endif
        Search_result res = go_noprint(&board, options, state);
        print_search_result_body(res, options, state);
        total.nodes += res.nodes;
        total.time += res.time;
    }
    std::cout << "total " << total.nodes << " nodes in " << ((double)total.time / 1000) << "s NPS " << calc_nps(total.nodes, total.time) << std::endl;
    #if USE_YBWC_SPLIT_STATISTICS
        ybwc_split_stats_print();
    #endif
}

void solve_problems_transcript_parallel(std::vector<std::string> arg, Options *options, State *state) {
    if (arg.size() < 1) {
        std::cerr << "[ERROR] [FATAL] please input problem file" << std::endl;
        return;
    }
    std::string file = arg[0];
    std::ifstream ifs(file);
    if (ifs.fail()) {
        std::cerr << "[ERROR] [FATAL] no problem file found" << std::endl;
        return;
    }
    uint64_t strt = tim();
    std::string line;
    std::vector<Board> board_list;
    Flip flip;
    Board board_start;
    while (std::getline(ifs, line)) {
        /*
        std::pair<Board, int> board_player = convert_board_from_str(line);
        if (board_player.second != BLACK && board_player.second != WHITE) {
            std::cerr << "[ERROR] can't convert board " << line << std::endl;
            std::exit(1);
        }
        board_list.emplace_back(board_player.first);
        */
        board_start.reset();
        for (int i = 0; i < (int)line.size() - 1; i += 2) {
            int x = line[i] - 'a';
            int y = line[i + 1] - '1';
            int coord = HW2_M1 - (y * HW + x);
            calc_flip(&flip, &board_start, coord);
            board_start.move_board(&flip);
            if (board_start.get_legal() == 0) {
                board_start.pass();
            }
        }
        board_list.emplace_back(board_start);
    }
    Search_result result;
    if (thread_pool.size() == 0) {
        for (int i = 0; i < (int)board_list.size(); ++i) {
            result = ai(board_list[i], options->level, true, 0, false, options->show_log);
            std::cout << board_list[i].to_str() << " " << result.value << std::endl;
        }
    } else {
        int print_task_idx = 0;
        std::vector<std::future<Search_result>> tasks;
        for (Board &board: board_list) {
            bool go_to_next_task = false;
            while (!go_to_next_task) {
                if (thread_pool.get_n_idle() && tasks.size() < board_list.size()) {
                    bool pushed = false;
                    tasks.emplace_back(thread_pool.push(&pushed, std::bind(&ai, board, options->level, true, 0, false, options->show_log)));
                    if (pushed) {
                        go_to_next_task = true;
                    } else {
                        tasks.pop_back();
                    }
                }
                if (tasks.size() > print_task_idx) {
                    if (tasks[print_task_idx].valid()) {
                        if (tasks[print_task_idx].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                            result = tasks[print_task_idx].get();
                            std::cout << board_list[print_task_idx].to_str() << " " << result.value << std::endl;
                            ++print_task_idx;
                        }
                    } else {
                        std::cerr << "[ERROR] task not valid" << std::endl;
                        std::exit(1);
                    }
                }
            }
        }
        while (print_task_idx < tasks.size()) {
            if (tasks[print_task_idx].valid()) {
                result = tasks[print_task_idx].get();
                std::cout << board_list[print_task_idx].to_str() << " " << result.value << std::endl;
                ++print_task_idx;
            } else {
                std::cerr << "[ERROR] task not valid" << std::endl;
                std::exit(1);
            }
        }
    }
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
}

void execute_special_tasks(Options options) {
    // move ordering tuning
    #if TUNE_MOVE_ORDERING
        std::cout << "tune move ordering ";
        tune_move_ordering(options.level);
        std::exit(0);
    #endif

    // probcut (midsearch)
    #if TUNE_PROBCUT_MID
        std::cout << "tune probcut (midsearch)" << std::endl;
        get_data_probcut_mid();
        std::exit(0);
    #endif

    // probcut (endsearch)
    #if TUNE_PROBCUT_END
        std::cout << "tune probcut (endsearch)" << std::endl;
        get_data_probcut_end();
        std::exit(0);
    #endif

    // local strategy
    #if TUNE_LOCAL_STRATEGY
        std::cout << "tune local strategy" << std::endl;
        tune_local_strategy();
        std::exit(0);
    #endif

    #if TEST_ENDGAME_ACCURACY
        endgame_accuracy_test();
        std::exit(0);
    #endif
}

bool execute_special_tasks_loop(Board_info *board, State *state, Options *options) {
    uint64_t start_time = tim();
    int player_before = board->player;
    if (options->mode == MODE_HUMAN_AI && board->player == WHITE && !board->board.is_end()) {
        go(board, options, state, start_time);
        return true;
    } else if (options->mode == MODE_AI_HUMAN && board->player == BLACK && !board->board.is_end()) {
        go(board, options, state, start_time);
        return true;
    } else if (options->mode == MODE_AI_AI && !board->board.is_end()) {
        go(board, options, state, start_time);
        return true;
    }
    return false;
}


std::string self_play_task(Board board_start, std::string pre_moves_transcript, Options *options, bool use_multi_thread, int n_random_moves_additional, int n_try) {
    Flip flip;
    Search_result result;
    std::string res = pre_moves_transcript;
    for (int j = 0; j < n_random_moves_additional && board_start.check_pass(); ++j) {
        uint64_t legal = board_start.get_legal();
        int random_idx = myrandrange(0, pop_count_ull(legal));
        int t = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
            if (t == random_idx) {
                calc_flip(&flip, &board_start, cell);
                break;
            }
            ++t;
        }
        res += idx_to_coord(flip.pos);
        board_start.move_board(&flip);
    }
    std::vector<int> prev_transcript;
    for (int i = 0; i < n_try; ++i) {
        Board board = board_start.copy();
        std::vector<int> transcript;
        while (board.check_pass()) {
            result = ai(board, options->level, true, 0, use_multi_thread, options->show_log);
            if (global_searching && is_valid_policy(result.policy)) {
                transcript.emplace_back(result.policy);
                calc_flip(&flip, &board, result.policy);
                board.move_board(&flip);
            } else {
                break;
            }
        }
        if (!global_searching) {
            if (n_try == 1 || i == n_try - 1) {
                prev_transcript.clear();
                for (int &elem: transcript) {
                    prev_transcript.emplace_back(elem);
                }
            }
            break;
        }
        bool break_flag = true;
        if (i < n_try - 1) {
            if (prev_transcript.size() != transcript.size()) {
                break_flag = false;
            } else {
                for (int i = 0; i < transcript.size(); ++i) {
                    if (transcript[i] != prev_transcript[i]) {
                        break_flag = false;
                        break;
                    }
                }
            }
        }
        prev_transcript.clear();
        for (int &elem: transcript) {
            prev_transcript.emplace_back(elem);
        }
        if (break_flag) {
            break;
        }
    }
    for (int &elem: prev_transcript) {
        res += idx_to_coord(elem);
    }
    return res;
}

struct Contest_record_scored_move {
    int policy;
    int value;

    Contest_record_scored_move(int policy_, int value_)
        : policy(policy_), value(value_) {}
};

using Contest_record_score_cache = std::unordered_map<Board, std::vector<Contest_record_scored_move>, Contest_book_hash>;

#define CONTEST_RECORD_THREAD_ID_BASE 10

struct Contest_record_state {
    Board board;
    std::string transcript;
    int loss_sum;

    Contest_record_state(Board board_, std::string transcript_, int loss_sum_)
        : board(board_), transcript(transcript_), loss_sum(loss_sum_) {}
};

struct Contest_record_leaf {
    Board board;
    std::string transcript;
    int loss_sum;
    int leaf_value;

    Contest_record_leaf(Board board_, std::string transcript_, int loss_sum_, int leaf_value_)
        : board(board_), transcript(transcript_), loss_sum(loss_sum_), leaf_value(leaf_value_) {}
};

struct Contest_record_expand_result {
    std::vector<Contest_record_state> next_states;
    std::vector<Contest_record_leaf> leaves;
};

struct Contest_record_running_task {
    int slot;
    std::future<Contest_record_expand_result> future;

    Contest_record_running_task(int slot_, std::future<Contest_record_expand_result> &&future_)
        : slot(slot_), future(std::move(future_)) {}
};

std::vector<Contest_record_scored_move> contest_record_score_moves(Board board, int level, bool use_multi_thread, thread_id_t thread_id) {
    std::vector<Contest_record_scored_move> res;
    uint64_t legal = board.get_legal();
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
        bool searching = true;
        Search_result search_result = ai_searching_thread_id(board, level, false, 0, use_multi_thread, false, thread_id, &searching);
        int value = search_result.value == SCORE_UNDEFINED ? SCORE_UNDEFINED : -search_result.value;
        res.emplace_back((int)cell, value);
        board.undo_board(&flip);
    }
    std::sort(res.begin(), res.end(), [](const Contest_record_scored_move &a, const Contest_record_scored_move &b) {
        if (a.value != b.value) {
            return a.value > b.value;
        }
        return a.policy < b.policy;
    });
    return res;
}

std::vector<Contest_record_scored_move> contest_record_score_moves_cached(
    Board board,
    int level,
    bool use_multi_thread,
    thread_id_t thread_id,
    Contest_record_score_cache *score_cache,
    std::mutex *score_cache_mtx
) {
    {
        std::lock_guard<std::mutex> lock(*score_cache_mtx);
        auto it = score_cache->find(board);
        if (it != score_cache->end()) {
            return it->second;
        }
    }
    std::vector<Contest_record_scored_move> res = contest_record_score_moves(board, level, use_multi_thread, thread_id);
    {
        std::lock_guard<std::mutex> lock(*score_cache_mtx);
        auto it = score_cache->find(board);
        if (it != score_cache->end()) {
            return it->second;
        }
        (*score_cache)[board] = res;
    }
    return res;
}

int contest_record_leaf_value(Board board, bool use_multi_thread, thread_id_t thread_id) {
    if (board.is_end()) {
        return board.score_player();
    }
    bool searching = true;
    Search_result search_result = ai_searching_thread_id(board, MAX_LEVEL, false, 0, use_multi_thread, false, thread_id, &searching);
    if (search_result.value != SCORE_UNDEFINED) {
        return search_result.value;
    }
    return mid_evaluate(&board);
}

bool contest_record_canonical_start(const std::string &raw_board, Board *board, std::string *canonical_board) {
    std::string compact = raw_board;
    compact.erase(std::remove_if(compact.begin(), compact.end(), ::isspace), compact.end());
    if (compact.size() != HW2 + 1) {
        std::cerr << "[ERROR] invalid contest start board length " << compact.size() << std::endl;
        return false;
    }
    if (!board->from_str(raw_board)) {
        return false;
    }
    int player_to_move = BLACK;
    if (is_white_like_char(compact[HW2])) {
        player_to_move = WHITE;
    } else if (!is_black_like_char(compact[HW2])) {
        std::cerr << "[ERROR] invalid contest start player" << std::endl;
        return false;
    }
    *canonical_board = board->to_str(player_to_move);
    return true;
}

void contest_record_write_record(
    std::ofstream &ofs,
    const std::string &initial_board,
    const std::string &transcript,
    Board board,
    int loss_sum,
    int leaf_value,
    int record_idx
) {
    ofs << "record: " << record_idx << '\n';
    ofs << "initial board: " << initial_board << '\n';
    ofs << "transcript: " << transcript << '\n';
    ofs << "leaf board: " << board.to_str() << '\n';
    ofs << "leaf empty: " << (HW2 - board.n_discs()) << '\n';
    ofs << "leaf value: " << leaf_value << '\n';
    ofs << "loss sum: " << loss_sum << '\n';
    ofs << '\n';
}

Contest_record_expand_result contest_record_expand_state(
    Contest_record_state state,
    Options *options,
    int max_loss_per_move,
    int target_loss_sum,
    int cut_empty,
    Contest_record_score_cache *score_cache,
    std::mutex *score_cache_mtx,
    bool use_multi_thread,
    thread_id_t thread_id
) {
    Contest_record_expand_result result;
    while (HW2 - state.board.n_discs() > cut_empty && state.board.get_legal() == 0ULL && !state.board.is_end()) {
        state.board.pass();
    }
    if (HW2 - state.board.n_discs() <= cut_empty || state.board.is_end()) {
        if (state.loss_sum == target_loss_sum) {
            result.leaves.emplace_back(
                state.board,
                state.transcript,
                state.loss_sum,
                contest_record_leaf_value(state.board, use_multi_thread, thread_id)
            );
        }
        return result;
    }

    std::vector<Contest_record_scored_move> scored_moves = contest_record_score_moves_cached(state.board, options->level, use_multi_thread, thread_id, score_cache, score_cache_mtx);
    if (scored_moves.empty() || scored_moves[0].value == SCORE_UNDEFINED) {
        if (state.loss_sum == target_loss_sum) {
            result.leaves.emplace_back(
                state.board,
                state.transcript,
                state.loss_sum,
                contest_record_leaf_value(state.board, use_multi_thread, thread_id)
            );
        }
        return result;
    }

    int best_value = scored_moves[0].value;
    Flip flip;
    for (const Contest_record_scored_move &move: scored_moves) {
        if (move.value == SCORE_UNDEFINED) {
            continue;
        }
        int move_loss = best_value - move.value;
        if (move_loss > max_loss_per_move || state.loss_sum + move_loss > target_loss_sum) {
            continue;
        }
        Board next_board = state.board.copy();
        calc_flip(&flip, &next_board, move.policy);
        next_board.move_board(&flip);
        result.next_states.emplace_back(
            next_board,
            state.transcript + idx_to_coord(move.policy),
            state.loss_sum + move_loss
        );
    }
    return result;
}

void contest_record_append_next_states(std::vector<Contest_record_state> *pending, const std::vector<Contest_record_state> &next_states) {
    for (auto it = next_states.rbegin(); it != next_states.rend(); ++it) {
        pending->emplace_back(*it);
    }
}

int contest_record_parallel_width() {
    return std::max(1, std::min(thread_pool.size() + 1, THREAD_ID_NONE - CONTEST_RECORD_THREAD_ID_BASE));
}

int contest_record_threads_for_progress(int progress_idx, int progress_width) {
    int total_threads = thread_pool.size() + 1;
    int base_threads = std::max(1, total_threads / progress_width);
    int extra_threads = total_threads % progress_width;
    return base_threads + (progress_idx < extra_threads ? 1 : 0);
}

bool contest_record_worker_slot_used(const std::vector<Contest_record_running_task> &tasks, int slot) {
    for (const Contest_record_running_task &task: tasks) {
        if (task.slot == slot) {
            return true;
        }
    }
    return false;
}

int contest_record_first_available_worker_slot(const std::vector<Contest_record_running_task> &tasks, int max_width) {
    for (int slot = 1; slot < max_width; ++slot) {
        if (!contest_record_worker_slot_used(tasks, slot)) {
            return slot;
        }
    }
    return -1;
}

void contest_record_process_expand_result(
    const Contest_record_expand_result &result,
    std::vector<Contest_record_state> *pending,
    const std::string &initial_board,
    int n_records_limit,
    int *n_generated,
    std::ofstream &ofs
) {
    for (const Contest_record_leaf &leaf: result.leaves) {
        if (*n_generated >= n_records_limit) {
            break;
        }
        contest_record_write_record(ofs, initial_board, leaf.transcript, leaf.board, leaf.loss_sum, leaf.leaf_value, *n_generated);
        ++(*n_generated);
        ofs.flush();
    }
    if (*n_generated < n_records_limit) {
        contest_record_append_next_states(pending, result.next_states);
    }
}

bool contest_record_collect_ready_tasks(
    std::vector<Contest_record_running_task> *tasks,
    std::vector<Contest_record_state> *pending,
    const std::string &initial_board,
    int n_records_limit,
    int *n_generated,
    std::ofstream &ofs,
    bool wait_for_one
) {
    bool collected = false;
    while (true) {
        for (int i = 0; i < (int)tasks->size(); ++i) {
            std::future_status status = (*tasks)[i].future.wait_for(wait_for_one ? std::chrono::milliseconds(1) : std::chrono::milliseconds(0));
            if (status != std::future_status::ready) {
                continue;
            }
            Contest_record_expand_result result = (*tasks)[i].future.get();
            tasks->erase(tasks->begin() + i);
            contest_record_process_expand_result(result, pending, initial_board, n_records_limit, n_generated, ofs);
            collected = true;
            --i;
            if (wait_for_one) {
                return true;
            }
        }
        if (collected || !wait_for_one || tasks->empty()) {
            return collected;
        }
    }
}

bool contest_record_launch_ready_tasks(
    std::vector<Contest_record_state> *pending,
    std::vector<Contest_record_running_task> *tasks,
    Options *options,
    int max_loss_per_move,
    int target_loss_sum,
    int cut_empty,
    Contest_record_score_cache *score_cache,
    std::mutex *score_cache_mtx
) {
    bool launched = false;
    int max_width = contest_record_parallel_width();
    while (!pending->empty()) {
        int target_width = std::min(max_width, (int)pending->size() + (int)tasks->size());
        int target_worker_count = std::max(0, target_width - 1);
        if ((int)tasks->size() >= target_worker_count) {
            break;
        }
        int slot = contest_record_first_available_worker_slot(*tasks, max_width);
        if (slot < 0) {
            break;
        }

        Contest_record_state state = pending->back();
        pending->pop_back();
        int progress_idx = (int)tasks->size() + 1;
        int threads_for_state = contest_record_threads_for_progress(progress_idx, target_width);
        thread_id_t thread_id = CONTEST_RECORD_THREAD_ID_BASE + slot;
        thread_pool.set_max_thread_size(thread_id, threads_for_state);

        bool pushed = false;
        std::future<Contest_record_expand_result> future = thread_pool.push(thread_id, &pushed, std::bind(
            contest_record_expand_state,
            state,
            options,
            max_loss_per_move,
            target_loss_sum,
            cut_empty,
            score_cache,
            score_cache_mtx,
            threads_for_state > 1,
            thread_id
        ));
        if (!pushed) {
            pending->emplace_back(state);
            break;
        }
        tasks->emplace_back(slot, std::move(future));
        launched = true;
    }
    return launched;
}

void contest_record_enumerate_exact_loss(
    Board board,
    const std::string &initial_board,
    Options *options,
    int max_loss_per_move,
    int target_loss_sum,
    int cut_empty,
    int n_records_limit,
    int *n_generated,
    std::ofstream &ofs,
    Contest_record_score_cache *score_cache
) {
    std::mutex score_cache_mtx;
    std::vector<Contest_record_state> pending;
    pending.emplace_back(board, "", 0);
    std::vector<Contest_record_running_task> tasks;
    int last_width = -1;
    int last_main_threads = -1;

    while (((!pending.empty() && *n_generated < n_records_limit) || !tasks.empty())) {
        if (*n_generated < n_records_limit) {
            contest_record_launch_ready_tasks(
                &pending,
                &tasks,
                options,
                max_loss_per_move,
                target_loss_sum,
                cut_empty,
                score_cache,
                &score_cache_mtx
            );
        } else {
            pending.clear();
        }

        bool progressed = contest_record_collect_ready_tasks(&tasks, &pending, initial_board, n_records_limit, n_generated, ofs, false);

        if (*n_generated < n_records_limit && !pending.empty()) {
            int target_width = std::min(contest_record_parallel_width(), (int)tasks.size() + 1);
            int main_threads_for_state = contest_record_threads_for_progress(0, target_width);
            if (target_width != last_width || main_threads_for_state != last_main_threads) {
                std::cerr << "contest record parallel width " << target_width
                          << " main_threads " << main_threads_for_state
                          << " worker_tasks " << tasks.size()
                          << " pending " << pending.size() << std::endl;
                last_width = target_width;
                last_main_threads = main_threads_for_state;
            }

            Contest_record_state state = pending.back();
            pending.pop_back();
            thread_id_t main_thread_id = CONTEST_RECORD_THREAD_ID_BASE;
            thread_pool.set_max_thread_size(main_thread_id, std::max(0, main_threads_for_state - 1));
            Contest_record_expand_result result = contest_record_expand_state(
                state,
                options,
                max_loss_per_move,
                target_loss_sum,
                cut_empty,
                score_cache,
                &score_cache_mtx,
                main_threads_for_state > 1,
                main_thread_id
            );
            contest_record_process_expand_result(result, &pending, initial_board, n_records_limit, n_generated, ofs);
            progressed = true;
        }

        if (!progressed && !tasks.empty()) {
            contest_record_collect_ready_tasks(&tasks, &pending, initial_board, n_records_limit, n_generated, ofs, true);
        }
    }
}

std::string contest_record_play_one_game(
    Board board_start,
    const std::string &initial_board,
    Options *options,
    int max_loss_per_move,
    int max_loss_total,
    int cut_empty,
    int game_idx
) {
    Board board = board_start.copy();
    std::string transcript;
    int loss_sum = 0;
    Flip flip;
    while (HW2 - board.n_discs() > cut_empty && board.check_pass()) {
        std::vector<Contest_record_scored_move> scored_moves = contest_record_score_moves(board, options->level, true, THREAD_ID_NONE);
        if (scored_moves.empty() || scored_moves[0].value == SCORE_UNDEFINED) {
            break;
        }
        int best_value = scored_moves[0].value;
        std::vector<Contest_record_scored_move> candidates;
        for (const Contest_record_scored_move &move: scored_moves) {
            if (move.value == SCORE_UNDEFINED) {
                continue;
            }
            int loss = best_value - move.value;
            if (loss <= max_loss_per_move && loss_sum + loss <= max_loss_total) {
                candidates.emplace_back(move);
            }
        }
        if (candidates.empty()) {
            candidates.emplace_back(scored_moves[0]);
        }
        int selected_idx = myrandrange(0, (int)candidates.size());
        Contest_record_scored_move selected = candidates[selected_idx];
        loss_sum += best_value - selected.value;
        transcript += idx_to_coord(selected.policy);
        calc_flip(&flip, &board, selected.policy);
        board.move_board(&flip);
    }
    int leaf_value = contest_record_leaf_value(board, true, THREAD_ID_NONE);
    std::ostringstream oss;
    oss << "record: " << game_idx << '\n';
    oss << "initial board: " << initial_board << '\n';
    oss << "transcript: " << transcript << '\n';
    oss << "leaf board: " << board.to_str() << '\n';
    oss << "leaf empty: " << (HW2 - board.n_discs()) << '\n';
    oss << "leaf value: " << leaf_value << '\n';
    oss << "loss sum: " << loss_sum << '\n';
    oss << '\n';
    return oss.str();
}

void contest_record_commandline(std::vector<std::string> arg, Options *options) {
    if (arg.size() < 6) {
        std::cerr << "[ERROR] [FATAL] please input <board> <n> <dir> <per_move_loss> <total_loss> <cut_empty>" << std::endl;
        std::exit(1);
    }
    int n_games = 0;
    int max_loss_per_move = 0;
    int max_loss_total = 0;
    int cut_empty = 0;
    try {
        n_games = std::stoi(arg[1]);
        max_loss_per_move = std::stoi(arg[3]);
        max_loss_total = std::stoi(arg[4]);
        cut_empty = std::stoi(arg[5]);
    } catch (const std::exception&) {
        std::cerr << "[ERROR] invalid contest record argument" << std::endl;
        std::exit(1);
    }
    if (n_games <= 0 || max_loss_per_move < 0 || max_loss_total < 0 || cut_empty < 0 || HW2 <= cut_empty) {
        std::cerr << "[ERROR] contest record argument out of range" << std::endl;
        std::exit(1);
    }
    Board board_start;
    std::string initial_board;
    if (!contest_record_canonical_start(arg[0], &board_start, &initial_board)) {
        std::exit(1);
    }

    std::filesystem::path out_dir(arg[2]);
    std::filesystem::create_directories(out_dir);
    std::filesystem::path out_file = out_dir / (get_current_datetime_for_file() + "_" + std::to_string(tim()) + ".txt");
    std::ofstream ofs(out_file);
    if (!ofs) {
        std::cerr << "[ERROR] can't open contest record file " << out_file.string() << std::endl;
        std::exit(1);
    }
    uint64_t strt = tim();
    std::cerr << "contest record generation start games " << n_games
              << " level " << options->level
              << " per_move_loss " << max_loss_per_move
              << " total_loss " << max_loss_total
              << " cut_empty " << cut_empty
              << " output " << out_file.string() << std::endl;
    int n_generated = 0;
    Contest_record_score_cache score_cache;
    for (int target_loss_sum = 0; target_loss_sum <= max_loss_total && n_generated < n_games; ++target_loss_sum) {
        int n_before = n_generated;
        std::cerr << "contest record enumerate loss " << target_loss_sum << std::endl;
        contest_record_enumerate_exact_loss(
            board_start,
            initial_board,
            options,
            max_loss_per_move,
            target_loss_sum,
            cut_empty,
            n_games,
            &n_generated,
            ofs,
            &score_cache
        );
        std::cerr << "contest record loss " << target_loss_sum
                  << " generated " << (n_generated - n_before)
                  << " total " << n_generated << "/" << n_games << std::endl;
    }
    std::cerr << "contest record generation done " << n_generated << "/" << n_games << " in " << tim() - strt << " ms" << std::endl;
}

void self_play(std::vector<std::string> arg, Options *options, State *state) {
    int n_games, n_random_moves;
    if (arg.size() < 2) {
        std::cerr << "[ERROR] [FATAL] please input arguments" << std::endl;
        std::exit(1);
    }
    std::string str_n_games = arg[0];
    std::string str_n_random_moves = arg[1];
    try{
        n_games = std::stoi(str_n_games);
        n_random_moves = std::stoi(str_n_random_moves);
    } catch (const std::invalid_argument& e) {
        std::cout << str_n_games << " " << str_n_random_moves << " invalid argument" << std::endl;
        std::exit(1);
    } catch (const std::out_of_range& e) {
        std::cout << str_n_games << " " << str_n_random_moves << " out of range" << std::endl;
        std::exit(1);
    }
    std::cerr << n_games << " games with " << n_random_moves << " random moves" << std::endl;
    uint64_t strt = tim();
    Board board_start;
    board_start.reset();
    if (thread_pool.size() == 0) {
        for (int i = 0; i < n_games; ++i) {
            std::string transcript = self_play_task(board_start, "", options, false, n_random_moves, SELF_PLAY_N_TRY);
            std::cout << transcript << std::endl;
        }
    } else {
        int n_games_done = 0;
        std::vector<std::future<std::string>> tasks;
        while (n_games_done < n_games) {
            if (thread_pool.get_n_idle() && (int)tasks.size() < n_games) {
                bool pushed = false;
                tasks.emplace_back(thread_pool.push(&pushed, std::bind(&self_play_task, board_start, "", options, false, n_random_moves, SELF_PLAY_N_TRY)));
                if (!pushed) {
                    tasks.pop_back();
                }
            }
            for (std::future<std::string> &task: tasks) {
                if (task.valid()) {
                    if (task.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                        std::string transcript = task.get();
                        std::cout << transcript << std::endl;
                        ++n_games_done;
                    }
                }
            }
            if (0 < n_games - n_games_done && n_games - n_games_done < thread_pool.size()) {
                std::vector<std::string> transcripts_mid;
                global_searching = false;
                    for (std::future<std::string> &task: tasks) {
                        if (task.valid()) {
                            std::string transcript_mid = task.get();
                            transcripts_mid.emplace_back(transcript_mid);
                        }
                    }
                global_searching = true;
                for (std::string &transcript_mid: transcripts_mid) {
                    Board board_start_mid = board_start.copy();
                    Flip flip;
                    for (int i = 0; i < transcript_mid.size(); i += 2) {
                        int x = transcript_mid[i] - 'a';
                        int y = transcript_mid[i + 1] - '1';
                        int coord = HW2_M1 - (y * HW + x);
                        calc_flip(&flip, &board_start_mid, coord);
                        board_start_mid.move_board(&flip);
                        if (board_start_mid.get_legal() == 0) {
                            board_start_mid.pass();
                        }
                    }
                    int n_random_moves_additional = std::max(0, n_random_moves - (int)transcript_mid.size() / 2);
                    std::string transcript = self_play_task(board_start_mid, transcript_mid, options, true, n_random_moves_additional, SELF_PLAY_N_TRY);
                    std::cout << transcript << std::endl;
                    ++n_games_done;
                }
            }
        }
    }
    global_searching = false;
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
}

void self_play_line(std::vector<std::string> arg, Options *options, State *state) {
    if (arg.size() < 1) {
        std::cerr << "please input opening file" << std::endl;
        std::exit(1);
    }
    std::string opening_file = arg[0];
    std::cerr << "selfplay with opening file " << opening_file << std::endl;
    std::ifstream ifs(opening_file);
    if (!ifs) {
        std::cerr << "[ERROR] can't open file " << opening_file << std::endl;
        std::exit(1);
    }
    uint64_t strt = tim();
    std::string line;
    Board board_start;
    Flip flip;
    Search_result result;
    std::vector<std::pair<std::string, Board>> board_list;
    while (std::getline(ifs, line)) {
        board_start.reset();
        for (int i = 0; i < (int)line.size() - 1; i += 2) {
            int x = line[i] - 'a';
            int y = line[i + 1] - '1';
            int coord = HW2_M1 - (y * HW + x);
            calc_flip(&flip, &board_start, coord);
            board_start.move_board(&flip);
            if (board_start.get_legal() == 0) {
                board_start.pass();
            }
        }
        board_list.emplace_back(std::make_pair(line, board_start));
    }
    if (thread_pool.size() == 0) {
        for (std::pair<std::string, Board> start_position: board_list) {
            std::string transcript = self_play_task(start_position.second, start_position.first, options, false, 0, SELF_PLAY_N_TRY);
            std::cout << transcript << std::endl;
        }
    } else {
        int task_idx = 0;
        int print_idx = 0;
        int n_running_task = 0;
        std::vector<std::future<std::string>> tasks;
        std::vector<std::string> results;
        int n_games = board_list.size();
        while (print_idx < n_games) {
            // add task
            if (thread_pool.get_n_idle() && tasks.size() < n_games) {
                bool pushed = false;
                tasks.emplace_back(thread_pool.push(&pushed, std::bind(&self_play_task, board_list[task_idx].second, board_list[task_idx].first, options, false, 0, SELF_PLAY_N_TRY)));
                if (pushed) {
                    ++task_idx;
                    ++n_running_task;
                    results.emplace_back("");
                } else {
                    tasks.pop_back();
                }
            }
            // check if task ends
            for (int i = 0; i < tasks.size(); ++i) {
                if (tasks[i].valid()) {
                    if (tasks[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                        --n_running_task;
                        results[i] = tasks[i].get();
                    }
                }
            }
            // print result
            if (tasks.size() > print_idx) {
                for (int i = print_idx; i < results.size(); ++i) {
                    if (results[i] != "") {
                        std::cout << results[i] << std::endl;
                        ++print_idx;
                    } else {
                        break;
                    }
                }
            }
            // special case
            if (task_idx == n_games && 0 < n_running_task && n_running_task < thread_pool.size()) {
                std::vector<std::pair<int, std::string>> mid_tasks;
                global_searching = false;
                    for (int i = 0; i < tasks.size(); ++i) {
                        if (tasks[i].valid()) {
                            std::string transcript_mid = tasks[i].get();
                            //std::cerr << "mid task " << i << " " << transcript_mid << std::endl;
                            mid_tasks.emplace_back(std::make_pair(i, transcript_mid));
                        }
                    }
                global_searching = true;
                for (std::pair<int, std::string> &mid_task: mid_tasks) {
                    Board board_start_mid;
                    board_start_mid.reset();
                    Flip flip;
                    for (int i = 0; i < mid_task.second.size(); i += 2) {
                        int x = mid_task.second[i] - 'a';
                        int y = mid_task.second[i + 1] - '1';
                        int coord = HW2_M1 - (y * HW + x);
                        calc_flip(&flip, &board_start_mid, coord);
                        board_start_mid.move_board(&flip);
                        if (board_start_mid.get_legal() == 0) {
                            board_start_mid.pass();
                        }
                    }
                    //std::cerr << "additional " << mid_task.first << " " << mid_task.second << " " << board_start_mid.to_str() << std::endl;
                    std::string transcript = self_play_task(board_start_mid, mid_task.second, options, true, 0, SELF_PLAY_N_TRY);
                    //std::cerr << "additional got " << mid_task.first << " " << transcript << std::endl;
                    results[mid_task.first] = transcript;
                    --n_running_task;
                    for (int i = print_idx; i < results.size(); ++i) {
                        if (results[i] != "") {
                            std::cout << results[i] << std::endl;
                            ++print_idx;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
}


void self_play_board(std::vector<std::string> arg, Options *options, State *state) {
    if (arg.size() < 1) {
        std::cerr << "please input opening board file" << std::endl;
        std::exit(1);
    }
    std::string opening_board_file = arg[0];
    std::cerr << "selfplay with opening board file " << opening_board_file << std::endl;
    std::ifstream ifs(opening_board_file);
    if (!ifs) {
        std::cerr << "[ERROR] can't open file " << opening_board_file << std::endl;
        std::exit(1);
    }
    uint64_t strt = tim();
    std::string line;
    Flip flip;
    Search_result result;
    std::vector<std::pair<std::string, Board>> board_list;
    while (std::getline(ifs, line)) {
        std::pair<Board, int> board_player = convert_board_from_str(line);
        if (board_player.second != BLACK && board_player.second != WHITE) {
            std::cerr << "[ERROR] can't convert board " << line << std::endl;
            std::exit(1);
        }
        board_list.emplace_back(std::make_pair(line, board_player.first));
    }
    if (thread_pool.size() == 0) {
        for (std::pair<std::string, Board> start_position: board_list) {
            std::string transcript = self_play_task(start_position.second, "", options, false, 0, SELF_PLAY_N_TRY);
            std::cout << start_position.first << " " << transcript << std::endl;
        }
    } else {
        int print_task_idx = 0;
        std::vector<std::future<std::string>> tasks;
        for (std::pair<std::string, Board> start_position: board_list) {
            bool go_to_next_task = false;
            while (!go_to_next_task) {
                if (thread_pool.get_n_idle() && tasks.size() < board_list.size()) {
                    bool pushed = false;
                    tasks.emplace_back(thread_pool.push(&pushed, std::bind(&self_play_task, start_position.second, "", options, false, 0, SELF_PLAY_N_TRY)));
                    if (pushed) {
                        go_to_next_task = true;
                    } else {
                        tasks.pop_back();
                    }
                }
                if (tasks.size() > print_task_idx) {
                    if (tasks[print_task_idx].valid()) {
                        if (tasks[print_task_idx].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                            std::string transcript = tasks[print_task_idx].get();
                            std::cout << board_list[print_task_idx].first << " " << transcript << std::endl;
                            ++print_task_idx;
                        }
                    } else {
                        std::cerr << "[ERROR] task not valid" << std::endl;
                        std::exit(1);
                    }
                }
            }
        }
        while (print_task_idx < tasks.size()) {
            if (tasks[print_task_idx].valid()) {
                std::string transcript = tasks[print_task_idx].get();
                std::cout << board_list[print_task_idx].first << " " << transcript << std::endl;
                ++print_task_idx;
            } else {
                std::cerr << "[ERROR] task not valid" << std::endl;
                std::exit(1);
            }
        }
    }
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
}

void self_play_lossless_lines_task(Board board, const std::string starting_board, Options *options, const int to_n_discs, std::vector<int> &transcript) {
    if (board.n_discs() >= to_n_discs) {
        std::cout << starting_board << " ";
        for (int &cell: transcript) {
            std::cout << idx_to_coord(cell);
        }
        //std::cout << " " << board.to_str() << std::endl;
        //Search_result accurate_search_result = ai(board, 28, true, 0, true, true);
        //int accurate_val = accurate_search_result.value;
        //std::cout << " " << accurate_val << std::endl;
        std::cout << std::endl;
        return;
    }
    if (board.is_end()) {
        std::cout << starting_board << " ";
        for (int &cell: transcript) {
            std::cout << idx_to_coord(cell);
        }
        //std::cout << " " << board.to_str() << " END" << std::endl;
        //std::cout << " " << board.score_player() << std::endl;
        std::cout << std::endl;
        return;
    }
    uint64_t legal = board.get_legal();
    if (legal == 0) {
        board.pass();
        legal = board.get_legal();
    }
    Flip flip;
    Search_result search_result = ai(board, options->level, true, 0, true, false);
    calc_flip(&flip, &board, search_result.policy);
    board.move_board(&flip);
    transcript.emplace_back(search_result.policy);
        self_play_lossless_lines_task(board, starting_board, options, to_n_discs, transcript);
    transcript.pop_back();
    board.undo_board(&flip);
    legal ^= 1ULL << search_result.policy;
    int best_score = search_result.value;
    //int best_move = search_result.policy;
    int alpha = best_score - 2; // accept best - 1
    int beta = best_score;
    while (legal) {
        search_result = ai_legal_window(board, alpha, beta, options->level, true, 0, true, false, legal);
        if (search_result.value <= alpha) {
            break;
        }
        //std::cerr << board.to_str() << " best " << idx_to_coord(best_move) << " " << best_score << " alt " << idx_to_coord(search_result.policy) << " " << search_result.value << " [" << alpha << "," << beta << "]" << std::endl;
        calc_flip(&flip, &board, search_result.policy);
        board.move_board(&flip);
        transcript.emplace_back(search_result.policy);
            self_play_lossless_lines_task(board, starting_board, options, to_n_discs, transcript);
        transcript.pop_back();
        board.undo_board(&flip);
        legal ^= 1ULL << search_result.policy;
    }
}

void self_play_board_lossless_lines(std::vector<std::string> arg, Options *options, State *state) {
    if (arg.size() < 2) {
        std::cerr << "please input opening board file and to_n_discs" << std::endl;
        std::exit(1);
    }
    std::string opening_board_file = arg[0];
    int to_n_discs = 0;
    try{
        to_n_discs = std::stoi(arg[1]);
    } catch (const std::invalid_argument& e) {
        std::cout << arg[1] << " invalid argument" << std::endl;
        std::exit(1);
    } catch (const std::out_of_range& e) {
        std::cout << arg[1] << " out of range" << std::endl;
        std::exit(1);
    }
    if (to_n_discs > HW2) {
        to_n_discs = HW2;
    }
    std::cerr << "selfplay with opening board file " << opening_board_file << " to " << to_n_discs << " discs" << std::endl;
    std::ifstream ifs(opening_board_file);
    if (!ifs) {
        std::cerr << "[ERROR] can't open file " << opening_board_file << std::endl;
        std::exit(1);
    }
    uint64_t strt = tim();
    std::string line;
    Flip flip;
    Search_result result;
    std::vector<std::pair<std::string, Board>> board_list;
    while (std::getline(ifs, line)) {
        std::pair<Board, int> board_player = convert_board_from_str(line);
        if (board_player.second != BLACK && board_player.second != WHITE) {
            std::cerr << "[ERROR] can't convert board " << line << std::endl;
            std::exit(1);
        }
        board_list.emplace_back(std::make_pair(line, board_player.first));
    }
    int idx = 0;
    for (std::pair<std::string, Board> start_position: board_list) {
        ++idx;
        double percent = (double)idx / board_list.size() * 100.0;
        std::cerr << idx << "/" << board_list.size() << " " << percent << "%" << std::endl;
        std::vector<int> transcript;
        self_play_lossless_lines_task(start_position.second, start_position.first, options, to_n_discs, transcript);
    }
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
}



Board get_random_board(int n_random_moves) {
    Board board;
    Flip flip;
    for (;;) {
        board.reset();
        for (int j = 0; j < n_random_moves && board.check_pass(); ++j) {
            uint64_t legal = board.get_legal();
            int random_idx = myrandrange(0, pop_count_ull(legal));
            int t = 0;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                if (t == random_idx) {
                    calc_flip(&flip, &board, cell);
                    break;
                }
                ++t;
            }
            board.move_board(&flip);
        }
        if (board.check_pass()) {
            return board;
        }
    }
    return board; // error
}


void solve_random(std::vector<std::string> arg, Options *options, State *state) {
    int n_boards, n_random_moves;
    if (arg.size() < 2) {
        std::cerr << "[ERROR] [FATAL] please input arguments" << std::endl;
        std::exit(1);
    }
    std::string str_n_boards = arg[0];
    std::string str_n_random_moves = arg[1];
    try{
        n_boards = std::stoi(str_n_boards);
        n_random_moves = std::stoi(str_n_random_moves);
    } catch (const std::invalid_argument& e) {
        std::cout << str_n_boards << " " << str_n_random_moves << " invalid argument" << std::endl;
        std::exit(1);
    } catch (const std::out_of_range& e) {
        std::cout << str_n_boards << " " << str_n_random_moves << " out of range" << std::endl;
        std::exit(1);
    }
    std::cerr << n_boards << " boards with " << n_random_moves << " random moves" << std::endl;
    uint64_t strt = tim();
    if (thread_pool.size() == 0) {
        for (int i = 0; i < n_boards; ++i) {
            Board board = get_random_board(n_random_moves);
            Search_result result = ai(board, options->level, true, 0, false, options->show_log);
            std::cout << board.to_str().substr(0, 64) << " " << result.value << std::endl;
        }
    } else {
        int n_boards_done = 0;
        std::vector<std::pair<Board, std::future<Search_result>>> tasks;
        while (n_boards_done < n_boards) {
            if (thread_pool.get_n_idle() && (int)tasks.size() < n_boards) {
                bool pushed = false;
                Board board = get_random_board(n_random_moves);
                tasks.emplace_back(std::make_pair(board, thread_pool.push(&pushed, std::bind(&ai, board, options->level, true, 0, false, options->show_log))));
                if (!pushed) {
                    tasks.pop_back();
                }
            }
            for (std::pair<Board, std::future<Search_result>> &task: tasks) {
                if (task.second.valid()) {
                    if (task.second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                        Search_result result = task.second.get();
                        std::cout << task.first.to_str().substr(0, 64) << " " << result.value << std::endl;
                        ++n_boards_done;
                    }
                }
            }
            if (0 < n_boards - n_boards_done && n_boards - n_boards_done < thread_pool.size()) {
                std::vector<Board> boards_mid;
                global_searching = false;
                    for (std::pair<Board, std::future<Search_result>> &task: tasks) {
                        if (task.second.valid()) {
                            task.second.get();
                            boards_mid.emplace_back(task.first);
                        }
                    }
                global_searching = true;
                for (Board &board: boards_mid) {
                    Search_result result = ai(board, options->level, true, 0, true, options->show_log);
                    std::cout << board.to_str().substr(0, 64) << " " << result.value << std::endl;
                    ++n_boards_done;
                }
            }
        }
    }
    global_searching = false;
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
}


void perft_commandline(std::vector<std::string> arg) {
    if (arg.size() < 2) {
        std::cerr << "please input <depth> <mode>" << std::endl;
        std::exit(1);
    }
    int depth, mode;
    std::string str_depth = arg[0];
    std::string str_mode = arg[1];
    try{
        depth = std::stoi(str_depth);
        mode = std::stoi(str_mode);
    } catch (const std::invalid_argument& e) {
        std::cout << str_depth << " " << str_mode << " invalid argument" << std::endl;
        std::exit(1);
    } catch (const std::out_of_range& e) {
        std::cout << str_depth << " " << str_mode << " out of range" << std::endl;
        std::exit(1);
    }
    if (mode != 1 && mode != 2) {
        std::cout << "mode must be 1 or 2, got " << mode << std::endl;
        std::exit(1);
    }
    if (depth <= 0 || 60 < depth) {
        std::cout << "depth must be in [1, 60], got " << depth << std::endl;
        std::exit(1);
    }
    Board board;
    board.reset();
    uint64_t strt = tim();
    uint64_t res;
    if (mode == 1) {
        res = perft(&board, depth, false);
    } else {
        res = perft_no_pass_count(&board, depth, false);
    }
    std::cout << "perft mode " << mode << " depth " << depth << " " << res << " leaves found in " << tim() - strt << " ms" << std::endl;
}

void minimax_commandline(std::vector<std::string> arg) {
    if (arg.size() < 1) {
        std::cerr << "please input <depth>" << std::endl;
        std::exit(1);
    }
    int depth;
    std::string str_depth = arg[0];
    try{
        depth = std::stoi(str_depth);
    } catch (const std::invalid_argument& e) {
        std::cout << str_depth << " invalid argument" << std::endl;
        std::exit(1);
    } catch (const std::out_of_range& e) {
        std::cout << str_depth << " out of range" << std::endl;
        std::exit(1);
    }
    if (depth <= 0 || 60 < depth) {
        std::cout << "depth must be in [1, 60], got " << depth << std::endl;
        std::exit(1);
    }
    Board board;
    board.reset();
    uint64_t strt = tim();
    int res = minimax(&board, depth);
    std::cout << "minimax depth " << depth << " value " << res << " in " << tim() - strt << " ms" << std::endl;
}
