/*
    Egaroucid Project

    @file ggs.hpp
        Telnet client for Generic Game Server https://skatgame.net/mburo/ggs/
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/
#pragma once
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include "./../engine/engine_all.hpp"
#include "option.hpp"
#include "util.hpp"
#pragma comment(lib, "ws2_32.lib")

#define GGS_URL "skatgame.net"
#define GGS_PORT 5000
#define GGS_READY "READY"
#define GGS_REPLY_HEADER "GGS RECV> "
#define GGS_SEND_HEADER "GGS SEND> "
#define GGS_INFO_HEADER "GGS INFO> "
#define GGS_RECEIVE_BUFFER_SIZE 20000
#define GGS_CONNECTION_CLOSED_TOKEN "__EGAROUCID_GGS_CONNECTION_CLOSED__"

#define TELNET_SE 240
#define TELNET_SB 250
#define TELNET_WILL 251
#define TELNET_WONT 252
#define TELNET_DO 253
#define TELNET_DONT 254
#define TELNET_IAC 255
#define TELNET_ECHO 1
#define TELNET_SUPPRESS_GO_AHEAD 3

#define GGS_NON_SYNCHRO_ID 0

#define GGS_SEND_EMPTY_INTERVAL 180000ULL // 3 minutes

#define GGS_USE_PONDER true
#define GGS_N_PONDER_PARALLEL 1

std::atomic_bool ggs_socket_closing(false);
#if IS_GGS_TOURNAMENT
    #ifndef GGS_TOURNAMENT_ENABLE_SYNCHRO_HINT_SEARCH_DELAY
        #define GGS_TOURNAMENT_ENABLE_SYNCHRO_HINT_SEARCH_DELAY true
    #endif
    #ifndef GGS_TOURNAMENT_EARLY_SYNCHRO_HINT_MOVE_WAIT_MAX_DISCS
        #define GGS_TOURNAMENT_EARLY_SYNCHRO_HINT_MOVE_WAIT_MAX_DISCS 30
    #endif
    #ifndef GGS_TOURNAMENT_NON_PRIORITIZED_THREADS
        #define GGS_TOURNAMENT_NON_PRIORITIZED_THREADS 0
    #endif
    #ifndef GGS_TOURNAMENT_NON_PRIORITIZED_MIN_FULL_THREADS
        #define GGS_TOURNAMENT_NON_PRIORITIZED_MIN_FULL_THREADS 6
    #endif
    #ifndef GGS_TOURNAMENT_TERMINATE_ALL_PONDERS_MAX_DISCS
        #define GGS_TOURNAMENT_TERMINATE_ALL_PONDERS_MAX_DISCS 36
    #endif
    #ifndef GGS_TOURNAMENT_CLEAR_TT_ON_MATCH_END
        #define GGS_TOURNAMENT_CLEAR_TT_ON_MATCH_END true
    #endif
constexpr bool GGS_ENABLE_SYNCHRO_HINT_SEARCH_DELAY = GGS_TOURNAMENT_ENABLE_SYNCHRO_HINT_SEARCH_DELAY;
constexpr int GGS_EARLY_SYNCHRO_HINT_MOVE_WAIT_MAX_DISCS = GGS_TOURNAMENT_EARLY_SYNCHRO_HINT_MOVE_WAIT_MAX_DISCS;
constexpr int GGS_NON_PRIORITIZED_THREADS = GGS_TOURNAMENT_NON_PRIORITIZED_THREADS;
constexpr int GGS_NON_PRIORITIZED_MIN_FULL_THREADS = GGS_TOURNAMENT_NON_PRIORITIZED_MIN_FULL_THREADS;
constexpr int GGS_TERMINATE_ALL_PONDERS_MAX_DISCS = GGS_TOURNAMENT_TERMINATE_ALL_PONDERS_MAX_DISCS;
constexpr bool GGS_CLEAR_TT_ON_MATCH_END = GGS_TOURNAMENT_CLEAR_TT_ON_MATCH_END;
#else
constexpr bool GGS_ENABLE_SYNCHRO_HINT_SEARCH_DELAY = true;
constexpr int GGS_EARLY_SYNCHRO_HINT_MOVE_WAIT_MAX_DISCS = 0;
constexpr int GGS_NON_PRIORITIZED_THREADS = 1;
constexpr int GGS_NON_PRIORITIZED_MIN_FULL_THREADS = 12;
constexpr int GGS_TERMINATE_ALL_PONDERS_MAX_DISCS = 0;
constexpr bool GGS_CLEAR_TT_ON_MATCH_END = true;
#endif

struct GGS_Clock_Params {
    bool initialized;
    uint64_t initial_msec;
    uint64_t increment_msec;
    uint64_t extension_msec;
    int initial_moves;
    int increment_moves;
    int extension_moves;
    bool initial_loss;
    bool increment_add;
    bool extension_add;

    GGS_Clock_Params()
        : initialized(false),
          initial_msec(0),
          increment_msec(0),
          extension_msec(0),
          initial_moves(0),
          increment_moves(1),
          extension_moves(0),
          initial_loss(true),
          increment_add(true),
          extension_add(true) {}
};

inline uint64_t ggs_time_safety_margin_msec(uint64_t remaining_time_msec) {
    if (remaining_time_msec > 120000ULL) {
        return 6000ULL;
    }
    if (remaining_time_msec > 60000ULL) {
        return 5000ULL;
    }
    if (remaining_time_msec > 30000ULL) {
        return 3500ULL;
    }
    if (remaining_time_msec > 10000ULL) {
        return 2500ULL;
    }
    if (remaining_time_msec > 3000ULL) {
        return 1500ULL;
    }
    if (remaining_time_msec > 1000ULL) {
        return 1000ULL;
    }
    return 1000ULL;
}

inline uint64_t ggs_clock_future_increment_msec(const Board &board, const GGS_Clock_Params &clock) {
    if (!clock.initialized || clock.increment_msec == 0ULL || !clock.increment_add || clock.increment_moves <= 0) {
        return 0ULL;
    }
    const int n_empties = HW2 - board.n_discs();
    const uint64_t remaining_own_moves = std::max(1, (n_empties + 1) / 2);
    return clock.increment_msec * remaining_own_moves / (uint64_t)clock.increment_moves;
}

inline uint64_t ggs_clock_adjusted_time_for_allocation(const Board &board, uint64_t safe_remaining_time_msec, uint64_t raw_remaining_time_msec, const GGS_Clock_Params &clock) {
    const uint64_t future_increment = ggs_clock_future_increment_msec(board, clock);
    if (future_increment == 0ULL) {
        return safe_remaining_time_msec;
    }
    const uint64_t activation_time = clock.increment_msec * 2ULL + ggs_time_safety_margin_msec(raw_remaining_time_msec);
    if (raw_remaining_time_msec <= activation_time) {
        return safe_remaining_time_msec;
    }
    return safe_remaining_time_msec + std::min<uint64_t>(future_increment, safe_remaining_time_msec * 2ULL);
}

inline uint64_t ggs_subtract_elapsed_or_floor(uint64_t remaining_time_msec, uint64_t elapsed_msec) {
    if (remaining_time_msec > elapsed_msec + 1ULL) {
        return remaining_time_msec - elapsed_msec;
    }
    return 1ULL;
}

inline bool ggs_engine_show_log(const Options *options) {
#if IS_GGS_TOURNAMENT
    return options->show_log || options->log_to_file;
#else
    return options->show_log;
#endif
}

Search_result ggs_fallback_search_result(Board board) {
    Search_result res;
    uint64_t legal = board.get_legal();
    if (legal == 0ULL) {
        res.policy = MOVE_PASS;
        res.value = 0;
        return res;
    }
    uint64_t preferred = legal & 0x8100000000000081ULL;
    uint64_t candidates = preferred ? preferred : legal;
    res.policy = first_bit(&candidates);
    res.value = mid_evaluate(&board);
    res.depth = 0;
    res.probability = 0;
    return res;
}

struct GGS_Board {
    std::string game_id;
    bool is_synchro;
    int synchro_id;
    int last_move;
    std::string player_black;
    uint64_t remaining_seconds_black;
    std::string player_white;
    uint64_t remaining_seconds_white;
    Board board;
    int player_to_move;
    GGS_Clock_Params clock;

    GGS_Board() {
        game_id = "";
        is_synchro = false;
        synchro_id = -1;
        last_move = -1;
        player_black = "";
        remaining_seconds_black = 0;
        player_white = "";
        remaining_seconds_white = 0;
        board.player = 0;
        board.opponent = 0;
        player_to_move = -1;
    }

    bool is_valid() const {
        return (board.player != 0 || board.opponent != 0) && player_to_move != -1;
    }
};

struct GGS_Match {
    std::string game_id;
    std::string initial_board;
    std::string transcript;
    int result_black;
    std::string player_black;
    std::string player_white;
    
    void init() {
        game_id = "";
    }

    bool is_initialized() {
        return game_id == "";
    }
};

void ggs_print_info(std::string str, Options *options);
void ggs_print_debug(std::string str, Options *options);

struct GGS_Move_Hint {
    int policy;
    int count;

    GGS_Move_Hint()
        : policy(MOVE_UNDEFINED), count(0) {}

    GGS_Move_Hint(int policy_, int count_)
        : policy(policy_), count(count_) {}
};

struct GGS_Pending_Move {
    bool active;
    GGS_Board board;
    Search_result result;
    uint64_t ready_time;
    uint64_t max_wait_msec;

    GGS_Pending_Move()
        : active(false), ready_time(0), max_wait_msec(0) {}
};

struct GGS_Pending_Search {
    bool active;
    GGS_Board board;
    int search_slot;
    thread_id_t thread_id;
    uint64_t ready_time;
    uint64_t max_wait_msec;

    GGS_Pending_Search()
        : active(false), search_slot(GGS_NON_SYNCHRO_ID), thread_id(THREAD_ID_NONE), ready_time(0), max_wait_msec(0) {}
};

struct GGS_Synchro_Search_Record {
    bool valid;
    std::string match_id;
    std::string game_id;
    int n_discs;
    Search_result result;
    uint64_t updated_time;

    GGS_Synchro_Search_Record()
        : valid(false), n_discs(0), updated_time(0) {}

    void clear() {
        valid = false;
        match_id.clear();
        game_id.clear();
        n_discs = 0;
        result = Search_result();
        updated_time = 0;
    }
};

struct GGS_Synchro_Time_Context {
    bool has_pair_result;
    std::string pair_game_id;
    int pair_n_discs;
    int pair_value;
    int pair_depth;
    int pair_probability;
    bool pair_is_end_search;

    GGS_Synchro_Time_Context()
        : has_pair_result(false),
          pair_n_discs(0),
          pair_value(0),
          pair_depth(0),
          pair_probability(0),
          pair_is_end_search(false) {}
};

using GGS_Move_Hint_Table = std::unordered_map<std::string, GGS_Move_Hint>;
using GGS_Ponder_Result_Table = std::unordered_map<std::string, Search_result>;

constexpr int GGS_GAME_LOG_WINNER_HINT_WEIGHT = 2;
constexpr int GGS_LIVE_OPPONENT_HINT_WEIGHT = 3;
constexpr int GGS_VERIFIED_ANALYSIS_HINT_WEIGHT = 9;
#if IS_GGS_TOURNAMENT
constexpr size_t GGS_TOURNAMENT_GAME_LOG_HINT_MAX_FILES = 256;
constexpr int GGS_LIVE_HINT_WAIT_MAX_DISCS = 30;
constexpr int GGS_SYNCHRO_PAIR_CONTEXT_MAX_DISC_GAP = 12;
constexpr int GGS_SYNCHRO_PAIR_TIME_BOOST_MAX_DISC = 36;
constexpr uint64_t GGS_SYNCHRO_PAIR_TIME_BOOST_MIN_RAW_REMAINING = 45000ULL;
#else
constexpr int GGS_LIVE_HINT_WAIT_MAX_DISCS = 28;
#endif

inline std::string ggs_move_hint_key(const Board &board) {
    return board.to_str();
}

inline std::string ggs_synchro_match_id(const std::string &game_id) {
    size_t last_dot = game_id.rfind('.');
    if (last_dot == std::string::npos) {
        return game_id;
    }
    return game_id.substr(0, last_dot);
}

inline bool ggs_is_legal_hint(const Board &board, int policy) {
    return is_valid_policy(policy) && (board.get_legal() & (1ULL << policy));
}

inline std::string ggs_policy_to_text(int policy) {
    if (policy == MOVE_PASS) {
        return "pa";
    }
    if (is_valid_policy(policy)) {
        return idx_to_coord(policy);
    }
    return "undefined";
}

inline bool ggs_is_usable_ponder_result(const Board &board, const Search_result &result) {
    if (
        !result.is_end_search ||
        result.depth < HW2 - board.n_discs() ||
        !is_valid_policy(result.policy) ||
        !(board.get_legal() & (1ULL << result.policy))
    ) {
        return false;
    }
    return result.probability == 100;
}

inline bool ggs_is_cacheable_ponder_result(const Search_result &result) {
    return result.is_end_search && result.probability == 100 && is_valid_policy(result.policy);
}

Search_result ggs_get_ponder_result(const GGS_Ponder_Result_Table &ponder_results, const Board &board) {
    auto it = ponder_results.find(board.to_str());
    if (it == ponder_results.end() || !ggs_is_usable_ponder_result(board, it->second)) {
        return Search_result();
    }
    return it->second;
}

Search_result ggs_hint_search_result(Board board, int policy) {
    Search_result res = ggs_fallback_search_result(board);
    res.policy = policy;
    res.value = mid_evaluate(&board);
    res.depth = 0;
    res.probability = 0;
    return res;
}

inline bool ggs_should_play_hint_without_search(const Board &board, int policy, int hint_count) {
    if (!ggs_is_legal_hint(board, policy)) {
        return false;
    }
#if IS_GGS_TOURNAMENT
    return board.n_discs() <= 20 && hint_count >= GGS_LIVE_OPPONENT_HINT_WEIGHT;
#else
    return board.n_discs() <= 20;
#endif
}

inline bool ggs_should_override_with_hint(const Board &board, int policy, int hint_count, const Search_result &search_result) {
    if (!ggs_is_legal_hint(board, policy) || policy == search_result.policy) {
        return false;
    }
#if IS_GGS_TOURNAMENT
    if (hint_count < GGS_GAME_LOG_WINNER_HINT_WEIGHT) {
        return false;
    }
#else
    (void)hint_count;
#endif
    const int n_discs = board.n_discs();
    if (!is_valid_policy(search_result.policy)) {
        return true;
    }
#if IS_GGS_TOURNAMENT
    const bool verified_analysis_hint = hint_count >= GGS_VERIFIED_ANALYSIS_HINT_WEIGHT;
    if (
        verified_analysis_hint &&
        n_discs <= 36 &&
        (!search_result.is_end_search || search_result.probability < 100)
    ) {
        return true;
    }
    const bool live_opponent_hint = hint_count >= GGS_LIVE_OPPONENT_HINT_WEIGHT;
    if (live_opponent_hint) {
        if (n_discs <= 24 && (search_result.value <= 8 || search_result.depth < 34 || search_result.probability < 93)) {
            return true;
        }
        if (n_discs <= 30 && (search_result.value <= 2 || search_result.depth < 32 || search_result.probability < 88)) {
            return true;
        }
        if (n_discs <= 32 && (search_result.value <= -2 || search_result.depth < 30)) {
            return true;
        }
    }
    if (n_discs <= 22 && (search_result.value <= -2 || search_result.depth < 29 || search_result.probability < 88)) {
        return true;
    }
    if (n_discs <= 30 && (search_result.value <= -6 || search_result.depth < 27)) {
        return true;
    }
    return false;
#else
    if (n_discs <= 24 && search_result.probability < 93) {
        return true;
    }
    if (n_discs <= 30 && (search_result.value <= -4 || search_result.depth < 27)) {
        return true;
    }
    return false;
#endif
}

int ggs_get_move_hint(const GGS_Move_Hint_Table &move_hints, const Board &board) {
    auto it = move_hints.find(ggs_move_hint_key(board));
    if (it == move_hints.end()) {
        return MOVE_UNDEFINED;
    }
    return ggs_is_legal_hint(board, it->second.policy) ? it->second.policy : MOVE_UNDEFINED;
}

int ggs_get_move_hint_count(const GGS_Move_Hint_Table &move_hints, const Board &board) {
    auto it = move_hints.find(ggs_move_hint_key(board));
    if (it == move_hints.end() || !ggs_is_legal_hint(board, it->second.policy)) {
        return 0;
    }
    return it->second.count;
}

std::string ggs_search_result_summary(
    const std::string &prefix,
    const GGS_Board &ggs_board,
    const Search_result &search_result,
    int hint_policy,
    int hint_count
) {
    std::string msg =
        prefix +
        " game " + ggs_board.game_id +
        " discs " + std::to_string(ggs_board.board.n_discs()) +
        " move " + ggs_policy_to_text(search_result.policy) +
        " value " + std::to_string(search_result.value) +
        " depth " + std::to_string(search_result.depth) +
        "@" + std::to_string(search_result.probability) + "%" +
        " time " + std::to_string(search_result.time) +
        " nodes " + std::to_string(search_result.nodes) +
        " nps " + std::to_string(search_result.nps) +
        " end " + std::to_string(search_result.is_end_search ? 1 : 0);
    if (ggs_is_legal_hint(ggs_board.board, hint_policy)) {
        msg += " hint " + ggs_policy_to_text(hint_policy) + "x" + std::to_string(hint_count);
    }
    msg += " board " + ggs_board.board.to_str();
    return msg;
}

inline void ggs_log_search_result_summary(
    const std::string &prefix,
    const GGS_Board &ggs_board,
    const Search_result &search_result,
    int hint_policy,
    int hint_count,
    Options *options
) {
#if IS_GGS_TOURNAMENT
    ggs_print_info(ggs_search_result_summary(prefix, ggs_board, search_result, hint_policy, hint_count), options);
#else
    if (options->show_log) {
        ggs_print_info(ggs_search_result_summary(prefix, ggs_board, search_result, hint_policy, hint_count), options);
    }
#endif
}

void ggs_store_synchro_search_record(
    GGS_Synchro_Search_Record records[],
    const GGS_Board &ggs_board,
    const Search_result &search_result
) {
    if (!ggs_board.is_synchro || ggs_board.synchro_id < 0 || ggs_board.synchro_id >= 2) {
        return;
    }
    if (!is_valid_policy(search_result.policy) && search_result.policy != MOVE_PASS) {
        return;
    }
    GGS_Synchro_Search_Record &record = records[ggs_board.synchro_id];
    record.valid = true;
    record.match_id = ggs_synchro_match_id(ggs_board.game_id);
    record.game_id = ggs_board.game_id;
    record.n_discs = ggs_board.board.n_discs();
    record.result = search_result;
    record.updated_time = tim();
}

GGS_Synchro_Time_Context ggs_make_synchro_time_context(
    const GGS_Board &ggs_board,
    const GGS_Synchro_Search_Record records[]
) {
    GGS_Synchro_Time_Context context;
    if (!ggs_board.is_synchro || ggs_board.synchro_id < 0 || ggs_board.synchro_id >= 2) {
        return context;
    }
    const GGS_Synchro_Search_Record &pair_record = records[ggs_board.synchro_id ^ 1];
    if (!pair_record.valid || pair_record.match_id != ggs_synchro_match_id(ggs_board.game_id)) {
        return context;
    }
#if IS_GGS_TOURNAMENT
    const int disc_gap = ggs_board.board.n_discs() > pair_record.n_discs ?
        ggs_board.board.n_discs() - pair_record.n_discs :
        pair_record.n_discs - ggs_board.board.n_discs();
    if (disc_gap > GGS_SYNCHRO_PAIR_CONTEXT_MAX_DISC_GAP) {
        return context;
    }
#endif
    if (pair_record.result.depth <= 0 && pair_record.result.probability <= 0 && !pair_record.result.is_end_search) {
        return context;
    }
    context.has_pair_result = true;
    context.pair_game_id = pair_record.game_id;
    context.pair_n_discs = pair_record.n_discs;
    context.pair_value = pair_record.result.value;
    context.pair_depth = pair_record.result.depth;
    context.pair_probability = pair_record.result.probability;
    context.pair_is_end_search = pair_record.result.is_end_search;
    return context;
}

uint64_t ggs_adjust_remaining_time_for_synchro_pair(
    const GGS_Board &ggs_board,
    uint64_t remaining_time_msec,
    uint64_t raw_remaining_time_msec,
    const GGS_Synchro_Time_Context &context,
    Options *options
) {
#if IS_GGS_TOURNAMENT
    if (
        !context.has_pair_result ||
        ggs_board.board.n_discs() > GGS_SYNCHRO_PAIR_TIME_BOOST_MAX_DISC ||
        raw_remaining_time_msec < GGS_SYNCHRO_PAIR_TIME_BOOST_MIN_RAW_REMAINING
    ) {
        return remaining_time_msec;
    }

    double bonus_coe = 0.0;
    uint64_t bonus_cap = 0ULL;
    std::string reason;
    if (context.pair_value <= -8) {
        bonus_coe = 0.45;
        bonus_cap = 70000ULL;
        reason = "large_deficit";
    } else if (context.pair_value <= -5) {
        bonus_coe = 0.35;
        bonus_cap = 55000ULL;
        reason = "deficit";
    } else if (context.pair_value >= 24) {
        bonus_coe = 0.28;
        bonus_cap = 45000ULL;
        reason = "large_surplus_guard";
    } else if (context.pair_value >= 13 && ggs_board.board.n_discs() <= 34) {
        bonus_coe = 0.18;
        bonus_cap = 30000ULL;
        reason = "surplus_guard";
    } else if (context.pair_value <= 4) {
        bonus_coe = 0.22;
        bonus_cap = 35000ULL;
        reason = "thin_margin";
    } else if (context.pair_value <= 12 && ggs_board.board.n_discs() <= 28) {
        bonus_coe = 0.10;
        bonus_cap = 18000ULL;
        reason = "protect_surplus";
    } else {
        return remaining_time_msec;
    }

    uint64_t bonus = std::min<uint64_t>(bonus_cap, (uint64_t)((double)raw_remaining_time_msec * bonus_coe));
    if (bonus == 0ULL) {
        return remaining_time_msec;
    }
    uint64_t adjusted_remaining_time_msec = remaining_time_msec + bonus;
    ggs_print_info(
        "synchro pair time boost game " + ggs_board.game_id +
        " pair " + context.pair_game_id +
        " pair_discs " + std::to_string(context.pair_n_discs) +
        " pair_value " + std::to_string(context.pair_value) +
        " pair_depth " + std::to_string(context.pair_depth) +
        "@" + std::to_string(context.pair_probability) + "%" +
        " reason " + reason +
        " bonus " + std::to_string(bonus) +
        " limit " + std::to_string(remaining_time_msec) +
        "->" + std::to_string(adjusted_remaining_time_msec),
        options
    );
    return adjusted_remaining_time_msec;
#else
    (void)ggs_board;
    (void)raw_remaining_time_msec;
    (void)context;
    (void)options;
    return remaining_time_msec;
#endif
}

void ggs_apply_move_hint_to_search_result(
    const GGS_Board &ggs_board,
    const GGS_Move_Hint_Table &move_hints,
    Search_result *search_result,
    Options *options
) {
    const int hint_policy = ggs_get_move_hint(move_hints, ggs_board.board);
    const int hint_count = ggs_get_move_hint_count(move_hints, ggs_board.board);
    if (
        hint_policy == search_result->policy ||
        (!ggs_should_play_hint_without_search(ggs_board.board, hint_policy, hint_count) && !ggs_should_override_with_hint(ggs_board.board, hint_policy, hint_count, *search_result))
    ) {
        return;
    }
#if IS_GGS_TOURNAMENT
    ggs_print_info(
        "search hint override game " + ggs_board.game_id +
        " discs " + std::to_string(ggs_board.board.n_discs()) +
        " from " + ggs_policy_to_text(search_result->policy) +
        " to " + ggs_policy_to_text(hint_policy) +
        " value " + std::to_string(search_result->value) +
        " depth " + std::to_string(search_result->depth) +
        "@" + std::to_string(search_result->probability) + "%" +
        " hintx" + std::to_string(hint_count),
        options
    );
#else
    if (options->show_log) {
        std::cerr << "ggs late synchro hint overrides search " << idx_to_coord(search_result->policy) << " -> " << idx_to_coord(hint_policy)
                  << " value " << search_result->value << " depth " << search_result->depth << "@" << search_result->probability << "%"
                  << " " << ggs_board.board.to_str() << std::endl;
    }
#endif
    *search_result = ggs_hint_search_result(ggs_board.board, hint_policy);
}

void ggs_record_move_hint(GGS_Move_Hint_Table *move_hints, const Board &board, int policy, int weight = 1) {
    if (!ggs_is_legal_hint(board, policy)) {
        return;
    }
    if (weight <= 0) {
        return;
    }
    std::string key = ggs_move_hint_key(board);
    auto it = move_hints->find(key);
    if (it == move_hints->end()) {
        (*move_hints)[key] = GGS_Move_Hint(policy, weight);
        return;
    }
    if (it->second.policy == policy) {
        it->second.count += weight;
    } else if (it->second.count < weight) {
        it->second = GGS_Move_Hint(policy, weight - it->second.count);
    } else if (it->second.count == weight) {
        move_hints->erase(it);
    } else {
        it->second.count -= weight;
    }
    if (move_hints->size() > 65536) {
        move_hints->clear();
    }
}

int ggs_seed_verified_analysis_hints(GGS_Move_Hint_Table *move_hints, Options *options) {
#if IS_GGS_TOURNAMENT
    struct Verified_Analysis_Hint {
        const char *board;
        const char *move;
    };
    constexpr Verified_Analysis_Hint hints[] = {
        {
            "------------O---OXXOOOX-OOOOOXO-OOOXXOO-OOXXOO--O-XXX-----O----- X",
            "b7"
        },
        {
            "----------O-O-----OOOOOO-OOOOXX--OOXXXXX-XXXXXX---OX------------ X",
            "a5"
        },
        {
            "-----------------XO-OO----XXOX----OXOX----O-OO----O------------- X",
            "d3"
        },
        {
            "----------O-------O-XO----OXO----XXXXO----XXOX------------------ X",
            "g5"
        }
    };

    int n_registered = 0;
    for (const Verified_Analysis_Hint &hint: hints) {
        std::pair<Board, int> board_player = convert_board_from_str(hint.board);
        if (board_player.second == -1) {
            continue;
        }
        Board board = board_player.first;
        const int policy = get_coord_from_chars(hint.move[0], hint.move[1]);
        const int before_count = ggs_get_move_hint_count(*move_hints, board);
        ggs_record_move_hint(move_hints, board, policy, GGS_VERIFIED_ANALYSIS_HINT_WEIGHT);
        const int after_count = ggs_get_move_hint_count(*move_hints, board);
        if (after_count > before_count) {
            n_registered += after_count - before_count;
        }
    }
    if (n_registered > 0 && options->show_log) {
        ggs_print_info("seeded verified analysis hints moves " + std::to_string(n_registered) + " boards " + std::to_string(move_hints->size()), options);
    }
    return n_registered;
#else
    (void)move_hints;
    (void)options;
    return 0;
#endif
}

void ggs_record_opponent_move_hint(
    GGS_Move_Hint_Table *move_hints,
    GGS_Board ggs_boards[][HW2 + 1],
    const GGS_Board &ggs_board,
    Options *options
) {
    if (!ggs_board.is_synchro || ggs_board.synchro_id < 0 || ggs_board.synchro_id >= 2 || !is_valid_policy(ggs_board.last_move)) {
        return;
    }
    const int last_player = ggs_board.player_to_move == BLACK ? WHITE : BLACK;
    const std::string last_player_name = last_player == BLACK ? ggs_board.player_black : ggs_board.player_white;
    if (last_player_name == options->ggs_username) {
        return;
    }

    const int n_discs = ggs_board.board.n_discs();
    if (n_discs <= 0) {
        return;
    }
    GGS_Board &previous = ggs_boards[ggs_board.synchro_id][n_discs - 1];
    if (!previous.is_valid() || !ggs_is_legal_hint(previous.board, ggs_board.last_move)) {
        return;
    }
    Board next = previous.board.copy();
    Flip flip;
    calc_flip(&flip, &next, ggs_board.last_move);
    next.move_board(&flip);
    if (next != ggs_board.board) {
        return;
    }

    ggs_record_move_hint(move_hints, previous.board, ggs_board.last_move, GGS_LIVE_OPPONENT_HINT_WEIGHT);
    ggs_print_debug("synchro hint " + previous.board.to_str() + " -> " + idx_to_coord(ggs_board.last_move), options);
}

inline bool ggs_line_starts_with(const std::string &line, const std::string &prefix) {
    return line.rfind(prefix, 0) == 0;
}

inline std::string ggs_line_value(const std::string &line, const std::string &prefix) {
    return ggs_line_starts_with(line, prefix) ? line.substr(prefix.size()) : "";
}

inline void ggs_write_log_lines(const std::string &prefix, const std::string &str, Options *options) {
    if (!options->ggs_log_to_file) {
        return;
    }
    std::ofstream ofs(options->ggs_log_file, std::ios::app);
    if (!ofs) {
        return;
    }
    std::stringstream ss(str);
    std::string line;
    while (std::getline(ss, line, '\n')) {
        ofs << prefix << line << std::endl;
    }
}

inline bool ggs_verbose_log(const Options *options) {
    return options->show_log;
}

inline void ggs_print_colored_lines(const std::string &prefix, const std::string &str, Options *options, const std::string &color) {
    std::stringstream ss(str);
    std::string line;
    std::ofstream ofs;
    if (options->ggs_log_to_file) {
        ofs.open(options->ggs_log_file, std::ios::app);
    }
    std::cout << color;
    while (std::getline(ss, line, '\n')) {
        std::cout << prefix << line << std::endl;
        if (options->ggs_log_to_file) {
            ofs << prefix << line << std::endl;
        }
    }
    std::cout << "\033[0m";
    if (options->ggs_log_to_file) {
        ofs.close();
    }
}

int ggs_seed_move_hints_from_game_log_file(GGS_Move_Hint_Table *move_hints, const std::filesystem::path &path, Options *options) {
    std::ifstream ifs(path);
    if (!ifs) {
        return 0;
    }

    std::string line;
    std::string black;
    std::string white;
    std::string initial_board;
    std::string transcript;
    int black_score = SCORE_UNDEFINED;
    while (std::getline(ifs, line)) {
        if (ggs_line_starts_with(line, "black: ")) {
            black = ggs_line_value(line, "black: ");
        } else if (ggs_line_starts_with(line, "white: ")) {
            white = ggs_line_value(line, "white: ");
        } else if (ggs_line_starts_with(line, "initial board: ")) {
            initial_board = ggs_line_value(line, "initial board: ");
        } else if (ggs_line_starts_with(line, "transcript: ")) {
            transcript = ggs_line_value(line, "transcript: ");
        } else if (ggs_line_starts_with(line, "black's score: ")) {
            try {
                black_score = std::stoi(ggs_line_value(line, "black's score: "));
            } catch (const std::exception&) {
                black_score = SCORE_UNDEFINED;
            }
        }
    }

    if (
        black_score == -99 ||
        black_score == SCORE_UNDEFINED ||
        initial_board.size() < HW2 + 2 ||
        transcript.size() < 2 ||
        (black != options->ggs_username && white != options->ggs_username)
    ) {
        return 0;
    }

    #if !IS_GGS_TOURNAMENT
        if (black_score == 0) {
            return 0;
        }
    #endif
    const int winner_player = black_score > 0 ? BLACK : WHITE;
    Board board(initial_board);
    int player_to_move = initial_board[initial_board.size() - 1] == 'O' ? WHITE : BLACK;
    int n_registered = 0;
    for (size_t i = 0; i + 1 < transcript.size(); i += 2) {
        uint64_t legal = board.get_legal();
        if (legal == 0ULL) {
            board.pass();
            player_to_move ^= 1;
            legal = board.get_legal();
            if (legal == 0ULL) {
                break;
            }
        }

        const int policy = get_coord_from_chars(transcript[i], transcript[i + 1]);
        if (!is_valid_policy(policy) || !(legal & (1ULL << policy))) {
            break;
        }

#if IS_GGS_TOURNAMENT
        const std::string move_player_name = player_to_move == BLACK ? black : white;
        const bool mover_did_not_lose = player_to_move == BLACK ? black_score >= 0 : black_score <= 0;
        if (move_player_name != options->ggs_username && mover_did_not_lose) {
            ggs_record_move_hint(move_hints, board, policy, GGS_GAME_LOG_WINNER_HINT_WEIGHT);
            n_registered += GGS_GAME_LOG_WINNER_HINT_WEIGHT;
        }
        #else
        if (player_to_move == winner_player) {
            ggs_record_move_hint(move_hints, board, policy, GGS_GAME_LOG_WINNER_HINT_WEIGHT);
            n_registered += GGS_GAME_LOG_WINNER_HINT_WEIGHT;
        }
        #endif

        Flip flip;
        calc_flip(&flip, &board, policy);
        board.move_board(&flip);
        player_to_move ^= 1;
    }
    return n_registered;
}

void ggs_seed_move_hints_from_game_logs(GGS_Move_Hint_Table *move_hints, Options *options) {
    if (!options->ggs_game_log_to_file || options->ggs_game_log_dir.empty()) {
        return;
    }
    int n_files = 0;
    int n_hints = 0;
    try {
        std::filesystem::path dir(options->ggs_game_log_dir);
        if (!std::filesystem::exists(dir)) {
            return;
        }
        struct Game_Log_File {
            std::filesystem::path path;
            std::filesystem::file_time_type write_time;
        };
        std::vector<Game_Log_File> game_log_files;
        for (const std::filesystem::directory_entry &entry: std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".txt") {
                continue;
            }
            game_log_files.push_back(Game_Log_File{entry.path(), entry.last_write_time()});
        }
        std::sort(game_log_files.begin(), game_log_files.end(), [](const Game_Log_File &a, const Game_Log_File &b) {
            return a.write_time > b.write_time;
        });
#if IS_GGS_TOURNAMENT
        if (game_log_files.size() > GGS_TOURNAMENT_GAME_LOG_HINT_MAX_FILES) {
            game_log_files.resize(GGS_TOURNAMENT_GAME_LOG_HINT_MAX_FILES);
        }
#endif
        for (const Game_Log_File &entry: game_log_files) {
            ++n_files;
            n_hints += ggs_seed_move_hints_from_game_log_file(move_hints, entry.path, options);
        }
    } catch (const std::exception &e) {
        #if IS_GGS_TOURNAMENT
        if (options->show_log) {
            std::cerr << "ggs failed to seed synchro hints: " << e.what() << std::endl;
        }
        #else
        ggs_print_info(std::string("failed to seed synchro hints: ") + e.what(), options);
        #endif
        return;
    }
    #if IS_GGS_TOURNAMENT
    if (options->show_log) {
        ggs_print_info("seeded safe opponent hints files " + std::to_string(n_files) + " moves " + std::to_string(n_hints) + " boards " + std::to_string(move_hints->size()), options);
    }
    #else
    ggs_print_info("seeded synchro hints files " + std::to_string(n_files) + " moves " + std::to_string(n_hints) + " boards " + std::to_string(move_hints->size()), options);
    #endif
}

void ggs_print_send(std::string str, Options *options) { // cyan
    ggs_print_colored_lines(GGS_SEND_HEADER, str, options, "\033[36m");
}

void ggs_print_receive(std::string str, Options *options) { // green
    ggs_print_colored_lines(GGS_REPLY_HEADER, str, options, "\033[32m");
}

void ggs_print_info(std::string str, Options *options) { // yellow
    ggs_print_colored_lines(GGS_INFO_HEADER, str, options, "\033[33m");
}

void ggs_report_error(std::string str, Options *options) {
    std::cout << "[ERROR] " << str << std::endl;
    ggs_write_log_lines(GGS_INFO_HEADER, "[ERROR] " + str, options);
}

void ggs_print_debug(std::string str, Options *options) {
    if (ggs_verbose_log(options)) {
        ggs_print_info(str, options);
    }
}

int ggs_send_raw(SOCKET &sock, const char *data, int len) {
    int sent_total = 0;
    while (sent_total < len) {
        int sent = send(sock, data + sent_total, len - sent_total, 0);
        if (sent == SOCKET_ERROR) {
            std::cerr << "Send failed. Error Code: " << WSAGetLastError() << std::endl;
            return 1;
        }
        if (sent == 0) {
            std::cerr << "Send failed. Socket sent 0 bytes." << std::endl;
            return 1;
        }
        sent_total += sent;
    }
    return 0;
}

std::string ggs_normalize_telnet_newlines(const std::string &msg) {
    std::string res;
    res.reserve(msg.size() + 8);
    for (size_t i = 0; i < msg.size(); ++i) {
        if (msg[i] == '\n' && (i == 0 || msg[i - 1] != '\r')) {
            res += "\r\n";
        } else {
            res += msg[i];
        }
    }
    return res;
}

inline bool ggs_accept_telnet_server_option(unsigned char option) {
    return option == TELNET_ECHO || option == TELNET_SUPPRESS_GO_AHEAD;
}

inline bool ggs_accept_telnet_client_option(unsigned char option) {
    return option == TELNET_SUPPRESS_GO_AHEAD;
}

void ggs_send_telnet_negotiation(SOCKET &sock, unsigned char command, unsigned char option) {
    unsigned char reply_command = TELNET_DONT;
    if (command == TELNET_WILL) {
        reply_command = ggs_accept_telnet_server_option(option) ? TELNET_DO : TELNET_DONT;
    } else if (command == TELNET_DO) {
        reply_command = ggs_accept_telnet_client_option(option) ? TELNET_WILL : TELNET_WONT;
    } else {
        return;
    }
    const unsigned char reply[3] = {TELNET_IAC, reply_command, option};
    ggs_send_raw(sock, reinterpret_cast<const char*>(reply), 3);
}

std::string ggs_telnet_command_name(unsigned char command) {
    switch (command) {
        case TELNET_WILL:
            return "WILL";
        case TELNET_WONT:
            return "WONT";
        case TELNET_DO:
            return "DO";
        case TELNET_DONT:
            return "DONT";
        default:
            return std::to_string(command);
    }
}

std::string ggs_sockaddr_to_text(const sockaddr *addr, int addr_len) {
    char host[NI_MAXHOST];
    char service[NI_MAXSERV];
    if (getnameinfo(addr, addr_len, host, sizeof(host), service, sizeof(service), NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
        return "(unknown)";
    }
    return std::string(host) + ":" + service;
}

std::string ggs_filter_telnet_control(SOCKET &sock, const char *data, int len, Options *options) {
    enum Telnet_State {
        TELNET_STATE_DATA,
        TELNET_STATE_IAC,
        TELNET_STATE_OPTION,
        TELNET_STATE_SUBNEGOTIATION,
        TELNET_STATE_SUBNEGOTIATION_IAC
    };
    static Telnet_State state = TELNET_STATE_DATA;
    static unsigned char option_command = 0;
    std::string text;
    text.reserve(len);

    for (int i = 0; i < len; ++i) {
        const unsigned char c = static_cast<unsigned char>(data[i]);
        switch (state) {
            case TELNET_STATE_DATA:
                if (c == TELNET_IAC) {
                    state = TELNET_STATE_IAC;
                } else {
                    text += static_cast<char>(c);
                }
                break;
            case TELNET_STATE_IAC:
                if (c == TELNET_IAC) {
                    text += static_cast<char>(c);
                    state = TELNET_STATE_DATA;
                } else if (c == TELNET_WILL || c == TELNET_WONT || c == TELNET_DO || c == TELNET_DONT) {
                    option_command = c;
                    state = TELNET_STATE_OPTION;
                } else if (c == TELNET_SB) {
                    state = TELNET_STATE_SUBNEGOTIATION;
                } else {
                    state = TELNET_STATE_DATA;
                }
                break;
            case TELNET_STATE_OPTION:
                if (options->show_log) {
                    ggs_print_debug("telnet recv " + ggs_telnet_command_name(option_command) + " option " + std::to_string(c), options);
                }
                ggs_send_telnet_negotiation(sock, option_command, c);
                state = TELNET_STATE_DATA;
                break;
            case TELNET_STATE_SUBNEGOTIATION:
                if (c == TELNET_IAC) {
                    state = TELNET_STATE_SUBNEGOTIATION_IAC;
                }
                break;
            case TELNET_STATE_SUBNEGOTIATION_IAC:
                state = c == TELNET_SE ? TELNET_STATE_DATA : TELNET_STATE_SUBNEGOTIATION;
                break;
        }
    }
    return text;
}

int ggs_connect(WSADATA &wsaData, struct sockaddr_in &server, SOCKET &sock, Options *options) {
    sock = INVALID_SOCKET;
    ggs_print_debug("winsock startup", options);
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "Failed to initialize Winsock. Error Code: " << WSAGetLastError() << std::endl;
        return 1;
    }

    const char* hostname = GGS_URL;
    const std::string port = std::to_string(GGS_PORT);
    ggs_print_debug("resolving " + std::string(hostname) + ":" + port, options);
    struct addrinfo hints;
    struct addrinfo *result = nullptr;
    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    const int resolve_error = getaddrinfo(hostname, port.c_str(), &hints, &result);
    if (resolve_error != 0) {
        std::cerr << "Failed to resolve hostname. Error Code: " << resolve_error << std::endl;
        WSACleanup();
        return 1;
    }
    ggs_print_debug("resolved " + std::string(hostname), options);

    ZeroMemory(&server, sizeof(server));
    for (struct addrinfo *addr = result; addr != nullptr; addr = addr->ai_next) {
        ggs_print_debug("connecting to " + ggs_sockaddr_to_text(addr->ai_addr, static_cast<int>(addr->ai_addrlen)), options);
        SOCKET candidate = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
        if (candidate == INVALID_SOCKET) {
            std::cerr << "Could not create socket. Error Code: " << WSAGetLastError() << std::endl;
            continue;
        }

        if (addr->ai_family == AF_INET && addr->ai_addrlen <= sizeof(server)) {
            memcpy(&server, addr->ai_addr, addr->ai_addrlen);
        }

        if (connect(candidate, addr->ai_addr, static_cast<int>(addr->ai_addrlen)) == 0) {
            sock = candidate;
            freeaddrinfo(result);
            ggs_print_debug("socket connected", options);
            return 0;
        }

        std::cerr << "Connection attempt failed. Error Code: " << WSAGetLastError() << std::endl;
        closesocket(candidate);
    }

    freeaddrinfo(result);
    WSACleanup();
    return 1;
}

void ggs_close(SOCKET &sock) {
    closesocket(sock);
    WSACleanup();
}

int ggs_send_message(SOCKET &sock, std::string msg, Options *options, std::string log_msg = "") {
    const std::string wire_msg = ggs_normalize_telnet_newlines(msg);
    if (ggs_send_raw(sock, wire_msg.c_str(), static_cast<int>(wire_msg.size())) != 0) {
        return 1;
    }
    ggs_print_send(log_msg.empty() ? msg : log_msg, options);
    return 0;
}

inline bool ggs_replies_connection_closed(const std::vector<std::string> &replies) {
    return std::find(replies.begin(), replies.end(), std::string(GGS_CONNECTION_CLOSED_TOKEN)) != replies.end();
}

std::vector<std::string> ggs_extract_ready_messages(std::string *buffer) {
    std::vector<std::string> res;
    size_t ready_pos = buffer->find(GGS_READY);
    while (ready_pos != std::string::npos) {
        res.emplace_back(buffer->substr(0, ready_pos));
        buffer->erase(0, ready_pos + std::string(GGS_READY).size());
        ready_pos = buffer->find(GGS_READY);
    }
    return res;
}

std::string& ggs_receive_buffer() {
    static std::string buffer;
    return buffer;
}

bool ggs_receive_append(SOCKET *sock, Options *options, std::vector<std::string> *res) {
    char server_reply[GGS_RECEIVE_BUFFER_SIZE];
    while (true) {
        int recv_size = recv(*sock, server_reply, GGS_RECEIVE_BUFFER_SIZE - 1, 0);
        if (recv_size == SOCKET_ERROR) {
            const int err = WSAGetLastError();
            if (err == WSAEINTR) {
                continue;
            }
            if (ggs_socket_closing.load()) {
                res->emplace_back(GGS_CONNECTION_CLOSED_TOKEN);
                return false;
            }
            std::cerr << "Recv failed. Error Code: " << err << std::endl;
            res->emplace_back(GGS_CONNECTION_CLOSED_TOKEN);
            return false;
        }
        if (recv_size == 0) {
            if (ggs_socket_closing.load()) {
                res->emplace_back(GGS_CONNECTION_CLOSED_TOKEN);
                return false;
            }
            std::cerr << "GGS connection closed by server." << std::endl;
            res->emplace_back(GGS_CONNECTION_CLOSED_TOKEN);
            return false;
        }

        ggs_print_debug("recv raw bytes " + std::to_string(recv_size), options);
        std::string text = ggs_filter_telnet_control(*sock, server_reply, recv_size, options);
        if (text.empty()) {
            continue;
        }
        ggs_receive_buffer() += text;
        ggs_print_receive(text, options);
        return true;
    }
}

std::vector<std::string> ggs_receive_message(SOCKET *sock, Options *options) {
    std::string &receive_buffer = ggs_receive_buffer();

    std::vector<std::string> res = ggs_extract_ready_messages(&receive_buffer);
    if (!res.empty()) {
        ggs_print_debug("using buffered READY reply count " + std::to_string(res.size()), options);
        return res;
    }

    while (true) {
        if (!ggs_receive_append(sock, options, &res)) {
            return res;
        }
        res = ggs_extract_ready_messages(&receive_buffer);
        if (!res.empty()) {
            ggs_print_debug("received READY reply count " + std::to_string(res.size()), options);
            return res;
        }
    }
}

bool ggs_receive_until_text(SOCKET *sock, Options *options, const std::string &stage, const std::vector<std::string> &markers, std::string *received = nullptr) {
    ggs_print_debug("waiting for " + stage, options);
    std::string &buffer = ggs_receive_buffer();
    std::vector<std::string> res;
    while (true) {
        for (const std::string &marker: markers) {
            const size_t pos = buffer.find(marker);
            if (pos != std::string::npos) {
                const size_t marker_end_pos = pos + marker.size();
                size_t end_pos = buffer.find('\n', marker_end_pos);
                end_pos = end_pos == std::string::npos ? marker_end_pos : end_pos + 1;
                if (received != nullptr) {
                    *received = buffer.substr(0, end_pos);
                }
                buffer.erase(0, end_pos);
                ggs_print_debug("received " + stage, options);
                return true;
            }
        }
        if (!ggs_receive_append(sock, options, &res)) {
            ggs_report_error("GGS connection closed while waiting for " + stage + ".", options);
            return false;
        }
    }
}

bool ggs_receive_required(SOCKET *sock, Options *options, const std::string &stage, std::vector<std::string> *received = nullptr) {
    ggs_print_debug("waiting for " + stage, options);
    std::vector<std::string> replies = ggs_receive_message(sock, options);
    if (received != nullptr) {
        *received = replies;
    }
    if (ggs_replies_connection_closed(replies)) {
        ggs_report_error("GGS connection closed while waiting for " + stage + ".", options);
        return false;
    }
    ggs_print_debug("received " + stage + " replies " + std::to_string(replies.size()), options);
    return true;
}

void ggs_log_server_errors(const std::vector<std::string> &server_replies, Options *options) {
    for (const std::string &server_reply: server_replies) {
        std::stringstream ss(server_reply);
        std::string line;
        while (std::getline(ss, line, '\n')) {
            if (line.find("ERR") != std::string::npos || line.find("+err") != std::string::npos || line.find("-err") != std::string::npos) {
                ggs_print_info("server error: " + line, options);
            }
        }
    }
}

std::string ggs_get_os_info(std::string str) {
    std::stringstream ss(str);
    std::string line;
    while (std::getline(ss, line, '\n')) {
        if (line.substr(0, 4) == "/os:") {
            return line;
        }
    }
    return "";
}

std::string ggs_get_user_input() {
    std::string res;
    if (!std::getline(std::cin, res)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        return "";
    }
    return res;
}

bool ggs_is_match_start(std::string line, std::string username) {
    std::vector<std::string> words = split_by_space(line);
    if (std::find(words.begin(), words.end(), username) != words.end()) {
        if (words.size() >= 3) {
            return words[1] == "+" && words[2] == "match";
        }
    }
    return false;
}

bool ggs_is_match_end(std::string line, std::string username) {
    std::vector<std::string> words = split_by_space(line);
    if (std::find(words.begin(), words.end(), username) != words.end()) {
        if (words.size() >= 3) {
            return words[1] == "-" && words[2] == "match";
        }
    }
    return false;
}

bool ggs_is_board_info(std::string line) {
    std::vector<std::string> words = split_by_space(line);
    if (words.size() >= 2) {
        return words[1] == "update" || words[1] == "join";
    }
    return false;
}

bool ggs_is_game_end(std::string line) {
    std::vector<std::string> words = split_by_space(line);
    return words.size() >= 9 && words[1] == "end";
}

bool ggs_parse_game_end(std::string line, std::string *game_id, std::string *first_player, int *score) {
    std::vector<std::string> words = split_by_space(line);
    if (words.size() < 9 || words[1] != "end") {
        return false;
    }
    try {
        *game_id = words[2];
        *first_player = words[4];
        *score = (int)std::round(std::stod(words.back()));
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

bool ggs_is_match_request(std::string line, std::string username) {
    std::vector<std::string> words = split_by_space(line);
    if (words.size() >= 10) {
        return words[1] == "+" && (words[7] == "R" || words[7] == "S") && (words[4] == username || words[9] == username);
        //return words[1] == "+" && words[7] == "R" && (words[4] == username || words[9] == username);
    }
    return false;
}

std::string ggs_match_request_get_id(std::string line) {
    std::vector<std::string> words = split_by_space(line);
    if (words.size() >= 3) {
        return words[2];
    }
    return "";
}

bool ggs_parse_clock_time_msec(const std::string &str, uint64_t *msec) {
    if (str.empty()) {
        *msec = 0ULL;
        return true;
    }
    std::vector<std::string> parts = split_by_delimiter(str, ":");
    if (parts.empty() || parts.size() > 3) {
        return false;
    }
    uint64_t values[3] = {0ULL, 0ULL, 0ULL};
    try {
        for (int i = 0; i < (int)parts.size(); ++i) {
            if (parts[i].empty()) {
                return false;
            }
            values[3 - parts.size() + i] = (uint64_t)std::stoull(parts[i]);
        }
    } catch (const std::exception&) {
        return false;
    }
    *msec = ((values[0] * 60ULL + values[1]) * 60ULL + values[2]) * 1000ULL;
    return true;
}

bool ggs_parse_clock_segment(
    const std::string &segment,
    uint64_t *time_msec,
    int *n_moves,
    bool *additive_or_loss,
    bool default_additive_or_loss
) {
    std::vector<std::string> parts = split_by_delimiter(segment, ",");
    if (parts.empty() || parts.size() > 2) {
        return false;
    }
    if (!ggs_parse_clock_time_msec(parts[0], time_msec)) {
        return false;
    }
    *additive_or_loss = default_additive_or_loss;
    if (parts.size() == 2) {
        std::string moves = parts[1];
        if (!moves.empty() && (moves[0] == 'N' || moves[0] == 'n')) {
            *additive_or_loss = false;
            moves.erase(0, 1);
        }
        try {
            *n_moves = moves.empty() ? 0 : std::stoi(moves);
        } catch (const std::exception&) {
            return false;
        }
    }
    return true;
}

bool ggs_parse_clock_params(const std::string &clock_str, GGS_Clock_Params *clock) {
    if (clock_str.empty()) {
        return false;
    }
    GGS_Clock_Params parsed;
    std::vector<std::string> segments = split_by_delimiter(clock_str, "/");
    if (segments.empty() || segments.size() > 3) {
        return false;
    }
    if (!ggs_parse_clock_segment(segments[0], &parsed.initial_msec, &parsed.initial_moves, &parsed.initial_loss, true)) {
        return false;
    }
    if (segments.size() >= 2 && !segments[1].empty()) {
        if (!ggs_parse_clock_segment(segments[1], &parsed.increment_msec, &parsed.increment_moves, &parsed.increment_add, true)) {
            return false;
        }
        if (parsed.increment_moves < 1) {
            return false;
        }
    } else if (segments.size() == 2) {
        return false;
    }
    if (segments.size() >= 3) {
        if (!ggs_parse_clock_segment(segments[2], &parsed.extension_msec, &parsed.extension_moves, &parsed.extension_add, true)) {
            return false;
        }
    }
    parsed.initialized = true;
    *clock = parsed;
    return true;
}

bool ggs_parse_board_clock_time_msec(const std::string &segment, uint64_t *msec) {
    const std::string time_part = segment.substr(0, segment.find(','));
    return !time_part.empty() && ggs_parse_clock_time_msec(time_part, msec);
}

bool ggs_parse_board_clock_params(const std::string &clock_str, GGS_Clock_Params *clock) {
    if (clock_str.empty()) {
        return false;
    }
    GGS_Clock_Params parsed;
    std::vector<std::string> segments = split_by_delimiter(clock_str, "/");
    if (segments.empty() || segments.size() > 3) {
        return false;
    }
    if (!ggs_parse_board_clock_time_msec(segments[0], &parsed.initial_msec)) {
        return false;
    }
    if (segments.size() >= 2 && !segments[1].empty()) {
        if (!ggs_parse_board_clock_time_msec(segments[1], &parsed.increment_msec)) {
            return false;
        }
    } else if (segments.size() == 2) {
        return false;
    }
    if (segments.size() >= 3 && !segments[2].empty()) {
        if (!ggs_parse_board_clock_time_msec(segments[2], &parsed.extension_msec)) {
            return false;
        }
    }
    parsed.initialized = true;
    *clock = parsed;
    return true;
}

std::string ggs_clock_summary(const GGS_Clock_Params &clock);

bool ggs_clock_control_params_equal(const GGS_Clock_Params &lhs, const GGS_Clock_Params &rhs) {
    return lhs.initialized == rhs.initialized &&
           lhs.increment_msec == rhs.increment_msec &&
           lhs.extension_msec == rhs.extension_msec &&
           lhs.increment_moves == rhs.increment_moves &&
           lhs.extension_moves == rhs.extension_moves &&
           lhs.increment_add == rhs.increment_add &&
           lhs.extension_add == rhs.extension_add;
}

void ggs_apply_board_clock(GGS_Board *ggs_board, GGS_Clock_Params *clock_params, Options *options) {
    if (ggs_board->clock.initialized) {
        if (!clock_params->initialized || !ggs_clock_control_params_equal(*clock_params, ggs_board->clock)) {
            *clock_params = ggs_board->clock;
            ggs_print_info("ggs clock inferred from board " + ggs_clock_summary(*clock_params), options);
            if (clock_params->extension_msec > 0ULL && clock_params->increment_msec == 0ULL) {
                ggs_print_info("ggs clock note: extension time is not per-move increment", options);
            }
        } else {
            ggs_board->clock = *clock_params;
        }
    } else {
        ggs_board->clock = *clock_params;
    }
}

std::string ggs_clock_summary(const GGS_Clock_Params &clock) {
    if (!clock.initialized) {
        return "uninitialized";
    }
    return "initial " + std::to_string(clock.initial_msec / 1000ULL) +
           "s inc " + std::to_string(clock.increment_msec / 1000ULL) +
           "s/" + std::to_string(clock.increment_moves) +
           (clock.increment_add ? " add" : " bronstein") +
           " ext " + std::to_string(clock.extension_msec / 1000ULL) +
           "s/" + std::to_string(clock.extension_moves) +
           (clock.extension_add ? " add" : " reset");
}

bool ggs_match_request_get_clock(std::string line, const std::string &username, GGS_Clock_Params *clock) {
    std::vector<std::string> words = split_by_space(line);
    if (words.size() < 10) {
        return false;
    }
    std::string clock_str;
    if (words[4] == username && words.size() >= 6) {
        clock_str = words[5];
    } else if (words[9] == username) {
        clock_str = (words.size() >= 11 && words[10].find('/') != std::string::npos) ? words[10] : words[5];
    }
    return ggs_parse_clock_params(clock_str, clock);
}

void ggs_accept_match_requests(std::vector<std::string> server_replies, SOCKET &sock, GGS_Clock_Params *clock_params, Options *options) {
    if (!options->ggs_accept_request) {
        return;
    }
    for (std::string server_reply: server_replies) {
        if (server_reply.size()) {
            std::string os_info = ggs_get_os_info(server_reply);
            if (ggs_is_match_request(os_info, options->ggs_username)) {
                std::string request_id = ggs_match_request_get_id(os_info);
                if (request_id.size()) {
                    GGS_Clock_Params request_clock;
                    if (ggs_match_request_get_clock(os_info, options->ggs_username, &request_clock)) {
                        *clock_params = request_clock;
                        if (ggs_verbose_log(options)) {
                            std::cerr << "ggs clock " << ggs_clock_summary(*clock_params) << std::endl;
                            if (clock_params->extension_msec > 0ULL && clock_params->increment_msec == 0ULL) {
                                std::cerr << "ggs clock note: extension time is not per-move increment" << std::endl;
                            }
                        }
                    }
                    std::string accept_cmd = "ts accept " + request_id;
                    ggs_send_message(sock, accept_cmd + "\n", options);
                    ggs_print_info("match request accepted " + request_id, options);
                }
            }
        }
    }
}

bool ggs_is_join_tournament_message(std::string line) {
    // something like "nyanyan: tell /td join .1" or "nyanyan: t /td join .1"
    std::vector<std::string> words = split_by_space(line);
    if (words.size() >= 5) {
        return (words[1] == "tell" || words[1] == "t") && words[2] == "/td" && words[3] == "join";
    }
    return false;
}

std::string ggs_join_tournament_get_cmd(std::string line) {
    // something like "nyanyan: tell /td join .1" or "nyanyan: t /td join .1"
    std::vector<std::string> words = split_by_space(line);
    std::string res = "";
    for (int i = 1; i < 5; ++i) {
        res += words[i];
        res += " ";
    }
    return res;
}

std::string ggs_board_get_id(std::string line) {
    std::vector<std::string> words = split_by_space(line);
    if (words.size() >= 3) {
        return words[2];
    }
    return "";
}

uint64_t ggs_parse_remaining_seconds(const std::string &clock_text) {
    std::string main_clock = clock_text.substr(0, clock_text.find(','));
    if (main_clock.empty()) {
        return 0ULL;
    }
    bool negative = false;
    if (main_clock[0] == '-' || main_clock[0] == '+') {
        negative = main_clock[0] == '-';
        main_clock.erase(main_clock.begin());
    }
    const size_t colon = main_clock.find(':');
    if (negative || colon == std::string::npos || colon == 0 || colon + 1 >= main_clock.size()) {
        return 0ULL;
    }
    try {
        const int minutes = std::stoi(main_clock.substr(0, colon));
        const int seconds = std::stoi(main_clock.substr(colon + 1));
        if (minutes < 0 || seconds < 0) {
            return 0ULL;
        }
        return (uint64_t)minutes * 60ULL + (uint64_t)seconds;
    } catch (const std::exception&) {
        return 0ULL;
    }
}

GGS_Board ggs_get_board(std::string str) {
    GGS_Board res;
    std::string os_info = ggs_get_os_info(str);
    std::vector<std::string> os_info_words = split_by_space(os_info);
    if (os_info_words.size() < 3) {
        std::cerr << "ggs_get_board failed: id invalid" << std::endl;
        return res;
    }
    bool is_join = os_info_words[1] == "join"; // /os: join .4.1 s8r18 K?
    res.game_id = os_info_words[2]; // /os: update .4.1 s8r18 K?
    int game_id_dot_count = 0;
    for (int i = 0; (i = res.game_id.find('.', i)) != std::string::npos; i++) {
        game_id_dot_count++;
    }
    res.is_synchro = game_id_dot_count == 2;
    if (res.is_synchro) {
        std::vector<std::string> ids = split_by_delimiter(res.game_id, ".");
        try {
            res.synchro_id = std::stoi(ids[ids.size() - 1]);
        } catch (const std::invalid_argument& e) {
            std::cerr << "ggs_get_board failed: synchro_id invalid" << std::endl;
            res.synchro_id = -1;
        } catch (const std::out_of_range& e) {
            std::cerr << "ggs_get_board failed: synchro_id out of range" << std::endl;
            res.synchro_id = -1;
        }
    }
    std::string board_str;
    std::stringstream ss(str);
    std::string line;
    int n_board_identifier_found = 0;
    int n_board_identifier_used = 1;
    while (std::getline(ss, line, '\n')) {
        std::vector<std::string> words = split_by_space(line);
        if (!line.empty() && line[0] == '|') {
            if (line.find(" move(s)") != std::string::npos) {
                if (line.substr(0, 10) != "|0 move(s)") { // happens in stored game
                    //std::cout << "stored game" << std::endl;
                    std::string line2;
                    while (line2.substr(0, 10) != "|* to move" && line2.substr(0, 10) != "|O to move") {
                        std::getline(ss, line2, '\n'); // skip starting board
                    }
                }
                continue;
            }
            // board
            if (line.find("A B C D E F G H") != std::string::npos) {
                ++n_board_identifier_found;
                continue;
            }
            if (n_board_identifier_found == 1) { // board info
                std::string board_str_part;
                for (char c : line) {
                    if (c == '-' || c == '*' || c == 'O') {
                        board_str_part += c;
                    }
                }
                board_str += remove_spaces(board_str_part);
                continue;
            }

            // which to move
            if (line.substr(0, 10) == "|* to move") {
                res.player_to_move = BLACK;
                continue;
            } else if (line.substr(0, 10) == "|O to move") {
                res.player_to_move = WHITE;
                continue;
            }

            // last move
            if (!is_join) {
                if (line.substr(0, 2) == "| ") {
                    if (words.size() >= 3) {
                        if (words[1][words[1].size() - 1] == ':' && words[2].size() >= 2) {
                            res.last_move = get_coord_from_chars(words[2][0], words[2][1]);
                            continue;
                        }
                    }
                }
            }

            // users
            if (words.size() >= 4) {
                std::string player_id = words[0].substr(1, words[0].size() - 1);
                uint64_t remaining_seconds = ggs_parse_remaining_seconds(words[3]);
                GGS_Clock_Params board_clock;
                if (ggs_parse_board_clock_params(words[3], &board_clock)) {
                    res.clock = board_clock;
                }
                if (words[2][0] == '*') {
                    res.player_black = player_id;
                    res.remaining_seconds_black = remaining_seconds;
                } else if (words[2][0] == 'O') {
                    res.player_white = player_id;
                    res.remaining_seconds_white = remaining_seconds;
                }
            }
        }
    }
    if (res.player_to_move == BLACK) {
        board_str += " *";
    } else if (res.player_to_move == WHITE) {
        board_str += " O";
    } else {
        return res;
    }
    if (remove_spaces(board_str).size() != HW2 + 1) {
        return res;
    }
    if (!res.board.from_str(board_str)) {
        res.board = Board();
        res.player_to_move = -1;
        return res;
    }

    // std::cerr << "game_id " << res.game_id << std::endl;
    // std::cerr << "is_synchro " << res.is_synchro << std::endl;
    // std::cerr << "synchro_id " << res.synchro_id << std::endl;
    // std::cerr << "black " << res.player_black << " " << res.remaining_seconds_black << std::endl;
    // std::cerr << "white " << res.player_white << " " << res.remaining_seconds_white << std::endl;
    // std::cerr << res.player_to_move << " to move" << std::endl;
    // std::cerr << board_str << std::endl;
    // res.board.print();

    return res;
}

Search_result ggs_search(
    GGS_Board ggs_board,
    Options *options,
    thread_id_t thread_id,
    bool *searching,
    int hint_policy,
    int hint_count,
    Search_result ponder_result,
    GGS_Synchro_Time_Context synchro_time_context
) {
    Search_result search_result;
    if (ggs_board.board.get_legal()) {
        if (ggs_is_usable_ponder_result(ggs_board.board, ponder_result)) {
            ggs_print_debug(
                "ggs exact ponder selected " + idx_to_coord(ponder_result.policy) +
                " value " + std::to_string(ponder_result.value) +
                " depth " + std::to_string(ponder_result.depth) + "@100% " +
                ggs_board.board.to_str(),
                options
            );
            ggs_log_search_result_summary("search exact ponder", ggs_board, ponder_result, hint_policy, hint_count, options);
            return ponder_result;
        }
        if (ggs_should_play_hint_without_search(ggs_board.board, hint_policy, hint_count)) {
            ggs_print_debug("ggs synchro hint selected without search " + idx_to_coord(hint_policy) + " " + ggs_board.board.to_str(), options);
            search_result = ggs_hint_search_result(ggs_board.board, hint_policy);
            ggs_log_search_result_summary("search hint only", ggs_board, search_result, hint_policy, hint_count, options);
            return search_result;
        }

        uint64_t remaining_time_msec = 0;
        if (ggs_board.player_to_move == BLACK) {
            remaining_time_msec = ggs_board.remaining_seconds_black * 1000;
        } else {
            remaining_time_msec = ggs_board.remaining_seconds_white * 1000;
        }
        const uint64_t raw_remaining_time_msec = remaining_time_msec;
        const uint64_t safety_margin = ggs_time_safety_margin_msec(raw_remaining_time_msec);
        if (remaining_time_msec > safety_margin) {
            remaining_time_msec -= safety_margin;
        } else {
            remaining_time_msec = std::max<uint64_t>(remaining_time_msec * 0.1, 1ULL);
        }
        remaining_time_msec = ggs_clock_adjusted_time_for_allocation(ggs_board.board, remaining_time_msec, raw_remaining_time_msec, ggs_board.clock);
        remaining_time_msec = ggs_adjust_remaining_time_for_synchro_pair(
            ggs_board,
            remaining_time_msec,
            raw_remaining_time_msec,
            synchro_time_context,
            options
        );

        // // special code for s8r14 5 min
        uint64_t strt = tim();
        // if (ggs_board.board.n_discs() == 14 && remaining_time_msec > 30000) {
        //     std::cerr << "s8r14 first move special selfplay" << std::endl;
        //     bool new_searching = true;
        //     std::future<std::vector<Ponder_elem>> ponder_future = std::async(std::launch::async, ai_ponder, ggs_board.board, true, thread_id, &new_searching);
        //     if (ponder_future.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
        //         new_searching = false;
        //     }
        //     ponder_future.get();
        //     /*
        //     std::vector<Ponder_elem> move_list = ai_get_values(ggs_board.board, true, 4000, thread_id);
        //     double best_value = move_list[0].value;
        //     int n_good_moves = 0;
        //     for (const Ponder_elem &elem: move_list) {
        //         if (elem.value >= best_value - AI_TL_ADDITIONAL_SEARCH_THRESHOLD * 3.0) {
        //             ++n_good_moves;
        //         }
        //     }
        //     if (n_good_moves >= 2) {
        //         ai_additional_selfplay(ggs_board.board, true, move_list, n_good_moves, AI_TL_ADDITIONAL_SEARCH_THRESHOLD * 3.0, 10000, thread_id);
        //     }
        //     */
        //     std::cerr << std::endl;
        // }
        remaining_time_msec = ggs_subtract_elapsed_or_floor(remaining_time_msec, tim() - strt);
#if IS_GGS_TOURNAMENT
        ggs_print_info(
            "search allocation game " + ggs_board.game_id +
            " discs " + std::to_string(ggs_board.board.n_discs()) +
            " raw " + std::to_string(raw_remaining_time_msec) +
            " safety " + std::to_string(safety_margin) +
            " limit " + std::to_string(remaining_time_msec) +
            " inc " + std::to_string(ggs_board.clock.increment_msec) +
            " ext " + std::to_string(ggs_board.clock.extension_msec) +
            (synchro_time_context.has_pair_result ?
                (" pair_value " + std::to_string(synchro_time_context.pair_value) +
                 " pair_game " + synchro_time_context.pair_game_id) : ""),
            options
        );
#else
        ggs_print_debug(
            "search allocation game " + ggs_board.game_id +
            " discs " + std::to_string(ggs_board.board.n_discs()) +
            " raw " + std::to_string(raw_remaining_time_msec) +
            " safety " + std::to_string(safety_margin) +
            " limit " + std::to_string(remaining_time_msec) +
            " inc " + std::to_string(ggs_board.clock.increment_msec) +
            " ext " + std::to_string(ggs_board.clock.extension_msec),
            options
        );
#endif

        if (remaining_time_msec <= 50ULL) {
            search_result = ggs_fallback_search_result(ggs_board.board);
        } else {
            search_result = ai_time_limit(ggs_board.board, !options->nobook, 0, true, ggs_engine_show_log(options), remaining_time_msec, thread_id, searching);
            const uint64_t legal = ggs_board.board.get_legal();
            if (ggs_should_override_with_hint(ggs_board.board, hint_policy, hint_count, search_result)) {
                ggs_print_debug(
                    "ggs synchro hint overrides search " + idx_to_coord(search_result.policy) + " -> " + idx_to_coord(hint_policy) +
                    " value " + std::to_string(search_result.value) +
                    " depth " + std::to_string(search_result.depth) + "@" + std::to_string(search_result.probability) + "% " +
                    ggs_board.board.to_str(),
                    options
                );
                search_result.policy = hint_policy;
                search_result.value = mid_evaluate(&ggs_board.board);
                search_result.depth = 0;
                search_result.probability = 0;
            }
            if (!is_valid_policy(search_result.policy) || !(legal & (1ULL << search_result.policy))) {
                search_result = ggs_fallback_search_result(ggs_board.board);
            }
        }
        ggs_log_search_result_summary("search result", ggs_board, search_result, hint_policy, hint_count, options);
    } else { // pass
        search_result.policy = MOVE_PASS;
    }
    return search_result;
}

void ggs_send_move(GGS_Board &ggs_board, SOCKET &sock, Search_result search_result, Options *options) {
    std::string ggs_move_cmd;
    std::string move_text;
    if (search_result.policy == MOVE_PASS) {
        ggs_move_cmd = "t /os play " + ggs_board.game_id + " pa";
        move_text = "pa";
    } else {
        move_text = idx_to_coord(search_result.policy);
        ggs_move_cmd = "t /os play " + ggs_board.game_id + " " + move_text + "/" + std::to_string(search_result.value);
    }
    ggs_send_message(sock, ggs_move_cmd + "\n", options);
    ggs_print_info("move sent game " + ggs_board.game_id + " move " + move_text + " value " + std::to_string(search_result.value), options);
}

inline bool ggs_is_my_turn(const GGS_Board &ggs_board, Options *options) {
    return (
        (ggs_board.player_black == options->ggs_username && ggs_board.player_to_move == BLACK) ||
        (ggs_board.player_white == options->ggs_username && ggs_board.player_to_move == WHITE)
    );
}

inline uint64_t ggs_remaining_time_msec_to_move(const GGS_Board &ggs_board) {
    return (ggs_board.player_to_move == BLACK ? ggs_board.remaining_seconds_black : ggs_board.remaining_seconds_white) * 1000ULL;
}

inline void ggs_subtract_elapsed_from_remaining_time_to_move(GGS_Board *ggs_board, uint64_t elapsed_msec) {
    uint64_t *remaining_seconds = ggs_board->player_to_move == BLACK ? &ggs_board->remaining_seconds_black : &ggs_board->remaining_seconds_white;
    const uint64_t elapsed_seconds = (elapsed_msec + 999ULL) / 1000ULL;
    if (*remaining_seconds > elapsed_seconds) {
        *remaining_seconds -= elapsed_seconds;
    } else {
        *remaining_seconds = 0ULL;
    }
}

inline uint64_t ggs_synchro_hint_wait_msec(const GGS_Board &ggs_board) {
    const int n_discs = ggs_board.board.n_discs();
    const uint64_t remaining_time_msec = ggs_remaining_time_msec_to_move(ggs_board);
    if (n_discs <= 20 && remaining_time_msec > 120000ULL) {
        return 1500ULL;
    }
    if (n_discs <= 28 && remaining_time_msec > 80000ULL) {
        return 900ULL;
    }
    if (remaining_time_msec > 50000ULL) {
        return 500ULL;
    }
    return 300ULL;
}

bool ggs_same_position_waiting_for_opponent(
    const GGS_Board &ggs_board,
    GGS_Board ggs_boards[][HW2 + 1],
    Options *options
) {
    if (!ggs_board.is_synchro || ggs_board.synchro_id < 0 || ggs_board.synchro_id >= 2) {
        return false;
    }
    const int n_discs = ggs_board.board.n_discs();
    const GGS_Board &paired_board = ggs_boards[ggs_board.synchro_id ^ 1][n_discs];
    return (
        paired_board.is_valid() &&
        paired_board.board == ggs_board.board &&
        paired_board.player_to_move == ggs_board.player_to_move &&
        !ggs_is_my_turn(paired_board, options)
    );
}

bool ggs_synchro_partner_may_provide_hint(
    const GGS_Board &ggs_board,
    GGS_Board ggs_boards[][HW2 + 1],
    const int ggs_boards_n_discs[],
    Options *options
) {
    if (ggs_same_position_waiting_for_opponent(ggs_board, ggs_boards, options)) {
        return true;
    }
    if (!ggs_board.is_synchro || ggs_board.synchro_id < 0 || ggs_board.synchro_id >= 2) {
        return false;
    }
    const int n_discs = ggs_board.board.n_discs();
    const int paired_n_discs = ggs_boards_n_discs[ggs_board.synchro_id ^ 1];
    return paired_n_discs < n_discs;
}

bool ggs_should_wait_for_synchro_hint(
    const GGS_Board &ggs_board,
    const Search_result &search_result,
    const GGS_Move_Hint_Table &move_hints,
    GGS_Board ggs_boards[][HW2 + 1],
    Options *options
) {
    if (ggs_get_move_hint(move_hints, ggs_board.board) != MOVE_UNDEFINED) {
        return false;
    }
    const int n_discs = ggs_board.board.n_discs();
    if (n_discs > GGS_LIVE_HINT_WAIT_MAX_DISCS || ggs_remaining_time_msec_to_move(ggs_board) < 45000ULL) {
        return false;
    }
#if IS_GGS_TOURNAMENT
    if (
        GGS_EARLY_SYNCHRO_HINT_MOVE_WAIT_MAX_DISCS > 0 &&
        n_discs <= GGS_EARLY_SYNCHRO_HINT_MOVE_WAIT_MAX_DISCS &&
        ggs_same_position_waiting_for_opponent(ggs_board, ggs_boards, options)
    ) {
        return true;
    }
    if (n_discs <= GGS_LIVE_HINT_WAIT_MAX_DISCS && search_result.value <= 4) {
        return ggs_same_position_waiting_for_opponent(ggs_board, ggs_boards, options);
    }
    if (search_result.value > -8 || search_result.depth >= 27) {
        return false;
    }
    return ggs_same_position_waiting_for_opponent(ggs_board, ggs_boards, options);
#else
    if (search_result.value > -16 || search_result.depth >= 24) {
        return false;
    }
    return ggs_same_position_waiting_for_opponent(ggs_board, ggs_boards, options);
#endif
}

inline uint64_t ggs_synchro_hint_search_delay_msec(const GGS_Board &ggs_board) {
    const int n_discs = ggs_board.board.n_discs();
    const uint64_t remaining_time_msec = ggs_remaining_time_msec_to_move(ggs_board);
#if IS_GGS_TOURNAMENT
    if (n_discs <= 16 && remaining_time_msec > 150000ULL) {
        return 2500ULL;
    }
    if (n_discs <= 20 && remaining_time_msec > 100000ULL) {
        return 1200ULL;
    }
    if (n_discs <= 24 && remaining_time_msec > 70000ULL) {
        return 700ULL;
    }
    if (n_discs <= GGS_LIVE_HINT_WAIT_MAX_DISCS && remaining_time_msec > 60000ULL) {
        return 350ULL;
    }
#endif
    if (n_discs <= 20 && remaining_time_msec > 120000ULL) {
        return 1200ULL;
    }
    if (n_discs <= GGS_LIVE_HINT_WAIT_MAX_DISCS && remaining_time_msec > 80000ULL) {
        return 700ULL;
    }
    return 350ULL;
}

bool ggs_should_delay_search_for_synchro_hint(
    const GGS_Board &ggs_board,
    const GGS_Move_Hint_Table &move_hints,
    GGS_Board ggs_boards[][HW2 + 1],
    const int ggs_boards_n_discs[],
    Options *options
) {
    if constexpr (!GGS_ENABLE_SYNCHRO_HINT_SEARCH_DELAY) {
        (void)ggs_board;
        (void)move_hints;
        (void)ggs_boards;
        (void)ggs_boards_n_discs;
        (void)options;
        return false;
    }
    if (ggs_get_move_hint(move_hints, ggs_board.board) != MOVE_UNDEFINED) {
        return false;
    }
    const int n_discs = ggs_board.board.n_discs();
    if (n_discs > GGS_LIVE_HINT_WAIT_MAX_DISCS || ggs_remaining_time_msec_to_move(ggs_board) < 50000ULL) {
        return false;
    }
#if IS_GGS_TOURNAMENT
    return ggs_same_position_waiting_for_opponent(ggs_board, ggs_boards, options);
#else
    return ggs_synchro_partner_may_provide_hint(ggs_board, ggs_boards, ggs_boards_n_discs, options);
#endif
}

void ggs_store_pending_move(GGS_Pending_Move *pending_move, const GGS_Board &ggs_board, const Search_result &search_result) {
    pending_move->active = true;
    pending_move->board = ggs_board;
    pending_move->result = search_result;
    pending_move->ready_time = tim();
    pending_move->max_wait_msec = ggs_synchro_hint_wait_msec(ggs_board);
}

void ggs_store_pending_search(
    GGS_Pending_Search *pending_search,
    const GGS_Board &ggs_board,
    int search_slot,
    thread_id_t thread_id
) {
    pending_search->active = true;
    pending_search->board = ggs_board;
    pending_search->search_slot = search_slot;
    pending_search->thread_id = thread_id;
    pending_search->ready_time = tim();
    pending_search->max_wait_msec = ggs_synchro_hint_search_delay_msec(ggs_board);
}

bool ggs_try_send_pending_move(
    GGS_Pending_Move *pending_move,
    const GGS_Move_Hint_Table &move_hints,
    GGS_Synchro_Search_Record synchro_search_records[],
    SOCKET &sock,
    Options *options
) {
    if (!pending_move->active) {
        return false;
    }
    const bool hint_arrived = ggs_get_move_hint(move_hints, pending_move->board.board) != MOVE_UNDEFINED;
    if (!hint_arrived && tim() - pending_move->ready_time < pending_move->max_wait_msec) {
        return false;
    }

    ggs_print_debug(
        "ggs pending move send " + pending_move->board.game_id +
        " reason " + std::string(hint_arrived ? "hint" : "timeout") +
        " waited " + std::to_string(tim() - pending_move->ready_time) +
        " " + pending_move->board.board.to_str(),
        options
    );
    ggs_apply_move_hint_to_search_result(pending_move->board, move_hints, &pending_move->result, options);
    const uint64_t legal = pending_move->board.board.get_legal();
    if (legal && (!is_valid_policy(pending_move->result.policy) || !(legal & (1ULL << pending_move->result.policy)))) {
        pending_move->result = ggs_fallback_search_result(pending_move->board.board);
    }
    ggs_log_search_result_summary(
        "search send pending",
        pending_move->board,
        pending_move->result,
        ggs_get_move_hint(move_hints, pending_move->board.board),
        ggs_get_move_hint_count(move_hints, pending_move->board.board),
        options
    );
    ggs_store_synchro_search_record(synchro_search_records, pending_move->board, pending_move->result);
    ggs_send_move(pending_move->board, sock, pending_move->result, options);
    pending_move->active = false;
    return true;
}

void ggs_launch_ai_search(
    std::future<Search_result> ai_futures[],
    bool ai_searchings[],
    GGS_Board ggs_boards_searching[],
    const GGS_Board &ggs_board,
    int search_slot,
    thread_id_t thread_id,
    const GGS_Move_Hint_Table &move_hints,
    const GGS_Ponder_Result_Table &ponder_results,
    const GGS_Synchro_Search_Record synchro_search_records[],
    Options *options
) {
    int hint_policy = ggs_get_move_hint(move_hints, ggs_board.board);
    int hint_count = ggs_get_move_hint_count(move_hints, ggs_board.board);
    Search_result ponder_result = ggs_get_ponder_result(ponder_results, ggs_board.board);
    GGS_Synchro_Time_Context synchro_time_context = ggs_make_synchro_time_context(ggs_board, synchro_search_records);
    if (is_valid_policy(hint_policy)) {
        ggs_print_debug("synchro hint candidate before search " + ggs_board.game_id + " " + idx_to_coord(hint_policy), options);
    }
#if !IS_GGS_TOURNAMENT
    if (options->show_log && ggs_is_usable_ponder_result(ggs_board.board, ponder_result)) {
        std::cerr << "exact ponder candidate before search " << ggs_board.game_id << " " << idx_to_coord(ponder_result.policy) << std::endl;
    }
#endif
    ai_searchings[search_slot] = true;
    ggs_boards_searching[search_slot] = ggs_board;
    ai_futures[search_slot] = std::async(
        std::launch::async,
        ggs_search,
        ggs_board,
        options,
        thread_id,
        &ai_searchings[search_slot],
        hint_policy,
        hint_count,
        ponder_result,
        synchro_time_context
    );
}

bool ggs_try_launch_pending_search(
    GGS_Pending_Search *pending_search,
    std::future<Search_result> ai_futures[],
    bool ai_searchings[],
    GGS_Board ggs_boards_searching[],
    const GGS_Move_Hint_Table &move_hints,
    const GGS_Ponder_Result_Table &ponder_results,
    const GGS_Synchro_Search_Record synchro_search_records[],
    Options *options
) {
    if (!pending_search->active) {
        return false;
    }
    if (ai_searchings[pending_search->search_slot]) {
        return false;
    }
    const bool hint_arrived = ggs_get_move_hint(move_hints, pending_search->board.board) != MOVE_UNDEFINED;
    if (!hint_arrived && tim() - pending_search->ready_time < pending_search->max_wait_msec) {
        return false;
    }
    ggs_print_debug(
        "ggs pending search launch " + pending_search->board.game_id +
        " reason " + std::string(hint_arrived ? "hint" : "timeout") +
        " waited " + std::to_string(tim() - pending_search->ready_time) +
        " " + pending_search->board.board.to_str(),
        options
    );
    GGS_Board search_board = pending_search->board;
    ggs_subtract_elapsed_from_remaining_time_to_move(&search_board, tim() - pending_search->ready_time);
    ggs_launch_ai_search(
        ai_futures,
        ai_searchings,
        ggs_boards_searching,
        search_board,
        pending_search->search_slot,
        pending_search->thread_id,
        move_hints,
        ponder_results,
        synchro_search_records,
        options
    );
    pending_search->active = false;
    return true;
}

int ggs_record_ponder_results(GGS_Ponder_Result_Table *ponder_results, const std::vector<Ponder_elem> &results) {
    int n_recorded = 0;
    for (const Ponder_elem &elem: results) {
        if (
            elem.response_board_key.empty() ||
            !ggs_is_cacheable_ponder_result(elem.response)
        ) {
            continue;
        }
        (*ponder_results)[elem.response_board_key] = elem.response;
        ++n_recorded;
    }
    if (ponder_results->size() > 4096) {
        ponder_results->clear();
    }
    return n_recorded;
}

void ggs_terminate_ponder(
    std::future<std::vector<Ponder_elem>> ponder_futures[][GGS_N_PONDER_PARALLEL],
    bool ponder_searchings[],
    int synchro_id,
    GGS_Ponder_Result_Table *ponder_results,
    Options *options
);

void ggs_start_ponder(
    std::future<std::vector<Ponder_elem>> ponder_futures[][GGS_N_PONDER_PARALLEL],
    Board board,
    bool show_log,
    int synchro_id,
    bool ponder_searchings[],
    GGS_Ponder_Result_Table *ponder_results,
    Options *options
) {
    if (synchro_id < 0) {
        synchro_id = GGS_NON_SYNCHRO_ID;
    }
    if (ponder_searchings[synchro_id]) {
        ggs_terminate_ponder(ponder_futures, ponder_searchings, synchro_id, ponder_results, options);
    }
    ponder_searchings[synchro_id] = true;
    for (int ponder_i = 0; ponder_i < GGS_N_PONDER_PARALLEL; ++ponder_i) {
        ponder_futures[synchro_id][ponder_i] = std::async(std::launch::async, ai_ponder, board, show_log, synchro_id, &ponder_searchings[synchro_id]); // set ponder
    }
}

void ggs_terminate_ponder(
    std::future<std::vector<Ponder_elem>> ponder_futures[][GGS_N_PONDER_PARALLEL],
    bool ponder_searchings[],
    int synchro_id,
    GGS_Ponder_Result_Table *ponder_results,
    Options *options
) {
    if (synchro_id < 0) {
        synchro_id = GGS_NON_SYNCHRO_ID;
    }
    ponder_searchings[synchro_id] = false; // terminate ponder
    std::vector<std::vector<Ponder_elem>> results;
    int n_recorded = 0;
    for (int ponder_i = 0; ponder_i < GGS_N_PONDER_PARALLEL; ++ponder_i) {
        if (ponder_futures[synchro_id][ponder_i].valid()) {
            results.emplace_back(ponder_futures[synchro_id][ponder_i].get());
            n_recorded += ggs_record_ponder_results(ponder_results, results.back());
        }
    }
    #if !IS_GGS_TOURNAMENT
    if (n_recorded > 0 && options->show_log) {
        ggs_print_info("stored exact ponder results " + std::to_string(n_recorded), options);
    }
    #endif
#if !IS_GGS_TOURNAMENT
    for (std::vector<Ponder_elem> result: results) {
        print_ponder_result(result);
    }
#endif
}

bool ggs_has_valid_ponder_future(std::future<std::vector<Ponder_elem>> ponder_futures[][GGS_N_PONDER_PARALLEL], int synchro_id) {
    if (synchro_id < 0) {
        synchro_id = GGS_NON_SYNCHRO_ID;
    }
    for (int ponder_i = 0; ponder_i < GGS_N_PONDER_PARALLEL; ++ponder_i) {
        if (ponder_futures[synchro_id][ponder_i].valid()) {
            return true;
        }
    }
    return false;
}

void ggs_terminate_all_ponders(
    std::future<std::vector<Ponder_elem>> ponder_futures[][GGS_N_PONDER_PARALLEL],
    bool ponder_searchings[],
    GGS_Ponder_Result_Table *ponder_results,
    Options *options
) {
    for (int i = 0; i < 2; ++i) {
        if (ponder_searchings[i] || ggs_has_valid_ponder_future(ponder_futures, i)) {
            ggs_terminate_ponder(ponder_futures, ponder_searchings, i, ponder_results, options);
        }
    }
}

void ggs_terminate_ponder_if_active(
    std::future<std::vector<Ponder_elem>> ponder_futures[][GGS_N_PONDER_PARALLEL],
    bool ponder_searchings[],
    int synchro_id,
    GGS_Ponder_Result_Table *ponder_results,
    Options *options
) {
    if (synchro_id < 0 || synchro_id >= 2) {
        return;
    }
    if (ponder_searchings[synchro_id] || ggs_has_valid_ponder_future(ponder_futures, synchro_id)) {
        ggs_terminate_ponder(ponder_futures, ponder_searchings, synchro_id, ponder_results, options);
    }
}

bool ggs_any_ai_searching(const bool ai_searchings[]) {
    return ai_searchings[0] || ai_searchings[1];
}

void ggs_client(Options *options) {
    WSADATA wsaData;
    SOCKET sock = INVALID_SOCKET;
    struct sockaddr_in server;
    ggs_socket_closing.store(false);
    
    // connect to GGS server
    ggs_print_info("GGS client start user " + options->ggs_username + " host " + std::string(GGS_URL) + ":" + std::to_string(GGS_PORT), options);
#if IS_GGS_TOURNAMENT
    transposition_table_auto_reset_importance = false;
    ggs_print_info("disabled TT auto importance reset for GGS tournament", options);
#endif
    if (ggs_connect(wsaData, server, sock, options) != 0) {
        std::cout << "[ERROR] [FATAL] Failed to Connect" << std::endl;
        return;
    }
    ggs_print_info("Connected to server!", options);
    if (!ggs_receive_until_text(&sock, options, "login prompt", {"Enter login"})) {
        ggs_close(sock);
        return;
    }

    // login
    ggs_print_debug("sending username", options);
    if (ggs_send_message(sock, options->ggs_username + "\n", options) != 0 || !ggs_receive_until_text(&sock, options, "password prompt", {"Enter your password"})) {
        ggs_close(sock);
        return;
    }
    ggs_print_debug("sending password", options);
    if (ggs_send_message(sock, options->ggs_password + "\n", options, "<password>\n") != 0 || !ggs_receive_required(&sock, options, "login READY")) {
        ggs_close(sock);
        return;
    }
    ggs_print_info("GGS login completed", options);

    // initialize
    std::vector<std::string> init_replies;
    GGS_Clock_Params ggs_clock_params;
    ggs_print_debug("initializing GGS subscriptions: ms /os", options);
    if (ggs_send_message(sock, "ms /os\n", options) != 0 || !ggs_receive_required(&sock, options, "ms /os response", &init_replies)) {
        ggs_close(sock);
        return;
    }
    ggs_accept_match_requests(init_replies, sock, &ggs_clock_params, options);
    ggs_print_debug("initializing GGS subscriptions: ts client -", options);
    if (ggs_send_message(sock, "ts client -\n", options) != 0 || !ggs_receive_required(&sock, options, "ts client response", &init_replies)) {
        ggs_close(sock);
        return;
    }
    ggs_accept_match_requests(init_replies, sock, &ggs_clock_params, options);
    ggs_print_info("GGS initialization completed; entering main loop", options);
    
    std::future<std::string> user_input_f;
    std::future<std::vector<std::string>> ggs_message_f;
    std::future<Search_result> ai_futures[2];
    bool ai_searchings[2] = {false, false};
    GGS_Board ggs_boards_searching[2];
    GGS_Pending_Move pending_moves[2];
    GGS_Pending_Search pending_searches[2];
    GGS_Synchro_Search_Record synchro_search_records[2];
    std::future<std::vector<Ponder_elem>> ponder_futures[2][GGS_N_PONDER_PARALLEL];
    bool ponder_searchings[2] = {false, false};
    GGS_Board ggs_boards[2][HW2 + 1];
    int ggs_boards_n_discs[2] = {0, 0};
    GGS_Move_Hint_Table move_hints;
    GGS_Move_Hint_Table seeded_move_hints;
    GGS_Ponder_Result_Table ponder_results;
    auto stop_calculations = [&]() {
        global_searching = false;
        for (int ai_i = 0; ai_i < 2; ++ai_i) {
            if (ai_futures[ai_i].valid()) {
                Search_result discarded_search_result = ai_futures[ai_i].get();
                (void)discarded_search_result;
            }
            for (int ponder_i = 0; ponder_i < GGS_N_PONDER_PARALLEL; ++ponder_i) {
                if (ponder_futures[ai_i][ponder_i].valid()) {
                    std::vector<Ponder_elem> discarded_ponder_result = ponder_futures[ai_i][ponder_i].get();
                    (void)discarded_ponder_result;
                }
            }
        }
    };
    ggs_seed_move_hints_from_game_logs(&seeded_move_hints, options);
    ggs_seed_verified_analysis_hints(&seeded_move_hints, options);
    move_hints = seeded_move_hints;
    bool match_playing = false;
#if IS_GGS_TOURNAMENT
    bool ggs_tt_clean = true;
#endif
    int thread_sizes[2];
    int thread_sizes_before[2];
    for (int i = 0; i < 2; ++i) {
        thread_sizes[i] = 0;
    }
    bool playing_synchro_game = false;
    bool playing_same_board = true;
    GGS_Match matches[2];
    for (int i = 0; i < 2; ++i) {
        matches[i].init();
    }
    uint64_t last_sent_time = tim();
    while (true) {
        if (tim() - last_sent_time > GGS_SEND_EMPTY_INTERVAL) {
            ggs_send_message(sock, "\n", options);
            last_sent_time = tim();
        }
        // check user input
        if (user_input_f.valid()) {
            if (user_input_f.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                std::string user_input = user_input_f.get();
                if (user_input == "quit") {
                    ggs_print_info("stdin command received: quit", options);
                    ggs_print_info("quit requested; closing GGS client", options);
                    ggs_send_message(sock, "quit\n", options);
                    ggs_socket_closing.store(true);
                    shutdown(sock, SD_BOTH);
                    stop_calculations();
                    break;
                } else if (!user_input.empty()) {
                    ggs_print_info("stdin command received: " + user_input, options);
                    ggs_send_message(sock, user_input + "\n", options);
                    ggs_print_info("stdin command forwarded: " + user_input, options);
                    last_sent_time = tim();
                }
            }
        } else {
            user_input_f = std::async(std::launch::async, ggs_get_user_input);
        }
        // check server reply
        std::vector<std::string> server_replies;
        if (ggs_message_f.valid()) {
            if (ggs_message_f.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                server_replies = ggs_message_f.get();
            }
        } else {
            ggs_message_f = std::async(std::launch::async, ggs_receive_message, &sock, options); // ask ggs message
        }
        if (ggs_replies_connection_closed(server_replies)) {
            ggs_report_error("GGS connection is no longer available. Shutting down GGS client.", options);
            stop_calculations();
            break;
        }
        bool new_calculation_start = false;
        if (server_replies.size()) {
            ggs_log_server_errors(server_replies, options);
            // match start
            for (std::string server_reply: server_replies) {
                if (server_reply.size()) {
                    std::string os_info = ggs_get_os_info(server_reply);
                    // match start
                    if (ggs_is_match_start(os_info, options->ggs_username)) {
                        ggs_print_info("match start!", options);
                        match_playing = true;
                        playing_same_board = true;
                        for (int i = 0; i < 2; ++i) {
                            ai_searchings[i] = false;
                            pending_moves[i].active = false;
                            pending_searches[i].active = false;
                            synchro_search_records[i].clear();
                            ggs_boards_n_discs[i] = 0;
                            for (int n_discs = 0; n_discs <= HW2; ++n_discs) {
                                ggs_boards[i][n_discs] = GGS_Board();
                            }
                        }
                        ggs_terminate_all_ponders(ponder_futures, ponder_searchings, &ponder_results, options);
#if IS_GGS_TOURNAMENT
                        if (!ggs_tt_clean) {
                            transposition_table.init();
                            ggs_print_info("cleared stale TT at match start", options);
                        }
                        ggs_tt_clean = false;
#endif
                        move_hints = seeded_move_hints;
                        #if IS_GGS_TOURNAMENT
                        ggs_print_info("restored seeded safe opponent hints boards " + std::to_string(move_hints.size()), options);
                        #else
                        ggs_print_debug("restored seeded synchro hints boards " + std::to_string(move_hints.size()), options);
                        #endif
                        for (int i = 0; i < 2; ++i) {
                            matches[i].init();
                        }
                    }
                }
            }
            // set board info & update match
            for (std::string server_reply: server_replies) {
                if (server_reply.size()) {
                    std::string os_info = ggs_get_os_info(server_reply);
                    if (ggs_is_board_info(os_info)) {
                        GGS_Board ggs_board = ggs_get_board(server_reply);
                        if (ggs_board.is_valid()) {
                            if (ggs_board.player_black == options->ggs_username || ggs_board.player_white == options->ggs_username) { // related to me
                                ggs_apply_board_clock(&ggs_board, &ggs_clock_params, options);
                                #if !IS_GGS_TOURNAMENT
                                ggs_print_info("ggs board synchro id " + std::to_string(ggs_board.synchro_id), options);
                                #endif
                                // set board info
                                if (ggs_board.is_synchro) { // synchro game
                                    int n_discs = ggs_board.board.n_discs();
                                    ggs_boards[ggs_board.synchro_id][n_discs] = ggs_board;
                                    ggs_boards_n_discs[ggs_board.synchro_id] = n_discs;
                                    ggs_record_opponent_move_hint(&move_hints, ggs_boards, ggs_board, options);
                                    //std::cerr << "synchro game memo " << "n_discs " << ggs_board.board.n_discs() << " " << ggs_board.board.to_str() << std::endl;
                                }
                                // update match
                                int match_idx = GGS_NON_SYNCHRO_ID;
                                if (ggs_board.is_synchro) {
                                    match_idx = ggs_board.synchro_id;
                                }
                                if (matches[match_idx].is_initialized()) {
                                    matches[match_idx].game_id = ggs_board.game_id;
                                    matches[match_idx].player_black = ggs_board.player_black;
                                    matches[match_idx].player_white = ggs_board.player_white;
                                    matches[match_idx].initial_board = ggs_board.board.to_str(ggs_board.player_to_move);
                                    matches[match_idx].result_black = -99;
                                    matches[match_idx].transcript = "";
                                }
                                #if !IS_GGS_TOURNAMENT
                                ggs_print_info("match log received " + std::to_string(match_idx) + " " + std::to_string(ggs_board.last_move) + " " + idx_to_coord(ggs_board.last_move), options);
                                #endif
                                if (is_valid_policy(ggs_board.last_move)) {
                                    matches[match_idx].transcript += idx_to_coord(ggs_board.last_move);
                                    if (
                                        (ggs_board.player_to_move == BLACK && ggs_board.player_black == options->ggs_username) || 
                                        (ggs_board.player_to_move == WHITE && ggs_board.player_white == options->ggs_username)
                                    ) {
                                        #if !IS_GGS_TOURNAMENT
                                        std::cerr << "opponent moved " << idx_to_coord(ggs_board.last_move) << std::endl;
                                        #endif
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // match end
            for (std::string server_reply: server_replies) {
                if (server_reply.size()) {
                    std::string os_info = ggs_get_os_info(server_reply);
                    if (ggs_is_game_end(os_info)) {
                        std::string game_id;
                        std::string first_player;
                        int first_player_score;
                        if (ggs_parse_game_end(os_info, &game_id, &first_player, &first_player_score)) {
                            for (int i = 0; i < 2; ++i) {
                                if (!matches[i].is_initialized() && matches[i].game_id == game_id) {
                                    matches[i].result_black = matches[i].player_black == first_player ? first_player_score : -first_player_score;
                                }
                            }
                        }
                    }
                }
            }
            // match end
            for (std::string server_reply: server_replies) {
                if (server_reply.size()) {
                    std::string os_info = ggs_get_os_info(server_reply);
                    // match end
                    if (ggs_is_match_end(os_info, options->ggs_username)) {
                        ggs_print_info("match end!", options);
                        match_playing = false;
                        for (int i = 0; i < 2; ++i) {
                            ai_searchings[i] = false;
                            pending_moves[i].active = false;
                            pending_searches[i].active = false;
                            synchro_search_records[i].clear();
                        }
                        ggs_terminate_all_ponders(ponder_futures, ponder_searchings, &ponder_results, options);
                        if (options->ggs_game_log_to_file) {
                            for (int i = 0; i < 2; ++i) {
                                if (!matches[i].is_initialized()) {
                                    std::string datetime = get_current_datetime_for_file();
                                    std::string filename = options->ggs_game_log_dir + "/" + datetime + "_" + matches[i].game_id + ".txt";
                                    std::ofstream ofs(filename, std::ios::app);
                                    if (!ofs) {
                                        ggs_print_info("Can't open gamelog file " + filename, options);
                                    } else {
                                        int black_score = 0;
                                        Board board(matches[i].initial_board);
                                        int player_sgn = matches[i].initial_board[matches[i].initial_board.size() - 1] == 'X' ? 1 : -1;
                                        Flip flip;
                                        for (int j = 0; j < matches[i].transcript.size(); j += 2) {
                                            if (board.get_legal() == 0) {
                                                board.pass();
                                                player_sgn *= -1;
                                            }
                                            int coord = get_coord_from_chars(matches[i].transcript[j], matches[i].transcript[j + 1]);
                                            calc_flip(&flip, &board, coord);
                                            board.move_board(&flip);
                                            player_sgn *= -1;
                                        }
                                        if (board.is_end()) {
                                            matches[i].result_black = player_sgn * board.score_player();
                                        } else if (matches[i].result_black == -99) {
                                            matches[i].result_black = -99;
                                        }
                                        ofs << matches[i].game_id << std::endl;
                                        ofs << "black: " << matches[i].player_black << std::endl;
                                        ofs << "white: " << matches[i].player_white << std::endl;
                                        ofs << "initial board: " << matches[i].initial_board << std::endl;
                                        ofs << "transcript: " << matches[i].transcript << std::endl;
                                        ofs << "black's score: " << matches[i].result_black << std::endl;
                                        ofs.close();
                                    }
                                }
                            }
                        }
                        if (GGS_CLEAR_TT_ON_MATCH_END) {
                            transposition_table.init();
#if IS_GGS_TOURNAMENT
                            ggs_tt_clean = true;
#endif
                            ggs_print_info("cleared TT after match", options);
                        }
                    }
                }
            }
            // receive match request
            if (options->ggs_accept_request) {
                if (!match_playing) {
                    for (std::string server_reply: server_replies) {
                        if (server_reply.size()) {
                            std::string os_info = ggs_get_os_info(server_reply);
                            // match end
                            if (ggs_is_match_request(os_info, options->ggs_username)) {
                                std::string request_id = ggs_match_request_get_id(os_info);
                                GGS_Clock_Params request_clock;
                                if (ggs_match_request_get_clock(os_info, options->ggs_username, &request_clock)) {
                                    ggs_clock_params = request_clock;
                                    if (ggs_verbose_log(options)) {
                                        std::cerr << "ggs clock " << ggs_clock_summary(ggs_clock_params) << std::endl;
                                        if (ggs_clock_params.extension_msec > 0ULL && ggs_clock_params.increment_msec == 0ULL) {
                                            std::cerr << "ggs clock note: extension time is not per-move increment" << std::endl;
                                        }
                                    }
                                }
                                std::string accept_cmd = "ts accept " + request_id;
                                ggs_send_message(sock, accept_cmd + "\n", options);
                                last_sent_time = tim();
                                ggs_print_info("match request accepted " + request_id, options);
                            }
                        }
                    }
                }
            }
            // join tournament
            if (options->ggs_route_join_tournament) {
                for (std::string server_reply: server_replies) {
                    if (server_reply.size()) {
                        if (ggs_is_join_tournament_message(server_reply)) {
                            std::string join_tournament_cmd = ggs_join_tournament_get_cmd(server_reply);
                            ggs_send_message(sock, join_tournament_cmd + "\n", options);
                            last_sent_time = tim();
                            ggs_print_info("join tournament " + join_tournament_cmd, options);
                        }
                    }
                }
            }
            // board processing
            for (std::string server_reply: server_replies) {
                if (server_reply.size()) {
                    //std::cout << "see " << server_reply << std::endl;
                    std::string os_info = ggs_get_os_info(server_reply);
                    // processing board
                    if (ggs_is_board_info(os_info)) {
                        //std::cout << "getting board info" << std::endl;
                        GGS_Board ggs_board = ggs_get_board(server_reply);
                        if (ggs_board.is_valid()) {
                            if (ggs_board.player_black == options->ggs_username || ggs_board.player_white == options->ggs_username) { // related to me
                                ggs_apply_board_clock(&ggs_board, &ggs_clock_params, options);
                                bool need_to_move = 
                                    (ggs_board.player_black == options->ggs_username && ggs_board.player_to_move == BLACK) || 
                                    (ggs_board.player_white == options->ggs_username && ggs_board.player_to_move == WHITE);
                                if (ggs_board.is_synchro) { // synchro game
                                    playing_synchro_game = true;
                                    int n_discs = ggs_board.board.n_discs();
                                    if (playing_same_board && (ggs_boards[0][n_discs].board == ggs_boards[1][n_discs].board || ggs_boards_n_discs[ggs_board.synchro_id] != ggs_boards_n_discs[ggs_board.synchro_id ^ 1])) {
                                        // std::string msg = "synchro playing same board or opponent has not played " + ggs_board.board.to_str();
                                        // ggs_print_info(msg);
                                        // playing_same_board = true;
                                    } else {
                                        // std::string msg = "synchro game separated " + ggs_board.board.to_str();
                                        // ggs_print_info(msg);
                                        playing_same_board = false;
                                    }
                                    if (need_to_move) { // Egaroucid should move
#if IS_GGS_TOURNAMENT
                                        if (ggs_board.board.n_discs() <= GGS_TERMINATE_ALL_PONDERS_MAX_DISCS) {
                                            ggs_terminate_all_ponders(ponder_futures, ponder_searchings, &ponder_results, options);
                                        } else {
                                            ggs_terminate_ponder_if_active(ponder_futures, ponder_searchings, ggs_board.synchro_id, &ponder_results, options);
                                        }
#else
                                        ggs_terminate_all_ponders(ponder_futures, ponder_searchings, &ponder_results, options);
#endif
                                        if (!ggs_board.board.is_end() && !ai_searchings[ggs_board.synchro_id] && !pending_searches[ggs_board.synchro_id].active) {
#if GGS_USE_PONDER
                                            const thread_id_t search_thread_id = ggs_board.synchro_id;
#else
                                            const thread_id_t search_thread_id = THREAD_ID_NONE;
#endif
                                            if (ggs_should_delay_search_for_synchro_hint(ggs_board, move_hints, ggs_boards, ggs_boards_n_discs, options)) {
                                                ggs_store_pending_search(&pending_searches[ggs_board.synchro_id], ggs_board, ggs_board.synchro_id, search_thread_id);
                                                ggs_print_debug(
                                                    "ggs pending search wait " + ggs_board.game_id +
                                                    " max " + std::to_string(pending_searches[ggs_board.synchro_id].max_wait_msec) +
                                                    " " + ggs_board.board.to_str(),
                                                    options
                                                );
                                            } else {
                                                ggs_launch_ai_search(ai_futures, ai_searchings, ggs_boards_searching, ggs_board, ggs_board.synchro_id, search_thread_id, move_hints, ponder_results, synchro_search_records, options);
                                            }
                                            // ggs_start_ponder(ponder_futures, ggs_board.board, options->show_log, ggs_board.synchro_id, ponder_searchings);
                                            new_calculation_start = true;
                                            std::string msg = "Egaroucid thinking... " + ggs_board.game_id + " " + ggs_board.board.to_str(ggs_board.player_to_move);
                                            ggs_print_info(msg, options);
                                        }
                                    } else { // Opponent's move
#if GGS_USE_PONDER
                                        if (!ggs_any_ai_searching(ai_searchings) && !ggs_board.board.is_end()) {
                                            ggs_start_ponder(ponder_futures, ggs_board.board, options->show_log, ggs_board.synchro_id, ponder_searchings, &ponder_results, options);
                                            // ponder_futures[ggs_board.synchro_id] = std::async(std::launch::async, ai_ponder, ggs_board.board, options->show_log, ggs_board.synchro_id, &ponder_searchings[ggs_board.synchro_id]); // set ponder
                                            new_calculation_start = true;
                                            std::string msg = "Egaroucid pondering... " + ggs_board.game_id + " " + ggs_board.board.to_str(ggs_board.player_to_move);
                                            ggs_print_info(msg, options);
                                        }
#endif
                                    }
                                } else { // non-synchro game
                                    playing_synchro_game = false;
                                    if (need_to_move) { // Egaroucid should move
                                        ggs_terminate_all_ponders(ponder_futures, ponder_searchings, &ponder_results, options);
                                        if (!ggs_board.board.is_end()) {
                                            ggs_launch_ai_search(ai_futures, ai_searchings, ggs_boards_searching, ggs_board, GGS_NON_SYNCHRO_ID, THREAD_ID_NONE, move_hints, ponder_results, synchro_search_records, options);
                                            // ggs_start_ponder(ponder_futures, ggs_board.board, options->show_log, GGS_NON_SYNCHRO_ID, ponder_searchings);
                                        }
                                    } else { // Opponent's move
#if GGS_USE_PONDER
                                        if (!ggs_any_ai_searching(ai_searchings) && !ggs_board.board.is_end()) {
                                            ggs_start_ponder(ponder_futures, ggs_board.board, options->show_log, ggs_board.synchro_id, ponder_searchings, &ponder_results, options);
                                            // ponder_futures[GGS_NON_SYNCHRO_ID] = std::async(std::launch::async, ai_ponder, ggs_board.board, options->show_log, THREAD_ID_NONE, &ponder_searchings[GGS_NON_SYNCHRO_ID]); // set ponder
                                        }
#endif
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i < 2; ++i) {
            if (ggs_try_launch_pending_search(&pending_searches[i], ai_futures, ai_searchings, ggs_boards_searching, move_hints, ponder_results, synchro_search_records, options)) {
                new_calculation_start = true;
            }
        }
        // Check completed searches after processing received boards so queued synchro moves can become hints before sending.
        if (match_playing) {
            for (int i = 0; i < 2; ++i) {
                if (ai_searchings[i]) {
                    if (ai_futures[i].valid()) {
                        if (ai_futures[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                            Search_result search_result = ai_futures[i].get();
                            ai_searchings[i] = false;
                            ggs_apply_move_hint_to_search_result(ggs_boards_searching[i], move_hints, &search_result, options);
                            ggs_log_search_result_summary(
                                "search ready",
                                ggs_boards_searching[i],
                                search_result,
                                ggs_get_move_hint(move_hints, ggs_boards_searching[i].board),
                                ggs_get_move_hint_count(move_hints, ggs_boards_searching[i].board),
                                options
                            );
                            ggs_store_synchro_search_record(synchro_search_records, ggs_boards_searching[i], search_result);
                            if (ggs_should_wait_for_synchro_hint(ggs_boards_searching[i], search_result, move_hints, ggs_boards, options)) {
                                ggs_store_pending_move(&pending_moves[i], ggs_boards_searching[i], search_result);
                                ggs_print_debug(
                                    "ggs pending move wait " + ggs_boards_searching[i].game_id +
                                    " max " + std::to_string(pending_moves[i].max_wait_msec) +
                                    " " + ggs_boards_searching[i].board.to_str(),
                                    options
                                );
                            } else {
                                ggs_send_move(ggs_boards_searching[i], sock, search_result, options);
                                last_sent_time = tim();
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < 2; ++i) {
                if (ponder_searchings[i]) {
                    for (int ponder_i = 0; ponder_i < GGS_N_PONDER_PARALLEL; ++ponder_i) {
                        if (ponder_futures[i][ponder_i].valid()) {
                            if (ponder_futures[i][ponder_i].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                                std::vector<Ponder_elem> ponder_result = ponder_futures[i][ponder_i].get();
                                int n_recorded = ggs_record_ponder_results(&ponder_results, ponder_result);
                                ponder_searchings[i] = false;
                                #if !IS_GGS_TOURNAMENT
                                ggs_print_info("ponder end " + std::to_string(i), options);
                                if (n_recorded > 0) {
                                    ggs_print_info("stored exact ponder results " + std::to_string(n_recorded), options);
                                }
                                #endif
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i < 2; ++i) {
            if (ggs_try_send_pending_move(&pending_moves[i], move_hints, synchro_search_records, sock, options)) {
                last_sent_time = tim();
            }
        }
#if GGS_USE_PONDER
        // thread manager
        thread_sizes_before[0] = thread_sizes[0];
        thread_sizes_before[1] = thread_sizes[1];
        if (playing_synchro_game) {
            int full_threads = thread_pool.size();
            int full_threads_enhanced = full_threads + std::max(1, full_threads / 4);
            int reduced_threads = std::max(1, full_threads / 2);
#if IS_GGS_TOURNAMENT
            int non_prioritized_threads = full_threads >= GGS_NON_PRIORITIZED_MIN_FULL_THREADS ? GGS_NON_PRIORITIZED_THREADS : 0;
#else
            int non_prioritized_threads = full_threads >= GGS_NON_PRIORITIZED_MIN_FULL_THREADS ? GGS_NON_PRIORITIZED_THREADS : 0;
#endif
            non_prioritized_threads = std::clamp(non_prioritized_threads, 0, std::max(0, full_threads - 1));
            int prioritized_threads = full_threads - non_prioritized_threads;
            prioritized_threads = std::min(prioritized_threads, full_threads);
            // int non_prioritized_threads = 1;
            // int prioritized_threads = full_threads - 1;
            // if (playing_same_board) {
            //     if (ai_searchings[0]) { // 0 is searching
            //         if (ai_searchings[1]) { // 1 is searching
            //             // not occurs
            //         } else if (ponder_searchings[1]) { // 1 is pondering
            //             thread_sizes[0] = full_threads; // full threads for 0
            //             thread_sizes[1] = 0; // off 1's ponder
            //         } else { // 1 is waiting
            //             thread_sizes[0] = full_threads; // full threads for 0
            //             thread_sizes[1] = 0; // off 1
            //         }
            //     } else if (ponder_searchings[0]) { // 0 is pondering
            //         if (ai_searchings[1]) { // 1 is searching
            //             thread_sizes[0] = 0; // off 0's ponder
            //             thread_sizes[1] = full_threads; // full threads for 1
            //         } else if (ponder_searchings[1]) { // 1 is pondering
            //             thread_sizes[0] = reduced_threads; // half & half
            //             thread_sizes[1] = reduced_threads; // half & half
            //         } else { // 1 is waiting
            //             thread_sizes[0] = full_threads; // full threads for 0
            //             thread_sizes[1] = 0; // off 1
            //         }
            //     } else { // 0 is waiting
            //         thread_sizes[0] = 0; // off 0
            //         thread_sizes[1] = full_threads; // full threads for 1
            //     }
            // } else {
                if (ai_searchings[0]) { // 0 is searching
                    if (ai_searchings[1]) { // 1 is searching
                        // not occurs
                    } else if (ponder_searchings[1]) { // 1 is pondering
                        thread_sizes[0] = prioritized_threads;
                        thread_sizes[1] = non_prioritized_threads;
                    } else { // 1 is waiting
                        thread_sizes[0] = full_threads;
                        thread_sizes[1] = 0;
                    }
                } else if (ponder_searchings[0]) { // 0 is pondering
                    if (ai_searchings[1]) { // 1 is searching
                        thread_sizes[0] = non_prioritized_threads;
                        thread_sizes[1] = prioritized_threads;
                    } else if (ponder_searchings[1]) { // 1 is pondering
                        thread_sizes[0] = reduced_threads;
                        thread_sizes[1] = reduced_threads;
                    } else { // 1 is waiting
                        thread_sizes[0] = full_threads;
                        thread_sizes[1] = 0;
                    }
                } else { // 0 is waiting
                    thread_sizes[0] = 0; // off 0
                    thread_sizes[1] = full_threads; // full threads for 1
                }
            // }
        } else {
            thread_sizes[GGS_NON_SYNCHRO_ID] = thread_pool.size(); // full threads for non-synchro game
        }
        // update thread size
        if (thread_sizes[0] != thread_sizes_before[0]) {
            thread_pool.set_max_thread_size(0, thread_sizes[0]);
        }
        if (thread_sizes[1] != thread_sizes_before[1]) {
            thread_pool.set_max_thread_size(1, thread_sizes[1]);
        }
        if (new_calculation_start || thread_sizes[0] != thread_sizes_before[0] || thread_sizes[1] != thread_sizes_before[1]) {
            std::string msg = "thread info synchro " + std::to_string(playing_synchro_game) + " same " + std::to_string(playing_same_board) + " ai " + std::to_string(ai_searchings[0]) + " " + std::to_string(ai_searchings[1]) + " ponder " + std::to_string(ponder_searchings[0]) + " " + std::to_string(ponder_searchings[1]) + " thread size " + std::to_string(thread_sizes[0]) + " " + std::to_string(thread_sizes[1]);
            ggs_print_info(msg, options);
        }
#endif
    }

    // close connection
    ggs_socket_closing.store(true);
    if (ggs_message_f.valid()) {
        shutdown(sock, SD_BOTH);
        std::vector<std::string> discarded_replies = ggs_message_f.get();
        (void)discarded_replies;
    }
    ggs_close(sock);
}
