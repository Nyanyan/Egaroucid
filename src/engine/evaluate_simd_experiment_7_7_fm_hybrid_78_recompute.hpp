/*
    Egaroucid Project

    @file evaluate_simd_experiment_7_7_fm_hybrid_78_recompute.hpp
        Experimental phase-selective hybrid of the 7.7 FM evaluator and the
        current EGEV2 linear evaluator. Linear phases recalculate their
        feature indices from the board at each evaluation.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#pragma once

#include "search.hpp"

#ifndef EVALUATE_EXPERIMENT_7_7_FM_FAST_COMMON_HEADER
    #define EVALUATE_EXPERIMENT_7_7_FM_FAST_COMMON_HEADER "evaluate_experiment_7_7_fm_grouped_common.hpp"
    #define EVAL77_HYBRID_DEFINED_FAST_COMMON_HEADER
#endif

#define load_eval_file eval77_hybrid_fm_load_eval_file
#define load_eval_move_ordering_end_file eval77_hybrid_fm_load_eval_move_ordering_end_file
#define evaluate_init eval77_hybrid_fm_evaluate_init
#define calc_pattern eval77_hybrid_fm_calc_pattern
#define calc_pattern_move_ordering_end eval77_hybrid_fm_calc_pattern_move_ordering_end
#define mid_evaluate eval77_hybrid_fm_mid_evaluate
#define mid_evaluate_diff eval77_hybrid_fm_mid_evaluate_diff
#define mid_evaluate_linear_move_ordering eval77_hybrid_fm_mid_evaluate_linear_move_ordering
#define mid_evaluate_move_ordering_end eval77_hybrid_fm_mid_evaluate_move_ordering_end
#define calc_feature_vector eval77_hybrid_fm_calc_feature_vector
#define calc_eval_features eval77_hybrid_fm_calc_eval_features
#define eval_move eval77_hybrid_fm_eval_move
#define eval_pass eval77_hybrid_fm_eval_pass
#define eval_move_endsearch eval77_hybrid_fm_eval_move_endsearch
#define eval_pass_endsearch eval77_hybrid_fm_eval_pass_endsearch
#define eval_undo eval77_hybrid_fm_eval_undo
#define eval_undo_endsearch eval77_hybrid_fm_eval_undo_endsearch
#include "evaluate_simd_experiment_7_7_fm_fast.hpp"
#undef load_eval_file
#undef load_eval_move_ordering_end_file
#undef evaluate_init
#undef calc_pattern
#undef calc_pattern_move_ordering_end
#undef mid_evaluate
#undef mid_evaluate_diff
#undef mid_evaluate_linear_move_ordering
#undef mid_evaluate_move_ordering_end
#undef calc_feature_vector
#undef calc_eval_features
#undef eval_move
#undef eval_pass
#undef eval_move_endsearch
#undef eval_pass_endsearch
#undef eval_undo
#undef eval_undo_endsearch

#if defined(EVAL77_HYBRID_DEFINED_FAST_COMMON_HEADER)
    #undef EVALUATE_EXPERIMENT_7_7_FM_FAST_COMMON_HEADER
    #undef EVAL77_HYBRID_DEFINED_FAST_COMMON_HEADER
#endif

class Eval78HybridSearchOwner {
public:
    Board board;
    Eval78HybridSearchState eval;

    explicit Eval78HybridSearchOwner(const Board *source)
        : board(source->copy()) {}

    int phase() const {
        return std::clamp(
            (board.n_discs() - 4) / PHASE_N_DISCS,
            0,
            N_PHASES - 1
        );
    }
};

#define Eval_search Eval78HybridSearchState
#define Search Eval78HybridSearchOwner
namespace eval78_recompute {
#include "evaluate_simd.hpp"
}
#undef Eval_search
#undef Search

constexpr uint64_t EVAL77_HYBRID_ALL_PHASES_MASK =
    (uint64_t{1} << N_PHASES) - uint64_t{1};

uint64_t eval77_hybrid_fm_phase_mask = EVAL77_HYBRID_ALL_PHASES_MASK;

inline bool eval77_hybrid_uses_fm(const int phase_idx) {
    return ((eval77_hybrid_fm_phase_mask >> phase_idx) & uint64_t{1}) != 0;
}

inline bool eval77_hybrid_parse_spec(
    const char *file,
    std::string *fm_file,
    std::string *linear_file,
    uint64_t *phase_mask
) {
    const std::string spec = file == nullptr ? "" : std::string(file);
    const size_t first = spec.find('@');
    if (first == std::string::npos) {
        *fm_file = spec;
        *linear_file = EXE_DIRECTORY_PATH + "resources/eval.egev2";
        *phase_mask = EVAL77_HYBRID_ALL_PHASES_MASK;
        return true;
    }
    const size_t second = spec.find('@', first + 1);
    if (second == std::string::npos) {
        std::cerr
            << "[ERROR] hybrid evaluation specification requires "
            << "FM_FILE@LINEAR_FILE@PHASE_MASK_HEX"
            << std::endl;
        return false;
    }
    *fm_file = spec.substr(0, first);
    *linear_file = spec.substr(first + 1, second - first - 1);
    const std::string mask_text = spec.substr(second + 1);
    try {
        size_t parsed = 0;
        *phase_mask = std::stoull(mask_text, &parsed, 16);
        if (parsed != mask_text.size() ||
            (*phase_mask & ~EVAL77_HYBRID_ALL_PHASES_MASK) != 0) {
            throw std::invalid_argument("phase mask");
        }
    } catch (const std::exception &) {
        std::cerr
            << "[ERROR] hybrid phase mask must be a hexadecimal integer "
            << "whose bits 60 and above are zero"
            << std::endl;
        return false;
    }
    return true;
}

inline bool load_eval_file(const char *file, bool show_log) {
    std::string fm_file;
    std::string linear_file;
    uint64_t phase_mask = 0;
    if (!eval77_hybrid_parse_spec(
            file, &fm_file, &linear_file, &phase_mask)) {
        return false;
    }
    if (!eval77_hybrid_fm_load_eval_file(fm_file.c_str(), show_log)) {
        return false;
    }
    if (!eval78_recompute::load_eval_file(linear_file.c_str(), show_log)) {
        return false;
    }
    eval77_hybrid_fm_phase_mask = phase_mask;
    if (show_log) {
        std::cerr
            << "hybrid FM file " << fm_file
            << " linear file " << linear_file
            << " FM phase mask 0x" << std::hex << phase_mask << std::dec
            << std::endl;
    }
    return true;
}

inline bool load_eval_move_ordering_end_file(
    const char *file,
    bool show_log
) {
    if (!eval77_hybrid_fm_load_eval_move_ordering_end_file(file, show_log)) {
        return false;
    }
    return eval78_recompute::load_eval_move_ordering_end_file(file, show_log);
}

inline bool evaluate_init(
    const char *file,
    const char *mo_end_nws_file,
    bool show_log
) {
    if (!load_eval_file(file, show_log)) {
        return false;
    }
    if (!load_eval_move_ordering_end_file(mo_end_nws_file, show_log)) {
        return false;
    }
    pre_calculate_eval_constant();
    eval78_recompute::pre_calculate_eval_constant();
    return true;
}

inline bool evaluate_init(
    const std::string file,
    const std::string mo_end_nws_file,
    bool show_log
) {
    return evaluate_init(file.c_str(), mo_end_nws_file.c_str(), show_log);
}

inline bool evaluate_init(bool show_log) {
    const std::string spec =
        EXE_DIRECTORY_PATH + "resources/eval.egev4@" +
        EXE_DIRECTORY_PATH + "resources/eval.egev2@" +
        "fffffffffffffff";
    return evaluate_init(
        spec.c_str(),
        (EXE_DIRECTORY_PATH + "resources/eval_move_ordering_end.egev").c_str(),
        show_log
    );
}

inline int calc_pattern(
    const int phase_idx,
    Eval_features *features,
    const int num0
) {
    return eval77_hybrid_fm_calc_pattern(phase_idx, features, num0);
}

inline int calc_pattern_move_ordering_end(Eval_features *features) {
    return eval77_hybrid_fm_calc_pattern_move_ordering_end(features);
}

inline int eval77_hybrid_linear_score(
    const int phase_idx,
    const int num0,
    Eval_features *features
) {
    int result = eval78_recompute::calc_pattern(
        phase_idx,
        features
    ) + eval78_recompute::eval_num_arr[phase_idx][num0];
    result += result >= 0 ? STEP_2 : -STEP_2;
    result /= STEP;
    return std::clamp(result, -SCORE_MAX, SCORE_MAX);
}

inline int eval77_hybrid_phase_from_disc_count(const int n_discs) {
    return std::clamp(
        (n_discs - 4) / PHASE_N_DISCS,
        0,
        N_PHASES - 1
    );
}

inline void eval77_hybrid_recalculate_fm_features(
    Board *board,
    Eval_features *features
) {
    int board_array[HW2 + 1];
    board->translate_to_arr_player_rev(board_array);
    board_array[COORD_NO] = 0;
    for (int vector = 0; vector < N_EVAL_VECTORS; ++vector) {
        eval77_hybrid_fm_calc_feature_vector(
            features->f256[vector],
            board_array,
            vector,
            MAX_PATTERN_CELLS - 1
        );
    }
}

inline void eval77_hybrid_recalculate_linear_features(
    Board *board,
    Eval_features *features
) {
    int board_array[HW2 + 1];
    board->translate_to_arr_player_rev(board_array);
    board_array[COORD_NO] = 0;
    for (int vector = 0; vector < N_EVAL_VECTORS; ++vector) {
        eval78_recompute::calc_feature_vector(
            features->f256[vector],
            board_array,
            vector,
            eval78_recompute::MAX_N_CELLS_GROUP[vector] - 1
        );
    }
    features->f256[0] = _mm256_add_epi16(
        features->f256[0],
        eval78_recompute::eval_simd_offsets_simple[0]
    );
    features->f256[1] = _mm256_add_epi16(
        features->f256[1],
        eval78_recompute::eval_simd_offsets_simple[1]
    );
}

#if defined(EVALUATE_EXPERIMENT_7_7_FM_EGEV2_MOVE_ORDERING)
inline void eval77_egev2_move_ordering_recalculate(
    Board *board,
    Eval_search *eval,
    const int feature_idx
) {
    eval77_hybrid_recalculate_linear_features(
        board,
        &eval->eval77_egev2_move_ordering_features[feature_idx]
    );
}

inline void eval77_egev2_move_ordering_move(
    Eval_search *eval,
    const Flip *flip,
    const Board *board
) {
    const int current_idx = eval->feature_idx;
    const int next_idx = current_idx + 1;
    const uint16_t *flipped_group = (const uint16_t*)&flip->flip;
    const uint16_t *player_group = (const uint16_t*)&board->player;
    const uint16_t *opponent_group = (const uint16_t*)&board->opponent;
    __m256i next[N_EVAL_VECTORS];
    for (int vector = 0; vector < N_EVAL_VECTORS; ++vector) {
        next[vector] = _mm256_sub_epi16(
            eval->eval77_egev2_move_ordering_features
                [current_idx].f256[vector],
            eval78_recompute::coord_to_feature_simd[flip->pos][vector]
        );
    }
    for (int group = 0; group < HW2 / 16; ++group) {
        const uint16_t unflipped_player =
            (uint16_t)(~flipped_group[group] & player_group[group]);
        const uint16_t unflipped_opponent =
            (uint16_t)(~flipped_group[group] & opponent_group[group]);
        for (int vector = 0; vector < N_EVAL_VECTORS; ++vector) {
            next[vector] = _mm256_add_epi16(
                next[vector],
                eval78_recompute::eval_move_unflipped_16bit
                    [unflipped_player][group][vector]
            );
            next[vector] = _mm256_sub_epi16(
                next[vector],
                eval78_recompute::eval_move_unflipped_16bit
                    [unflipped_opponent][group][vector]
            );
        }
    }
    for (int vector = 0; vector < N_EVAL_VECTORS; ++vector) {
        eval->eval77_egev2_move_ordering_features
            [next_idx].f256[vector] = next[vector];
    }
}

inline void eval77_egev2_move_ordering_pass(
    Eval_search *eval,
    const Board *board
) {
    const int feature_idx = eval->feature_idx;
    const uint16_t *player_group = (const uint16_t*)&board->player;
    const uint16_t *opponent_group = (const uint16_t*)&board->opponent;
    for (int group = 0; group < HW2 / 16; ++group) {
        for (int vector = 0; vector < N_EVAL_VECTORS; ++vector) {
            __m256i &feature =
                eval->eval77_egev2_move_ordering_features
                    [feature_idx].f256[vector];
            feature = _mm256_add_epi16(
                feature,
                eval78_recompute::eval_move_unflipped_16bit
                    [player_group[group]][group][vector]
            );
            feature = _mm256_sub_epi16(
                feature,
                eval78_recompute::eval_move_unflipped_16bit
                    [opponent_group[group]][group][vector]
            );
        }
    }
}
#endif

inline void eval77_hybrid_recalculate_after_move(
    Eval_search *eval,
    const Flip *flip,
    const Board *board,
    const bool next_uses_fm
) {
    Board next_board = board->copy();
    next_board.move_board(flip);
    ++eval->feature_idx;
    Eval_features *next = &eval->features[eval->feature_idx];
    if (next_uses_fm) {
        eval77_hybrid_recalculate_fm_features(&next_board, next);
    } else {
        eval77_hybrid_recalculate_linear_features(&next_board, next);
    }
}

inline void eval77_hybrid_linear_move(
    Eval_search *eval,
    const Flip *flip,
    const Board *board
) {
    const uint16_t *flipped_group = (const uint16_t*)&flip->flip;
    const uint16_t *player_group = (const uint16_t*)&board->player;
    const uint16_t *opponent_group = (const uint16_t*)&board->opponent;
    __m256i next[N_EVAL_VECTORS];
    for (int vector = 0; vector < N_EVAL_VECTORS; ++vector) {
        next[vector] = _mm256_sub_epi16(
            eval->features[eval->feature_idx].f256[vector],
            eval78_recompute::coord_to_feature_simd[flip->pos][vector]
        );
    }
    for (int group = 0; group < HW2 / 16; ++group) {
        const uint16_t unflipped_player =
            (uint16_t)(~flipped_group[group] & player_group[group]);
        const uint16_t unflipped_opponent =
            (uint16_t)(~flipped_group[group] & opponent_group[group]);
        for (int vector = 0; vector < N_EVAL_VECTORS; ++vector) {
            next[vector] = _mm256_add_epi16(
                next[vector],
                eval78_recompute::eval_move_unflipped_16bit
                    [unflipped_player][group][vector]
            );
            next[vector] = _mm256_sub_epi16(
                next[vector],
                eval78_recompute::eval_move_unflipped_16bit
                    [unflipped_opponent][group][vector]
            );
        }
    }
    ++eval->feature_idx;
    for (int vector = 0; vector < N_EVAL_VECTORS; ++vector) {
        eval->features[eval->feature_idx].f256[vector] = next[vector];
    }
}

inline void eval77_hybrid_linear_move_endsearch(
    Eval_search *eval,
    const Flip *flip,
    const Board *board
) {
    const uint16_t *flipped_group = (const uint16_t*)&flip->flip;
    const uint16_t *player_group = (const uint16_t*)&board->player;
    const uint16_t *opponent_group = (const uint16_t*)&board->opponent;
    __m256i next = _mm256_sub_epi16(
        eval->features[eval->feature_idx].f256[2],
        eval78_recompute::coord_to_feature_simd[flip->pos][2]
    );
    for (int group = 0; group < HW2 / 16; ++group) {
        const uint16_t unflipped_player =
            (uint16_t)(~flipped_group[group] & player_group[group]);
        const uint16_t unflipped_opponent =
            (uint16_t)(~flipped_group[group] & opponent_group[group]);
        next = _mm256_add_epi16(
            next,
            eval78_recompute::eval_move_unflipped_16bit
                [unflipped_player][group][2]
        );
        next = _mm256_sub_epi16(
            next,
            eval78_recompute::eval_move_unflipped_16bit
                [unflipped_opponent][group][2]
        );
    }
    ++eval->feature_idx;
    eval->features[eval->feature_idx].f256[2] = next;
}

inline void eval77_hybrid_linear_pass(
    Eval_search *eval,
    const Board *board
) {
    const uint16_t *player_group = (const uint16_t*)&board->player;
    const uint16_t *opponent_group = (const uint16_t*)&board->opponent;
    for (int group = 0; group < HW2 / 16; ++group) {
        for (int vector = 0; vector < N_EVAL_VECTORS; ++vector) {
            Eval_features *current = &eval->features[eval->feature_idx];
            current->f256[vector] = _mm256_add_epi16(
                current->f256[vector],
                eval78_recompute::eval_move_unflipped_16bit
                    [player_group[group]][group][vector]
            );
            current->f256[vector] = _mm256_sub_epi16(
                current->f256[vector],
                eval78_recompute::eval_move_unflipped_16bit
                    [opponent_group[group]][group][vector]
            );
        }
    }
}

inline void eval77_hybrid_linear_pass_endsearch(
    Eval_search *eval,
    const Board *board
) {
    const uint16_t *player_group = (const uint16_t*)&board->player;
    const uint16_t *opponent_group = (const uint16_t*)&board->opponent;
    __m256i current = eval->features[eval->feature_idx].f256[2];
    for (int group = 0; group < HW2 / 16; ++group) {
        current = _mm256_add_epi16(
            current,
            eval78_recompute::eval_move_unflipped_16bit
                [player_group[group]][group][2]
        );
        current = _mm256_sub_epi16(
            current,
            eval78_recompute::eval_move_unflipped_16bit
                [opponent_group[group]][group][2]
        );
    }
    eval->features[eval->feature_idx].f256[2] = current;
}

inline int mid_evaluate(Board *board) {
    Search search(board);
    const int phase_idx = search.phase();
    const int num0 = pop_count_ull(search.board.player);
    return eval77_hybrid_uses_fm(phase_idx) ?
        eval77_hybrid_fm_calc_pattern(
            phase_idx,
            &search.eval.features[search.eval.feature_idx],
            num0
        ) :
        eval77_hybrid_linear_score(
            phase_idx,
            num0,
            &search.eval.features[search.eval.feature_idx]
        );
}

inline int mid_evaluate_diff(Search *search) {
    const int phase_idx = search->phase();
    const int num0 = pop_count_ull(search->board.player);
    return eval77_hybrid_uses_fm(phase_idx) ?
        eval77_hybrid_fm_mid_evaluate_diff(search) :
        eval77_hybrid_linear_score(
            phase_idx,
            num0,
            &search->eval.features[search->eval.feature_idx]
        );
}

#if defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_MOVE_ORDERING)
inline int mid_evaluate_linear_move_ordering(Search *search) {
    const int phase_idx = search->phase();
    const int num0 = pop_count_ull(search->board.player);
    if (eval77_hybrid_uses_fm(phase_idx)) {
        return eval77_hybrid_fm_mid_evaluate_linear_move_ordering(search);
    }
    return eval77_hybrid_linear_score(
        phase_idx,
        num0,
        &search->eval.features[search->eval.feature_idx]
    );
}
#endif

#if defined(EVALUATE_EXPERIMENT_7_7_FM_EGEV2_MOVE_ORDERING)
inline int mid_evaluate_linear_move_ordering(Search *search) {
    const int feature_idx = search->eval.feature_idx;
    return eval77_hybrid_linear_score(
        search->phase(),
        pop_count_ull(search->board.player),
        &search->eval.eval77_egev2_move_ordering_features[feature_idx]
    );
}
#endif

inline int mid_evaluate_move_ordering_end(Search *search) {
    const int phase_idx = search->phase();
    if (eval77_hybrid_uses_fm(phase_idx)) {
        return eval77_hybrid_fm_mid_evaluate_move_ordering_end(search);
    }
    int result = eval78_recompute::calc_pattern_move_ordering_end(
        &search->eval.features[search->eval.feature_idx]
    );
    result += result >= 0 ? STEP_2_MO_END : -STEP_2_MO_END;
    return result / STEP_MO_END;
}

inline void calc_eval_features(Board *board, Eval_search *eval) {
    eval->feature_idx = 0;
#if defined(EVALUATE_EXPERIMENT_7_7_FM_EGEV2_MOVE_ORDERING)
    eval77_hybrid_recalculate_fm_features(board, &eval->features[0]);
    eval77_egev2_move_ordering_recalculate(board, eval, 0);
#else
    const int phase_idx =
        eval77_hybrid_phase_from_disc_count(board->n_discs());
    if (eval77_hybrid_uses_fm(phase_idx)) {
        eval77_hybrid_recalculate_fm_features(board, &eval->features[0]);
    } else {
        eval77_hybrid_recalculate_linear_features(board, &eval->features[0]);
    }
#endif
}

inline void eval_move(
    Eval_search *eval,
    const Flip *flip,
    const Board *board
) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_EGEV2_MOVE_ORDERING)
    eval77_egev2_move_ordering_move(eval, flip, board);
    eval77_hybrid_fm_eval_move(eval, flip, board);
#else
    const int current_phase =
        eval77_hybrid_phase_from_disc_count(board->n_discs());
    const int next_phase =
        eval77_hybrid_phase_from_disc_count(board->n_discs() + 1);
    const bool current_uses_fm = eval77_hybrid_uses_fm(current_phase);
    const bool next_uses_fm = eval77_hybrid_uses_fm(next_phase);
    if (current_uses_fm != next_uses_fm) {
        eval77_hybrid_recalculate_after_move(
            eval,
            flip,
            board,
            next_uses_fm
        );
    } else if (next_uses_fm) {
        eval77_hybrid_fm_eval_move(eval, flip, board);
    } else {
        eval77_hybrid_linear_move(eval, flip, board);
    }
#endif
}

inline void eval_undo(Eval_search *eval) {
    --eval->feature_idx;
}

inline void eval_pass(Eval_search *eval, const Board *board) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_EGEV2_MOVE_ORDERING)
    eval77_egev2_move_ordering_pass(eval, board);
    eval77_hybrid_fm_eval_pass(eval, board);
#else
    const int phase_idx =
        eval77_hybrid_phase_from_disc_count(board->n_discs());
    if (eval77_hybrid_uses_fm(phase_idx)) {
        eval77_hybrid_fm_eval_pass(eval, board);
    } else {
        eval77_hybrid_linear_pass(eval, board);
    }
#endif
}

inline void eval_move_endsearch(
    Eval_search *eval,
    const Flip *flip,
    const Board *board
) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_EGEV2_MOVE_ORDERING)
    eval77_hybrid_fm_eval_move_endsearch(eval, flip, board);
#else
    const int current_phase =
        eval77_hybrid_phase_from_disc_count(board->n_discs());
    const int next_phase =
        eval77_hybrid_phase_from_disc_count(board->n_discs() + 1);
    const bool current_uses_fm = eval77_hybrid_uses_fm(current_phase);
    const bool next_uses_fm = eval77_hybrid_uses_fm(next_phase);
    if (current_uses_fm != next_uses_fm) {
        eval77_hybrid_recalculate_after_move(
            eval,
            flip,
            board,
            next_uses_fm
        );
    } else if (next_uses_fm) {
        eval77_hybrid_fm_eval_move_endsearch(eval, flip, board);
    } else {
        eval77_hybrid_linear_move_endsearch(eval, flip, board);
    }
#endif
}

inline void eval_undo_endsearch(Eval_search *eval) {
    --eval->feature_idx;
}

inline void eval_pass_endsearch(Eval_search *eval, const Board *board) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_EGEV2_MOVE_ORDERING)
    eval77_hybrid_fm_eval_pass_endsearch(eval, board);
#else
    const int phase_idx =
        eval77_hybrid_phase_from_disc_count(board->n_discs());
    if (eval77_hybrid_uses_fm(phase_idx)) {
        eval77_hybrid_fm_eval_pass_endsearch(eval, board);
    } else {
        eval77_hybrid_linear_pass_endsearch(eval, board);
    }
#endif
}
