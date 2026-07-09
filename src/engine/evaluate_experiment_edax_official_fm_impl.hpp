/*
    Egaroucid Project

    @file evaluate_experiment_edax_official_fm_impl.hpp
        Isolated official-Edax linear evaluation plus FM interaction experiment
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#pragma once

#define load_eval_file edax_official_fm_load_linear_eval_file
#define load_eval_move_ordering_end_file edax_official_fm_load_linear_move_ordering_end_file
#define evaluate_init edax_official_fm_linear_evaluate_init
#define calc_pattern_generic edax_official_fm_linear_calc_pattern_generic
#define calc_pattern edax_official_fm_linear_calc_pattern
#define calc_pattern_move_ordering_end edax_official_fm_linear_calc_pattern_move_ordering_end
#define mid_evaluate edax_official_fm_linear_mid_evaluate
#define mid_evaluate_diff edax_official_fm_linear_mid_evaluate_diff
#define mid_evaluate_move_ordering_end edax_official_fm_linear_mid_evaluate_move_ordering_end
#define calc_eval_features edax_official_fm_linear_calc_eval_features
#define eval_move edax_official_fm_linear_eval_move
#define eval_pass edax_official_fm_linear_eval_pass
#define eval_move_endsearch edax_official_fm_linear_eval_move_endsearch
#define eval_pass_endsearch edax_official_fm_linear_eval_pass_endsearch
#define eval_undo edax_official_fm_linear_eval_undo
#define eval_undo_endsearch edax_official_fm_linear_eval_undo_endsearch
#include "evaluate_experiment_edax_official_impl.hpp"
#undef load_eval_file
#undef load_eval_move_ordering_end_file
#undef evaluate_init
#undef calc_pattern_generic
#undef calc_pattern
#undef calc_pattern_move_ordering_end
#undef mid_evaluate
#undef mid_evaluate_diff
#undef mid_evaluate_move_ordering_end
#undef calc_eval_features
#undef eval_move
#undef eval_pass
#undef eval_move_endsearch
#undef eval_pass_endsearch
#undef eval_undo
#undef eval_undo_endsearch

#include "evaluate_experiment_edax_official_fm_common.hpp"

inline void edax_official_fm_split_eval_spec(const char *file, std::string *linear_file, std::string *fm_file) {
    const std::string spec = file == nullptr ? "" : std::string(file);
    const size_t sep = spec.find('@');
    if (sep == std::string::npos) {
        *linear_file = spec;
        fm_file->clear();
        return;
    }
    *linear_file = spec.substr(0, sep);
    *fm_file = spec.substr(sep + 1);
}

inline bool load_eval_file(const char* file, bool show_log) {
    std::string linear_file;
    std::string fm_file;
    edax_official_fm_split_eval_spec(file, &linear_file, &fm_file);
    if (show_log) {
        std::cerr << "Edax official + FM linear request " << linear_file
                  << " FM request " << (fm_file.empty() ? "(zero vectors)" : fm_file) << std::endl;
    }
    if (!edax_official_fm_load_linear_eval_file(linear_file.c_str(), show_log)) {
        return false;
    }
    return edax_official_fm_load_egev4(fm_file.c_str(), show_log);
}

inline bool load_eval_move_ordering_end_file(const char *file, bool show_log) {
    return edax_official_fm_load_linear_move_ordering_end_file(file, show_log);
}

inline bool evaluate_init(const char* file, const char* mo_end_nws_file, bool show_log) {
    if (!load_eval_file(file, show_log)) {
        std::cerr << "[ERROR] [FATAL] Edax official + FM evaluation file not loaded" << std::endl;
        return false;
    }
    if (!load_eval_move_ordering_end_file(mo_end_nws_file, show_log)) {
        return false;
    }
    build_coord_to_feature();
    if (show_log) {
        std::cerr << "Edax official + FM experiment evaluation function initialized" << std::endl;
    }
    return true;
}

bool evaluate_init(const std::string file, std::string mo_end_nws_file, bool show_log) {
    return evaluate_init(file.c_str(), mo_end_nws_file.c_str(), show_log);
}

bool evaluate_init(bool show_log) {
    return evaluate_init(EXE_DIRECTORY_PATH + "resources/eval_edax.dat", EXE_DIRECTORY_PATH + "resources/eval_move_ordering_end.egev", show_log);
}

inline int edax_official_fm_active_id(const int feature, const int feature_value) {
    return edax_official_weight_base[feature] + edax_official_feature_offset[feature] + feature_value;
}

#if !USE_SIMD
inline int calc_pattern_generic(const int phase_idx, const Eval_search *eval) {
    const int phase = edax_official_normalize_phase(phase_idx);
    const uint16_t *values = eval->features[eval->feature_idx];
    const bool reversed = eval->reversed[eval->feature_idx] != 0;
    const int linear_raw = edax_official_calc_raw(phase, values, reversed);
    int active_ids[N_PATTERN_FEATURES];
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        const int feature_value = reversed
            ? edax_official_swap_player_idx(values[i], pattern_sizes[feature_to_pattern[i]])
            : values[i];
        active_ids[i] = edax_official_fm_active_id(i, feature_value);
    }
    return edax_official_fm_round_score(linear_raw, edax_official_fm_interaction_from_ids(phase, active_ids, N_PATTERN_FEATURES));
}
#endif

#if USE_SIMD
inline int calc_pattern(const int phase_idx, Eval_features *features) {
    const int phase = edax_official_normalize_phase(phase_idx);
    uint16_t values[EDAX_OFFICIAL_N_FEATURES_CEIL];
    unpack_simd_features(features, values);
    const int linear_raw = edax_official_calc_raw(phase, 0, values);
    int active_ids[N_PATTERN_FEATURES];
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        active_ids[i] = edax_official_fm_active_id(i, values[i]);
    }
    return edax_official_fm_round_score(linear_raw, edax_official_fm_interaction_from_ids(phase, active_ids, N_PATTERN_FEATURES));
}
#else
inline int calc_pattern(const int phase_idx, Eval_search *eval) {
    return calc_pattern_generic(phase_idx, eval);
}
#endif

inline int calc_pattern_move_ordering_end(
#if USE_SIMD
    Eval_features*
#else
    Eval_search*
#endif
) {
    return 0;
}

inline int mid_evaluate(Board *board) {
    Search search(board);
    edax_official_fm_linear_calc_eval_features(&(search.board), &(search.eval));
    int phase_idx = search.phase();
#if USE_SIMD
    return calc_pattern(phase_idx, &search.eval.features[search.eval.feature_idx]);
#else
    return calc_pattern(phase_idx, &search.eval);
#endif
}

inline int mid_evaluate_diff(Search *search) {
    int phase_idx = search->phase();
#if USE_SIMD
    return calc_pattern(phase_idx, &search->eval.features[search->eval.feature_idx]);
#else
    return calc_pattern(phase_idx, &search->eval);
#endif
}

inline int mid_evaluate_move_ordering_end(Search*) {
    return 0;
}

inline void calc_eval_features(Board *board, Eval_search *eval) {
    edax_official_fm_linear_calc_eval_features(board, eval);
}

#if USE_SIMD
inline void eval_move(Eval_search *eval, const Flip *flip, const Board *board) {
    edax_official_fm_linear_eval_move(eval, flip, board);
}

inline void eval_pass(Eval_search *eval, const Board *board) {
    edax_official_fm_linear_eval_pass(eval, board);
}

inline void eval_move_endsearch(Eval_search *eval, const Flip *flip, const Board *board) {
    edax_official_fm_linear_eval_move_endsearch(eval, flip, board);
}

inline void eval_pass_endsearch(Eval_search *eval, const Board *board) {
    edax_official_fm_linear_eval_pass_endsearch(eval, board);
}
#else
inline void eval_move(Eval_search *eval, const Flip *flip) {
    edax_official_fm_linear_eval_move(eval, flip);
}

inline void eval_pass(Eval_search *eval) {
    edax_official_fm_linear_eval_pass(eval);
}

inline void eval_move_endsearch(Eval_search *eval, const Flip *flip) {
    edax_official_fm_linear_eval_move_endsearch(eval, flip);
}

inline void eval_pass_endsearch(Eval_search *eval) {
    edax_official_fm_linear_eval_pass_endsearch(eval);
}
#endif

inline void eval_undo(Eval_search *eval) {
    edax_official_fm_linear_eval_undo(eval);
}

inline void eval_undo_endsearch(Eval_search *eval) {
    edax_official_fm_linear_eval_undo_endsearch(eval);
}
