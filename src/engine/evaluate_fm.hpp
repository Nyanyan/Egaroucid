/*
    Egaroucid Project

    @file evaluate_fm.hpp
        Factorization Machine extension for evaluation function
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

constexpr char EVAL_FM_FILE_MAGIC[8] = {'E', 'G', 'F', 'M', '0', '0', '1', '\0'};
constexpr uint32_t EVAL_FM_FILE_VERSION = 1;
constexpr int EVAL_FM_MAX_DIM = 64;
constexpr uint32_t EVAL_FM_FLAG_PHASE_RANGE = 0x80000000U;
constexpr uint32_t EVAL_FM_PHASE_START_SHIFT = 16;
constexpr uint32_t EVAL_FM_PHASE_END_SHIFT = 22;
constexpr uint32_t EVAL_FM_PHASE_FLAG_MASK = 0x3FU;

bool eval_fm_enabled = false;
bool eval_fm_has_phase_range = false;
uint32_t eval_fm_n_phases = 0;
uint32_t eval_fm_dim = 0;
int32_t eval_fm_scale = 1;
int64_t eval_fm_score_denom = 2;
int32_t eval_fm_score_denom_shift = 1;
uint64_t eval_fm_total_vectors = 0;
std::array<uint8_t, N_PHASES> eval_fm_phase_table;
std::array<uint8_t, N_PHASES> eval_fm_phase_enabled;
std::array<uint64_t, N_PHASES> eval_fm_phase_vector_offsets;
uint32_t eval_fm_active_pattern_mask = 0;
std::vector<int8_t> eval_fm_vectors;
std::vector<uint16_t> eval_fm_vectors_dim2_packed;
std::array<int, N_PATTERN_FEATURES> eval_fm_feature_offsets;
std::array<uint8_t, N_PATTERN_FEATURES> eval_fm_active_feature_indices;
std::array<uint8_t, N_PATTERN_FEATURES> eval_fm_active_lane_indices;
std::array<uint16_t, N_PATTERN_FEATURES> eval_fm_active_pack_offsets;
std::array<uint16_t, N_PATTERN_FEATURES> eval_fm_active_feature_limits;
std::array<uint32_t, N_PATTERN_FEATURES> eval_fm_active_vector_offsets;
std::array<uint32_t, N_PATTERN_FEATURES> eval_fm_active_lane_to_vector_offsets;
uint32_t eval_fm_n_active_features = 0;
uint32_t eval_fm_active_feature_vector_mask = 0;

template<typename T>
inline bool eval_fm_read_scalar(FILE *fp, T *v) {
    return fread(v, sizeof(T), 1, fp) == 1;
}

inline void eval_fm_disable() {
    eval_fm_enabled = false;
    eval_fm_has_phase_range = false;
    eval_fm_n_phases = 0;
    eval_fm_dim = 0;
    eval_fm_scale = 1;
    eval_fm_score_denom = 2;
    eval_fm_score_denom_shift = 1;
    eval_fm_total_vectors = 0;
    eval_fm_phase_table.fill(0);
    eval_fm_phase_enabled.fill(0);
    eval_fm_phase_vector_offsets.fill(0);
    eval_fm_active_pattern_mask = 0;
    eval_fm_n_active_features = 0;
    eval_fm_active_feature_vector_mask = 0;
    eval_fm_vectors.clear();
    eval_fm_vectors_dim2_packed.clear();
}

inline void eval_fm_init_feature_offsets() {
    int offset = 0;
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        eval_fm_feature_offsets[i] = offset;
        offset += pow3[pattern_sizes[i >> 2]];
    }
    eval_fm_total_vectors = (uint64_t)offset;
}

inline uint32_t eval_fm_phase(const int phase_idx) {
    return eval_fm_phase_table[phase_idx];
}

inline uint64_t eval_fm_phase_vector_offset(const int phase_idx) {
    return eval_fm_phase_vector_offsets[phase_idx];
}

inline int32_t eval_fm_power_of_two_shift(const int64_t v) {
    if (v <= 0 || (v & (v - 1)) != 0) {
        return -1;
    }
    int32_t shift = 0;
    int64_t x = v;
    while (x > 1) {
        x >>= 1;
        ++shift;
    }
    return shift;
}

inline bool eval_fm_feature_active(const int feature_idx) {
    return eval_fm_active_pattern_mask == 0 || (eval_fm_active_pattern_mask & (1U << (feature_idx >> 2))) != 0;
}

inline bool eval_fm_parse_phase_range(const uint32_t flags, uint32_t *start_phase, uint32_t *end_phase) {
    if ((flags & EVAL_FM_FLAG_PHASE_RANGE) == 0) {
        *start_phase = 0;
        *end_phase = N_PHASES - 1;
        return true;
    }
    *start_phase = (flags >> EVAL_FM_PHASE_START_SHIFT) & EVAL_FM_PHASE_FLAG_MASK;
    *end_phase = (flags >> EVAL_FM_PHASE_END_SHIFT) & EVAL_FM_PHASE_FLAG_MASK;
    return *start_phase <= *end_phase && *end_phase < N_PHASES;
}

inline void eval_fm_init_active_features() {
    eval_fm_n_active_features = 0;
    eval_fm_active_feature_vector_mask = 0;
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        if (eval_fm_feature_active(i)) {
            const uint32_t active_idx = eval_fm_n_active_features++;
            eval_fm_active_feature_vector_mask |= 1U << (i / 16);
            eval_fm_active_feature_indices[active_idx] = (uint8_t)i;
            eval_fm_active_lane_indices[active_idx] = (uint8_t)((i / 16) * 16 + (15 - (i & 15)));
            eval_fm_active_pack_offsets[active_idx] = (uint16_t)(i < 32 ? pattern_starts[i >> 2] : 0);
            eval_fm_active_feature_limits[active_idx] = (uint16_t)pow3[pattern_sizes[i >> 2]];
            eval_fm_active_vector_offsets[active_idx] = (uint32_t)eval_fm_feature_offsets[i];
            eval_fm_active_lane_to_vector_offsets[active_idx] =
                eval_fm_active_vector_offsets[active_idx] - eval_fm_active_pack_offsets[active_idx];
        }
    }
}

inline bool load_eval_fm_file(
    const char *file,
    bool show_log,
    std::vector<int16_t> *linear_params,
    bool *failed,
    bool *handled
) {
    *failed = false;
    *handled = false;
    eval_fm_disable();

    FILE *fp;
    if (!file_open(&fp, file, "rb")) {
        std::cerr << "[ERROR] [FATAL] can't open eval " << file << std::endl;
        *failed = true;
        return false;
    }

    char magic[8];
    if (fread(magic, 1, sizeof(magic), fp) < sizeof(magic)) {
        fclose(fp);
        *failed = true;
        return false;
    }
    if (std::memcmp(magic, EVAL_FM_FILE_MAGIC, sizeof(magic)) != 0) {
        fclose(fp);
        return false;
    }
    *handled = true;

    uint32_t version = 0;
    uint32_t n_phases = 0;
    uint32_t linear_params_per_phase = 0;
    uint32_t n_fm_phases = 0;
    uint32_t n_features = 0;
    uint32_t fm_dim = 0;
    int32_t fm_scale = 0;
    uint32_t flags = 0;
    uint64_t linear_count = 0;
    uint64_t fm_count = 0;

    bool ok =
        eval_fm_read_scalar(fp, &version) &&
        eval_fm_read_scalar(fp, &n_phases) &&
        eval_fm_read_scalar(fp, &linear_params_per_phase) &&
        eval_fm_read_scalar(fp, &n_fm_phases) &&
        eval_fm_read_scalar(fp, &n_features) &&
        eval_fm_read_scalar(fp, &fm_dim) &&
        eval_fm_read_scalar(fp, &fm_scale) &&
        eval_fm_read_scalar(fp, &flags) &&
        eval_fm_read_scalar(fp, &linear_count) &&
        eval_fm_read_scalar(fp, &fm_count);

    const uint32_t expected_linear_params_per_phase = N_PATTERN_PARAMS_RAW + MAX_STONE_NUM;
    const uint64_t expected_linear_count = (uint64_t)N_PHASES * expected_linear_params_per_phase;
    eval_fm_init_feature_offsets();
    const uint64_t expected_fm_count = (uint64_t)n_fm_phases * eval_fm_total_vectors * fm_dim;
    uint32_t apply_start_phase = 0;
    uint32_t apply_end_phase = N_PHASES - 1;
    const bool phase_range_ok = eval_fm_parse_phase_range(flags, &apply_start_phase, &apply_end_phase);

    ok = ok &&
        version == EVAL_FM_FILE_VERSION &&
        n_phases == N_PHASES &&
        linear_params_per_phase == expected_linear_params_per_phase &&
        n_fm_phases > 0 &&
        n_features == N_PATTERN_FEATURES &&
        fm_dim > 0 &&
        fm_dim <= EVAL_FM_MAX_DIM &&
        fm_scale > 0 &&
        linear_count == expected_linear_count &&
        fm_count == expected_fm_count &&
        phase_range_ok;

    if (!ok) {
        std::cerr << "[ERROR] [FATAL] evaluation FM file header invalid: " << file << std::endl;
        fclose(fp);
        *failed = true;
        return true;
    }

    linear_params->resize((size_t)linear_count);
    if (fread(linear_params->data(), sizeof(int16_t), (size_t)linear_count, fp) < linear_count) {
        std::cerr << "[ERROR] [FATAL] evaluation FM file linear payload broken: " << file << std::endl;
        fclose(fp);
        *failed = true;
        return true;
    }
    eval_fm_vectors.resize((size_t)fm_count);
    if (fread(eval_fm_vectors.data(), sizeof(int8_t), (size_t)fm_count, fp) < fm_count) {
        std::cerr << "[ERROR] [FATAL] evaluation FM file FM payload broken: " << file << std::endl;
        fclose(fp);
        *failed = true;
        return true;
    }
    fclose(fp);

    if (fm_dim == 2) {
        const size_t n_rows = (size_t)(fm_count / 2);
        eval_fm_vectors_dim2_packed.resize(n_rows);
        for (size_t i = 0; i < n_rows; ++i) {
            const uint16_t x0 = (uint16_t)(uint8_t)eval_fm_vectors[i * 2];
            const uint16_t x1 = (uint16_t)(uint8_t)eval_fm_vectors[i * 2 + 1];
            eval_fm_vectors_dim2_packed[i] = (uint16_t)(x0 | (x1 << 8));
        }
    }

    eval_fm_enabled = true;
    eval_fm_has_phase_range = (flags & EVAL_FM_FLAG_PHASE_RANGE) != 0;
    eval_fm_n_phases = n_fm_phases;
    eval_fm_dim = fm_dim;
    eval_fm_scale = fm_scale;
    eval_fm_score_denom = 2LL * eval_fm_scale * eval_fm_scale;
    eval_fm_score_denom_shift = eval_fm_power_of_two_shift(eval_fm_score_denom);
    for (int phase = 0; phase < N_PHASES; ++phase) {
        eval_fm_phase_table[phase] = (uint8_t)std::min<uint32_t>(
            eval_fm_n_phases - 1,
            (uint32_t)((phase * (int)eval_fm_n_phases) / N_PHASES)
        );
        eval_fm_phase_enabled[phase] = apply_start_phase <= (uint32_t)phase && (uint32_t)phase <= apply_end_phase;
        eval_fm_phase_vector_offsets[phase] = (uint64_t)eval_fm_phase_table[phase] * eval_fm_total_vectors * eval_fm_dim;
    }
    eval_fm_active_pattern_mask = flags & 0xFFFFU;
    eval_fm_init_active_features();
    if (show_log) {
        std::cerr << "FM evaluation file " << file
                  << " linear_params " << linear_count
                  << " fm_phases " << eval_fm_n_phases
                  << " dim " << eval_fm_dim
                  << " scale " << eval_fm_scale
                  << " apply_phase_range " << apply_start_phase << "-" << apply_end_phase
                  << " fm_pattern_features ";
        if (eval_fm_active_pattern_mask == 0) {
            std::cerr << "all";
        } else {
            std::cerr << "subset_flags 0x" << std::hex << eval_fm_active_pattern_mask << std::dec;
        }
        std::cerr << " flags " << flags << std::endl;
    }
    return true;
}

inline int eval_fm_finalize_score(const int64_t sum_square_minus_square_sum) {
    if (eval_fm_score_denom_shift >= 0) {
        const int64_t half = 1LL << (eval_fm_score_denom_shift - 1);
        if (sum_square_minus_square_sum >= 0) {
            return (int)((sum_square_minus_square_sum + half) >> eval_fm_score_denom_shift);
        }
        return -(int)((-sum_square_minus_square_sum + half) >> eval_fm_score_denom_shift);
    }
    if (sum_square_minus_square_sum >= 0) {
        return (int)((sum_square_minus_square_sum + eval_fm_score_denom / 2) / eval_fm_score_denom);
    }
    return -(int)((-sum_square_minus_square_sum + eval_fm_score_denom / 2) / eval_fm_score_denom);
}

#if USE_SIMD_EVALUATION
inline int eval_fm_hsum_epi32(const __m256i x) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_cvtsi128_si32(sum128) + _mm_extract_epi32(sum128, 1);
}

inline int eval_fm_calc_dim8_avx2(const int phase_idx, const uint16_t active_raw_features[N_PATTERN_FEATURES]) {
    __m256i sum_acc = _mm256_setzero_si256();
    __m256i sq_sum_acc = _mm256_setzero_si256();
    const uint64_t phase_offset = eval_fm_phase_vector_offset(phase_idx);
    const int8_t *base = eval_fm_vectors.data() + phase_offset;
    for (uint32_t j = 0; j < eval_fm_n_active_features; j += 2) {
        const uint16_t raw0 = active_raw_features[j];
        const uint16_t raw1 = active_raw_features[j + 1];
        if (raw0 >= eval_fm_active_feature_limits[j] || raw1 >= eval_fm_active_feature_limits[j + 1]) {
            return 0;
        }
        const int8_t *row0 = base + (uint64_t)(eval_fm_active_vector_offsets[j] + raw0) * 8;
        const int8_t *row1 = base + (uint64_t)(eval_fm_active_vector_offsets[j + 1] + raw1) * 8;
        const __m128i r0 = _mm_loadl_epi64((const __m128i*)row0);
        const __m128i r1 = _mm_loadl_epi64((const __m128i*)row1);
        const __m128i combined = _mm_unpacklo_epi64(r0, r1);
        const __m256i v16 = _mm256_cvtepi8_epi16(combined);
        sum_acc = _mm256_add_epi16(sum_acc, v16);
        sq_sum_acc = _mm256_add_epi32(sq_sum_acc, _mm256_madd_epi16(v16, v16));
    }

    const __m128i sum_lo = _mm256_castsi256_si128(sum_acc);
    const __m128i sum_hi = _mm256_extracti128_si256(sum_acc, 1);
    const __m128i folded16 = _mm_add_epi16(sum_lo, sum_hi);
    const __m256i folded32 = _mm256_cvtepi16_epi32(folded16);
    const int32_t sum_square = eval_fm_hsum_epi32(_mm256_mullo_epi32(folded32, folded32));
    const int32_t square_sum = eval_fm_hsum_epi32(sq_sum_acc);
    return eval_fm_finalize_score((int64_t)sum_square - square_sum);
}
#endif

inline int eval_fm_calc_dim1_unrolled(const int phase_idx, const uint16_t active_raw_features[N_PATTERN_FEATURES]) {
    int32_t sum0 = 0;
    int32_t square_sum0 = 0;
    const uint64_t phase_offset = eval_fm_phase_vector_offset(phase_idx);
    const int8_t *base = eval_fm_vectors.data() + phase_offset;
    for (uint32_t j = 0; j < eval_fm_n_active_features; ++j) {
        const uint16_t raw = active_raw_features[j];
        if (raw >= eval_fm_active_feature_limits[j]) {
            return 0;
        }
        const int32_t x0 = base[eval_fm_active_vector_offsets[j] + raw];
        sum0 += x0;
        square_sum0 += x0 * x0;
    }
    return eval_fm_finalize_score((int64_t)sum0 * sum0 - square_sum0);
}

inline int eval_fm_calc_dim1_from_active_lanes(const int phase_idx, const uint16_t lanes[N_PATTERN_FEATURES]) {
    int32_t sum0 = 0;
    int32_t square_sum0 = 0;
    const int8_t *base = eval_fm_vectors.data() + eval_fm_phase_vector_offset(phase_idx);
    for (uint32_t j = 0; j < eval_fm_n_active_features; ++j) {
        const uint16_t lane = lanes[eval_fm_active_lane_indices[j]];
#ifndef NDEBUG
        const uint16_t raw = lane - eval_fm_active_pack_offsets[j];
        if (raw >= eval_fm_active_feature_limits[j]) {
            return 0;
        }
#endif
        const int32_t x0 = base[eval_fm_active_lane_to_vector_offsets[j] + lane];
        sum0 += x0;
        square_sum0 += x0 * x0;
    }
    return eval_fm_finalize_score((int64_t)sum0 * sum0 - square_sum0);
}

template<uint32_t J>
inline bool eval_fm_dim2_accumulate_lane_value(
    const uint16_t lane,
    const uint16_t *base,
    int32_t &sum0,
    int32_t &sum1,
    int32_t &square_sum0,
    int32_t &square_sum1
) {
#ifndef NDEBUG
    const uint16_t raw = lane - eval_fm_active_pack_offsets[J];
    if (raw >= eval_fm_active_feature_limits[J]) {
        return false;
    }
#endif
    const uint16_t packed = base[eval_fm_active_lane_to_vector_offsets[J] + lane];
    const int32_t x0 = (int8_t)(packed & 0xFFU);
    const int32_t x1 = (int8_t)(packed >> 8);
    sum0 += x0;
    sum1 += x1;
    square_sum0 += x0 * x0;
    square_sum1 += x1 * x1;
    return true;
}

template<uint32_t J>
inline bool eval_fm_dim2_accumulate_lane(
    const uint16_t lanes[N_PATTERN_FEATURES],
    const uint16_t *base,
    int32_t &sum0,
    int32_t &sum1,
    int32_t &square_sum0,
    int32_t &square_sum1
) {
    return eval_fm_dim2_accumulate_lane_value<J>(
        lanes[eval_fm_active_lane_indices[J]],
        base,
        sum0,
        sum1,
        square_sum0,
        square_sum1
    );
}

inline int eval_fm_calc_dim2_from_active_lanes(const int phase_idx, const uint16_t lanes[N_PATTERN_FEATURES]) {
    int32_t sum0 = 0;
    int32_t sum1 = 0;
    int32_t square_sum0 = 0;
    int32_t square_sum1 = 0;
    const uint16_t *base = eval_fm_vectors_dim2_packed.data() + (eval_fm_phase_vector_offset(phase_idx) >> 1);
    if (eval_fm_n_active_features == 12) {
        if (!eval_fm_dim2_accumulate_lane<0>(lanes, base, sum0, sum1, square_sum0, square_sum1)) return 0;
        if (!eval_fm_dim2_accumulate_lane<1>(lanes, base, sum0, sum1, square_sum0, square_sum1)) return 0;
        if (!eval_fm_dim2_accumulate_lane<2>(lanes, base, sum0, sum1, square_sum0, square_sum1)) return 0;
        if (!eval_fm_dim2_accumulate_lane<3>(lanes, base, sum0, sum1, square_sum0, square_sum1)) return 0;
        if (!eval_fm_dim2_accumulate_lane<4>(lanes, base, sum0, sum1, square_sum0, square_sum1)) return 0;
        if (!eval_fm_dim2_accumulate_lane<5>(lanes, base, sum0, sum1, square_sum0, square_sum1)) return 0;
        if (!eval_fm_dim2_accumulate_lane<6>(lanes, base, sum0, sum1, square_sum0, square_sum1)) return 0;
        if (!eval_fm_dim2_accumulate_lane<7>(lanes, base, sum0, sum1, square_sum0, square_sum1)) return 0;
        if (!eval_fm_dim2_accumulate_lane<8>(lanes, base, sum0, sum1, square_sum0, square_sum1)) return 0;
        if (!eval_fm_dim2_accumulate_lane<9>(lanes, base, sum0, sum1, square_sum0, square_sum1)) return 0;
        if (!eval_fm_dim2_accumulate_lane<10>(lanes, base, sum0, sum1, square_sum0, square_sum1)) return 0;
        if (!eval_fm_dim2_accumulate_lane<11>(lanes, base, sum0, sum1, square_sum0, square_sum1)) return 0;
        const int64_t diff = (int64_t)sum0 * sum0 + (int64_t)sum1 * sum1 - square_sum0 - square_sum1;
        return eval_fm_finalize_score(diff);
    }
    for (uint32_t j = 0; j < eval_fm_n_active_features; ++j) {
        const uint16_t lane = lanes[eval_fm_active_lane_indices[j]];
#ifndef NDEBUG
        const uint16_t raw = lane - eval_fm_active_pack_offsets[j];
        if (raw >= eval_fm_active_feature_limits[j]) {
            return 0;
        }
#endif
        const uint16_t packed = base[eval_fm_active_lane_to_vector_offsets[j] + lane];
        const int32_t x0 = (int8_t)(packed & 0xFFU);
        const int32_t x1 = (int8_t)(packed >> 8);
        sum0 += x0;
        sum1 += x1;
        square_sum0 += x0 * x0;
        square_sum1 += x1 * x1;
    }
    const int64_t diff = (int64_t)sum0 * sum0 + (int64_t)sum1 * sum1 - square_sum0 - square_sum1;
    return eval_fm_finalize_score(diff);
}

#if !USE_SIMD_EVALUATION
inline int eval_fm_calc_dim1_from_eval_search(const int phase_idx, Eval_search *eval) {
    int32_t sum0 = 0;
    int32_t square_sum0 = 0;
    const int8_t *base = eval_fm_vectors.data() + eval_fm_phase_vector_offset(phase_idx);
    const bool reversed = eval->reversed[eval->feature_idx];
    for (uint32_t j = 0; j < eval_fm_n_active_features; ++j) {
        const int i = eval_fm_active_feature_indices[j];
        const uint16_t feature = eval->features[eval->feature_idx][i];
        const uint16_t raw = reversed ? swap_player_idx(feature, pattern_sizes[i >> 2]) : feature;
        if (raw >= eval_fm_active_feature_limits[j]) {
            return 0;
        }
        const int32_t x0 = base[eval_fm_active_vector_offsets[j] + raw];
        sum0 += x0;
        square_sum0 += x0 * x0;
    }
    return eval_fm_finalize_score((int64_t)sum0 * sum0 - square_sum0);
}

inline int eval_fm_calc_dim2_from_eval_search(const int phase_idx, Eval_search *eval) {
    int32_t sum0 = 0;
    int32_t sum1 = 0;
    int32_t square_sum0 = 0;
    int32_t square_sum1 = 0;
    const uint16_t *base = eval_fm_vectors_dim2_packed.data() + (eval_fm_phase_vector_offset(phase_idx) >> 1);
    const bool reversed = eval->reversed[eval->feature_idx];
    for (uint32_t j = 0; j < eval_fm_n_active_features; ++j) {
        const int i = eval_fm_active_feature_indices[j];
        const uint16_t feature = eval->features[eval->feature_idx][i];
        const uint16_t raw = reversed ? swap_player_idx(feature, pattern_sizes[i >> 2]) : feature;
        if (raw >= eval_fm_active_feature_limits[j]) {
            return 0;
        }
        const uint16_t packed = base[eval_fm_active_vector_offsets[j] + raw];
        const int32_t x0 = (int8_t)(packed & 0xFFU);
        const int32_t x1 = (int8_t)(packed >> 8);
        sum0 += x0;
        sum1 += x1;
        square_sum0 += x0 * x0;
        square_sum1 += x1 * x1;
    }
    const int64_t diff = (int64_t)sum0 * sum0 + (int64_t)sum1 * sum1 - square_sum0 - square_sum1;
    return eval_fm_finalize_score(diff);
}
#endif

inline int eval_fm_calc_dim2_unrolled(const int phase_idx, const uint16_t active_raw_features[N_PATTERN_FEATURES]) {
    int32_t sum0 = 0;
    int32_t sum1 = 0;
    int32_t square_sum0 = 0;
    int32_t square_sum1 = 0;
    const uint16_t *base = eval_fm_vectors_dim2_packed.data() + (eval_fm_phase_vector_offset(phase_idx) >> 1);
    for (uint32_t j = 0; j < eval_fm_n_active_features; ++j) {
        const uint16_t raw = active_raw_features[j];
        if (raw >= eval_fm_active_feature_limits[j]) {
            return 0;
        }
        const uint16_t packed = base[eval_fm_active_vector_offsets[j] + raw];
        const int32_t x0 = (int8_t)(packed & 0xFFU);
        const int32_t x1 = (int8_t)(packed >> 8);
        sum0 += x0;
        sum1 += x1;
        square_sum0 += x0 * x0;
        square_sum1 += x1 * x1;
    }
    const int64_t diff = (int64_t)sum0 * sum0 + (int64_t)sum1 * sum1 - square_sum0 - square_sum1;
    return eval_fm_finalize_score(diff);
}

inline int eval_fm_calc_dim4_unrolled(const int phase_idx, const uint16_t active_raw_features[N_PATTERN_FEATURES]) {
    int32_t sum0 = 0;
    int32_t sum1 = 0;
    int32_t sum2 = 0;
    int32_t sum3 = 0;
    int32_t square_sum0 = 0;
    int32_t square_sum1 = 0;
    int32_t square_sum2 = 0;
    int32_t square_sum3 = 0;
    const uint64_t phase_offset = eval_fm_phase_vector_offset(phase_idx);
    const int8_t *base = eval_fm_vectors.data() + phase_offset;
    for (uint32_t j = 0; j < eval_fm_n_active_features; ++j) {
        const uint16_t raw = active_raw_features[j];
        if (raw >= eval_fm_active_feature_limits[j]) {
            return 0;
        }
        const int8_t *v = base + (uint64_t)(eval_fm_active_vector_offsets[j] + raw) * 4;
        const int32_t x0 = v[0];
        const int32_t x1 = v[1];
        const int32_t x2 = v[2];
        const int32_t x3 = v[3];
        sum0 += x0;
        sum1 += x1;
        sum2 += x2;
        sum3 += x3;
        square_sum0 += x0 * x0;
        square_sum1 += x1 * x1;
        square_sum2 += x2 * x2;
        square_sum3 += x3 * x3;
    }
    const int64_t diff =
        (int64_t)sum0 * sum0 + (int64_t)sum1 * sum1 + (int64_t)sum2 * sum2 + (int64_t)sum3 * sum3
        - square_sum0 - square_sum1 - square_sum2 - square_sum3;
    return eval_fm_finalize_score(diff);
}

inline int eval_fm_calc_from_active_raw_features(const int phase_idx, const uint16_t active_raw_features[N_PATTERN_FEATURES]) {
    if (!eval_fm_enabled || (eval_fm_has_phase_range && !eval_fm_phase_enabled[phase_idx])) {
        return 0;
    }
    if (eval_fm_dim == 1) {
        return eval_fm_calc_dim1_unrolled(phase_idx, active_raw_features);
    }
    if (eval_fm_dim == 2) {
        return eval_fm_calc_dim2_unrolled(phase_idx, active_raw_features);
    }
    if (eval_fm_dim == 4) {
        return eval_fm_calc_dim4_unrolled(phase_idx, active_raw_features);
    }
#if USE_SIMD_EVALUATION
    if (eval_fm_dim == 8) {
        return eval_fm_calc_dim8_avx2(phase_idx, active_raw_features);
    }
#endif
    int32_t sum[EVAL_FM_MAX_DIM] = {};
    int32_t square_sum[EVAL_FM_MAX_DIM] = {};
    const uint64_t phase_offset = eval_fm_phase_vector_offset(phase_idx);
    for (uint32_t j = 0; j < eval_fm_n_active_features; ++j) {
        const uint16_t raw = active_raw_features[j];
        if (raw >= eval_fm_active_feature_limits[j]) {
            return 0;
        }
        const uint64_t vector_idx = phase_offset + (uint64_t)(eval_fm_active_vector_offsets[j] + raw) * eval_fm_dim;
        const int8_t *v = eval_fm_vectors.data() + vector_idx;
        for (uint32_t d = 0; d < eval_fm_dim; ++d) {
            const int32_t x = v[d];
            sum[d] += x;
            square_sum[d] += x * x;
        }
    }
    int64_t diff = 0;
    for (uint32_t d = 0; d < eval_fm_dim; ++d) {
        diff += (int64_t)sum[d] * sum[d] - square_sum[d];
    }
    return eval_fm_finalize_score(diff);
}

#if USE_SIMD_EVALUATION
inline int eval_fm_calc(const int phase_idx, Eval_features *features) {
    if (!eval_fm_enabled || (eval_fm_has_phase_range && !eval_fm_phase_enabled[phase_idx])) {
        return 0;
    }
    alignas(32) uint16_t lanes[N_PATTERN_FEATURES];
    for (int v = 0; v < N_EVAL_VECTORS; ++v) {
        if (eval_fm_active_feature_vector_mask & (1U << v)) {
            _mm256_store_si256((__m256i*)(lanes + v * 16), features->f256[v]);
        }
    }
    if (eval_fm_dim == 1) {
        return eval_fm_calc_dim1_from_active_lanes(phase_idx, lanes);
    }
    if (eval_fm_dim == 2) {
        return eval_fm_calc_dim2_from_active_lanes(phase_idx, lanes);
    }
    uint16_t active_raw_features[N_PATTERN_FEATURES];
    for (uint32_t j = 0; j < eval_fm_n_active_features; ++j) {
        active_raw_features[j] = lanes[eval_fm_active_lane_indices[j]] - eval_fm_active_pack_offsets[j];
    }
    return eval_fm_calc_from_active_raw_features(phase_idx, active_raw_features);
}
#else
inline int eval_fm_calc(const int phase_idx, Eval_search *eval) {
    if (!eval_fm_enabled || (eval_fm_has_phase_range && !eval_fm_phase_enabled[phase_idx])) {
        return 0;
    }
    if (eval_fm_dim == 1) {
        return eval_fm_calc_dim1_from_eval_search(phase_idx, eval);
    }
    if (eval_fm_dim == 2) {
        return eval_fm_calc_dim2_from_eval_search(phase_idx, eval);
    }
    uint16_t active_raw_features[N_PATTERN_FEATURES];
    const bool reversed = eval->reversed[eval->feature_idx];
    for (uint32_t j = 0; j < eval_fm_n_active_features; ++j) {
        const int i = eval_fm_active_feature_indices[j];
        const uint16_t feature = eval->features[eval->feature_idx][i];
        active_raw_features[j] = reversed ? swap_player_idx(feature, pattern_sizes[i >> 2]) : feature;
    }
    return eval_fm_calc_from_active_raw_features(phase_idx, active_raw_features);
}
#endif
