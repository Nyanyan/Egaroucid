#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif

#define N 10000000ULL

#define STEP 32
#define STEP_2 16
#define SCORE_MAX 64

#define EVAL_NNUE_N_INPUT 128
#define EVAL_NNUE_N_NODES_LAYER 16

__m256i eval_nnue_layer_A_bias;
__m256i eval_nnue_layer_A_weight[EVAL_NNUE_N_INPUT];
__m256i eval_nnue_layer_B_bias;
__m256i eval_nnue_layer_B_weight[EVAL_NNUE_N_NODES_LAYER];
int eval_nnue_layer_out_bias;
__m256i eval_nnue_layer_out_weight;

inline __m256i clipped_ReLU(__m256i a){
    a = _mm256_max_epi16(a, _mm256_set1_epi16(-127));
    a = _mm256_min_epi16(a, _mm256_set1_epi16(127));
    return a;
}

/*
    @brief midgame evaluation function

    @param search               search information
    @return evaluation value
*/
inline int mid_evaluate(__m256i layer_A){
    int16_t layer_B_in_arr[EVAL_NNUE_N_NODES_LAYER];
    _mm256_storeu_si256((__m256i*)layer_B_in_arr, clipped_ReLU(layer_A));
    __m256i layer_B_out = eval_nnue_layer_B_bias;
    // layer B
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        layer_B_out = _mm256_add_epi16(layer_B_out, _mm256_mullo_epi16(eval_nnue_layer_B_weight[i], _mm256_set1_epi16(layer_B_in_arr[i])));
    }
    layer_B_out = clipped_ReLU(layer_B_out);
    // output layer
    __m256i out = _mm256_mullo_epi16(layer_B_out, eval_nnue_layer_out_weight);
    int16_t out_arr[EVAL_NNUE_N_NODES_LAYER];
    _mm256_storeu_si256((__m256i*)out_arr, out);
    int res = eval_nnue_layer_out_bias;
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        res += out_arr[i];
    }
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    //res = std::clamp(res, -SCORE_MAX, SCORE_MAX);
    return res;
}














/*
    @brief timing function

    @return time in milliseconds
*/
inline uint64_t tim(){
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

std::mt19937 raw_myrandom(tim());

/*
    @brief random function

    @return random value from 0.0 to 1.0 (not including 1.0)
*/
inline double myrandom(){
    return (double)raw_myrandom() / std::mt19937::max();
}

/*
    @brief randrange function

    @param s                    minimum integer
    @param e                    maximum integer
    @return random integer from s to e - 1
*/
inline int32_t myrandrange(int32_t s, int32_t e){
    return s +(int)((e - s) * myrandom());
}

/*
    @brief random integer function

    @return random 64bit integer
*/
inline uint64_t myrand_ull(){
    return ((uint64_t)raw_myrandom() << 32) | (uint64_t)raw_myrandom();
}







void nnue_speed(){
    eval_nnue_layer_A_bias = _mm256_set_epi64x(myrand_ull(), myrand_ull(), myrand_ull(), myrand_ull());
    for (int i = 0; i < EVAL_NNUE_N_INPUT; ++i){
        eval_nnue_layer_A_weight[i] = _mm256_set_epi64x(myrand_ull(), myrand_ull(), myrand_ull(), myrand_ull());
    }
    eval_nnue_layer_B_bias = _mm256_set_epi64x(myrand_ull(), myrand_ull(), myrand_ull(), myrand_ull());
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        eval_nnue_layer_B_weight[i] = _mm256_set_epi64x(myrand_ull(), myrand_ull(), myrand_ull(), myrand_ull());
    }
    eval_nnue_layer_out_bias = 1;
    eval_nnue_layer_out_weight = _mm256_set_epi64x(myrand_ull(), myrand_ull(), myrand_ull(), myrand_ull());

    __m256i *data = (__m256i*)malloc(sizeof(__m256i) * N);
    for (uint64_t i = 0; i < N; ++i){
        data[i] = _mm256_set_epi64x(myrand_ull(), myrand_ull(), myrand_ull(), myrand_ull());
    }
    
    std::cerr << "start!" << std::endl;
    uint64_t res = 0;
    uint64_t strt = tim();
    for (uint64_t i = 0; i < N; ++i){
        res += mid_evaluate(data[i]);
        //std::cerr << i << std::endl;
    }
    uint64_t elapsed = tim() - strt;
    uint64_t nps = N * 1000ULL / (elapsed + 1);
    std::cerr << res << std::endl;
    std::cerr << "NNUE " << elapsed << " ms NPS=" << nps << std::endl;
    free(data);
}






#define N_PATTERNS 16
#define MAX_PATTERN_CELLS 10
#define MAX_CELL_PATTERNS 13
#define MAX_EVALUATE_IDX 59049
#define N_SYMMETRY_PATTERNS 62
#define N_SIMD_EVAL_FEATURES 4 // 16 (elems per 256 bit vector) * N_SIMD_EVAL_FEATURES >= N_SYMMETRY_PATTERNS
#define HW2 64
#define N_16BIT 65536 // 2 ^ 16
#define CEIL_N_SYMMETRY_PATTERNS 64         // N_SYMMETRY_PATTRENS + dummy
#define N_PATTERN_PARAMS (521478 + 2)       // +2 for byte bound & dummy for d8
#define SIMD_EVAL_MAX_VALUE 4092            // evaluate range [-4092, 4092]
#define N_SIMD_EVAL_FEATURES_SIMPLE 2
#define N_SIMD_EVAL_FEATURES_COMP 2
#define N_PATTERN_PARAMS_BEFORE_DUMMY 29403
#define SIMD_EVAL_DUMMY_ADDR 29404
#define N_PATTERN_PARAMS_AFTER_DUMMY 492075
#define N_SIMD_EVAL_FEATURE_CELLS 16
#define N_SIMD_EVAL_FEATURE_GROUP 4
// additional features
#define MAX_SURROUND 64
#define MAX_STONE_NUM 65

// evaluation phase definition
#define N_PHASES 60
#define PHASE_N_DISCS 1

__m256i eval_lower_mask;
__m256i feature_to_coord_simd_mul[N_SIMD_EVAL_FEATURES][MAX_PATTERN_CELLS - 1];
__m256i feature_to_coord_simd_cell[N_SIMD_EVAL_FEATURES][MAX_PATTERN_CELLS][2];
__m256i coord_to_feature_simd[HW2][N_SIMD_EVAL_FEATURES];
__m256i eval_move_unflipped_16bit[N_16BIT][N_SIMD_EVAL_FEATURE_GROUP][N_SIMD_EVAL_FEATURES];
__m256i eval_simd_offsets_simple[N_SIMD_EVAL_FEATURES_SIMPLE]; // 16bit * 16 * N
__m256i eval_simd_offsets_comp[N_SIMD_EVAL_FEATURES_COMP * 2]; // 32bit * 8 * N
__m256i eval_surround_mask;
__m128i eval_surround_shift1879;

/*
    @brief evaluation parameters
*/
int16_t pattern_arr[N_PHASES][N_PATTERN_PARAMS];
int16_t eval_num_arr[N_PHASES][MAX_STONE_NUM];
int16_t eval_sur0_sur1_arr[N_PHASES][MAX_SURROUND][MAX_SURROUND];

#ifdef __GNUC__
    #define	pop_count_ull(x) (int)__builtin_popcountll(x)
    #define pop_count_uint(x) (int)__builtin_popcount(x)
    #define pop_count_uchar(x) (int)__builtin_popcount(x)
#else
    #define	pop_count_ull(x) (int)__popcnt64(x)
    #define pop_count_uint(x) (int)__popcnt(x)
    #define pop_count_uchar(x) (int)__popcnt(x)
#endif

union Eval_features{
    __m256i f256[N_SIMD_EVAL_FEATURES];
    __m128i f128[N_SIMD_EVAL_FEATURES * 2];
};

inline int calc_surround(const uint64_t discs, const uint64_t empties){
    __m256i pl = _mm256_set1_epi64x(discs);
    pl = _mm256_and_si256(pl, eval_surround_mask);
    pl = _mm256_or_si256(_mm256_sll_epi64(pl, eval_surround_shift1879), _mm256_srl_epi64(pl, eval_surround_shift1879));
    __m128i res = _mm_or_si128(_mm256_castsi256_si128(pl), _mm256_extracti128_si256(pl, 1));
    res = _mm_or_si128(res, _mm_shuffle_epi32(res, 0x4e));
    return pop_count_ull(_mm_cvtsi128_si64(res));
}
#define CALC_SURROUND_FUNCTION

/*
    @brief pattern evaluation

    @param phase_idx            evaluation phase
    @param search               search information
    @return pattern evaluation value
*/
inline __m256i calc_idx8_comp(const __m128i feature, const int i){
    return _mm256_add_epi32(_mm256_cvtepu16_epi32(feature), eval_simd_offsets_comp[i]);
}

inline __m256i gather_eval(const int *start_addr, const __m256i idx8){
    return _mm256_i32gather_epi32(start_addr, idx8, 2); // stride is 2 byte, because 16 bit array used, HACK: if (SIMD_EVAL_MAX_VALUE * 2) * (N_ADD=8) < 2 ^ 16, AND is unnecessary
    // return _mm256_and_si256(_mm256_i32gather_epi32(start_addr, idx8, 2), eval_lower_mask);
}

inline int calc_pattern(const int phase_idx, Eval_features *features){
    const int *start_addr = (int*)pattern_arr[phase_idx];
    __m256i res256 =                  gather_eval(start_addr, _mm256_cvtepu16_epi32(features->f128[0]));    // hv4 d5
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, _mm256_cvtepu16_epi32(features->f128[1])));   // hv2 hv3
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, _mm256_cvtepu16_epi32(features->f128[2])));   // d8 corner9
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, _mm256_cvtepu16_epi32(features->f128[3])));   // d6 d7
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, calc_idx8_comp(features->f128[4], 0)));       // corner+block cross
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, calc_idx8_comp(features->f128[5], 1)));       // edge+2X triangle
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, calc_idx8_comp(features->f128[6], 2)));       // fish kite
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, calc_idx8_comp(features->f128[7], 3)));       // edge+2Y narrow_triangle
    res256 = _mm256_and_si256(res256, eval_lower_mask);
    __m128i res128 = _mm_add_epi32(_mm256_castsi256_si128(res256), _mm256_extracti128_si256(res256, 1));
    res128 = _mm_hadd_epi32(res128, res128);
    return _mm_cvtsi128_si32(res128) + _mm_extract_epi32(res128, 1) - SIMD_EVAL_MAX_VALUE * N_SYMMETRY_PATTERNS;
}


/*
    @brief midgame evaluation function

    @param search               search information
    @return evaluation value
*/
inline int mid_evaluate_diff(int phase, uint64_t player, uint64_t opponent, Eval_features features){
    int phase_idx, sur0, sur1, num0;
    uint64_t empties;
    phase_idx = phase;
    empties = ~(player | opponent);
    sur0 = calc_surround(player, empties);
    sur1 = calc_surround(opponent, empties);
    num0 = pop_count_ull(player);
    int res = calc_pattern(phase_idx, &features) + 
        eval_num_arr[phase_idx][num0] + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1];
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    //res = std::clamp(res, -SCORE_MAX, SCORE_MAX);
    return res;
}


void pattern_speed(){
    Eval_features *data = (Eval_features*)malloc(sizeof(Eval_features) * N);
    uint64_t *data_player = (uint64_t*)malloc(sizeof(uint64_t) * N);
    uint64_t *data_opponent = (uint64_t*)malloc(sizeof(uint64_t) * N);
    int *data_phase = (int*)malloc(sizeof(int) * N);
    for (uint64_t i = 0; i < N; ++i){
        for (int j = 0; j < N_SIMD_EVAL_FEATURES; ++j){
            data[i].f256[j] = _mm256_set_epi64x(myrand_ull(), myrand_ull(), myrand_ull(), myrand_ull());
        }
        data_player[i] = myrand_ull();
        data_opponent[i] = myrand_ull();
        data_phase[i] = myrandrange(0, N_PHASES);
    }
    
    std::cerr << "start!" << std::endl;
    uint64_t res = 0;
    uint64_t strt = tim();
    for (uint64_t i = 0; i < N; ++i){
        res += mid_evaluate_diff(data_phase[i], data_player[i], data_opponent[i], data[i]);
    }
    uint64_t elapsed = tim() - strt;
    uint64_t nps = N * 1000ULL / (elapsed + 1);
    std::cerr << res << std::endl;
    std::cerr << "pattern " << elapsed << " ms NPS=" << nps << std::endl;
    free(data);
}



int main(){
    pattern_speed();
    nnue_speed();
}