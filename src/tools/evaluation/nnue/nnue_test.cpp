#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif







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








#define STEP 8
#define STEP_2 4

#define SCORE_MAX 64

#define EVAL_NNUE_N_INPUT 128
#define EVAL_NNUE_N_NODES_LAYER 16
#define EVAL_NNUE_N_MID_LAYER 3
#define EVAL_NNUE_N_SHIFT_CRELU 6

__m256i eval_nnue_first_layer_bias;
__m256i eval_nnue_first_layer_weight[EVAL_NNUE_N_INPUT];
__m256i eval_nnue_layer_bias[EVAL_NNUE_N_MID_LAYER];
__m256i eval_nnue_layer_weight[EVAL_NNUE_N_MID_LAYER][EVAL_NNUE_N_NODES_LAYER];
int eval_nnue_out_layer_bias;
__m256i eval_nnue_out_layer_weight;

inline __m256i clipped_ReLU(__m256i a){
    a = _mm256_srai_epi16(a, EVAL_NNUE_N_SHIFT_CRELU);
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
    int16_t layer_in_arr[EVAL_NNUE_N_NODES_LAYER];
    __m256i layer_out = layer_A;
    // mid layer
    for (int i = 0; i < EVAL_NNUE_N_MID_LAYER; ++i){
        _mm256_storeu_si256((__m256i*)layer_in_arr, clipped_ReLU(layer_out));
        layer_out = eval_nnue_layer_bias[i];
        for (int j = 0; j < EVAL_NNUE_N_NODES_LAYER; ++j){
            layer_out = _mm256_add_epi16(layer_out, _mm256_mullo_epi16(eval_nnue_layer_weight[i][j], _mm256_set1_epi16(layer_in_arr[j])));
        }
        layer_out = clipped_ReLU(layer_out);
    }
    // output layer
    __m256i out = _mm256_mullo_epi16(layer_out, eval_nnue_out_layer_weight);
    int16_t out_arr[EVAL_NNUE_N_NODES_LAYER];
    _mm256_storeu_si256((__m256i*)out_arr, out);
    int res = eval_nnue_out_layer_bias;
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        res += out_arr[i];
    }
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    //res = std::clamp(res, -SCORE_MAX, SCORE_MAX);
    return res;
}






int16_t generic_eval_nnue_first_layer_bias[EVAL_NNUE_N_NODES_LAYER];
int16_t generic_eval_nnue_first_layer_weight[EVAL_NNUE_N_INPUT][EVAL_NNUE_N_NODES_LAYER];
int16_t generic_eval_nnue_layer_bias[EVAL_NNUE_N_MID_LAYER][EVAL_NNUE_N_NODES_LAYER];
int16_t generic_eval_nnue_layer_weight[EVAL_NNUE_N_MID_LAYER][EVAL_NNUE_N_NODES_LAYER][EVAL_NNUE_N_NODES_LAYER];
int generic_eval_nnue_out_layer_bias;
int16_t generic_eval_nnue_out_layer_weight[EVAL_NNUE_N_NODES_LAYER];

inline void clipped_ReLU(int16_t a[], int16_t dst[]){
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        dst[i] = std::min(std::max((int)a[i] >> EVAL_NNUE_N_SHIFT_CRELU, -127), 127);
    }
}


inline int mid_evaluate(int16_t layer_A[]){ // TBD
    int16_t layer_B_in_arr[EVAL_NNUE_N_NODES_LAYER];
    clipped_ReLU(layer_A, layer_B_in_arr);
    int16_t layer_B_out[EVAL_NNUE_N_NODES_LAYER];
    for (int j = 0; j < EVAL_NNUE_N_NODES_LAYER; ++j){
        layer_B_out[j] = generic_eval_nnue_layer_bias[i][j];
    }
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        for (int j = 0; j < EVAL_NNUE_N_NODES_LAYER; ++j){
            layer_B_out[i] += layer_B_in_arr[j] * generic_eval_nnue_layer_B_weight[j][i];
        }
    }
    clipped_ReLU(layer_B_out, layer_B_out);
    int res = generic_eval_nnue_layer_out_bias;
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        res += layer_B_out[i] * generic_eval_nnue_layer_out_weight[i];
    }
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    //res = std::clamp(res, -SCORE_MAX, SCORE_MAX);
    return res;
}

/*
#define N 10000000ULL
__m256i data[N];
int16_t data_generic[N][EVAL_NNUE_N_NODES_LAYER];
*/

int main(){
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        //generic_eval_nnue_layer_B_bias[i] = i % 16 - 7;
        generic_eval_nnue_layer_B_bias[i] = myrandrange(-7, 8);
    }
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        for (int j = 0; j < EVAL_NNUE_N_NODES_LAYER; ++j){
            //generic_eval_nnue_layer_B_weight[i][j] = (i + j) % 16 - 7;
            generic_eval_nnue_layer_B_weight[i][j] = myrandrange(-7, 8);
        }
    }
    generic_eval_nnue_layer_out_bias = 1;
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        //generic_eval_nnue_layer_out_weight[i] = i % 256 - 127;
        generic_eval_nnue_layer_out_weight[i] = myrandrange(-7, 8);
    }


    eval_nnue_layer_B_bias = _mm256_load_si256((__m256i*)generic_eval_nnue_layer_B_bias);
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        eval_nnue_layer_B_weight[i] = _mm256_load_si256((__m256i*)generic_eval_nnue_layer_B_weight[i]);
    }
    eval_nnue_layer_out_bias = generic_eval_nnue_layer_out_bias;
    eval_nnue_layer_out_weight = _mm256_load_si256((__m256i*)generic_eval_nnue_layer_out_weight);





    
    for (int ii = 0; ii < 100000; ++ii){
        int16_t generic_test_data[EVAL_NNUE_N_NODES_LAYER];
        for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
            generic_test_data[i] = myrandrange(-9000, 9000);
        }
        __m256i test_data = _mm256_load_si256((__m256i*)generic_test_data);
        int res = mid_evaluate(test_data);
        int res_generic = mid_evaluate(generic_test_data);
        //std::cerr << res << std::endl;
        if (res != res_generic){
            std::cerr << "err" << std::endl;
            std::cerr << res << " " << res_generic << std::endl;
            for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
                std::cerr << generic_test_data[i] << " ";
            }
            std::cerr << std::endl;
        }
    }
    std::cerr << "done" << std::endl;
    





    /*
    for (uint64_t i = 0; i < N; ++i){
        for (int j = 0; j < EVAL_NNUE_N_NODES_LAYER; ++j){
            data_generic[i][j] = myrandrange(-127, 128);
        }
        data[i] = _mm256_load_si256((__m256i*)data_generic[i]);
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
    std::cerr << "NNUE SIMD " << elapsed << " ms NPS=" << nps << std::endl;

    std::cerr << "start!" << std::endl;
    uint64_t res_generic = 0;
    uint64_t strt_generic = tim();
    for (uint64_t i = 0; i < N; ++i){
        res_generic += mid_evaluate(data_generic[i]);
        //std::cerr << i << std::endl;
    }
    uint64_t elapsed_generic = tim() - strt_generic;
    uint64_t nps_generic = N * 1000ULL / (elapsed_generic + 1);
    std::cerr << res_generic << std::endl;
    std::cerr << "NNUE Generic " << elapsed_generic << " ms NPS=" << nps_generic << std::endl;
    */

    /*
    int16_t layer_B_in_arr[EVAL_NNUE_N_NODES_LAYER];
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        layer_B_in_arr[i] = i;
    }
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        std::cerr << layer_B_in_arr[i] << std::endl;
    }
    __m256i a = _mm256_load_si256((__m256i*)layer_B_in_arr);
    // __m256i a = _mm256_set_epi16(
    //     layer_B_in_arr[15], layer_B_in_arr[14], layer_B_in_arr[13], layer_B_in_arr[12], layer_B_in_arr[11], layer_B_in_arr[10], layer_B_in_arr[9], layer_B_in_arr[8], 
    //     layer_B_in_arr[7], layer_B_in_arr[6], layer_B_in_arr[5], layer_B_in_arr[4], layer_B_in_arr[3], layer_B_in_arr[2], layer_B_in_arr[1], layer_B_in_arr[0]
      
    // );
    _mm256_storeu_si256((__m256i*)layer_B_in_arr, a);
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        std::cerr << layer_B_in_arr[i] << std::endl;
    }
    */
}

