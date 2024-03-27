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
    return res;
    /*
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    //res = std::clamp(res, -SCORE_MAX, SCORE_MAX);
    return res;
    */
}






int16_t generic_eval_nnue_layer_A_bias[EVAL_NNUE_N_NODES_LAYER];
int16_t generic_eval_nnue_layer_A_weight[EVAL_NNUE_N_INPUT][EVAL_NNUE_N_NODES_LAYER];
int16_t generic_eval_nnue_layer_B_bias[EVAL_NNUE_N_NODES_LAYER];
int16_t generic_eval_nnue_layer_B_weight[EVAL_NNUE_N_NODES_LAYER][EVAL_NNUE_N_NODES_LAYER];
int generic_eval_nnue_layer_out_bias;
int16_t generic_eval_nnue_layer_out_weight[EVAL_NNUE_N_NODES_LAYER];

inline void clipped_ReLU(int16_t a[], int16_t dst[]){
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        dst[i] = std::min(std::max((int)a[i], -127), 127);
    }
}


inline int mid_evaluate(int16_t layer_A[]){
    int16_t layer_B_in_arr[EVAL_NNUE_N_NODES_LAYER];
    clipped_ReLU(layer_A, layer_B_in_arr);
    int16_t layer_B_out[EVAL_NNUE_N_NODES_LAYER];
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        layer_B_out[i] = generic_eval_nnue_layer_B_bias[i];
    }
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        for (int j = 0; j < EVAL_NNUE_N_NODES_LAYER; ++j){
            layer_B_out[i] += layer_B_in_arr[j] * generic_eval_nnue_layer_B_weight[i][j];
        }
    }
    clipped_ReLU(layer_B_out, layer_B_out);
    int res = generic_eval_nnue_layer_out_bias;
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        res += layer_B_out[i] * generic_eval_nnue_layer_out_weight[i];
    }
    return res;
    /*
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    //res = std::clamp(res, -SCORE_MAX, SCORE_MAX);
    return res;
    */
}


void mm256_print_epi32(__m256i v){
    int* varray = (int*)&v;
    for (int i = 0; i < 8; ++i){
        std::cerr << varray[i] << " ";
    }
    std::cerr << std::endl;
}

void mm256_print_epi16(__m256i v){
    int16_t* varray = (int16_t*)&v;
    for (int i = 0; i < 16; ++i){
        std::cerr << varray[i] << " ";
    }
    std::cerr << std::endl;
}

int main(){
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        generic_eval_nnue_layer_B_bias[i] = i % 16 - 7;
    }
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        for (int j = 0; j < EVAL_NNUE_N_NODES_LAYER; ++j){
            generic_eval_nnue_layer_B_weight[i][j] = (i + j) % 16 - 7;
        }
    }
    generic_eval_nnue_layer_out_bias = 1;
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        generic_eval_nnue_layer_out_weight[i] = i % 256 - 127;
    }


    eval_nnue_layer_B_bias = _mm256_load_si256((__m256i*)generic_eval_nnue_layer_B_bias);
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        eval_nnue_layer_B_weight[i] = _mm256_load_si256((__m256i*)generic_eval_nnue_layer_B_weight[i]);
    }
    eval_nnue_layer_out_bias = generic_eval_nnue_layer_out_bias;
    eval_nnue_layer_out_weight = _mm256_load_si256((__m256i*)generic_eval_nnue_layer_out_weight);

    int16_t generic_test_data[EVAL_NNUE_N_NODES_LAYER];
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        generic_test_data[i] = i % 256 - 127;
    }
    __m256i test_data = _mm256_load_si256((__m256i*)generic_test_data);

    std::cerr << mid_evaluate(generic_test_data) << " " << mid_evaluate(test_data) << std::endl;


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

