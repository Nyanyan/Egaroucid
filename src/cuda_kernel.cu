#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cassert>
#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
/*
#include "setting.hpp"
#include "evaluate.hpp"
*/
#include "cuda_board.hpp"

#define HW 8
#define HW_P1 9
#define HW_M1 7
#define SCORE_MAX 64
#define INF 100000000

using namespace std;

#define CUDA_MAX_DEPTH 10 + 1
#define THREADS_PER_BLOCK 128
#define SIMD_WIDTH 4
constexpr int NODES_PER_BLOCK = THREADS_PER_BLOCK / SIMD_WIDTH;

struct Node_cuda {
    Board_simple board;
    uint64_t puttable;
    int alpha;
    int beta;
    bool pass;
    bool passed;
    int depth;
    __device__ void update(char value) {
        alpha = max(alpha, -value);
    }
};

__shared__ Node_cuda nodes_stack[NODES_PER_BLOCK][CUDA_MAX_DEPTH + 1];
__shared__ int count_cuda[THREADS_PER_BLOCK];

__device__ inline uint64_t puttable(Board_simple *b){
    uint64_t hmask = b->opponent & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = b->opponent & 0x00FFFFFFFFFFFF00ULL;
    uint64_t hvmask = b->opponent & 0x007E7E7E7E7E7E00ULL;
    uint64_t res = (hmask << 1) | (hmask >> 1) | (vmask << HW) | (vmask >> HW) | (hvmask >> HW_P1) | (hvmask << HW_P1) | (hvmask >> HW_M1) | (hvmask << HW_M1);
    return res & ~(b->player | b->opponent);
}

__device__ inline uint64_t puttable(uint64_t player, uint64_t opponent){
    uint64_t hmask = opponent & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = opponent & 0x00FFFFFFFFFFFF00ULL;
    uint64_t hvmask = opponent & 0x0055555555555500ULL;
    uint64_t res = (hmask << 1) | (hmask >> 1) | (vmask << HW) | (vmask >> HW) | (hvmask >> HW_P1) | (hvmask << HW_P1) | (hvmask >> HW_M1) | (hvmask << HW_M1);
    return res & ~(player | opponent);
}
/*
__device__ inline uint64_t calc_legal_cuda(uint64_t player, uint64_t opponent){
    uint64_t hmask = opponent & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = opponent & 0x00FFFFFFFFFFFF00ULL;
    uint64_t hvmask = opponent & 0x0055555555555500ULL;
    uint64_t vacant = ~(player | opponent);
    uint64_t tmp, legal;

    tmp = hmask & (player << 1);
    tmp |= hmask & (tmp << 1);
    tmp |= hmask & (tmp << 1);
    tmp |= hmask & (tmp << 1);
    tmp |= hmask & (tmp << 1);
    tmp |= hmask & (tmp << 1);
    legal = vacant & (tmp << 1);

    tmp = hmask & (player >> 1);
    tmp |= hmask & (tmp >> 1);
    tmp |= hmask & (tmp >> 1);
    tmp |= hmask & (tmp >> 1);
    tmp |= hmask & (tmp >> 1);
    tmp |= hmask & (tmp >> 1);
    legal |= vacant & (tmp >> 1);

    tmp = vmask & (player << 8);
    tmp |= vmask & (tmp << 8);
    tmp |= vmask & (tmp << 8);
    tmp |= vmask & (tmp << 8);
    tmp |= vmask & (tmp << 8);
    tmp |= vmask & (tmp << 8);
    legal |= vacant & (tmp << 8);

    tmp = vmask & (player >> 8);
    tmp |= vmask & (tmp >> 8);
    tmp |= vmask & (tmp >> 8);
    tmp |= vmask & (tmp >> 8);
    tmp |= vmask & (tmp >> 8);
    tmp |= vmask & (tmp >> 8);
    legal |= vacant & (tmp >> 8);

    tmp = hvmask & (player << 7);
    tmp |= hvmask & (tmp << 7);
    tmp |= hvmask & (tmp << 7);
    tmp |= hvmask & (tmp << 7);
    tmp |= hvmask & (tmp << 7);
    tmp |= hvmask & (tmp << 7);
    legal |= vacant & (tmp << 7);

    tmp = hvmask & (player << 9);
    tmp |= hvmask & (tmp << 9);
    tmp |= hvmask & (tmp << 9);
    tmp |= hvmask & (tmp << 9);
    tmp |= hvmask & (tmp << 9);
    tmp |= hvmask & (tmp << 9);
    legal |= vacant & (tmp << 9);

    tmp = hvmask & (player >> 9);
    tmp |= hvmask & (tmp >> 9);
    tmp |= hvmask & (tmp >> 9);
    tmp |= hvmask & (tmp >> 9);
    tmp |= hvmask & (tmp >> 9);
    tmp |= hvmask & (tmp >> 9);
    legal |= vacant & (tmp >> 9);

    tmp = hvmask & (player >> 7);
    tmp |= hvmask & (tmp >> 7);
    tmp |= hvmask & (tmp >> 7);
    tmp |= hvmask & (tmp >> 7);
    tmp |= hvmask & (tmp >> 7);
    tmp |= hvmask & (tmp >> 7);
    legal |= vacant & (tmp >> 7);

    return legal;
}

__device__ inline uint64_t calc_surround_part_cuda(const uint64_t player, const int dr){
    return (player << dr | player >> dr);
}

__device__ inline int calc_surround_cuda(const uint64_t player, const uint64_t empties){
    return __popcll(empties & (
        calc_surround_part_cuda(player & 0b0111111001111110011111100111111001111110011111100111111001111110ULL, 1) | 
        calc_surround_part_cuda(player & 0b0000000011111111111111111111111111111111111111111111111100000000ULL, HW) | 
        calc_surround_part_cuda(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, HW_M1) | 
        calc_surround_part_cuda(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, HW_P1)
    ));
}

inline uint8_t join_v_line_cuda(uint64_t x, int c){
    x = (x >> c) & 0b0000000100000001000000010000000100000001000000010000000100000001ULL;
    return (x * 0b0000000100000010000001000000100000010000001000000100000010000000ULL) >> 56;
}

__device__ inline uint64_t full_stability_h_cuda(uint64_t full){
    full &= full >> 1;
    full &= full >> 2;
    full &= full >> 4;
    return (full & 0x0101010101010101) * 0xff;
}

__device__ inline uint64_t full_stability_v_cuda(uint64_t full){
    full &= (full >> 8) | (full << 56);
    full &= (full >> 16) | (full << 48);
    full &= (full >> 32) | (full << 32);
    return full;
}

__device__ inline void full_stability_d_cuda(uint64_t full, uint64_t *full_d7, uint64_t *full_d9){
    static const uint64_t edge = 0xFF818181818181FF;
    static const uint64_t e7[4] = {
        0xFFFF030303030303, 0xC0C0C0C0C0C0FFFF, 0xFFFFFFFF0F0F0F0F, 0xF0F0F0F0FFFFFFFF};
    static const uint64_t e9[3] = {
        0xFFFFC0C0C0C0C0C0, 0x030303030303FFFF, 0x0F0F0F0FF0F0F0F0};
    uint64_t l7, r7, l9, r9;
    l7 = r7 = full;
    l7 &= edge | (l7 >> 7);		r7 &= edge | (r7 << 7);
    l7 &= e7[0] | (l7 >> 14);	r7 &= e7[1] | (r7 << 14);
    l7 &= e7[2] | (l7 >> 28);	r7 &= e7[3] | (r7 << 28);
    *full_d7 = l7 & r7;

    l9 = r9 = full;
    l9 &= edge | (l9 >> 9);		r9 &= edge | (r9 << 9);
    l9 &= e9[0] | (l9 >> 18);	r9 &= e9[1] | (r9 << 18);
    *full_d9 = l9 & r9 & (e9[2] | (l9 >> 36) | (r9 << 36));
}

__device__ inline void full_stability_cuda(uint64_t player, uint64_t opponent, uint64_t *h, uint64_t *v, uint64_t *d7, uint64_t *d9){
    const uint64_t stones = (player | opponent);
    *h = full_stability_h_cuda(stones);
    *v = full_stability_v_cuda(stones);
    full_stability_d_cuda(stones, d7, d9);
}

inline void calc_stability_cuda(Board_simple *b, int *stab0, int *stab1){
    uint64_t full_h, full_v, full_d7, full_d9;
    uint64_t edge_stability = 0, player_stability = 0, opponent_stability = 0, n_stability;
    uint64_t h, v, d7, d9;
    const uint64_t player_mask = b->player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    const uint64_t opponent_mask = b->opponent & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    uint8_t pl, op;
    pl = b->player & 0b11111111U;
    op = b->opponent & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = (b->player >> 56) & 0b11111111U;
    op = (b->opponent >> 56) & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0];
    pl = join_v_line_cuda(b->player, 0);
    op = join_v_line_cuda(b->opponent, 0);
    edge_stability |= stability_edge_arr[pl][op][1] << 7;
    pl = join_v_line_cuda(b->player, 7);
    op = join_v_line_cuda(b->opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][1];
    full_stability_cuda(b->player, b->opponent, &full_h, &full_v, &full_d7, &full_d9);

    n_stability = (edge_stability & b->player) | (full_h & full_v & full_d7 & full_d9 & player_mask);
    while (n_stability & ~player_stability){
        player_stability |= n_stability;
        h = (player_stability >> 1) | (player_stability << 1) | full_h;
        v = (player_stability >> HW) | (player_stability << HW) | full_v;
        d7 = (player_stability >> HW_M1) | (player_stability << HW_M1) | full_d7;
        d9 = (player_stability >> HW_P1) | (player_stability << HW_P1) | full_d9;
        n_stability = h & v & d7 & d9 & player_mask;
    }

    n_stability = (edge_stability & b->opponent) | (full_h & full_v & full_d7 & full_d9 & opponent_mask);
    while (n_stability & ~opponent_stability){
        opponent_stability |= n_stability;
        h = (opponent_stability >> 1) | (opponent_stability << 1) | full_h;
        v = (opponent_stability >> HW) | (opponent_stability << HW) | full_v;
        d7 = (opponent_stability >> HW_M1) | (opponent_stability << HW_M1) | full_d7;
        d9 = (opponent_stability >> HW_P1) | (opponent_stability << HW_P1) | full_d9;
        n_stability = h & v & d7 & d9 & opponent_mask;
    }

    *stab0 = __popcll(player_stability);
    *stab1 = __popcll(opponent_stability);
}

inline int calc_pattern_first_cuda(const int phase_idx, Board_simple *b){
    uint_fast8_t b_arr[HW2];
    for (int i = 0; i < HW2; ++i)
        b_arr[HW2_M1 - i] = 2 - (1 & (b->player >> i)) * 2 - (1 & (b->opponent >> i));
    return 
        pick_pattern(phase_idx, 0, b_arr, 8, 9, 10, 11, 12, 13, 14, 15) + pick_pattern(phase_idx, 0, b_arr, 1, 9, 17, 25, 33, 41, 49, 57) + pick_pattern(phase_idx, 0, b_arr, 48, 49, 50, 51, 52, 53, 54, 55) + pick_pattern(phase_idx, 0, b_arr, 6, 14, 22, 30, 38, 46, 54, 62) + 
        pick_pattern(phase_idx, 1, b_arr, 16, 17, 18, 19, 20, 21, 22, 23) + pick_pattern(phase_idx, 1, b_arr, 2, 10, 18, 26, 34, 42, 50, 58) + pick_pattern(phase_idx, 1, b_arr, 40, 41, 42, 43, 44, 45, 46, 47) + pick_pattern(phase_idx, 1, b_arr, 5, 13, 21, 29, 37, 45, 53, 61) + 
        pick_pattern(phase_idx, 2, b_arr, 24, 25, 26, 27, 28, 29, 30, 31) + pick_pattern(phase_idx, 2, b_arr, 3, 11, 19, 27, 35, 43, 51, 59) + pick_pattern(phase_idx, 2, b_arr, 32, 33, 34, 35, 36, 37, 38, 39) + pick_pattern(phase_idx, 2, b_arr, 4, 12, 20, 28, 36, 44, 52, 60) + 
        pick_pattern(phase_idx, 3, b_arr, 3, 12, 21, 30, 39) + pick_pattern(phase_idx, 3, b_arr, 4, 11, 18, 25, 32) + pick_pattern(phase_idx, 3, b_arr, 24, 33, 42, 51, 60) + pick_pattern(phase_idx, 3, b_arr, 59, 52, 45, 38, 31) + 
        pick_pattern(phase_idx, 4, b_arr, 2, 11, 20, 29, 38, 47) + pick_pattern(phase_idx, 4, b_arr, 5, 12, 19, 26, 33, 40) + pick_pattern(phase_idx, 4, b_arr, 16, 25, 34, 43, 52, 61) + pick_pattern(phase_idx, 4, b_arr, 58, 51, 44, 37, 30, 23) + 
        pick_pattern(phase_idx, 5, b_arr, 1, 10, 19, 28, 37, 46, 55) + pick_pattern(phase_idx, 5, b_arr, 6, 13, 20, 27, 34, 41, 48) + pick_pattern(phase_idx, 5, b_arr, 8, 17, 26, 35, 44, 53, 62) + pick_pattern(phase_idx, 5, b_arr, 57, 50, 43, 36, 29, 22, 15) + 
        pick_pattern(phase_idx, 6, b_arr, 0, 9, 18, 27, 36, 45, 54, 63) + pick_pattern(phase_idx, 6, b_arr, 7, 14, 21, 28, 35, 42, 49, 56) + 
        pick_pattern(phase_idx, 7, b_arr, 9, 0, 1, 2, 3, 4, 5, 6, 7, 14) + pick_pattern(phase_idx, 7, b_arr, 9, 0, 8, 16, 24, 32, 40, 48, 56, 49) + pick_pattern(phase_idx, 7, b_arr, 49, 56, 57, 58, 59, 60, 61, 62, 63, 54) + pick_pattern(phase_idx, 7, b_arr, 54, 63, 55, 47, 39, 31, 23, 15, 7, 14) + 
        pick_pattern(phase_idx, 8, b_arr, 0, 1, 2, 3, 8, 9, 10, 16, 17, 24) + pick_pattern(phase_idx, 8, b_arr, 7, 6, 5, 4, 15, 14, 13, 23, 22, 31) + pick_pattern(phase_idx, 8, b_arr, 63, 62, 61, 60, 55, 54, 53, 47, 46, 39) + pick_pattern(phase_idx, 8, b_arr, 56, 57, 58, 59, 48, 49, 50, 40, 41, 32) + 
        pick_pattern(phase_idx, 9, b_arr, 0, 2, 3, 4, 5, 7, 10, 11, 12, 13) + pick_pattern(phase_idx, 9, b_arr, 0, 16, 24, 32, 40, 56, 17, 25, 33, 41) + pick_pattern(phase_idx, 9, b_arr, 56, 58, 59, 60, 61, 63, 50, 51, 52, 53) + pick_pattern(phase_idx, 9, b_arr, 7, 23, 31, 39, 47, 63, 22, 30, 38, 46) + 
        pick_pattern(phase_idx, 10, b_arr, 0, 9, 18, 27, 1, 10, 19, 8, 17, 26) + pick_pattern(phase_idx, 10, b_arr, 7, 14, 21, 28, 6, 13, 20, 15, 22, 29) + pick_pattern(phase_idx, 10, b_arr, 56, 49, 42, 35, 57, 50, 43, 48, 41, 34) + pick_pattern(phase_idx, 10, b_arr, 63, 54, 45, 36, 62, 53, 44, 55, 46, 37) + 
        pick_pattern(phase_idx, 11, b_arr, 0, 1, 2, 8, 9, 10, 16, 17, 18) + pick_pattern(phase_idx, 11, b_arr, 7, 6, 5, 15, 14, 13, 23, 22, 21) + pick_pattern(phase_idx, 11, b_arr, 56, 57, 58, 48, 49, 50, 40, 41, 42) + pick_pattern(phase_idx, 11, b_arr, 63, 62, 61, 55, 54, 53, 47, 46, 45) + 
        pick_pattern(phase_idx, 12, b_arr, 10, 0, 1, 2, 3, 4, 5, 6, 7, 13) + pick_pattern(phase_idx, 12, b_arr, 17, 0, 8, 16, 24, 32, 40, 48, 56, 41) + pick_pattern(phase_idx, 12, b_arr, 50, 56, 57, 58, 59, 60, 61, 62, 63, 53) + pick_pattern(phase_idx, 12, b_arr, 46, 63, 55, 47, 39, 31, 23, 15, 7, 22) + 
        pick_pattern(phase_idx, 13, b_arr, 0, 1, 2, 3, 4, 8, 9, 16, 24, 32) + pick_pattern(phase_idx, 13, b_arr, 7, 6, 5, 4, 3, 15, 14, 23, 31, 39) + pick_pattern(phase_idx, 13, b_arr, 63, 62, 61, 60, 59, 55, 54, 47, 39, 31) + pick_pattern(phase_idx, 13, b_arr, 56, 57, 58, 59, 60, 48, 49, 40, 32, 24) + 
        pick_pattern(phase_idx, 14, b_arr, 0, 1, 8, 9, 10, 11, 17, 18, 25, 27) + pick_pattern(phase_idx, 14, b_arr, 7, 6, 15, 14, 13, 12, 22, 21, 30, 28) + pick_pattern(phase_idx, 14, b_arr, 56, 57, 48, 49, 50, 51, 41, 42, 33, 35) + pick_pattern(phase_idx, 14, b_arr, 63, 62, 55, 54, 53, 52, 46, 45, 38, 36) + 
        pick_pattern(phase_idx, 15, b_arr, 0, 1, 8, 9, 10, 11, 12, 17, 25, 33) + pick_pattern(phase_idx, 15, b_arr, 7, 6, 15, 14, 13, 12, 11, 22, 30, 38) + pick_pattern(phase_idx, 15, b_arr, 56, 57, 48, 49, 50, 51, 52, 41, 33, 25) + pick_pattern(phase_idx, 15, b_arr, 63, 62, 55, 54, 53, 52, 51, 46, 38, 30);
}

inline int create_canput_line_h_cuda(uint64_t b, uint64_t w, int t){
    return (((w >> (HW * t)) & 0b11111111) << HW) | ((b >> (HW * t)) & 0b11111111);
}

inline int create_canput_line_v_cuda(uint64_t b, uint64_t w, int t){
    return (join_v_line(w, t) << HW) | join_v_line(b, t);
}

inline int calc_canput_pattern_cuda(const int phase_idx, const uint64_t player_mobility, const uint64_t opponent_mobility){
    return 
        eval_canput_pattern[phase_idx][0][create_canput_line_h_cuda(player_mobility, opponent_mobility, 0)] + 
        eval_canput_pattern[phase_idx][0][create_canput_line_h_cuda(player_mobility, opponent_mobility, 7)] + 
        eval_canput_pattern[phase_idx][0][create_canput_line_v_cuda(player_mobility, opponent_mobility, 0)] + 
        eval_canput_pattern[phase_idx][0][create_canput_line_v_cuda(player_mobility, opponent_mobility, 7)] + 
        eval_canput_pattern[phase_idx][1][create_canput_line_h_cuda(player_mobility, opponent_mobility, 1)] + 
        eval_canput_pattern[phase_idx][1][create_canput_line_h_cuda(player_mobility, opponent_mobility, 6)] + 
        eval_canput_pattern[phase_idx][1][create_canput_line_v_cuda(player_mobility, opponent_mobility, 1)] + 
        eval_canput_pattern[phase_idx][1][create_canput_line_v_cuda(player_mobility, opponent_mobility, 6)] + 
        eval_canput_pattern[phase_idx][2][create_canput_line_h_cuda(player_mobility, opponent_mobility, 2)] + 
        eval_canput_pattern[phase_idx][2][create_canput_line_h_cuda(player_mobility, opponent_mobility, 5)] + 
        eval_canput_pattern[phase_idx][2][create_canput_line_v_cuda(player_mobility, opponent_mobility, 2)] + 
        eval_canput_pattern[phase_idx][2][create_canput_line_v_cuda(player_mobility, opponent_mobility, 5)] + 
        eval_canput_pattern[phase_idx][3][create_canput_line_h_cuda(player_mobility, opponent_mobility, 3)] + 
        eval_canput_pattern[phase_idx][3][create_canput_line_h_cuda(player_mobility, opponent_mobility, 4)] + 
        eval_canput_pattern[phase_idx][3][create_canput_line_v_cuda(player_mobility, opponent_mobility, 3)] + 
        eval_canput_pattern[phase_idx][3][create_canput_line_v_cuda(player_mobility, opponent_mobility, 4)];
}

inline int mid_evaluate_cuda(Board_simple *b){
    int phase_idx, sur0, sur1, canput0, canput1, stab0, stab1, num0, num1;
    uint64_t player_mobility, opponent_mobility, empties;
    player_mobility = calc_legal_cuda(b->player, b->opponent);
    opponent_mobility = calc_legal_cuda(b->opponent, b->player);
    canput0 = min(MAX_CANPUT - 1, __popcll(player_mobility));
    canput1 = min(MAX_CANPUT - 1, __popcll(opponent_mobility));
    if (canput0 == 0 && canput1 == 0)
        return end_evaluate_cuda(b);
    phase_idx = (__popcll(b->player | b->opponent) - 4) / PHASE_N_STONES;
    empties = ~(b->player | b->opponent);
    sur0 = min(MAX_SURROUND - 1, calc_surround_cuda(b->player, empties));
    sur1 = min(MAX_SURROUND - 1, calc_surround_cuda(b->opponent, empties));
    calc_stability_cuda(b, &stab0, &stab1);
    num0 = __popcll(b->player);
    num1 = __popcll(b->opponent);
    //cerr << calc_pattern(phase_idx, b) << " " << eval_sur0_sur1_arr[phase_idx][sur0][sur1] << " " << eval_canput0_canput1_arr[phase_idx][canput0][canput1] << " "
    //    << eval_stab0_stab1_arr[phase_idx][stab0][stab1] << " " << eval_num0_num1_arr[phase_idx][num0][num1] << " " << calc_canput_pattern(phase_idx, b, player_mobility, opponent_mobility) << endl;
    int res = calc_pattern_first_cuda(phase_idx, b) + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1] + 
        eval_canput0_canput1_arr[phase_idx][canput0][canput1] + 
        eval_stab0_stab1_arr[phase_idx][stab0][stab1] + 
        eval_num0_num1_arr[phase_idx][num0][num1] + 
        calc_canput_pattern_cuda(phase_idx, player_mobility, opponent_mobility);
    //return score_modification(phase_idx, res);
    //cerr << res << endl;
    #if EVALUATION_STEP_WIDTH_MODE == 0
        if (res > 0)
            res += STEP_2;
        else if (res < 0)
            res -= STEP_2;
        res /= STEP;
    #elif EVALUATION_STEP_WIDTH_MODE == 1
        if (res > 0)
            res += STEP;
        else if (res < 0)
            res -= STEP;
        res /= STEP * 2;
    #elif EVALUATION_STEP_WIDTH_MODE == 2
        if (res > 0)
            res += STEP / 4;
        else if (res < 0)
            res -= STEP / 4;
        res /= STEP_2;
    
    #elif EVALUATION_STEP_WIDTH_MODE == 3
        if (res > 0)
            res += STEP / 8;
        else if (res < 0)
            res -= STEP / 8;
        res /= STEP / 4;
    #elif EVALUATION_STEP_WIDTH_MODE == 4
        if (res > 0)
            res += STEP / 16;
        else if (res < 0)
            res -= STEP / 16;
        res /= STEP / 8;
    #elif EVALUATION_STEP_WIDTH_MODE == 5
        if (res > 0)
            res += STEP / 32;
        else if (res < 0)
            res -= STEP / 32;
        res /= STEP / 16;
    #elif EVALUATION_STEP_WIDTH_MODE == 6
        if (res > 0)
            res += STEP / 64;
        else if (res < 0)
            res -= STEP / 64;
        res /= STEP / 32;
    #endif
    //cerr << res << " " << value_to_score_double(res) << endl;
    return max(-SCORE_MAX, min(SCORE_MAX, res));
}
*/

__device__ inline int end_evaluate_cuda(Board_simple *b){
    int p = __popcll(b->player);
    int o = __popcll(b->opponent);
    if (p == o) return 0;
    if (p > o) return 64 - 2 * o;
    else return 2 * p - 64;
}

// original code https://gist.github.com/primenumber/82443a22f8a789b54213229efdc17ba9
// modified by Nyanyan
__constant__ uint64_t mask1[4] = {
    0x0080808080808080ULL,
    0x7f00000000000000ULL,
    0x0102040810204000ULL,
    0x0040201008040201ULL
};
__constant__ uint64_t mask2[4] = {
    0x0101010101010100ULL,
    0x00000000000000feULL,
    0x0002040810204080ULL,
    0x8040201008040200ULL
};

__device__ uint64_t flip(const Board_simple &bd, int pos, int index) {
    uint64_t om = bd.opponent;
    if (index) om &= 0x7E7E7E7E7E7E7E7EULL;
    uint64_t mask = mask1[index] >> (63 - pos);
    uint64_t outflank = (0x8000000000000000ULL >> __clzll(~om & mask)) & bd.player;
    uint64_t flipped = (-outflank * 2) & mask;
    mask = mask2[index] << pos;
    outflank = mask & ((om | ~mask) + 1) & bd.player;
    flipped |= (outflank - (outflank != 0)) & mask;
    return flipped;
}

__device__ uint64_t flip_all(const Board_simple &bd, int pos) {
    return flip(bd, pos, 0) | flip(bd, pos, 1) | flip(bd, pos, 2) | flip(bd, pos, 3);
}

struct Arrays {
    const Board_simple *bd_ary;
    int *res_ary;
    int *nodes_count;
    size_t index;
    const size_t size;
};

__device__ bool get_next_node(Arrays &arys, const int node_index, const int simd_index, const int res) {
    if (simd_index == 0) {
        arys.res_ary[arys.index] = res;
        arys.nodes_count[arys.index] = count_cuda[threadIdx.x];
    }
    arys.index += (blockDim.x * gridDim.x) / SIMD_WIDTH;
    if (arys.index >= arys.size) return true;
    count_cuda[threadIdx.x] = 1; 
    if (simd_index == 0) {
        Node_cuda &root = nodes_stack[node_index][0];
        root.board = arys.bd_ary[arys.index];
        root.puttable = puttable(root.board.player, root.board.opponent);
        root.alpha = -64;
        root.beta = 64;
        root.pass = true;
        root.passed = false;
    }
    return false;
}

__device__ void alpha_beta(Arrays &arys) {
    int node_index = threadIdx.x / SIMD_WIDTH;
    int simd_index = threadIdx.x % SIMD_WIDTH;
    int stack_index = 0;
    while (true) {
        Node_cuda &node = nodes_stack[node_index][stack_index];
        /*
        if (node.depth == 0){
            if (get_next_node(arys, node_index, simd_index, -mid_evaluate_cuda(&node.board)))
                return;
        }
        */
        if (node.puttable == 0) {
            if (node.pass) {
                if (node.passed) {
                    if (stack_index) {
                        Node_cuda &parent = nodes_stack[node_index][stack_index-1];
                        if (simd_index == 0) {
                            parent.update(-end_evaluate_cuda(&node.board));
                        }
                        --stack_index;
                } else {
                    if (get_next_node(arys, node_index, simd_index, -end_evaluate_cuda(&node.board)))
                        return;
                }
                } else {
                    if (simd_index == 0) {
                        uint64_t swap_tmp = node.board.player;
                        node.board.player = node.board.opponent;
                        node.board.opponent = swap_tmp;
                        node.puttable = puttable(node.board.player, node.board.opponent);
                        int tmp = node.alpha;
                        node.alpha = -node.beta;
                        node.beta = -tmp;
                        node.passed = true;
                    }
                }
            } else {
                if (stack_index) {
                    Node_cuda &parent = nodes_stack[node_index][stack_index-1];
                    if (simd_index == 0) {
                        parent.update((node.passed ? -1 : 1) * node.alpha);
                    }
                    --stack_index;
                } else {
                    if (get_next_node(arys, node_index, simd_index, (node.passed ? -1 : 1) * node.alpha))
                        return;
                }
            }
        } else if (node.alpha >= node.beta) {
            if (stack_index) {
                Node_cuda &parent = nodes_stack[node_index][stack_index-1];
                if (simd_index == 0) {
                    parent.update((node.passed ? -1 : 1) * node.alpha);
                }
                --stack_index;
            } else {
                if (get_next_node(arys, node_index, simd_index, (node.passed ? -1 : 1) * node.alpha))
                    return;
            }
        } else {
            uint64_t bit = node.puttable & -node.puttable;
            if (simd_index == 0) {
                node.puttable ^= bit;
            }
            int pos = __popcll(bit-1);
            uint64_t flipped = flip(node.board, pos, simd_index);
            flipped |= __shfl_xor(flipped, 1);
            flipped |= __shfl_xor(flipped, 2);
            if (flipped) {
                ++stack_index;
                if (simd_index == 0) {
                    Node_cuda &next = nodes_stack[node_index][stack_index];
                    node.pass = false;
                    next.board.player = node.board.opponent ^ flipped;
                    next.board.opponent = (node.board.player ^ flipped) | bit;
                    next.puttable = puttable(next.board.player, next.board.opponent);
                    next.alpha = -node.beta;
                    next.beta = -node.alpha;
                    next.pass = true;
                    next.passed = false;
                    next.depth = node.depth - 1;
                    ++count_cuda[threadIdx.x];
                }
            }
        }
    }
}

__global__ void search_cuda(const Board_simple *bd_ary, int *res_ary, int *nodes_count, const size_t size, int depth) {
    size_t index = (threadIdx.x + blockIdx.x * blockDim.x) / SIMD_WIDTH;
    int simd_index = threadIdx.x % SIMD_WIDTH;
    int node_index = threadIdx.x / SIMD_WIDTH;
    count_cuda[threadIdx.x] = 1; 
    if (simd_index == 0) {
        Node_cuda &root = nodes_stack[node_index][0];
        root.board = bd_ary[index];
        root.puttable = puttable(root.board.player, root.board.opponent);
        root.alpha = -SCORE_MAX;
        root.beta = SCORE_MAX;
        root.pass = true;
        root.passed = false;
        root.depth = depth;
    }
    Arrays arys = {
        bd_ary,
        res_ary,
        nodes_count,
        index,
        size
    };
    alpha_beta(arys);
}

// end of modification

extern "C" int do_search_cuda(vector<Board_simple> &boards, int depth, bool is_end_search){
    if (is_end_search)
        ++depth;
    const size_t n = boards.size();
    //cerr << "start gpu search depth = " << depth << " size = " << n << endl;
    Board_simple *bd_ary_host = (Board_simple*)malloc(sizeof(Board_simple) * n);
    for (int i = 0; i < n; ++i){
        bd_ary_host[i].player = boards[i].player;
        bd_ary_host[i].opponent = boards[i].opponent;
    }
    Board_simple *bd_ary;
    int *res_ary;
    int *nodes_count;
    cudaMalloc(&bd_ary, sizeof(Board_simple) * n);
    cudaMallocManaged(&res_ary, sizeof(int) * n);
    cudaMallocManaged(&nodes_count, sizeof(int) * n);
    cudaMemcpy(bd_ary, bd_ary_host, sizeof(Board_simple) * n, cudaMemcpyHostToDevice);
    cudaMemset(res_ary, 0, sizeof(int) * n);
    cudaMemset(nodes_count, 0, sizeof(int) * n);
    search_cuda<<<256, THREADS_PER_BLOCK>>>(bd_ary, res_ary, nodes_count, n, depth);
    cudaDeviceSynchronize();
    //cerr << "gpu calculating done" << endl;
    //for (int i = 0; i < n; ++i)
    //    cerr << res_ary[i] << endl;
    int res = -INF;
    for (int i = 0; i < n; ++i)
        res = max(res, -res_ary[i]);
    //cerr << "done val = " << res << endl;
    return res;
}