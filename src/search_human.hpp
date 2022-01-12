#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "transpose_table.hpp"

using namespace std;

#define search_epsilon 1
constexpr int cache_hit = 100;
constexpr int cache_both = 100;
constexpr int parity_vacant_bonus = 10;
constexpr int canput_bonus = 0;

#define mpc_min_depth 3
#define mpc_max_depth 10
#define mpc_min_depth_final 9
#define mpc_max_depth_final 30

#define simple_mid_threshold 3
#define simple_end_threshold 7

#define po_max_depth 8

#define extra_stability_threshold 58

#define n_kernels 32
#define n_board_input 2
#define kernel_size 3
#define n_residual 2
#define conv_size (hw_p1 - kernel_size)
#define conv_padding (kernel_size / 2)
#define conv_padding2 (conv_padding * 2)

const int cell_weight[hw2] = {
    10, 3, 9, 7, 7, 9, 3, 10, 
    3, 2, 4, 5, 5, 4, 2, 3, 
    9, 4, 8, 6, 6, 8, 4, 9, 
    7, 5, 6, 0, 0, 6, 5, 7, 
    7, 5, 6, 0, 0, 6, 5, 7, 
    9, 4, 8, 6, 6, 8, 4, 9, 
    3, 2, 4, 5, 5, 4, 2, 3, 
    10, 3, 9, 7, 7, 9, 3, 10
};

const int mpcd[32] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 8, 9};
#if USE_MID_SMOOTH
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={

    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {

    };
#else
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
    };
#endif
unsigned long long can_be_flipped[hw2];

unsigned long long searched_nodes;
vector<int> vacant_lst;

struct search_result{
    int policy;
    int value;
    int depth;
    int nps;
};

struct principal_variation{
    int policy;
    int value;
    int depth;
    int nps;
    vector<int> pv;
};

struct search_result_pv{
    int policy;
    int value;
    int divergence[6];
    double line_distance;
    double concat_value;
    int depth;
    int nps;
    bool operator<(const search_result_pv& another) const {
        return concat_value > another.concat_value;
    }
};

class line_distance{
    private:
        double conv1[n_kernels][n_board_input][kernel_size][kernel_size];
        double conv_residual[n_residual][n_kernels][n_kernels][kernel_size][kernel_size];
        double dense1[hw2][n_kernels];
        double bias1[hw2];
    
    public:
        inline void init(){
            ifstream ifs("resources/line_distance.txt");
            if (ifs.fail()){
                cerr << "evaluation file not exist" << endl;
                exit(1);
            }
            string line;
            int i, j, k, l, ri;
            for (i = 0; i < n_kernels; ++i){
                for (j = 0; j < n_board_input; ++j){
                    for (k = 0; k < kernel_size; ++k){
                        for (l = 0; l < kernel_size; ++l){
                            getline(ifs, line);
                            conv1[i][j][l][k] = stof(line);
                        }
                    }
                }
            }
            for (ri = 0; ri < n_residual; ++ri){
                for (i = 0; i < n_kernels; ++i){
                    for (j = 0; j < n_kernels; ++j){
                        for (k = 0; k < kernel_size; ++k){
                            for (l = 0; l < kernel_size; ++l){
                                getline(ifs, line);
                                conv_residual[ri][i][j][l][k] = stof(line);
                            }
                        }
                    }
                }
            }
            for (i = 0; i < n_kernels; ++i){
                for (j = 0; j < hw2; ++j){
                    getline(ifs, line);
                    dense1[j][i] = stof(line);
                }
            }
            for (i = 0; i < hw2; ++i){
                getline(ifs, line);
                bias1[i] = stof(line);
            }
            cerr << "line distance initialized" << endl;
        }

        inline void predict(board b, double res[hw2]){
            int board_arr[hw2];
            double board_input[n_board_input][hw + conv_padding2][hw + conv_padding2];
            double hidden_conv1[n_kernels][hw + conv_padding2][hw + conv_padding2];
            double hidden_conv2[n_kernels][hw + conv_padding2][hw + conv_padding2];
            double after_conv[n_kernels];
            double max_res = -inf, sum_softmax = 0.0;
            int ri, i, j, y, x, sy, sx;
            // reshape board
            b.translate_to_arr(board_arr);
            for (i = 0; i < hw + conv_padding2; ++i){
                for (j = 0; j < hw + conv_padding2; ++j){
                    board_input[0][i][j] = 0.0;
                    board_input[1][i][j] = 0.0;
                }
            }
            for (i = 0; i < hw; ++i){
                for (j = 0; j < hw; ++j){
                    if (board_arr[i * hw + j] == b.p)
                        board_input[0][i + conv_padding][j + conv_padding] = 1.0;
                    else if (board_arr[i * hw + j] == 1 - b.p)
                        board_input[1][i + conv_padding][j + conv_padding] = 1.0;
                }
            }
            // conv and leaky-relu
            for (i = 0; i < n_kernels; ++i){
                for (y = 0; y < hw + conv_padding2; ++y){
                    for (x = 0; x < hw + conv_padding2; ++x)
                        hidden_conv1[i][y][x] = 0.0;
                }
                for (j = 0; j < n_board_input; ++j){
                    for (sy = 0; sy < hw; ++sy){
                        for (sx = 0; sx < hw; ++sx){
                            for (y = 0; y < kernel_size; ++y){
                                for (x = 0; x < kernel_size; ++x)
                                    hidden_conv1[i][sy + conv_padding][sx + conv_padding] += conv1[i][j][y][x] * board_input[j][sy + y][sx + x];
                            }
                        }
                    }
                }
                for (y = conv_padding; y < hw + conv_padding; ++y){
                    for (x = conv_padding; x < hw + conv_padding; ++x)
                        hidden_conv1[i][y][x] = leaky_relu(hidden_conv1[i][y][x]);
                }
            }
            // residual block
            for (ri = 0; ri < n_residual; ++ri){
                for (i = 0; i < n_kernels; ++i){
                    for (y = 0; y < hw + conv_padding2; ++y){
                        for (x = 0; x < hw + conv_padding2; ++x)
                            hidden_conv2[i][y][x] = 0.0;
                    }
                    for (j = 0; j < n_kernels; ++j){
                        for (sy = 0; sy < hw; ++sy){
                            for (sx = 0; sx < hw; ++sx){
                                for (y = 0; y < kernel_size; ++y){
                                    for (x = 0; x < kernel_size; ++x)
                                        hidden_conv2[i][sy + conv_padding][sx + conv_padding] += conv_residual[ri][i][j][y][x] * hidden_conv1[j][sy + y][sx + x];
                                }
                            }
                        }
                    }
                }
                for (i = 0; i < n_kernels; ++i){
                    for (y = conv_padding; y < hw + conv_padding; ++y){
                        for (x = conv_padding; x < hw + conv_padding; ++x)
                            hidden_conv1[i][y][x] = leaky_relu(hidden_conv1[i][y][x] + hidden_conv2[i][y][x]);
                    }
                }
            }
            // global-average-pooling and leaky_relu
            for (i = 0; i < n_kernels; ++i){
                after_conv[i] = 0.0;
                for (y = 0; y < hw; ++y){
                    for (x = 0; x < hw; ++x)
                        after_conv[i] += hidden_conv1[i][y + conv_padding][x + conv_padding];
                }
                after_conv[i] = leaky_relu(after_conv[i] / hw2);
            }
            // dense1 for policy
            for (j = 0; j < hw2; ++j){
                res[j] = bias1[j];
                for (i = 0; i < n_kernels; ++i)
                    res[j] += dense1[j][i] * after_conv[i];
                max_res = max(max_res, res[j]);
            }
            // softmax
            for (i = 0; i < hw2; ++i){
                res[i] = exp(res[i] - max_res);
                sum_softmax += res[i];
            }
            for (i = 0; i < hw2; ++i)
                res[i] /= sum_softmax;
        }

    private:
        inline double leaky_relu(double x){
            return max(x, 0.01 * x);
        }
};

line_distance line_distance;

inline void search_init(){
    #if !MPC_MODE
        line_distance.init();
    #endif
    int i;
    for (int cell = 0; cell < hw2; ++cell){
        can_be_flipped[cell] = 0b1111111110000001100000011000000110000001100000011000000111111111;
        for (i = 0; i < hw; ++i){
            if (global_place[place_included[cell][0]][i] != -1)
                can_be_flipped[cell] |= 1ULL << global_place[place_included[cell][0]][i];
        }
        for (i = 0; i < hw; ++i){
            if (global_place[place_included[cell][1]][i] != -1)
                can_be_flipped[cell] |= 1ULL << global_place[place_included[cell][1]][i];
        }
        for (i = 0; i < hw; ++i){
            if (global_place[place_included[cell][2]][i] != -1)
                can_be_flipped[cell] |= 1ULL << global_place[place_included[cell][2]][i];
        }
        if (place_included[cell][3] != -1){
            for (i = 0; i < hw; ++i){
                if (global_place[place_included[cell][3]][i] != -1)
                    can_be_flipped[cell] |= 1ULL << global_place[place_included[cell][3]][i];
            }
        }
    }
    cerr << "search initialized" << endl;
}

int cmp_vacant(int p, int q){
    return cell_weight[p] > cell_weight[q];
}

inline void move_ordering(board *b){
    int l, u;
    transpose_table.get_prev(b, b->hash() & search_hash_mask, &l, &u);
    if (u != inf && l != -inf)
        b->v = -(u + l) / 2 + cache_hit + cache_both;
    else if (u != inf)
        b->v = -mid_evaluate(b) + cache_hit;
    else if (l != -inf)
        b->v = -mid_evaluate(b) + cache_hit;
    else
        b->v = -mid_evaluate(b);
}

inline void move_ordering_eval(board *b){
    b->v = -mid_evaluate(b);
}

inline void calc_extra_stability(board *b, int p, unsigned long long extra_stability, int *pres, int *ores){
    *pres = 0;
    *ores = 0;
    int y, x;
    extra_stability >>= hw;
    for (y = 1; y < hw_m1; ++y){
        extra_stability >>= 1;
        for (x = 1; x < hw_m1; ++x){
            if ((extra_stability & 1) == 0){
                if (pop_digit[b->b[y]][x] == p)
                    ++*pres;
                else if (pop_digit[b->b[y]][x] == 1 - p)
                    ++*ores;
            }
            extra_stability >>= 1;
        }
        extra_stability >>= 1;
    }
}

inline unsigned long long calc_extra_stability_ull(board *b){
    unsigned long long extra_stability = 0b1111111110000001100000011000000110000001100000011000000111111111;
    for (const int &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == vacant)
            extra_stability |= can_be_flipped[cell];
    }
    return extra_stability;
}

inline bool stability_cut(board *b, int *alpha, int *beta){
    if (b->n >= extra_stability_threshold){
        int ps, os;
        calc_extra_stability(b, b->p, calc_extra_stability_ull(b), &ps, &os);
        *alpha = max(*alpha, (2 * (calc_stability(b, b->p) + ps) - hw2));
        *beta = min(*beta, (hw2 - 2 * (calc_stability(b, 1 - b->p) + os)));
    } else{
        *alpha = max(*alpha, (2 * calc_stability(b, b->p) - hw2));
        *beta = min(*beta, (hw2 - 2 * calc_stability(b, 1 - b->p)));
    }
    return *alpha >= *beta;
}

inline int calc_canput_exact(board *b){
    int res = 0;
    for (const int &cell: vacant_lst)
        res += b->legal(cell);
    return res;
}
