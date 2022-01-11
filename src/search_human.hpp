#pragma once
#include <iostream>
#include <fstream>
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
#define mpc_max_depth 24
#define mpc_min_depth_final 9
#define mpc_max_depth_final 30

#define simple_mid_threshold 3
#define simple_end_threshold 7

#define po_max_depth 8

#define extra_stability_threshold 58

#define n_kernels 16
#define n_board_input 2
#define kernel_size 3
#define n_residual 3

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
        {0.921, 1.051, 0.89, 1.431, 1.02, 1.042, 1.074, 1.353, 0.956, 1.063, 0.511, 0.772, 0.577, 0.977, 0.598, 0.832, 1.077, 0.797, 1.238, 0.562, 0.445, 0.616},
        {0.991, 1.212, 1.41, 1.971, 1.49, 1.712, 1.833, 1.613, 2.058, 1.24, 1.546, 1.577, 1.204, 1.43, 1.503, 1.099, 0.991, 1.773, 1.109, 0.721, 0.751, 0.663},
        {2.169, 1.839, 1.479, 2.111, 2.023, 1.629, 2.343, 2.304, 2.185, 2.104, 1.569, 2.341, 1.225, 2.163, 1.396, 1.264, 1.563, 1.511, 2.153, 1.046, 0.911, 0.897},
        {2.286, 2.172, 1.928, 2.496, 2.732, 1.956, 2.741, 2.365, 2.902, 2.358, 1.959, 2.713, 2.457, 1.883, 1.87, 1.518, 1.586, 1.502, 1.194, 1.045, 1.162, 1.104},
        {2.379, 2.499, 2.445, 3.675, 2.447, 2.651, 2.897, 2.719, 2.819, 2.56, 2.063, 2.818, 2.09, 2.573, 2.442, 2.123, 1.706, 2.011, 1.251, 1.261, 1.246, 0.798},
        {2.901, 2.56, 2.116, 3.136, 2.585, 2.713, 3.294, 3.384, 3.385, 2.72, 2.345, 3.155, 2.207, 2.278, 2.322, 2.273, 1.911, 1.689, 1.267, 1.484, 1.022, 1.788},
        {3.057, 2.745, 2.584, 3.108, 3.612, 2.723, 2.84, 3.081, 3.51, 3.052, 2.794, 3.14, 2.775, 3.169, 2.915, 3.213, 2.721, 2.458, 1.642, 1.841, 1.486, 1.347},
        {3.321, 2.976, 2.852, 3.75, 3.775, 3.725, 3.806, 3.809, 5.513, 4.158, 4.163, 3.103, 4.225, 3.46, 2.623, 2.913, 2.312, 3.955, 1.682, 1.961, 1.635, 1.519},
        {3.581, 3.33, 3.037, 5.371, 3.448, 3.639, 4.9, 4.205, 4.986, 4.194, 2.427, 3.268, 3.135, 2.507, 2.732, 2.143, 1.321, 3.856, 1.982, 1.586, 1.871, 1.303},
        {2.971, 2.486, 2.12, 3.224, 1.877, 1.947, 2.174, 1.634, 2.218, 1.147, 0.592, 1.954, 0.64, 0.315, 0.898, 0.288, 0.0, 0.0, 2.029, 1.18, 0.2, 0.0}
    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        5.044, 4.774, 5.326, 5.007, 4.952, 5.185, 4.948, 4.743, 4.985, 4.68, 4.657, 4.652, 4.417, 4.718, 4.643, 4.437, 4.268, 4.111, 4.032, 3.933, 3.905, 4.491
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
                            conv1[i][j][k][l] = stof(line);
                        }
                    }
                }
            }
            for (ri = 0; ri < n_residual; ++ri){
                for (i = 0; i < n_kernels; ++i){
                    for (j = 0; j < n_board_input; ++j){
                        for (k = 0; k < kernel_size; ++k){
                            for (l = 0; l < kernel_size; ++l){
                                getline(ifs, line);
                                conv_residual[ri][i][j][k][l] = stof(line);
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
                bias1[j][i] = stof(line);
            }
        }

        inline void predict(board b, double res[hw2]){
            
        }

    private:
        inline double leaky_relu(double x){
            return max(x, 0.01 * x);
        }
};

line_distance line_distance;

inline void search_init(){
    line_distance.init();
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

pair<int, vector<int>> create_principal_variation(board *b, bool skipped, int depth, int alpha, int beta){
    ++searched_nodes;
    pair<int, vector<int>> res;
    if (depth == 0){
        res.first = mid_evaluate(b);
        return res;
    }
    int hash = (int)(b->hash() & search_hash_mask);
    int l, u;
    transpose_table.get_now(b, hash, &l, &u);
    alpha = max(alpha, l);
    beta = min(beta, u);
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            nb.emplace_back(b->move(cell));
            move_ordering(&nb[canput]);
            ++canput;
        }
    }
    if (canput == 0){
        if (skipped){
            res.first = end_evaluate(b);
            return res;
        }
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        res = create_principal_variation(&rb, true, depth, -beta, -alpha);
        res.first = -res.first;
        return res;
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int v = -inf;
    pair<int, vector<int>> fail_low_res;
    for (board nnb: nb){
        res = create_principal_variation(&nnb, false, depth - 1, -beta, -alpha);
        res.first = -res.first;
        if (beta <= res.first){
            res.second.emplace_back(nnb.policy);
            return res;
        }
        alpha = max(alpha, res.first);
        if (v < res.first){
            v = res.first;
            fail_low_res.first = res.first;
            fail_low_res.second.clear();
            for (const int &elem: res.second)
                fail_low_res.second.emplace_back(elem);
            fail_low_res.second.emplace_back(nnb.policy);
        }
    }
    return fail_low_res;
}

inline vector<principal_variation> midsearch_pv(board b, long long strt, int max_depth){
    cerr << "start pv midsearch depth " << max_depth << endl;
    vector<principal_variation> res;
    vector<board> nb;
    for (const int &cell: vacant_lst){
        if (b.legal(cell)){
            nb.push_back(b.move(cell));
        }
    }
    int canput = nb.size();
    cerr << "canput: " << canput << endl;
    int g;
    searched_nodes = 0;
    transpose_table.hash_get = 0;
    transpose_table.hash_reg = 0;
    bool use_mpc = max_depth >= 11 ? true : false;
    double use_mpct = 2.0;
    if (max_depth >= 13)
        use_mpct = 1.7;
    if (max_depth >= 15)
        use_mpct = 1.5;
    if (max_depth >= 17)
        use_mpct = 1.3;
    if (max_depth >= 19)
        use_mpct = 1.1;
    if (max_depth >= 21)
        use_mpct = 0.8;
    if (max_depth >= 23)
        use_mpct = 0.6;
    for (board nnb: nb){
        transpose_table.init_now();
        transpose_table.init_prev();
        for (int depth = min(11, max(0, max_depth - 5)); depth <= min(hw2 - b.n, max_depth - 1); ++depth){
            transpose_table.init_now();
            g = -mtd(&nnb, false, depth, -hw2, hw2, use_mpc, use_mpct);
            swap(transpose_table.now, transpose_table.prev);
        }
        principal_variation pv;
        pv.value = g;
        pv.depth = min(hw2 - b.n, max_depth - 1) + 1;
        pv.nps = 0;
        pv.policy = nnb.policy;
        pv.pv = create_principal_variation(&nnb, false, min(hw2 - b.n, max_depth - 1), -hw2, hw2).second;
        pv.pv.emplace_back(nnb.policy);
        reverse(pv.pv.begin(), pv.pv.end());
        //cerr << "value: " << g << endl;
        cerr << "principal variation: ";
        for (const int &elem: pv.pv)
            cerr << elem << " ";
        cerr << endl;
        res.emplace_back(pv);
    }
    return res;
}

inline double calc_divergence_distance(board b, vector<int> pv, int divergence[6], int depth){
    double res = 0.0;
    for (int i = 0; i < 6; ++i)
        divergence[i] = 0;
    int g, player = b.p;
    double possibility[hw2];
    for (const int &policy: pv){
        b = b.move(policy);
        vector<board> nb;
        for (const int &cell: vacant_lst){
            if (b.legal(cell))
                nb.push_back(b.move(cell));
        }
        if (nb.size() == 0){
            b.p = 1 - b.p;
            for (const int &cell: vacant_lst){
                if (b.legal(cell))
                    nb.push_back(b.move(cell));
            }
            if (nb.size() == 0)
                break;
        }
        line_distance.predict(b, possibility);
        for (board nnb: nb){
            g = -mtd(&nnb, false, depth, -hw2, hw2, false, -1.0);
            if (b.p == player){
                res += (double)g * possibility[nnb.policy];
                if (g > 0)
                    ++divergence[0];
                else if (g == 0)
                    ++divergence[1];
                else if (g < 0)
                    ++divergence[2];
            } else{
                res -= (double)g * possibility[nnb.policy];
                if (g > 0)
                    ++divergence[3];
                else if (g == 0)
                    ++divergence[4];
                else if (g < 0)
                    ++divergence[5];
            }
        }
    }
    return res;
}

inline double evaluate_human(int value, int divergence[6], double line_distance){
    double res = 
        (double)value * 0.01 + 
        (double)(divergence[0] - divergence[3]) / (double)(divergence[0] + divergence[3]) + 
        (double)(divergence[5] - divergence[2]) / (double)(divergence[5] + divergence[2]) + 
        line_distance;
    return res;
}