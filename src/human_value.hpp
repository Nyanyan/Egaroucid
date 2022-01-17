#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "book.hpp"
#include "evaluate.hpp"
#include "transpose_table.hpp"
#include "midsearch.hpp"

#define n_kernels 64
#define n_board_input 2
#define kernel_size 3
#define n_residual 2
#define conv_size (hw_p1 - kernel_size)
#define conv_padding (kernel_size / 2)
#define conv_padding2 (conv_padding * 2)


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

inline void human_value_init(){
    line_distance.init();
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
    /*
    if (l != u && depth > simple_mid_threshold){
        res.first = inf;
        return res;
    }
    */
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

inline vector<principal_variation> search_pv(board b, long long strt, int max_depth){
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
        //for (int depth = min(7, max(0, max_depth - 5)); depth <= min(hw2 - b.n, max_depth - 1); ++depth){
        //    swap(transpose_table.now, transpose_table.prev);
        //    transpose_table.init_now();
        //    g = -mtd(&nnb, false, depth, -hw2, hw2, use_mpc, use_mpct);
        //}
        g = -mtd(&nnb, false, min(hw2 - b.n, max_depth - 1), -hw2, hw2, use_mpc, use_mpct);
        principal_variation pv;
        pv.value = g;
        g = book.get(&nnb);
        if (g != -inf)
            pv.value = -g;
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

inline double calc_divergence_distance(board b, vector<int> pv, int divergence[6], int max_depth){
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
            g = -book.get(&nnb);
            if (g == inf)
                g = -nega_alpha(&nnb, false, max_depth, -search_epsilon, search_epsilon);
            if (b.p == player){
                res += (double)max(-1, min(1, g)) * possibility[nnb.policy] / (double)nb.size();
                if (g > 0)
                    ++divergence[0];
                else if (g == 0)
                    ++divergence[1];
                else if (g < 0)
                    ++divergence[2];
            } else{
                res -= (double)max(-1, min(1, g)) * possibility[nnb.policy] / (double)nb.size();
                if (g > 0)
                    ++divergence[3];
                else if (g == 0)
                    ++divergence[4];
                else if (g < 0)
                    ++divergence[5];
            }
        }
    }
    res /= (double)pv.size();
    return res;
}

inline double evaluate_human(int value, int divergence[6], double line_distance){
    double val = (double)value * 0.75;
    double divergence1 = (double)(divergence[0] - divergence[3]) / (double)(divergence[0] + divergence[3]);
    double divergence2 = (double)(divergence[5] - divergence[2]) / (double)(divergence[5] + divergence[2]);
    double line_dist = line_distance * 0.01;
    //cerr << val << " " << divergence1 << " " << divergence2 << " " << line_dist << endl;
    return val + divergence1 + divergence2 + line_dist;
}

inline vector<search_result_pv> search_human(board b, long long strt, int max_depth, int sub_depth){
    cerr << "start midsearch human" << endl;
    vector<search_result_pv> res;
    vector<principal_variation> pv_value = search_pv(b, tim(), max_depth);
    for (principal_variation pv: pv_value){
        search_result_pv res_elem;
        res_elem.value = pv.value;
        res_elem.depth = pv.depth;
        res_elem.nps = pv.nps;
        res_elem.policy = pv.policy;
        res_elem.value = pv.value;
        res_elem.line_distance = calc_divergence_distance(b, pv.pv, res_elem.divergence, sub_depth);
        res_elem.concat_value = res_elem.line_distance * 10; //evaluate_human(res_elem.value, res_elem.divergence, res_elem.line_distance);
        cerr << "value: " << res_elem.value << " human value: " << res_elem.concat_value << " policy: " << res_elem.policy << endl;
        //cerr << "divergence cout: ";
        //for (int i = 0; i < 6; ++i)
        //    cerr << res_elem.divergence[i] << " ";
        //cerr << endl;
        res.emplace_back(res_elem);
    }
    sort(res.begin(), res.end());
    return res;
}