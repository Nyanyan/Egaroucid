// Egaroucid4

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <unordered_map>
#include <random>
#include <time.h>

using namespace std;

#define hw 8
#define hw_m1 7
#define hw_p1 9
#define hw2 64
#define hw22 128
#define hw2_m1 63
#define hw2_mhw 56
#define hw2_p1 65
#define n_line 6561
#define max_evaluate_idx 59049
#define inf 1000000
#define b_idx_num 38

#define book_hash_table_size 8192
constexpr int book_hash_mask = book_hash_table_size - 1;
#define book_stones 64

#define search_hash_table_size 1048576
constexpr int search_hash_mask = search_hash_table_size - 1;

#define n_phases 4
#define n_patterns 11
#define n_dense0 16
#define n_dense1 16
#define n_add_input 3
#define n_add_dense0 8
#define n_add_dense1 8
#define n_all_input 19
#define max_canput 30
#define max_surround 50

#define epsilon 1
#define sc_w 6400
#define step 100
#define cache_hit 10000
#define cache_both 1000
#define mtd_threshold 400

#define mpc_min_depth 1
#define mpc_max_depth 10
#define mpc_min_depth_final 10
#define mpc_max_depth_final 28
#define mpct_final 1.1

#define p31 3
#define p32 9
#define p33 27
#define p34 81
#define p35 243
#define p36 729
#define p37 2187
#define p38 6561
#define p39 19683
#define p310 59049

struct board{
    int b[b_idx_num];
    int p;
    int policy;
    int v;
    int n;
    bool operator<(const board& another) const {
        return v > another.v;
    }
};

struct book_node{
    int k[4];
    int policies[35];
    int size;
    book_node* p_n_node;
};

struct search_node{
    bool reg;
    int k[4];
    int l;
    int u;
    //search_node* p_n_node;
};

struct search_result{
    int policy;
    int value;
    int depth;
    int nps;
};

const int idx_n_cell[b_idx_num] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3};
const int move_offset[b_idx_num] = {1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
const int global_place[b_idx_num][hw] = {
    {0, 1, 2, 3, 4, 5, 6, 7},{8, 9, 10, 11, 12, 13, 14, 15},{16, 17, 18, 19, 20, 21, 22, 23},{24, 25, 26, 27, 28, 29, 30, 31},{32, 33, 34, 35, 36, 37, 38, 39},{40, 41, 42, 43, 44, 45, 46, 47},{48, 49, 50, 51, 52, 53, 54, 55},{56, 57, 58, 59, 60, 61, 62, 63},
    {0, 8, 16, 24, 32, 40, 48, 56},{1, 9, 17, 25, 33, 41, 49, 57},{2, 10, 18, 26, 34, 42, 50, 58},{3, 11, 19, 27, 35, 43, 51, 59},{4, 12, 20, 28, 36, 44, 52, 60},{5, 13, 21, 29, 37, 45, 53, 61},{6, 14, 22, 30, 38, 46, 54, 62},{7, 15, 23, 31, 39, 47, 55, 63},
    {5, 14, 23, -1, -1, -1, -1, -1},{4, 13, 22, 31, -1, -1, -1, -1},{3, 12, 21, 30, 39, -1, -1, -1},{2, 11, 20, 29, 38, 47, -1, -1},{1, 10, 19, 28, 37, 46, 55, -1},{0, 9, 18, 27, 36, 45, 54, 63},{8, 17, 26, 35, 44, 53, 62, -1},{16, 25, 34, 43, 52, 61, -1, -1},{24, 33, 42, 51, 60, -1, -1, -1},{32, 41, 50, 59, -1, -1, -1, -1},{40, 49, 58, -1, -1, -1, -1, -1},
    {2, 9, 16, -1, -1, -1, -1, -1},{3, 10, 17, 24, -1, -1, -1, -1},{4, 11, 18, 25, 32, -1, -1, -1},{5, 12, 19, 26, 33, 40, -1, -1},{6, 13, 20, 27, 34, 41, 48, -1},{7, 14, 21, 28, 35, 42, 49, 56},{15, 22, 29, 36, 43, 50, 57, -1},{23, 30, 37, 44, 51, 58, -1, -1},{31, 38, 45, 52, 59, -1, -1, -1},{39, 46, 53, 60, -1, -1, -1, -1},{47, 54, 61, -1, -1, -1, -1, -1}
};
int place_included[hw2][4];
int pow3[11];
int mod3[n_line][hw];
int move_arr[2][n_line][hw][2];
bool legal_arr[2][n_line][hw];
int flip_arr[2][n_line][hw];
int put_arr[2][n_line][hw];
int local_place[b_idx_num][hw2];
const int cell_weight[hw2] = {
    120, -20, 20, 5, 5, 20, -20, 120,
    -20, -40, -5, -5, -5, -5, -40, -20,
    20, -5, 15, 3, 3, 15, -5, 20,
    5, -5, 3, 3, 3, 3, -5, 5,
    5, -5, 3, 3, 3, 3, -5, 5,
    20, -5, 15, 3, 3, 15, -5, 20,
    -20, -40, -5, -5, -5, -5, -40, -20,
    120, -20, 20, 5, 5, 20, -20, 120
};
int count_arr[n_line];
int count_all_arr[n_line];
int pop_digit[n_line][hw];
int pop_mid[n_line][hw][hw];
int reverse_board[n_line];
int canput_arr[2][n_line];
int surround_arr[2][n_line];
const int mpcd[30]={0,0,0,1,2,1,2,3,4,3,4,3,4,5,6,5,6,5,6,7,6,7,8,9,8,9,10,11,10,11};
const double mpct[6]={1.6,1.6,1.6,1.5,1.5,1.4};
const double mpcsd[6][mpc_max_depth-mpc_min_depth+1]={
    {482,512,352,298,474,372,349,323,463,335},
    {312,381,310,261,354,322,291,313,389,371},
    {389,463,392,352,548,402,422,441,530,538},
    {417,490,436,405,570,494,452,438,527,524},
    {486,554,519,463,635,665,555,550,635,581},
    {433,517,430,391,560,556,383,345,567,332}
};
const double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
    1564, 1309, 1440, 1453, 1152, 1580, 1394, 1655, 1381, 1578, 1660, 1452, 1574, 1725, 1655, 1514, 1378, 1622, 1489
};
int mpctsd[6][mpc_max_depth + 1];
int mpctsd_final[mpc_max_depth_final + 1];

vector<int> vacant_lst;
book_node *book[book_hash_table_size];
search_node search_replace_table[2][search_hash_table_size];
long long searched_nodes, hash_conf, hash_get, hash_reg;
int f_search_table_idx;
double pattern_arr[n_phases][n_patterns][max_evaluate_idx];
double add_arr[n_phases][max_canput * 2 + 1][max_surround + 1][max_surround + 1][n_add_dense1];
double all_dense[n_phases][n_all_input];
double all_bias[n_phases];

mt19937 raw_myrandom(time(0));

inline double myrandom(){
    return (double)raw_myrandom() / mt19937::max();
}

inline int myrandrange(int s, int e){
    return s +(int)((e - s) * myrandom());
}

inline long long tim(){
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

inline unsigned long long calc_hash(const int *p){
    return
        p[0] + 
        p[1] * 17 + 
        p[2] * 289 + 
        p[3] * 4913 + 
        p[4] * 83521 + 
        p[5] * 1419857 + 
        p[6] * 24137549 + 
        p[7] * 410338673;
}

inline bool compare_key(const int a[], const int b[]){
    for (int i = 0; i < 4; ++i){
        if (a[i * 2] + a[i * 2 + 1] * n_line != b[i])
            return false;
    }
    return true;
}

inline void print_board_line(int tmp){
    int j;
    string res = "";
    for (j = 0; j < hw; ++j){
        if (tmp % 3 == 0)
            res = "X " + res;
        else if (tmp % 3 == 1)
            res = "O " + res;
        else
            res = ". " + res;
        tmp /= 3;
    }
    cerr << res;
}

inline void print_board(const int* board){
    int i, j, tmp;
    string res;
    for (i = 0; i < hw; ++i){
        tmp = board[i];
        res = "";
        for (j = 0; j < hw; ++j){
            if (tmp % 3 == 0)
                res = "X " + res;
            else if (tmp % 3 == 1)
                res = "O " + res;
            else
                res = ". " + res;
            tmp /= 3;
        }
        cerr << res << endl;
    }
    cerr << endl;
}

inline int create_one_color(int idx, const int k){
    int res = 0;
    for (int i = 0; i < hw; ++i){
        if (idx % 3 == k){
            res |= 1 << i;
        }
        idx /= 3;
    }
    return res;
}

inline int trans(const int pt, const int k) {
    if (k == 0)
        return pt << 1;
    else
        return pt >> 1;
}

inline int move_line_half(const int p, const int o, const int place, const int k) {
    int mask;
    int res = 0;
    int pt = 1 << (hw_m1 - place);
    if (pt & p || pt & o)
        return res;
    mask = trans(pt, k);
    while (mask && (mask & o)) {
        ++res;
        mask = trans(mask, k);
        if (mask & p)
            return res;
    }
    return 0;
}

inline void init_pow(){
    int idx;
    pow3[0] = 1;
    for (idx = 1; idx < 11; ++idx)
        pow3[idx] = pow3[idx- 1] * 3;
}

inline void init_move(){
    int idx, b, w, place;
    for (idx = 0; idx < n_line; ++idx){
        b = create_one_color(idx, 0);
        w = create_one_color(idx, 1);
        count_arr[idx] = 0;
        count_all_arr[idx] = 0;
        reverse_board[idx] = 0;
        canput_arr[0][idx] = 0;
        canput_arr[1][idx] = 0;
        surround_arr[0][idx] = 0;
        surround_arr[1][idx] = 0;
        for (place = 0; place < hw; ++place){
            count_arr[idx] += 1 & (b >> place);
            count_arr[idx] -= 1 & (w >> place);
            count_all_arr[idx] += 1 & (b >> place);
            count_all_arr[idx] += 1 & (w >> place);
            reverse_board[idx] *= 3;
            if (1 & (b >> place))
                reverse_board[idx] += 0;
            else if (1 & (w >> place)) 
                reverse_board[idx] += 1;
            else
                reverse_board[idx] += 2;
            if (place > 0){
                if ((1 & (b >> (place - 1))) == 0 && (1 & (w >> (place - 1))) == 0){
                    if (1 & (b >> place))
                        ++surround_arr[0][idx];
                    else if (1 & (w >> place))
                        ++surround_arr[1][idx];
                }
            }
            if (place < hw_m1){
                if ((1 & (b >> (place + 1))) == 0 && (1 & (w >> (place + 1))) == 0){
                    if (1 & (b >> place))
                        ++surround_arr[0][idx];
                    else if (1 & (w >> place))
                        ++surround_arr[1][idx];
                }
            }
        }
        for (place = 0; place < hw; ++place){
            move_arr[0][idx][place][0] = move_line_half(b, w, place, 0);
            move_arr[0][idx][place][1] = move_line_half(b, w, place, 1);
            if (move_arr[0][idx][place][0] || move_arr[0][idx][place][1])
                legal_arr[0][idx][place] = true;
            else
                legal_arr[0][idx][place] = false;
            move_arr[1][idx][place][0] = move_line_half(w, b, place, 0);
            move_arr[1][idx][place][1] = move_line_half(w, b, place, 1);
            if (move_arr[1][idx][place][0] || move_arr[1][idx][place][1])
                legal_arr[1][idx][place] = true;
            else
                legal_arr[1][idx][place] = false;
            if (legal_arr[0][idx][place])
                ++canput_arr[0][idx];
            if (legal_arr[1][idx][place])
                ++canput_arr[1][idx];
        }
        for (place = 0; place < hw; ++place){
            flip_arr[0][idx][place] = idx;
            flip_arr[1][idx][place] = idx;
            put_arr[0][idx][place] = idx;
            put_arr[1][idx][place] = idx;
            if (b & (1 << (hw_m1 - place)))
                flip_arr[1][idx][place] += pow3[hw_m1 - place];
            else if (w & (1 << (hw_m1 - place)))
                flip_arr[0][idx][place] -= pow3[hw_m1 - place];
            else{
                put_arr[0][idx][place] -= pow3[hw_m1 - place] * 2;
                put_arr[1][idx][place] -= pow3[hw_m1 - place];
            }
        }
    }
}

inline void init_local_place(){
    int idx, place, l_place;
    for (idx = 0; idx < b_idx_num; ++idx){
        for (place = 0; place < hw2; ++place){
            local_place[idx][place] = -1;
            for (l_place = 0; l_place < hw; ++l_place){
                if (global_place[idx][l_place] == place)
                    local_place[idx][place] = l_place;
            }
        }
    }
}

inline void init_included(){
    int idx, place, l_place, inc_idx;
    for (place = 0; place < hw2; ++place){
        inc_idx = 0;
        for (idx = 0; idx < b_idx_num; ++idx){
            for (l_place = 0; l_place < hw; ++l_place){
                if (global_place[idx][l_place] == place)
                    place_included[place][inc_idx++] = idx;
            }
        }
        if (inc_idx == 3)
            place_included[place][inc_idx] = -1;
    }
}

inline void init_pop_digit(){
    int i, j;
    for (i = 0; i < n_line; ++i){
        for (j = 0; j < hw; ++j)
            pop_digit[i][j] = (i / pow3[hw_m1 - j]) % 3;
    }
}

inline void init_mod3(){
    int i, j;
    for (i = 0; i < n_line; ++i){
        for (j = 0; j < hw; ++j)
            mod3[i][j] = i % pow3[j];
    }
}

inline void init_pop_mid(){
    int i, j, k;
    for (i = 0; i < n_line; ++i){
        for (j = 0; j < hw; ++j){
            for (k = 0; k < hw; ++k)
                pop_mid[i][j][k] = (i - i / pow3[j] * pow3[j]) / pow3[k];
        }
    }
}

inline void init_mpc(){
    int i, j;
    for (i = 0; i < 6; ++i){
        for (j = 0; j < mpc_max_depth - mpc_min_depth + 1; ++j)
            mpctsd[i][mpc_min_depth + j] = (int)(mpct[i] * mpcsd[i][j]);
    }
    for (i = 0; i < mpc_max_depth_final - mpc_min_depth_final + 1; ++i)
        mpctsd_final[mpc_min_depth_final + i] = (int)(mpct_final * mpcsd_final[i]);
}

inline void flip(const board *b, board *res, int g_place){
    res->b[place_included[g_place][0]] = flip_arr[b->p][res->b[place_included[g_place][0]]][local_place[place_included[g_place][0]][g_place]];
    res->b[place_included[g_place][1]] = flip_arr[b->p][res->b[place_included[g_place][1]]][local_place[place_included[g_place][1]][g_place]];
    res->b[place_included[g_place][2]] = flip_arr[b->p][res->b[place_included[g_place][2]]][local_place[place_included[g_place][2]][g_place]];
    if (place_included[g_place][3] != -1)
        res->b[place_included[g_place][3]] = flip_arr[b->p][res->b[place_included[g_place][3]]][local_place[place_included[g_place][3]][g_place]];
}

inline void move_p(const board *b, board *res, int global_place, int i){
    int j, place;
    place = local_place[place_included[global_place][i]][global_place];
    for (j = 1; j <= move_arr[b->p][b->b[place_included[global_place][i]]][place][0]; ++j)
        flip(b, res, global_place - move_offset[place_included[global_place][i]] * j);
    for (j = 1; j <= move_arr[b->p][b->b[place_included[global_place][i]]][place][1]; ++j)
        flip(b, res, global_place + move_offset[place_included[global_place][i]] * j);
}

inline board move(const board *b, const int global_place){
    board res;
    int i;
    for (i = 0; i < b_idx_num; ++i)
        res.b[i] = b->b[i];
    for (i = 0; i < 3; ++i)
        move_p(b, &res, global_place, i);
    if (place_included[global_place][3] != -1){
        move_p(b, &res, global_place, 3);
    }
    for (i = 0; i < 3; ++i)
        res.b[place_included[global_place][i]] = put_arr[b->p][res.b[place_included[global_place][i]]][local_place[place_included[global_place][i]][global_place]];
    if (place_included[global_place][3] != -1)
        res.b[place_included[global_place][3]] = put_arr[b->p][res.b[place_included[global_place][3]]][local_place[place_included[global_place][3]][global_place]];
    res.p = 1 - b->p;
    res.n = b->n + 1;
    res.policy = global_place;
    return res;
}

inline void book_hash_table_init(book_node** hash_table){
    for(int i = 0; i < book_hash_table_size; ++i)
        hash_table[i] = NULL;
}

inline book_node* book_node_init(const int *key, int policy){
    book_node* p_node = NULL;
    p_node = (book_node*)malloc(sizeof(book_node));
    for (int i = 0; i < 4; ++i)
        p_node->k[i] = key[i * 2] + key[i * 2 + 1] * n_line;
    p_node->policies[0] = policy;
    p_node->size = 1;
    p_node->p_n_node = NULL;
    return p_node;
}

inline void register_book(book_node** hash_table, const int *key, int hash, int policy){
    if(hash_table[hash] == NULL){
        hash_table[hash] = book_node_init(key, policy);
    } else {
        book_node *p_node = hash_table[hash];
        book_node *p_pre_node = NULL;
        p_pre_node = p_node;
        while(p_node != NULL){
            if(compare_key(key, p_node->k)){
                p_node->policies[p_node->size++] = policy;
                return;
            }
            p_pre_node = p_node;
            p_node = p_node->p_n_node;
        }
        p_pre_node->p_n_node = book_node_init(key, policy);
    }
}

inline int get_book(const int *key){
    book_node *p_node = book[calc_hash(key) & book_hash_mask];
    while(p_node != NULL){
        if(compare_key(key, p_node->k)){
            return p_node->policies[myrandrange(0, p_node->size)];
        }
        p_node = p_node->p_n_node;
    }
    return -1;
}

inline void init_book(){
    int i;
    unordered_map<char, int> char_keys;
    const string book_chars = "!#$&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abc";
    string param_compressed1;
    for (i = 0; i < hw2; ++i)
        char_keys[book_chars[i]] = i;
    ifstream ifs("resources/book.txt");
    if (ifs.fail()){
        cerr << "book file not exist" << endl;
        exit(1);
    }
    getline(ifs, param_compressed1);
    int ln = param_compressed1.length();
    int coord;
    board fb;
    const int first_board[b_idx_num] = {6560, 6560, 6560, 6425, 6326, 6560, 6560, 6560, 6560, 6560, 6560, 6425, 6344, 6506, 6560, 6560, 6560, 6560, 6560, 6560, 6344, 6425, 6398, 6560, 6560, 6560, 6560, 6560, 6560, 6560, 6560, 6479, 6344, 6398, 6074, 6560, 6560, 6560};
    book_hash_table_init(book);
    int data_idx = 0;
    int n_book = 0;
    int y, x;
    int tmp[16];
    while (data_idx < ln){
        fb.p = 1;
        for (i = 0; i < b_idx_num; ++i)
            fb.b[i] = first_board[i];
        while (true){
            if (param_compressed1[data_idx] == ' '){
                ++data_idx;
                break;
            }
            coord = char_keys[param_compressed1[data_idx++]];
            fb = move(&fb, coord);
        }
        coord = char_keys[param_compressed1[data_idx++]];
        y = coord / hw;
        x = coord % hw;
        register_book(book, fb.b, calc_hash(fb.b) & book_hash_mask, y * hw + x);
        for (i = 0; i < 8; ++i)
            swap(fb.b[i], fb.b[8 + i]);
        register_book(book, fb.b, calc_hash(fb.b) & book_hash_mask, x * hw + y);
        for (i = 0; i < 16; ++i)
            tmp[i] = fb.b[i];
        for (i = 0; i < 8; ++i)
            fb.b[i] = reverse_board[tmp[7 - i]];
        for (i = 0; i < 8; ++i)
            fb.b[8 + i] = reverse_board[tmp[15 - i]];
        register_book(book, fb.b, calc_hash(fb.b) & book_hash_mask, (hw_m1 - x) * hw + (hw_m1 - y));
        for (i = 0; i < 8; ++i)
            swap(fb.b[i], fb.b[8 + i]);
        register_book(book, fb.b, calc_hash(fb.b) & book_hash_mask, (hw_m1 - y) * hw + (hw_m1 - x));
        n_book += 4;
    }
    cerr << n_book << " boards in book" << endl;
}

inline double leaky_relu(double x){
    return max(0.01 * x, x);
}

inline double predict(int pattern_size, double in_arr[], double dense0[n_dense0][20], double bias0[n_dense0], double dense1[n_dense1][n_dense0], double bias1[n_dense1], double dense2[n_dense1], double bias2){
    double hidden0[16], hidden1;
    int i, j;
    for (i = 0; i < n_dense0; ++i){
        hidden0[i] = bias0[i];
        for (j = 0; j < pattern_size * 2; ++j)
            hidden0[i] += in_arr[j] * dense0[i][j];
        hidden0[i] = leaky_relu(hidden0[i]);
    }
    double res = bias2;
    for (i = 0; i < n_dense1; ++i){
        hidden1 = bias1[i];
        for (j = 0; j < n_dense0; ++j)
            hidden1 += hidden0[j] * dense1[i][j];
        hidden1 = leaky_relu(hidden1);
        res += hidden1 * dense2[i];
    }
    res = leaky_relu(res);
    return res;
}

inline int calc_pop(int a, int b, int s){
    return (a / pow3[s - 1 - b]) % 3;
}

inline int calc_rev_idx(int pattern_idx, int pattern_size, int idx){
    int res = 0;
    if (pattern_idx <= 7){
        for (int i = 0; i < pattern_size; ++i)
            res += pow3[i] * calc_pop(idx, i, pattern_size);
    } else if (pattern_idx == 8){
        res += p39 * calc_pop(idx, 0, pattern_size);
        res += p38 * calc_pop(idx, 4, pattern_size);
        res += p37 * calc_pop(idx, 7, pattern_size);
        res += p36 * calc_pop(idx, 9, pattern_size);
        res += p35 * calc_pop(idx, 1, pattern_size);
        res += p34 * calc_pop(idx, 5, pattern_size);
        res += p33 * calc_pop(idx, 8, pattern_size);
        res += p32 * calc_pop(idx, 2, pattern_size);
        res += p31 * calc_pop(idx, 6, pattern_size);
        res += calc_pop(idx, 3, pattern_size);
    } else if (pattern_idx == 9){
        res += p39 * calc_pop(idx, 5, pattern_size);
        res += p38 * calc_pop(idx, 4, pattern_size);
        res += p37 * calc_pop(idx, 3, pattern_size);
        res += p36 * calc_pop(idx, 2, pattern_size);
        res += p35 * calc_pop(idx, 1, pattern_size);
        res += p34 * calc_pop(idx, 0, pattern_size);
        res += p33 * calc_pop(idx, 9, pattern_size);
        res += p32 * calc_pop(idx, 8, pattern_size);
        res += p31 * calc_pop(idx, 7, pattern_size);
        res += calc_pop(idx, 6, pattern_size);
    } else if (pattern_idx == 10){
        res += p39 * calc_pop(idx, 0, pattern_size);
        res += p38 * calc_pop(idx, 1, pattern_size);
        res += p37 * calc_pop(idx, 2, pattern_size);
        res += p36 * calc_pop(idx, 3, pattern_size);
        res += p35 * calc_pop(idx, 7, pattern_size);
        res += p34 * calc_pop(idx, 8, pattern_size);
        res += p33 * calc_pop(idx, 9, pattern_size);
        res += p32 * calc_pop(idx, 4, pattern_size);
        res += p31 * calc_pop(idx, 5, pattern_size);
        res += calc_pop(idx, 6, pattern_size);
    }
    return res;
}

inline void pre_evaluation(int pattern_idx, int phase_idx, int evaluate_idx, int pattern_size, double dense0[n_dense0][20], double bias0[n_dense0], double dense1[n_dense1][n_dense0], double bias1[n_dense1], double dense2[n_dense1], double bias2){
    int digit, idx, i;
    double arr[20], tmp_pattern_arr[max_evaluate_idx];
    for (idx = 0; idx < pow3[pattern_size]; ++idx){
        for (i = 0; i < pattern_size; ++i){
            digit = (idx / pow3[pattern_size - 1 - i]) % 3;
            if (digit == 0){
                arr[i] = 1.0;
                arr[pattern_size + i] = 0.0;
            } else if (digit == 1){
                arr[i] = 0.0;
                arr[pattern_size + i] = 1.0;
            } else{
                arr[i] = 0.0;
                arr[pattern_size + i] = 0.0;
            }
        }
        pattern_arr[phase_idx][evaluate_idx][idx] = predict(pattern_size, arr, dense0, bias0, dense1, bias1, dense2, bias2);
        tmp_pattern_arr[calc_rev_idx(pattern_idx, pattern_size, idx)] = pattern_arr[phase_idx][evaluate_idx][idx];
    }
    for (idx = 0; idx < pow3[pattern_size]; ++idx)
        pattern_arr[phase_idx][evaluate_idx][idx] += tmp_pattern_arr[idx];
}

inline void predict_add(int canput, int sur0, int sur1, double dense0[n_add_dense0][n_add_input], double bias0[n_add_dense0], double dense1[n_add_dense1][n_add_dense0], double bias1[n_add_dense1], double res[n_add_dense1]){
    double hidden0[n_add_dense0], in_arr[n_add_input];
    int i, j;
    in_arr[0] = (double)canput / 30.0;
    in_arr[1] = ((double)sur0 - 15.0) / 15.0;
    in_arr[2] = ((double)sur1 - 15.0) / 15.0;
    for (i = 0; i < n_add_dense0; ++i){
        hidden0[i] = bias0[i];
        for (j = 0; j < n_add_input; ++j)
            hidden0[i] += in_arr[j] * dense0[i][j];
        hidden0[i] = leaky_relu(hidden0[i]);
    }
    for (i = 0; i < n_add_dense1; ++i){
        res[i] = bias1[i];
        for (j = 0; j < n_add_dense0; ++j)
            res[i] += hidden0[j] * dense1[i][j];
        res[i] = leaky_relu(res[i]);
    }
}

inline void pre_evaluation_add(int phase_idx, double dense0[n_add_dense0][n_add_input], double bias0[n_add_dense0], double dense1[n_add_dense1][n_add_dense0], double bias1[n_add_dense1]){
    int canput, sur0, sur1;
    for (canput = -max_canput; canput <= max_canput; ++canput){
        for (sur0 = 0; sur0 <= max_surround; ++sur0){
            for (sur1 = 0; sur1 <= max_surround; ++sur1)
                predict_add(canput, sur0, sur1, dense0, bias0, dense1, bias1, add_arr[phase_idx][canput + max_canput][sur0][sur1]);
        }
    }
}

inline void init_evaluation(){
    ifstream ifs("resources/param.txt");
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    string line;
    int i, j, phase_idx, pattern_idx;
    double dense0[n_dense0][20];
    double bias0[n_dense0];
    double dense1[n_dense1][n_dense0];
    double bias1[n_dense1];
    double dense2[n_dense1];
    double bias2;
    double add_dense0[n_add_dense0][n_add_input];
    double add_bias0[n_add_dense0];
    double add_dense1[n_add_dense1][n_add_dense0];
    double add_bias1[n_add_dense1];
    const int pattern_sizes[n_patterns] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10};
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        for (pattern_idx = 0; pattern_idx < n_patterns; ++pattern_idx){
            for (i = 0; i < pattern_sizes[pattern_idx] * 2; ++i){
                for (j = 0; j < n_dense0; ++j){
                    getline(ifs, line);
                    dense0[j][i] = stof(line);
                }
            }
            for (i = 0; i < n_dense0; ++i){
                getline(ifs, line);
                bias0[i] = stof(line);
            }
            for (i = 0; i < n_dense0; ++i){
                for (j = 0; j < n_dense1; ++j){
                    getline(ifs, line);
                    dense1[j][i] = stof(line);
                }
            }
            for (i = 0; i < n_dense1; ++i){
                getline(ifs, line);
                bias1[i] = stof(line);
            }
            for (i = 0; i < n_dense1; ++i){
                getline(ifs, line);
                dense2[i] = stof(line);
            }
            getline(ifs, line);
            bias2 = stof(line);
            pre_evaluation(pattern_idx, phase_idx, pattern_idx, pattern_sizes[pattern_idx], dense0, bias0, dense1, bias1, dense2, bias2);
        }
        for (i = 0; i < n_add_input; ++i){
            for (j = 0; j < n_add_dense0; ++j){
                getline(ifs, line);
                add_dense0[j][i] = stof(line);
            }
        }
        for (i = 0; i < n_add_dense0; ++i){
            getline(ifs, line);
            add_bias0[i] = stof(line);
        }
        for (i = 0; i < n_add_dense0; ++i){
            for (j = 0; j < n_add_dense1; ++j){
                getline(ifs, line);
                add_dense1[j][i] = stof(line);
            }
        }
        for (i = 0; i < n_add_dense1; ++i){
            getline(ifs, line);
            add_bias1[i] = stof(line);
        }
        pre_evaluation_add(phase_idx, add_dense0, add_bias0, add_dense1, add_bias1);
        for (i = 0; i < n_all_input; ++i){
            getline(ifs, line);
            all_dense[phase_idx][i] = stof(line);
        }
        getline(ifs, line);
        all_bias[phase_idx] = stof(line);
    }
}

inline void search_hash_table_init(const int table_idx){
    for(int i = 0; i < search_hash_table_size; ++i)
        search_replace_table[table_idx][i].reg = false;
}

inline void register_search(const int table_idx, const int *key, int hash, int l, int u){
    ++hash_reg;
    search_replace_table[table_idx][hash].reg = true;
    for (int i = 0; i < 4; ++i)
        search_replace_table[table_idx][hash].k[i] = key[i * 2] + key[i * 2 + 1] * n_line;
    search_replace_table[table_idx][hash].l = l;
    search_replace_table[table_idx][hash].u = u;
}

inline void get_search(const int *key, const int hash, const int table_idx, int *l, int *u){
    if (search_replace_table[table_idx][hash].reg){
        if (compare_key(key, search_replace_table[table_idx][hash].k)){
            *l = search_replace_table[table_idx][hash].l;
            *u = search_replace_table[table_idx][hash].u;
            ++hash_get;
            return;
        }
    }
    *l = -inf;
    *u = -inf;
}

inline int calc_canput_exact(const board *b){
    int res = 0;
    bool legal;
    for (const int &global_place: vacant_lst){
        legal = legal_arr[b->p][b->b[place_included[global_place][0]]][local_place[place_included[global_place][0]][global_place]] || 
                legal_arr[b->p][b->b[place_included[global_place][1]]][local_place[place_included[global_place][1]][global_place]] || 
                legal_arr[b->p][b->b[place_included[global_place][2]]][local_place[place_included[global_place][2]][global_place]];
        if (place_included[global_place][3] != -1)
            legal |= legal_arr[b->p][b->b[place_included[global_place][3]]][local_place[place_included[global_place][3]][global_place]];
        res += legal;
    }
    return res;
}

inline int sfill5(int b){
    return pop_digit[b][2] != 2 ? b - p35 + 1 : b;
}

inline int sfill4(int b){
    return pop_digit[b][3] != 2 ? b - p34 + 1 : b;
}

inline int sfill3(int b){
    return pop_digit[b][4] != 2 ? b - p33 + 1 : b;
}

inline int sfill2(int b){
    return pop_digit[b][5] != 2 ? b - p32 + 1 : b;
}

inline int sfill1(int b){
    return pop_digit[b][6] != 2 ? b - p31 + 1 : b;
}

inline int calc_canput(const board *b){
    return (b->p ? -1 : 1) * 
        canput_arr[b->p][b->b[0]] + canput_arr[b->p][b->b[1]] + canput_arr[b->p][b->b[2]] + canput_arr[b->p][b->b[3]] + 
        canput_arr[b->p][b->b[4]] + canput_arr[b->p][b->b[5]] + canput_arr[b->p][b->b[6]] + canput_arr[b->p][b->b[7]] + 
        canput_arr[b->p][b->b[8]] + canput_arr[b->p][b->b[9]] + canput_arr[b->p][b->b[10]] + canput_arr[b->p][b->b[11]] + 
        canput_arr[b->p][b->b[12]] + canput_arr[b->p][b->b[13]] + canput_arr[b->p][b->b[14]] + canput_arr[b->p][b->b[15]] + 
        canput_arr[b->p][b->b[16] - p35 + 1] + canput_arr[b->p][b->b[26] - p35 + 1] + canput_arr[b->p][b->b[27] - p35 + 1] + canput_arr[b->p][b->b[37] - p35 + 1] + 
        canput_arr[b->p][b->b[17] - p34 + 1] + canput_arr[b->p][b->b[25] - p34 + 1] + canput_arr[b->p][b->b[28] - p34 + 1] + canput_arr[b->p][b->b[36] - p34 + 1] + 
        canput_arr[b->p][b->b[18] - p33 + 1] + canput_arr[b->p][b->b[24] - p33 + 1] + canput_arr[b->p][b->b[29] - p33 + 1] + canput_arr[b->p][b->b[35] - p33 + 1] + 
        canput_arr[b->p][b->b[19] - p32 + 1] + canput_arr[b->p][b->b[23] - p32 + 1] + canput_arr[b->p][b->b[30] - p32 + 1] + canput_arr[b->p][b->b[34] - p32 + 1] + 
        canput_arr[b->p][b->b[20] - p31 + 1] + canput_arr[b->p][b->b[22] - p31 + 1] + canput_arr[b->p][b->b[31] - p31 + 1] + canput_arr[b->p][b->b[33] - p31 + 1] + 
        canput_arr[b->p][b->b[21]] + canput_arr[b->p][b->b[32]];
}

inline int calc_surround(const board *b, int p){
    return surround_arr[p][b->b[0]] + surround_arr[p][b->b[1]] + surround_arr[p][b->b[2]] + surround_arr[p][b->b[3]] + 
        surround_arr[p][b->b[4]] + surround_arr[p][b->b[5]] + surround_arr[p][b->b[6]] + surround_arr[p][b->b[7]] + 
        surround_arr[p][b->b[8]] + surround_arr[p][b->b[9]] + surround_arr[p][b->b[10]] + surround_arr[p][b->b[11]] + 
        surround_arr[p][b->b[12]] + surround_arr[p][b->b[13]] + surround_arr[p][b->b[14]] + surround_arr[p][b->b[15]] + 
        surround_arr[p][sfill5(b->b[16])] + surround_arr[p][sfill5(b->b[26])] + surround_arr[p][sfill5(b->b[27])] + surround_arr[p][sfill5(b->b[37])] + 
        surround_arr[p][sfill4(b->b[17])] + surround_arr[p][sfill4(b->b[25])] + surround_arr[p][sfill4(b->b[28])] + surround_arr[p][sfill4(b->b[36])] + 
        surround_arr[p][sfill3(b->b[18])] + surround_arr[p][sfill3(b->b[24])] + surround_arr[p][sfill3(b->b[29])] + surround_arr[p][sfill3(b->b[35])] + 
        surround_arr[p][sfill2(b->b[19])] + surround_arr[p][sfill2(b->b[23])] + surround_arr[p][sfill2(b->b[30])] + surround_arr[p][sfill2(b->b[34])] + 
        surround_arr[p][sfill1(b->b[20])] + surround_arr[p][sfill1(b->b[22])] + surround_arr[p][sfill1(b->b[31])] + surround_arr[p][sfill1(b->b[33])] + 
        surround_arr[p][b->b[21]] + surround_arr[p][b->b[32]];
}

inline int calc_phase_idx(const board *b){
    return max(0, min(3, (b->n - 4) / 10 - 2));
}

inline double edge_2x(int phase_idx, const int b[], int x, int y){
    return pattern_arr[phase_idx][7][pop_digit[b[x]][1] * p39 + b[y] * p31 + pop_digit[b[x]][6]];
}

inline double triangle0(int phase_idx, const int b[], int w, int x, int y, int z){
    return pattern_arr[phase_idx][8][b[w] / p34 * p36 + b[x] / p35 * p33 + b[y] / p36 * p31 + b[z] / p37];
}

inline double triangle1(int phase_idx, const int b[], int w, int x, int y, int z){
    return pattern_arr[phase_idx][8][reverse_board[b[w]] / p34 * p36 + reverse_board[b[x]] / p35 * p33 + reverse_board[b[y]] / p36 * p31 + reverse_board[b[z]] / p37];
}

inline double edge_block(int phase_idx, const int b[], int x, int y){
    return pattern_arr[phase_idx][9][pop_digit[b[x]][0] * p39 + pop_mid[b[x]][6][2] * p35 + pop_digit[b[x]][7] * p34 + pop_mid[b[y]][6][2]];
}

inline double cross(int phase_idx, const int b[], int x, int y, int z){
    return pattern_arr[phase_idx][10][b[x] / p34 * p36 + b[y] / p35 * p33 + b[z] / p35] + 
        pattern_arr[phase_idx][10][reverse_board[b[x]] / p34 * p36 + pop_mid[reverse_board[b[y]]][7][4] * p33 + pop_mid[reverse_board[b[z]]][7][4]];
}

inline double calc_pattern(int phase_idx, const board *b){
    return all_dense[phase_idx][0] * (pattern_arr[phase_idx][0][b->b[1]] + pattern_arr[phase_idx][0][b->b[6]] + pattern_arr[phase_idx][0][b->b[9]] + pattern_arr[phase_idx][0][b->b[14]]) + 
        all_dense[phase_idx][1] * (pattern_arr[phase_idx][1][b->b[2]] + pattern_arr[phase_idx][1][b->b[5]] + pattern_arr[phase_idx][1][b->b[10]] + pattern_arr[phase_idx][1][b->b[13]]) + 
        all_dense[phase_idx][2] * (pattern_arr[phase_idx][2][b->b[3]] + pattern_arr[phase_idx][2][b->b[4]] + pattern_arr[phase_idx][2][b->b[11]] + pattern_arr[phase_idx][2][b->b[12]]) + 
        all_dense[phase_idx][3] * (pattern_arr[phase_idx][3][b->b[18] / p33] + pattern_arr[phase_idx][3][b->b[24] / p33] + pattern_arr[phase_idx][3][b->b[29] / p33] + pattern_arr[phase_idx][3][b->b[35] / p33]) + 
        all_dense[phase_idx][4] * (pattern_arr[phase_idx][4][b->b[19] / p32] + pattern_arr[phase_idx][4][b->b[23] / p32] + pattern_arr[phase_idx][4][b->b[30] / p32] + pattern_arr[phase_idx][4][b->b[34] / p32]) + 
        all_dense[phase_idx][5] * (pattern_arr[phase_idx][5][b->b[20] / p31] + pattern_arr[phase_idx][5][b->b[22] / p31] + pattern_arr[phase_idx][5][b->b[31] / p31] + pattern_arr[phase_idx][5][b->b[33] / p31]) + 
        all_dense[phase_idx][6] * (pattern_arr[phase_idx][6][b->b[21]] + pattern_arr[phase_idx][6][b->b[32]]) + 
        all_dense[phase_idx][7] * (edge_2x(phase_idx, b->b, 1, 0) + edge_2x(phase_idx, b->b, 6, 7) + edge_2x(phase_idx, b->b, 9, 8) + edge_2x(phase_idx, b->b, 14, 15)) + 
        all_dense[phase_idx][8] * (triangle0(phase_idx, b->b, 0, 1, 2, 3) + triangle0(phase_idx, b->b, 7, 6, 5, 4) + triangle0(phase_idx, b->b, 15, 14, 13, 12) + triangle1(phase_idx, b->b, 15, 14, 13, 12)) + 
        all_dense[phase_idx][9] * (edge_block(phase_idx, b->b, 0, 1) + edge_block(phase_idx, b->b, 7, 6) + edge_block(phase_idx, b->b, 8, 9) + edge_block(phase_idx, b->b, 15, 14)) + 
        all_dense[phase_idx][10] * (cross(phase_idx, b->b, 21, 20, 22) + cross(phase_idx, b->b, 32, 31, 33));
}

inline int evaluate(const board *b){
    int phase_idx = calc_phase_idx(b), canput, sur0, sur1;
    canput = min(max_canput * 2, max(0, max_canput + calc_canput(b)));
    sur0 = min(max_surround, calc_surround(b, 0));
    sur1 = min(max_surround, calc_surround(b, 1));
    double res = all_bias[phase_idx] + calc_pattern(phase_idx, b) + 
        all_dense[phase_idx][11] * add_arr[phase_idx][canput][sur0][sur1][0] + all_dense[phase_idx][12] * add_arr[phase_idx][canput][sur0][sur1][1] + all_dense[phase_idx][13] * add_arr[phase_idx][canput][sur0][sur1][2] + all_dense[phase_idx][14] * add_arr[phase_idx][canput][sur0][sur1][3] + 
        all_dense[phase_idx][15] * add_arr[phase_idx][canput][sur0][sur1][4] + all_dense[phase_idx][16] * add_arr[phase_idx][canput][sur0][sur1][5] + all_dense[phase_idx][17] * add_arr[phase_idx][canput][sur0][sur1][6] + all_dense[phase_idx][18] * add_arr[phase_idx][canput][sur0][sur1][7];
    if (b->p)
        res = -res;
    return (int)(max(-1.0, min(1.0, res)) * sc_w);
}

inline int end_game(const board *b){
    int count = (b->p ? -1 : 1) * 
        (count_arr[b->b[0]] + count_arr[b->b[1]] + count_arr[b->b[2]] + count_arr[b->b[3]] + 
        count_arr[b->b[4]] + count_arr[b->b[5]] + count_arr[b->b[6]] + count_arr[b->b[7]]);
    int vacant = hw2 - count_all_arr[b->b[0]] - count_all_arr[b->b[1]] - count_all_arr[b->b[2]] - count_all_arr[b->b[3]] - 
        count_all_arr[b->b[4]] - count_all_arr[b->b[5]] - count_all_arr[b->b[6]] - count_all_arr[b->b[7]];
    if (count > 0)
        count += vacant;
    else if (count < 0)
        count -= vacant;
    return count * step;
}

int nega_alpha(const board *b, bool skipped, int depth, int alpha, int beta);

inline bool mpc_higher(const board *b, bool skipped, int depth, int beta){
    int bound = beta + mpctsd[(b->n - 4) / 10][depth];
    return nega_alpha(b, skipped, mpcd[depth], bound - epsilon, bound) >= bound;
}

inline bool mpc_lower(const board *b, bool skipped, int depth, int alpha){
    int bound = alpha - mpctsd[(b->n - 4) / 10][depth];
    return nega_alpha(b, skipped, mpcd[depth], bound, bound + epsilon) <= bound;
}

inline bool mpc_higher_final(const board *b, bool skipped, int depth, int beta){
    int bound = beta + mpctsd_final[depth];
    return nega_alpha(b, skipped, mpcd[depth], bound - epsilon, bound) >= bound;
}

inline bool mpc_lower_final(const board *b, bool skipped, int depth, int alpha){
    int bound = alpha - mpctsd_final[depth];
    return nega_alpha(b, skipped, mpcd[depth], bound, bound + epsilon) <= bound;
}

inline void move_ordering(board *b){
    int l, u;
    get_search(b->b, calc_hash(b->b) & search_hash_mask, f_search_table_idx, &l, &u);
    b->v = -max(l, u);
    if (u != -inf && l != -inf)
        b->v += cache_both;
    if (u != -inf || l != -inf)
        b->v += cache_hit;
    else
        b->v = -evaluate(b);
}

inline int last1(const board *b, bool skipped, int p0){
    ++searched_nodes;
    int i, before_score = 0;
    for (i = 0; i < hw; ++i)
        before_score += count_arr[b->b[i]];
    if (b->p)
        before_score = -before_score;
    int score = before_score + 1 + (
        move_arr[b->p][b->b[place_included[p0][0]]][local_place[place_included[p0][0]][p0]][0] + move_arr[b->p][b->b[place_included[p0][0]]][local_place[place_included[p0][0]][p0]][1] + 
        move_arr[b->p][b->b[place_included[p0][1]]][local_place[place_included[p0][1]][p0]][0] + move_arr[b->p][b->b[place_included[p0][1]]][local_place[place_included[p0][1]][p0]][1] + 
        move_arr[b->p][b->b[place_included[p0][2]]][local_place[place_included[p0][2]][p0]][0] + move_arr[b->p][b->b[place_included[p0][2]]][local_place[place_included[p0][2]][p0]][1]) * 2;
    if (place_included[p0][3] != -1)
        score += (move_arr[b->p][b->b[place_included[p0][3]]][local_place[place_included[p0][3]][p0]][0] + move_arr[b->p][b->b[place_included[p0][3]]][local_place[place_included[p0][3]][p0]][1]) * 2;
    if (score == before_score + 1){
        if (skipped)
            return end_game(b);
        board rb;
        for (i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -last1(&rb, true, p0);
    }
    return score * step;
}

inline int last2(const board *b, bool skipped, int alpha, int beta, int p0, int p1){
    ++searched_nodes;
    board nb;
    bool passed = true, legal;
    int v = -inf, g;
    legal = legal_arr[b->p][b->b[place_included[p0][0]]][local_place[place_included[p0][0]][p0]] || 
            legal_arr[b->p][b->b[place_included[p0][1]]][local_place[place_included[p0][1]][p0]] || 
            legal_arr[b->p][b->b[place_included[p0][2]]][local_place[place_included[p0][2]][p0]];
    if (place_included[p0][3] != -1)
        legal |= legal_arr[b->p][b->b[place_included[p0][3]]][local_place[place_included[p0][3]][p0]];
    if (legal){
        passed = false;
        nb = move(b, p0);
        g = -last1(&nb, false, p1);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    legal = legal_arr[b->p][b->b[place_included[p1][0]]][local_place[place_included[p1][0]][p1]] || 
            legal_arr[b->p][b->b[place_included[p1][1]]][local_place[place_included[p1][1]][p1]] || 
            legal_arr[b->p][b->b[place_included[p1][2]]][local_place[place_included[p1][2]][p1]];
    if (place_included[p1][3] != -1)
        legal |= legal_arr[b->p][b->b[place_included[p1][3]]][local_place[place_included[p1][3]][p1]];
    if (legal){
        passed = false;
        nb = move(b, p1);
        g = -last1(&nb, false, p0);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (passed){
        if (skipped)
            return end_game(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -last2(&rb, true, -beta, -alpha, p0, p1);
    }
    return v;
}

inline int last3(const board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2){
    ++searched_nodes;
    board nb;
    bool passed = true, legal;
    int v = -inf, g;
    legal = legal_arr[b->p][b->b[place_included[p0][0]]][local_place[place_included[p0][0]][p0]] || 
            legal_arr[b->p][b->b[place_included[p0][1]]][local_place[place_included[p0][1]][p0]] || 
            legal_arr[b->p][b->b[place_included[p0][2]]][local_place[place_included[p0][2]][p0]];
    if (place_included[p0][3] != -1)
        legal |= legal_arr[b->p][b->b[place_included[p0][3]]][local_place[place_included[p0][3]][p0]];
    if (legal){
        passed = false;
        nb = move(b, p0);
        g = -last2(&nb, false, -beta, -alpha, p1, p2);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    legal = legal_arr[b->p][b->b[place_included[p1][0]]][local_place[place_included[p1][0]][p1]] || 
            legal_arr[b->p][b->b[place_included[p1][1]]][local_place[place_included[p1][1]][p1]] || 
            legal_arr[b->p][b->b[place_included[p1][2]]][local_place[place_included[p1][2]][p1]];
    if (place_included[p1][3] != -1)
        legal |= legal_arr[b->p][b->b[place_included[p1][3]]][local_place[place_included[p1][3]][p1]];
    if (legal){
        passed = false;
        nb = move(b, p1);
        g = -last2(&nb, false, -beta, -alpha, p0, p2);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    legal = legal_arr[b->p][b->b[place_included[p2][0]]][local_place[place_included[p2][0]][p2]] || 
            legal_arr[b->p][b->b[place_included[p2][1]]][local_place[place_included[p2][1]][p2]] || 
            legal_arr[b->p][b->b[place_included[p2][2]]][local_place[place_included[p2][2]][p2]];
    if (place_included[p2][3] != -1)
        legal |= legal_arr[b->p][b->b[place_included[p2][3]]][local_place[place_included[p2][3]][p2]];
    if (legal){
        passed = false;
        nb = move(b, p2);
        g = -last2(&nb, false, -beta, -alpha, p0, p1);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (passed){
        if (skipped)
            return end_game(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -last3(&rb, true, -beta, -alpha, p0, p1, p2);
    }
    return v;
}

inline void pick_vacant(const board *b, int cells[]){
    int idx = 0;
    for (const int &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == 2)
            cells[idx++] = cell;
    }
}

int nega_alpha_final(const board *b, bool skipped, int depth, int alpha, int beta){
    ++searched_nodes;
    if (b->n >= hw2 - 3){
        int cells[3];
        pick_vacant(b, cells);
        if (b->n == hw2 - 3)
            return last3(b, skipped, alpha, beta, cells[0], cells[1], cells[2]);
        if (b->n == hw2 - 2)
            return last2(b, skipped, alpha, beta, cells[0], cells[1]);
        if (b->n == hw2 - 1)
            return last1(b, skipped, cells[0]);
        return end_game(b);
    }
    board nb;
    bool passed = true;
    int g, v = -inf;
    bool legal;
    for (const int &cell: vacant_lst){
        legal = legal_arr[b->p][b->b[place_included[cell][0]]][local_place[place_included[cell][0]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][1]]][local_place[place_included[cell][1]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][2]]][local_place[place_included[cell][2]][cell]];
        if (place_included[cell][3] != -1)
            legal |= legal_arr[b->p][b->b[place_included[cell][3]]][local_place[place_included[cell][3]][cell]];
        if (legal){
            passed = false;
            nb = move(b, cell);
            g = -nega_alpha_final(&nb, false, depth - 1, -beta, -alpha);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    }
    if (passed){
        if (skipped)
            return end_game(b);
        for (int i = 0; i < b_idx_num; ++i)
            nb.b[i] = b->b[i];
        nb.p = 1 - b->p;
        nb.n = b->n;
        return -nega_alpha_final(&nb, true, depth, -beta, -alpha);
    }
    return v;
}

int nega_alpha_ordering_final(const board *b, const long long strt, bool skipped, int depth, int alpha, int beta){
    ++searched_nodes;
    if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final){
        if (mpc_higher_final(b, skipped, depth, beta))
            return beta + step;
        if (mpc_lower_final(b, skipped, depth, beta))
            return alpha - step;
    }
    if (depth <= 9)
        return nega_alpha_final(b, skipped, depth, alpha, beta);
    vector<board> nb;
    int canput = 0;
    bool legal;
    for (const int &cell: vacant_lst){
        legal = legal_arr[b->p][b->b[place_included[cell][0]]][local_place[place_included[cell][0]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][1]]][local_place[place_included[cell][1]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][2]]][local_place[place_included[cell][2]][cell]];
        if (place_included[cell][3] != -1)
            legal |= legal_arr[b->p][b->b[place_included[cell][3]]][local_place[place_included[cell][3]][cell]];
        if (legal){
            nb.push_back(move(b, cell));
            move_ordering(&(nb[canput]));
            nb[canput].v -= 100 * calc_canput_exact(&(nb[canput]));
            ++canput;
        }
    }
    if (canput == 0){
        if (skipped)
            return end_game(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_alpha_ordering_final(&rb, strt, true, depth, -beta, -alpha);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int hash = (int)(calc_hash(b->b) & search_hash_mask);
    int l, u;
    get_search(b->b, hash, 1 - f_search_table_idx, &l, &u);
    if (l != -inf){
        if (l == u)
            return l;
        alpha = max(alpha, l);
        if (alpha >= beta)
            return alpha;
    }
    if (u != -inf){
        beta = min(beta, u);
        if (alpha >= beta)
            return alpha;
    }
    int g, v = -inf, first_alpha = alpha;
    for (const board &nnb: nb){
        g = -nega_alpha_ordering_final(&nnb, strt, false, depth - 1, -beta, -alpha);
        alpha = max(alpha, g);
        if (beta <= alpha){
            if (l < g)
                register_search(1 - f_search_table_idx, b->b, hash, alpha, u);
            return alpha;
        }
        v = max(v, g);
    }
    if (v <= first_alpha)
        register_search(1 - f_search_table_idx, b->b, hash, l, v);
    else
        register_search(1 - f_search_table_idx, b->b, hash, v, v);
    return v;
}

int nega_scout_final(const board *b, const long long strt, bool skipped, int depth, int alpha, int beta){
    ++searched_nodes;
    if (depth <= 9)
        return nega_alpha_final(b, skipped, depth, alpha, beta);
    vector<board> nb;
    int canput = 0;
    bool legal;
    for (const int &cell: vacant_lst){
        legal = legal_arr[b->p][b->b[place_included[cell][0]]][local_place[place_included[cell][0]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][1]]][local_place[place_included[cell][1]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][2]]][local_place[place_included[cell][2]][cell]];
        if (place_included[cell][3] != -1)
            legal |= legal_arr[b->p][b->b[place_included[cell][3]]][local_place[place_included[cell][3]][cell]];
        if (legal){
            nb.push_back(move(b, cell));
            move_ordering(&(nb[canput]));
            nb[canput].v -= 100 * calc_canput_exact(&(nb[canput]));
            ++canput;
        }
    }
    if (canput == 0){
        if (skipped)
            return end_game(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_scout_final(&rb, strt, true, depth, -beta, -alpha);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int hash = (int)(calc_hash(b->b) & search_hash_mask);
    int l, u;
    get_search(b->b, hash, 1 - f_search_table_idx, &l, &u);
    if (l != -inf){
        if (l == u)
            return l;
        alpha = max(alpha, l);
        if (alpha >= beta)
            return alpha;
    }
    if (u != -inf){
        beta = min(beta, u);
        if (alpha >= beta)
            return alpha;
    }
    int g = alpha, v = -inf, first_alpha = alpha;
    for (const board &nnb: nb){
        if (&nnb - &nb[0]){
            g = -nega_alpha_ordering_final(&nnb, strt, false, depth - 1, -alpha - step, -alpha);
            if (beta <= g){
                if (l < g)
                    register_search(1 - f_search_table_idx, b->b, hash, g, u);
                return g;
            }
            v = max(v, g);
        }
        if (alpha <= g){
            g = -nega_scout_final(&nnb, strt, false, depth - 1, -beta, -g);
            if (beta <= g){
                if (l < g)
                    register_search(1 - f_search_table_idx, b->b, hash, g, u);
                return g;
            }
            alpha = max(alpha, g);
            v = max(v, g);
        }
    }
    if (v <= first_alpha)
        register_search(1 - f_search_table_idx, b->b, hash, l, v);
    else
        register_search(1 - f_search_table_idx, b->b, hash, v, v);
    return v;
}

int nega_alpha(const board *b, bool skipped, int depth, int alpha, int beta){
    ++searched_nodes;
    if (depth == 0){
        if (b->n < hw2)
            return evaluate(b);
        else
            return end_game(b);
    }
    if (mpc_min_depth <= depth && depth <= mpc_max_depth){
        if (mpc_higher(b, skipped, depth, beta + epsilon))
            return beta + epsilon;
        if (mpc_lower(b, skipped, depth, alpha - epsilon))
            return alpha - epsilon;
    }
    board nb;
    bool passed = true;
    int g, v = -inf;
    bool legal;
    for (const int &cell: vacant_lst){
        legal = legal_arr[b->p][b->b[place_included[cell][0]]][local_place[place_included[cell][0]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][1]]][local_place[place_included[cell][1]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][2]]][local_place[place_included[cell][2]][cell]];
        if (place_included[cell][3] != -1)
            legal |= legal_arr[b->p][b->b[place_included[cell][3]]][local_place[place_included[cell][3]][cell]];
        if (legal){
            passed = false;
            nb = move(b, cell);
            g = -nega_alpha(&nb, false, depth - 1, -beta, -alpha);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    }
    if (passed){
        if (skipped)
            return end_game(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_alpha(&rb, true, depth, -beta, -alpha);
    }
    return v;
}

int nega_alpha_ordering(const board *b, const long long strt, bool skipped, int depth, int alpha, int beta){
    ++searched_nodes;
    if (mpc_min_depth <= depth && depth <= mpc_max_depth){
        if (mpc_higher(b, skipped, depth, beta + epsilon))
            return beta + epsilon;
        if (mpc_lower(b, skipped, depth, alpha - epsilon))
            return alpha - epsilon;
    }
    if (depth <= 3)
        return nega_alpha(b, skipped, depth, alpha, beta);
    int hash = (int)(calc_hash(b->b) & search_hash_mask);
    int l, u;
    get_search(b->b, hash, 1 - f_search_table_idx, &l, &u);
    if (l != -inf){
        if (l == u)
            return l;
        alpha = max(alpha, l);
        if (alpha >= beta)
            return alpha;
    }
    if (u != -inf){
        beta = min(beta, u);
        if (alpha >= beta)
            return alpha;
    }
    vector<board> nb;
    int canput = 0;
    bool legal;
    for (const int &cell: vacant_lst){
        legal = legal_arr[b->p][b->b[place_included[cell][0]]][local_place[place_included[cell][0]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][1]]][local_place[place_included[cell][1]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][2]]][local_place[place_included[cell][2]][cell]];
        if (place_included[cell][3] != -1)
            legal |= legal_arr[b->p][b->b[place_included[cell][3]]][local_place[place_included[cell][3]][cell]];
        if (legal){
            nb.emplace_back(move(b, cell));
            move_ordering(&(nb[canput]));
            ++canput;
        }
    }
    if (canput == 0){
        if (skipped)
            return end_game(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_alpha_ordering(&rb, strt, true, depth, -beta, -alpha);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int first_alpha = alpha, g, v = -inf;
    for (const board &nnb: nb){
        g = -nega_alpha_ordering(&nnb, strt, false, depth - 1, -beta, -alpha);
        if (beta <= g){
            if (l < g)
                register_search(1 - f_search_table_idx, b->b, hash, g, u);
            return g;
        }
        alpha = max(alpha, g);
        v = max(v, g);
    }
    if (v <= first_alpha)
        register_search(1 - f_search_table_idx, b->b, hash, l, v);
    else
        register_search(1 - f_search_table_idx, b->b, hash, v, v);
    return v;
}

int nega_scout(const board *b, const long long strt, bool skipped, int depth, int alpha, int beta){
    ++searched_nodes;
    if (depth <= 3)
        return nega_alpha(b, skipped, depth, alpha, beta);
    int hash = (int)(calc_hash(b->b) & search_hash_mask);
    int l, u;
    get_search(b->b, hash, 1 - f_search_table_idx, &l, &u);
    if (l != -inf){
        if (l == u)
            return l;
        alpha = max(alpha, l);
        if (alpha >= beta)
            return alpha;
    }
    if (u != -inf){
        beta = min(beta, u);
        if (alpha >= beta)
            return alpha;
    }
    vector<board> nb;
    int canput = 0;
    bool legal;
    for (const int &cell: vacant_lst){
        legal = legal_arr[b->p][b->b[place_included[cell][0]]][local_place[place_included[cell][0]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][1]]][local_place[place_included[cell][1]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][2]]][local_place[place_included[cell][2]][cell]];
        if (place_included[cell][3] != -1)
            legal |= legal_arr[b->p][b->b[place_included[cell][3]]][local_place[place_included[cell][3]][cell]];
        if (legal){
            nb.emplace_back(move(b, cell));
            move_ordering(&(nb[canput]));
            ++canput;
        }
    }
    if (canput == 0){
        if (skipped)
            return end_game(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_scout(&rb, strt, true, depth, -beta, -alpha);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g = alpha, v = -inf, first_alpha = alpha;
    for (const board &nnb: nb){
        if (&nnb - &nb[0]){
            g = -nega_alpha_ordering(&nnb, strt, false, depth - 1, -alpha - epsilon, -alpha);
            if (beta <= g){
                if (l < g)
                    register_search(1 - f_search_table_idx, b->b, hash, g, u);
                return g;
            }
            v = max(v, g);
        }
        if (alpha <= g){
            g = -nega_scout(&nnb, strt, false, depth - 1, -beta, -g);
            if (beta <= g){
                if (l < g)
                    register_search(1 - f_search_table_idx, b->b, hash, g, u);
                return g;
            }
            alpha = max(alpha, g);
            v = max(v, g);
        }
    }
    if (v <= first_alpha)
        register_search(1 - f_search_table_idx, b->b, hash, l, v);
    else
        register_search(1 - f_search_table_idx, b->b, hash, v, v);
    return v;
}

int mtd(const board *b, const long long strt, bool skipped, int depth, int l, int u){
    int g = evaluate(b), beta;
    while (u - l > mtd_threshold){
        beta = g;
        g = nega_alpha_ordering(b, strt, skipped, depth, beta - epsilon, beta);
        if (g < beta)
            u = g;
        else
            l = g;
        g = (l + u) / 2;
    }
    return nega_scout(b, strt, skipped, depth, l, u);
}

int mtd_final(const board *b, const long long strt, bool skipped, int depth, int max_beta){
    int l = -hw2, u = hw2;
    max_beta /= step;
    int g = evaluate(b) / step, beta;
    while (u - l > 0){
        beta = max(l + 1, g);
        g = nega_alpha_ordering_final(b, strt, skipped, depth, beta * step - epsilon, beta * step);
        //cerr << l << " " << g << " " << u << "  " << beta << endl;
        if (g < beta * step)
            u = g / step;
        else
            l = g / step;
        cerr << l << " " << u << "  " << max_beta << endl;
        if (g / step >= max_beta)
            return g;
        g = (l + u) / 2;
    }
    cerr << l << endl;
    return l * step;
}

inline search_result search(const board b, long long strt, int max_depth){
    vector<board> nb;
    int i;
    for (const int &cell: vacant_lst){
        for (i = 0; i < 4; ++i){
            if (place_included[cell][i] == -1)
                break;
            if (legal_arr[b.p][b.b[place_included[cell][i]]][local_place[place_included[cell][i]][cell]]){
                cerr << cell << " ";
                nb.push_back(move(&b, cell));
                break;
            }
        }
    }
    cerr << endl;
    int canput = nb.size();
    cerr << "canput: " << canput << endl;
    int res_depth;
    int policy = -1;
    int tmp_policy;
    int alpha, beta, g, value;
    searched_nodes = 0;
    hash_conf = 0;
    hash_get = 0;
    hash_reg = 0;
    int order_l, order_u;
    for (int depth = min(5, max(1, max_depth - 5)); depth < min(hw2 - b.n, max_depth); ++depth){
        alpha = -sc_w - epsilon;
        beta = sc_w + epsilon;
        search_hash_table_init(1 - f_search_table_idx);
        for (i = 0; i < canput; ++i){
            get_search(nb[i].b, calc_hash(nb[i].b) & search_hash_mask, f_search_table_idx, &order_l, &order_u);
            nb[i].v = -max(order_l, order_u);
            if (order_l != -inf && order_u != -inf)
                nb[i].v += 100000;
            if (order_l == -inf && order_u == -inf)
                nb[i].v = -evaluate(&nb[i]);
            else
                nb[i].v += cache_hit;
        }
        if (canput >= 2)
            sort(nb.begin(), nb.end());
        g = -mtd(&nb[0], strt, false, depth, -beta, -alpha);
        if (g == inf)
            break;
        register_search(1 - f_search_table_idx, nb[0].b, (int)(calc_hash(nb[0].b) & search_hash_mask), g, g);
        alpha = max(alpha, g);
        tmp_policy = nb[0].policy;
        for (i = 1; i < canput; ++i){
            g = -nega_alpha_ordering(&nb[i], strt, false, depth, -alpha - epsilon, -alpha);
            if (alpha < g){
                alpha = g;
                g = -mtd(&nb[i], strt, false, depth, -beta, -alpha);
                register_search(1 - f_search_table_idx, nb[i].b, (int)(calc_hash(nb[i].b) & search_hash_mask), g, g);
                if (alpha < g){
                    alpha = g;
                    tmp_policy = nb[i].policy;
                }
            } else{
                register_search(1 - f_search_table_idx, nb[i].b, (int)(calc_hash(nb[i].b) & search_hash_mask), -inf, g);
            }
        }
        f_search_table_idx = 1 - f_search_table_idx;
        policy = tmp_policy;
        value = alpha;
        res_depth = depth + 1;
        cerr << "depth: " << depth + 1 << " time: " << tim() - strt << " policy: " << policy << " value: " << alpha << " nodes: " << searched_nodes << " nps: " << (long long)searched_nodes * 1000 / max(1LL, tim() - strt) << " get: " << hash_get << " reg: " << hash_reg << endl;
    }
    search_result res;
    res.policy = policy;
    res.value = value;
    res.depth = res_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    return res;
}

inline search_result final_search(const board b, long long strt){
    vector<board> nb;
    int i;
    for (const int &cell: vacant_lst){
        for (i = 0; i < 4; ++i){
            if (place_included[cell][i] == -1)
                break;
            if (legal_arr[b.p][b.b[place_included[cell][i]]][local_place[place_included[cell][i]][cell]]){
                cerr << cell << " ";
                nb.push_back(move(&b, cell));
                break;
            }
        }
    }
    cerr << endl;
    int canput = nb.size();
    cerr << "canput: " << canput << endl;
    int policy = -1;
    int tmp_policy;
    int alpha, beta, g, value;
    searched_nodes = 0;
    hash_conf = 0;
    hash_get = 0;
    hash_reg = 0;
    int order_l, order_u;
    int max_depth = hw2 - b.n;
    if (max_depth - 9 > 0)
        search(b, strt, max_depth - 9);
    cerr << "start final search" << endl;
    alpha = -sc_w;
    beta = sc_w;
    search_hash_table_init(1 - f_search_table_idx);
    for (i = 0; i < canput; ++i){
        get_search(nb[i].b, calc_hash(nb[i].b) & search_hash_mask, f_search_table_idx, &order_l, &order_u);
        nb[i].v = -max(order_l, order_u);
        if (order_l != -inf && order_u != -inf)
            nb[i].v += cache_both;
        if (order_l == -inf && order_u == -inf)
            nb[i].v = -evaluate(&nb[i]);
        else
            nb[i].v += cache_hit;
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    alpha = -nega_scout_final(&nb[0], strt, false, max_depth, -beta, -alpha);
    tmp_policy = nb[0].policy;
    for (i = 1; i < canput; ++i){
        g = -nega_alpha_ordering_final(&nb[i], strt, false, max_depth, -alpha - step, -alpha);
        if (alpha < g){
            g = -nega_scout_final(&nb[i], strt, false, max_depth, -beta, -g);
            if (alpha <= g){
                alpha = g;
                tmp_policy = nb[i].policy;
            }
        }
    }
    f_search_table_idx = 1 - f_search_table_idx;
    policy = tmp_policy;
    value = alpha;
    cerr << "final depth: " << max_depth << " time: " << tim() - strt << " policy: " << policy << " value: " << alpha << " nodes: " << searched_nodes << " nps: " << (long long)searched_nodes * 1000 / max(1LL, tim() - strt) << " get: " << hash_get << " reg: " << hash_reg << endl;
    search_result res;
    res.policy = policy;
    res.value = value;
    res.depth = max_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    return res;
}

int cmp_vacant(int p, int q){
    return cell_weight[p] > cell_weight[q];
}

inline int input_board(int (&board)[b_idx_num]){
    int i, j;
    unsigned long long b = 0, w = 0;
    char elem;
    int n_stones = 0;
    vacant_lst.clear();
    for (i = 0; i < hw; ++i){
        string raw_board;
        cin >> raw_board; cin.ignore();
        cerr << raw_board << endl;
        for (j = 0; j < hw; ++j){
            elem = raw_board[j];
            if (elem != '.'){
                b |= (unsigned long long)(elem == '0') << (i * hw + j);
                w |= (unsigned long long)(elem == '1') << (i * hw + j);
                ++n_stones;
            } else{
                vacant_lst.push_back(i * hw + j);
            }
        }
    }
    if (n_stones < hw2_m1)
        sort(vacant_lst.begin(), vacant_lst.end(), cmp_vacant);
    for (i = 0; i < b_idx_num; ++i){
        board[i] = n_line - 1;
        for (j = 0; j < idx_n_cell[i]; ++j){
            if (1 & (b >> global_place[i][j]))
                board[i] -= pow3[hw_m1 - j] * 2;
            else if (1 & (w >> global_place[i][j]))
                board[i] -= pow3[hw_m1 - j];
        }
    }
    return n_stones;
}

inline double calc_result_value(int v){
    return (double)v * hw2 / sc_w;
}

inline void print_result(int policy, int value){
    cout << policy / hw << " " << policy % hw << " " << calc_result_value(value) << endl;
}

inline void print_result(search_result result){
    cout << result.policy / hw << " " << result.policy % hw << " " << calc_result_value(result.value) << endl;
}

int main(){
    cerr << myrandom() << endl;
    int policy, n_stones, ai_player, depth, final_depth;
    board b;
    const int first_moves[4] = {19, 26, 37, 44};
    cin >> ai_player;
    depth = 16;
    final_depth = 24;
    long long strt = tim();
    search_result result;
    cerr << "initializing" << endl;
    init_pow();
    init_mod3();
    init_pop_mid();
    init_move();
    init_local_place();
    init_included();
    init_pop_digit();
    init_mpc();
    init_book();
    cerr << "book initialized in " << tim() - strt << " ms" << endl;
    init_evaluation();
    f_search_table_idx = 0;
    search_hash_table_init(f_search_table_idx);
    search_hash_table_init(1 - f_search_table_idx);
    cerr << "iniitialized in " << tim() - strt << " ms" << endl;
    while (true){
        n_stones = input_board(b.b);
        strt = tim();
        cerr << "n_stones: " << n_stones << endl;
        cerr << n_stones - 4 << " moves" << endl;
        b.n = n_stones;
        b.p = ai_player;
        cerr << "value: " << evaluate(&b) << endl;
        if (n_stones == 4){
            policy = first_moves[myrandrange(0, 4)];
            print_result(policy, 0);
            continue;
        }
        if (n_stones < book_stones){
            policy = get_book(b.b);
            cerr << "book policy " << policy << endl;
            if (policy != -1){
                b = move(&b, policy);
                ++n_stones;
                result = search(b, strt, 10);
                print_result(policy, -result.value);
                continue;
            }
        }
        if (n_stones >= hw2 - final_depth)
            result = final_search(b, strt);
        else
            result = search(b, strt, depth);
        print_result(result);
    }
    return 0;
}