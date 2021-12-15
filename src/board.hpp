#pragma once
#include <iostream>
#include "setting.hpp"

using namespace std;

#define hw 8
#define hw_m1 7
#define hw_p1 9
#define hw2 64
#define hw22 128
#define hw2_m1 63
#define hw2_mhw 56
#define hw2_p1 65
#define b_idx_num 38
#define n_line 6561
#define black 0
#define white 1
#define vacant 2

const int idx_n_cell[b_idx_num] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3};
const int move_offset[b_idx_num] = {1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
const int global_place[b_idx_num][hw] = {
    {0, 1, 2, 3, 4, 5, 6, 7},{8, 9, 10, 11, 12, 13, 14, 15},{16, 17, 18, 19, 20, 21, 22, 23},{24, 25, 26, 27, 28, 29, 30, 31},{32, 33, 34, 35, 36, 37, 38, 39},{40, 41, 42, 43, 44, 45, 46, 47},{48, 49, 50, 51, 52, 53, 54, 55},{56, 57, 58, 59, 60, 61, 62, 63},
    {0, 8, 16, 24, 32, 40, 48, 56},{1, 9, 17, 25, 33, 41, 49, 57},{2, 10, 18, 26, 34, 42, 50, 58},{3, 11, 19, 27, 35, 43, 51, 59},{4, 12, 20, 28, 36, 44, 52, 60},{5, 13, 21, 29, 37, 45, 53, 61},{6, 14, 22, 30, 38, 46, 54, 62},{7, 15, 23, 31, 39, 47, 55, 63},
    {5, 14, 23, -1, -1, -1, -1, -1},{4, 13, 22, 31, -1, -1, -1, -1},{3, 12, 21, 30, 39, -1, -1, -1},{2, 11, 20, 29, 38, 47, -1, -1},{1, 10, 19, 28, 37, 46, 55, -1},{0, 9, 18, 27, 36, 45, 54, 63},{8, 17, 26, 35, 44, 53, 62, -1},{16, 25, 34, 43, 52, 61, -1, -1},{24, 33, 42, 51, 60, -1, -1, -1},{32, 41, 50, 59, -1, -1, -1, -1},{40, 49, 58, -1, -1, -1, -1, -1},
    {2, 9, 16, -1, -1, -1, -1, -1},{3, 10, 17, 24, -1, -1, -1, -1},{4, 11, 18, 25, 32, -1, -1, -1},{5, 12, 19, 26, 33, 40, -1, -1},{6, 13, 20, 27, 34, 41, 48, -1},{7, 14, 21, 28, 35, 42, 49, 56},{15, 22, 29, 36, 43, 50, 57, -1},{23, 30, 37, 44, 51, 58, -1, -1},{31, 38, 45, 52, 59, -1, -1, -1},{39, 46, 53, 60, -1, -1, -1, -1},{47, 54, 61, -1, -1, -1, -1, -1}
};
uint_fast16_t move_arr[2][n_line][hw][2];
bool legal_arr[2][n_line][hw];
uint_fast16_t flip_arr[2][n_line][hw];
uint_fast16_t put_arr[2][n_line][hw];
uint_fast16_t redo_arr[2][n_line][hw];
int_fast8_t local_place[b_idx_num][hw2];
int_fast8_t place_included[hw2][4];
uint_fast16_t reverse_board[n_line];
uint_fast16_t pow3[11];
uint_fast16_t pop_digit[n_line][hw];
uint_fast16_t pop_mid[n_line][hw][hw];

const int cell_div4[hw2] = {
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8
};

inline int create_one_color(int idx, const int k) {
    int res = 0;
    for (int i = 0; i < hw; ++i) {
        if (idx % 3 == k) {
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
    int pt = 1 << (hw - 1 - place);
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

void board_init() {
    int idx, b, w, place, i, j, k, l_place, inc_idx;
    pow3[0] = 1;
    for (idx = 1; idx < 11; ++idx)
        pow3[idx] = pow3[idx- 1] * 3;
    for (i = 0; i < n_line; ++i){
        for (j = 0; j < hw; ++j)
            pop_digit[i][j] = (i / pow3[hw_m1 - j]) % 3;
    }
    for (i = 0; i < n_line; ++i){
        for (j = 0; j < hw; ++j){
            for (k = 0; k < hw; ++k)
                pop_mid[i][j][k] = (i - i / pow3[j] * pow3[j]) / pow3[k];
        }
    }
    for (idx = 0; idx < n_line; ++idx) {
        b = create_one_color(idx, 0);
        w = create_one_color(idx, 1);
        for (place = 0; place < hw; ++place) {
            reverse_board[idx] *= 3;
            if (1 & (b >> place))
                reverse_board[idx] += 0;
            else if (1 & (w >> place)) 
                reverse_board[idx] += 1;
            else
                reverse_board[idx] += 2;
        }
        for (place = 0; place < hw; ++place) {
            move_arr[black][idx][place][0] = move_line_half(b, w, place, 0);
            move_arr[black][idx][place][1] = move_line_half(b, w, place, 1);
            if (move_arr[black][idx][place][0] || move_arr[black][idx][place][1])
                legal_arr[black][idx][place] = true;
            else
                legal_arr[black][idx][place] = false;
            move_arr[white][idx][place][0] = move_line_half(w, b, place, 0);
            move_arr[white][idx][place][1] = move_line_half(w, b, place, 1);
            if (move_arr[white][idx][place][0] || move_arr[white][idx][place][1])
                legal_arr[white][idx][place] = true;
            else
                legal_arr[white][idx][place] = false;
        }
        for (place = 0; place < hw; ++place) {
            flip_arr[black][idx][place] = idx;
            flip_arr[white][idx][place] = idx;
            put_arr[black][idx][place] = idx;
            put_arr[white][idx][place] = idx;
            redo_arr[black][idx][place] = idx;
            redo_arr[white][idx][place] = idx;
            if (b & (1 << (hw - 1 - place))){
                flip_arr[white][idx][place] += pow3[hw_m1 - place];
                redo_arr[black][idx][place] += pow3[hw_m1 - place] * 2;
            } else if (w & (1 << (hw - 1 - place))){
                flip_arr[black][idx][place] -= pow3[hw_m1 - place];
                redo_arr[white][idx][place] += pow3[hw_m1 - place];
            } else{
                put_arr[black][idx][place] -= pow3[hw_m1 - place] * 2;
                put_arr[white][idx][place] -= pow3[hw_m1 - place];
            }
        }
    }
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
    for (idx = 0; idx < b_idx_num; ++idx){
        for (place = 0; place < hw2; ++place){
            local_place[idx][place] = -1;
            for (l_place = 0; l_place < hw; ++l_place){
                if (global_place[idx][l_place] == place)
                    local_place[idx][place] = l_place;
            }
        }
    }
    cerr << "board initialized" << endl;
}

class board {
    public:
        uint_fast16_t b[b_idx_num];
        int p;
        int policy;
        int v;
        int n;
        int parity;
        //int move_log[4][2];

    public:
        bool operator<(const board& another) const {
            return v > another.v;
        }

        unsigned long long hash(){
            return
                this->p + 
                this->b[0] + 
                this->b[1] * 17 + 
                this->b[2] * 289 + 
                this->b[3] * 4913 + 
                this->b[4] * 83521 + 
                this->b[5] * 1419857 + 
                this->b[6] * 24137549 + 
                this->b[7] * 410338673;
        }

        inline void print() {
            int i, j, tmp;
            string res;
            for (i = 0; i < hw; ++i) {
                tmp = this->b[i];
                res = "";
                for (j = 0; j < hw; ++j) {
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

        inline bool legal(int g_place) {
            bool res = 
                legal_arr[this->p][this->b[place_included[g_place][0]]][local_place[place_included[g_place][0]][g_place]] || 
                legal_arr[this->p][this->b[place_included[g_place][1]]][local_place[place_included[g_place][1]][g_place]] || 
                legal_arr[this->p][this->b[place_included[g_place][2]]][local_place[place_included[g_place][2]][g_place]];
            if (place_included[g_place][3] != -1)
                res |= legal_arr[this->p][this->b[place_included[g_place][3]]][local_place[place_included[g_place][3]][g_place]];
            return res;
        }

        inline board move(const int g_place) {
            board res;
            for (int i = 0; i < b_idx_num; ++i)
                res.b[i] = this->b[i];
            move_p(&res, g_place, 0);
            move_p(&res, g_place, 1);
            move_p(&res, g_place, 2);
            if (place_included[g_place][3] != -1)
                move_p(&res, g_place, 3);
            res.b[place_included[g_place][0]] = put_arr[this->p][res.b[place_included[g_place][0]]][local_place[place_included[g_place][0]][g_place]];
            res.b[place_included[g_place][1]] = put_arr[this->p][res.b[place_included[g_place][1]]][local_place[place_included[g_place][1]][g_place]];
            res.b[place_included[g_place][2]] = put_arr[this->p][res.b[place_included[g_place][2]]][local_place[place_included[g_place][2]][g_place]];
            if (place_included[g_place][3] != -1)
                res.b[place_included[g_place][3]] = put_arr[this->p][res.b[place_included[g_place][3]]][local_place[place_included[g_place][3]][g_place]];
            res.p = 1 - this->p;
            res.n = this->n + 1;
            res.policy = g_place;
            res.parity = this->parity ^ cell_div4[g_place];
            return res;
        }

        inline void move(const int g_place, board *res) {
            for (int i = 0; i < b_idx_num; ++i)
                res->b[i] = this->b[i];
            move_p(res, g_place, 0);
            move_p(res, g_place, 1);
            move_p(res, g_place, 2);
            if (place_included[g_place][3] != -1)
                move_p(res, g_place, 3);
            res->b[place_included[g_place][0]] = put_arr[this->p][res->b[place_included[g_place][0]]][local_place[place_included[g_place][0]][g_place]];
            res->b[place_included[g_place][1]] = put_arr[this->p][res->b[place_included[g_place][1]]][local_place[place_included[g_place][1]][g_place]];
            res->b[place_included[g_place][2]] = put_arr[this->p][res->b[place_included[g_place][2]]][local_place[place_included[g_place][2]][g_place]];
            if (place_included[g_place][3] != -1)
                res->b[place_included[g_place][3]] = put_arr[this->p][res->b[place_included[g_place][3]]][local_place[place_included[g_place][3]][g_place]];
            res->p = 1 - this->p;
            res->n = this->n + 1;
            res->policy = g_place;
            res->parity = this->parity ^ cell_div4[g_place];
        }

        /*
        inline void self_move(const int g_place) {
            move_p_log(g_place, 0);
            move_p_log(g_place, 1);
            move_p_log(g_place, 2);
            if (place_included[g_place][3] != -1)
                move_p_log(g_place, 3);
            this->b[place_included[g_place][0]] = put_arr[this->p][this->b[place_included[g_place][0]]][local_place[place_included[g_place][0]][g_place]];
            this->b[place_included[g_place][1]] = put_arr[this->p][this->b[place_included[g_place][1]]][local_place[place_included[g_place][1]][g_place]];
            this->b[place_included[g_place][2]] = put_arr[this->p][this->b[place_included[g_place][2]]][local_place[place_included[g_place][2]][g_place]];
            if (place_included[g_place][3] != -1)
                this->b[place_included[g_place][3]] = put_arr[this->p][this->b[place_included[g_place][3]]][local_place[place_included[g_place][3]][g_place]];
            this->p = 1 - this->p;
            ++this->n;
            this->policy = g_place;
            this->parity ^= cell_div4[g_place];
        }

        inline void redo(){
            redo_p(this->policy, 0);
            redo_p(this->policy, 1);
            redo_p(this->policy, 2);
            if (place_included[this->policy][3] != -1)
                redo_p(this->policy, 3);
            this->b[place_included[this->policy][0]] = redo_arr[1 - this->p][this->b[place_included[this->policy][0]]][local_place[place_included[this->policy][0]][this->policy]];
            this->b[place_included[this->policy][1]] = redo_arr[1 - this->p][this->b[place_included[this->policy][1]]][local_place[place_included[this->policy][1]][this->policy]];
            this->b[place_included[this->policy][2]] = redo_arr[1 - this->p][this->b[place_included[this->policy][2]]][local_place[place_included[this->policy][2]][this->policy]];
            if (place_included[this->policy][3] != -1)
                this->b[place_included[this->policy][3]] = redo_arr[1 - this->p][this->b[place_included[this->policy][3]]][local_place[place_included[this->policy][3]][this->policy]];
            this->p = 1 - this->p;
            --this->n;
            this->parity ^= cell_div4[this->policy];
            this->policy = -1;
        }
        */

        inline void translate_to_arr(int res[]) {
            int i, j;
            for (i = 0; i < hw; ++i) {
                for (j = 0; j < hw; ++j)
                    res[i * hw + j] = pop_digit[this->b[i]][j];
            }
        }

        inline void translate_from_arr(const int arr[], int player) {
            int i, j;
            for (i = 0; i < b_idx_num; ++i)
                this->b[i] = n_line - 1;
            this->n = hw2;
            for (i = 0; i < hw2; ++i) {
                for (j = 0; j < 4; ++j) {
                    if (place_included[i][j] == -1)
                        continue;
                    if (arr[i] == black)
                        this->b[place_included[i][j]] -= 2 * pow3[hw_m1 - local_place[place_included[i][j]][i]];
                    else if (arr[i] == white)
                        this->b[place_included[i][j]] -= pow3[hw_m1 - local_place[place_included[i][j]][i]];
                    else if (j == 0)
                        --this->n;
                }
            }
            this->p = player;
        }

    private:
        inline void flip(board *res, int g_place) {
            res->b[place_included[g_place][0]] = flip_arr[this->p][res->b[place_included[g_place][0]]][local_place[place_included[g_place][0]][g_place]];
            res->b[place_included[g_place][1]] = flip_arr[this->p][res->b[place_included[g_place][1]]][local_place[place_included[g_place][1]][g_place]];
            res->b[place_included[g_place][2]] = flip_arr[this->p][res->b[place_included[g_place][2]]][local_place[place_included[g_place][2]][g_place]];
            if (place_included[g_place][3] != -1)
                res->b[place_included[g_place][3]] = flip_arr[this->p][res->b[place_included[g_place][3]]][local_place[place_included[g_place][3]][g_place]];
        }

        /*
        inline void flip(int g_place) {
            this->b[place_included[g_place][0]] = flip_arr[this->p][this->b[place_included[g_place][0]]][local_place[place_included[g_place][0]][g_place]];
            this->b[place_included[g_place][1]] = flip_arr[this->p][this->b[place_included[g_place][1]]][local_place[place_included[g_place][1]][g_place]];
            this->b[place_included[g_place][2]] = flip_arr[this->p][this->b[place_included[g_place][2]]][local_place[place_included[g_place][2]][g_place]];
            if (place_included[g_place][3] != -1)
                this->b[place_included[g_place][3]] = flip_arr[this->p][this->b[place_included[g_place][3]]][local_place[place_included[g_place][3]][g_place]];
        }
        */

        inline void move_p(board *res, int g_place, int i) {
            int j, place = local_place[place_included[g_place][i]][g_place];
            for (j = 1; j <= move_arr[this->p][this->b[place_included[g_place][i]]][place][0]; ++j)
                flip(res, g_place - move_offset[place_included[g_place][i]] * j);
            for (j = 1; j <= move_arr[this->p][this->b[place_included[g_place][i]]][place][1]; ++j)
                flip(res, g_place + move_offset[place_included[g_place][i]] * j);
        }

        /*
        inline void move_p_log(int g_place, int i) {
            int j, place = local_place[place_included[g_place][i]][g_place];
            this->move_log[i][0] = move_arr[this->p][this->b[place_included[g_place][i]]][place][0];
            this->move_log[i][1] = move_arr[this->p][this->b[place_included[g_place][i]]][place][1];
            for (j = 1; j <= this->move_log[i][0]; ++j)
                flip(g_place - move_offset[place_included[g_place][i]] * j);
            for (j = 1; j <= this->move_log[i][1]; ++j)
                flip(g_place + move_offset[place_included[g_place][i]] * j);
        }

        inline void redo_p(int g_place, int i){
            int j;
            for (j = 1; j <= this->move_log[i][0]; ++j)
                flip(g_place - move_offset[place_included[g_place][i]] * j);
            for (j = 1; j <= this->move_log[i][1]; ++j)
                flip(g_place + move_offset[place_included[g_place][i]] * j);
        }
        */

};