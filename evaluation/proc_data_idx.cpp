#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "board.hpp"

using namespace std;

#define black 0
#define white 1

#define n_phases 15
#define phase_n_stones 4

#define max_surround 80
#define max_canput 50

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
#define p31m 2
#define p32m 8
#define p33m 26
#define p34m 80
#define p35m 242
#define p36m 728
#define p37m 2186
#define p38m 6560
#define p39m 19682
#define p310m 59048

int count_black_arr[n_line];
int count_both_arr[n_line];
int mobility_arr[2][n_line];
int mobility_arr2[2][n_line * n_line];
int surround_arr[2][n_line];
int stability_edge_arr[2][n_line];
int stability_corner_arr[2][n_line];

inline int calc_phase_idx(const board *b){
    return (b->n - 4) / phase_n_stones;
}

inline void init_evaluation_base() {
    int idx, place, b, w;
    bool full_black, full_white;
    for (idx = 0; idx < n_line; ++idx) {
        b = create_one_color(idx, 0);
        w = create_one_color(idx, 1);
        mobility_arr[black][idx] = 0;
        mobility_arr[white][idx] = 0;
        surround_arr[black][idx] = 0;
        surround_arr[white][idx] = 0;
        count_black_arr[idx] = 0;
        count_both_arr[idx] = 0;
        stability_edge_arr[black][idx] = 0;
        stability_edge_arr[white][idx] = 0;
        stability_corner_arr[black][idx] = 0;
        stability_corner_arr[white][idx] = 0;
        for (place = 0; place < hw; ++place) {
            if (1 & (b >> place)){
                ++count_black_arr[idx];
                ++count_both_arr[idx];
            } else if (1 & (w >> place)){
                --count_black_arr[idx];
                ++count_both_arr[idx];
            }
            if (place > 0) {
                if ((1 & (b >> (place - 1))) == 0 && (1 & (w >> (place - 1))) == 0) {
                    if (1 & (b >> place))
                        ++surround_arr[black][idx];
                    else if (1 & (w >> place))
                        ++surround_arr[white][idx];
                }
            }
            if (place < hw - 1) {
                if ((1 & (b >> (place + 1))) == 0 && (1 & (w >> (place + 1))) == 0) {
                    if (1 & (b >> place))
                        ++surround_arr[black][idx];
                    else if (1 & (w >> place))
                        ++surround_arr[white][idx];
                }
            }
        }
        for (place = 0; place < hw; ++place) {
            if (legal_arr[black][idx][place])
                ++mobility_arr[black][idx];
            if (legal_arr[white][idx][place])
                ++mobility_arr[white][idx];
        }
        if (count_both_arr[idx] == hw){
            stability_edge_arr[black][idx] = (count_black_arr[idx] + hw) / 2;
            stability_edge_arr[white][idx] += hw - stability_edge_arr[black][idx];
        } else{
            full_black = true;
            full_white = true;
            for (place = 0; place < hw; ++place){
                full_black &= 1 & (b >> place);
                full_white &= 1 & (w >> place);
                stability_edge_arr[black][idx] += full_black;
                stability_edge_arr[white][idx] += full_white;
            }
            full_black = true;
            full_white = true;
            for (place = hw_m1; place >= 0; --place){
                full_black &= 1 & (b >> place);
                full_white &= 1 & (w >> place);
                stability_edge_arr[black][idx] += full_black;
                stability_edge_arr[white][idx] += full_white;
            }
            if (1 & b){
                --stability_edge_arr[black][idx];
                ++stability_corner_arr[black][idx];
            }
            if (1 & (b >> hw_m1)){
                --stability_edge_arr[black][idx];
                ++stability_corner_arr[black][idx];
            }
            if (1 & w){
                --stability_edge_arr[white][idx];
                ++stability_corner_arr[white][idx];
            }
            if (1 & (w >> hw_m1)){
                --stability_edge_arr[white][idx];
                ++stability_corner_arr[white][idx];
            }
        }
    }
}

inline int sfill5(int b){
    return pop_digit[b][2] != 2 ? b - p35m : b;
}

inline int sfill4(int b){
    return pop_digit[b][3] != 2 ? b - p34m : b;
}

inline int sfill3(int b){
    return pop_digit[b][4] != 2 ? b - p33m : b;
}

inline int sfill2(int b){
    return pop_digit[b][5] != 2 ? b - p32m : b;
}

inline int sfill1(int b){
    return pop_digit[b][6] != 2 ? b - p31m : b;
}

inline int calc_canput(const board *b){
    return (b->p ? -1 : 1) * (
        mobility_arr[b->p][b->b[0]] + mobility_arr[b->p][b->b[1]] + mobility_arr[b->p][b->b[2]] + mobility_arr[b->p][b->b[3]] + 
        mobility_arr[b->p][b->b[4]] + mobility_arr[b->p][b->b[5]] + mobility_arr[b->p][b->b[6]] + mobility_arr[b->p][b->b[7]] + 
        mobility_arr[b->p][b->b[8]] + mobility_arr[b->p][b->b[9]] + mobility_arr[b->p][b->b[10]] + mobility_arr[b->p][b->b[11]] + 
        mobility_arr[b->p][b->b[12]] + mobility_arr[b->p][b->b[13]] + mobility_arr[b->p][b->b[14]] + mobility_arr[b->p][b->b[15]] + 
        mobility_arr[b->p][b->b[16] - p35m] + mobility_arr[b->p][b->b[26] - p35m] + mobility_arr[b->p][b->b[27] - p35m] + mobility_arr[b->p][b->b[37] - p35m] + 
        mobility_arr[b->p][b->b[17] - p34m] + mobility_arr[b->p][b->b[25] - p34m] + mobility_arr[b->p][b->b[28] - p34m] + mobility_arr[b->p][b->b[36] - p34m] + 
        mobility_arr[b->p][b->b[18] - p33m] + mobility_arr[b->p][b->b[24] - p33m] + mobility_arr[b->p][b->b[29] - p33m] + mobility_arr[b->p][b->b[35] - p33m] + 
        mobility_arr[b->p][b->b[19] - p32m] + mobility_arr[b->p][b->b[23] - p32m] + mobility_arr[b->p][b->b[30] - p32m] + mobility_arr[b->p][b->b[34] - p32m] + 
        mobility_arr[b->p][b->b[20] - p31m] + mobility_arr[b->p][b->b[22] - p31m] + mobility_arr[b->p][b->b[31] - p31m] + mobility_arr[b->p][b->b[33] - p31m] + 
        mobility_arr[b->p][b->b[21]] + mobility_arr[b->p][b->b[32]]);
}

inline int calc_surround(const board *b, int p){
    return 
        surround_arr[p][b->b[0]] + surround_arr[p][b->b[1]] + surround_arr[p][b->b[2]] + surround_arr[p][b->b[3]] + 
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

inline int edge_2x(int phase_idx, const board *b, int x, int y){
    return pop_digit[b->b[x]][1] * p39 + b->b[y] * p31 + pop_digit[b->b[x]][6];
}

inline int triangle0(int phase_idx, const board *b, int w, int x, int y, int z){
    return b->b[w] / p34 * p36 + b->b[x] / p35 * p33 + b->b[y] / p36 * p31 + b->b[z] / p37;
}

inline int triangle1(int phase_idx, const board *b, int w, int x, int y, int z){
    return reverse_board[b->b[w]] / p34 * p36 + reverse_board[b->b[x]] / p35 * p33 + reverse_board[b->b[y]] / p36 * p31 + reverse_board[b->b[z]] / p37;
}

inline int edge_block(int phase_idx, const board *b, int x, int y){
    return pop_digit[b->b[x]][0] * p39 + pop_mid[b->b[x]][6][2] * p35 + pop_digit[b->b[x]][7] * p34 + pop_mid[b->b[y]][6][2];
}

inline int cross0(int phase_idx, const board *b, int x, int y, int z){
    return b->b[x] / p34 * p36 + b->b[y] / p35 * p33 + b->b[z] / p35;
}

inline int cross1(int phase_idx, const board *b, int x, int y, int z){
    return reverse_board[b->b[x]] / p34 * p36 + pop_mid[reverse_board[b->b[y]]][7][4] * p33 + pop_mid[reverse_board[b->b[z]]][7][4];
}

inline int corner90(int phase_idx, const board *b, int x, int y, int z){
    return b->b[x] / p35 * p36 + b->b[y] / p35 * p33 + b->b[z] / p35;
}

inline int corner91(int phase_idx, const board *b, int x, int y, int z){
    return reverse_board[b->b[x]] / p35 * p36 + reverse_board[b->b[y]] / p35 * p33 + reverse_board[b->b[z]] / p35;
}

inline int edge_2y(int phase_idx, const board *b, int x, int y){
    return pop_digit[b->b[x]][2] * p39 + b->b[y] * p31 + pop_digit[b->b[x]][5];
}

inline int narrow_triangle0(int phase_idx, const board *b, int v, int w, int x, int y, int z){
    return b->b[v] / p33 * p35 + b->b[w] / p36 * p33 + b->b[x] / p37 * p32 + b->b[y] / p37 * p31 + b->b[z] / p37;
}

inline int narrow_triangle1(int phase_idx, const board *b, int v, int w, int x, int y, int z){
    return reverse_board[b->b[v]] / p33 * p35 + reverse_board[b->b[w]] / p36 * p33 + pop_digit[b->b[x]][7] * p32 + pop_digit[b->b[y]][7] / p37 * p31 + pop_digit[b->b[z]][7] / p37;
}

inline void calc_idx(int phase_idx, const board *b, int idxes[]){
    idxes[0] = b->b[1];
    idxes[1] = b->b[6];
    idxes[2] = b->b[9];
    idxes[3] = b->b[14];
    idxes[4] = b->b[2];
    idxes[5] = b->b[5];
    idxes[6] = b->b[10];
    idxes[7] = b->b[13];
    idxes[8] = b->b[3];
    idxes[9] = b->b[4];
    idxes[10] = b->b[11];
    idxes[11] = b->b[12];
    idxes[12] = b->b[18] / p33;
    idxes[13] = b->b[24] / p33;
    idxes[14] = b->b[29] / p33;
    idxes[15] = b->b[35] / p33;
    idxes[16] = b->b[19] / p32;
    idxes[17] = b->b[23] / p32;
    idxes[18] = b->b[30] / p32;
    idxes[19] = b->b[34] / p32;
    idxes[20] = b->b[20] / p31;
    idxes[21] = b->b[22] / p31;
    idxes[22] = b->b[31] / p31;
    idxes[23] = b->b[33] / p31;
    idxes[24] = b->b[21];
    idxes[25] = b->b[32];
    idxes[26] = edge_2x(phase_idx, b, 1, 0);
    idxes[27] = edge_2x(phase_idx, b, 6, 7);
    idxes[28] = edge_2x(phase_idx, b, 9, 8);
    idxes[29] = edge_2x(phase_idx, b, 14, 15);
    idxes[30] = triangle0(phase_idx, b, 0, 1, 2, 3);
    idxes[31] = triangle0(phase_idx, b, 7, 6, 5, 4);
    idxes[32] = triangle0(phase_idx, b, 15, 14, 13, 12);
    idxes[33] = triangle1(phase_idx, b, 15, 14, 13, 12);
    idxes[34] = edge_block(phase_idx, b, 0, 1);
    idxes[35] = edge_block(phase_idx, b, 7, 6);
    idxes[36] = edge_block(phase_idx, b, 8, 9);
    idxes[37] = edge_block(phase_idx, b, 15, 14);
    idxes[38] = cross0(phase_idx, b, 21, 20, 22);
    idxes[39] = cross1(phase_idx, b, 21, 20, 22);
    idxes[40] = cross0(phase_idx, b, 32, 31, 33);
    idxes[41] = cross1(phase_idx, b, 32, 31, 33);
    idxes[42] = corner90(phase_idx, b, 0, 1, 2);
    idxes[43] = corner91(phase_idx, b, 0, 1, 2);
    idxes[44] = corner90(phase_idx, b, 7, 6, 5);
    idxes[45] = corner91(phase_idx, b, 7, 6, 5);
    idxes[46] = edge_2y(phase_idx, b, 1, 0);
    idxes[47] = edge_2y(phase_idx, b, 6, 7);
    idxes[48] = edge_2y(phase_idx, b, 9, 8);
    idxes[49] = edge_2y(phase_idx, b, 14, 15);
    //idxes[50] = narrow_triangle0(phase_idx, b, 0, 1, 2, 3, 4);
    //idxes[51] = narrow_triangle0(phase_idx, b, 7, 6, 5, 4, 3);
    //idxes[52] = narrow_triangle1(phase_idx, b, 0, 1, 2, 3, 4);
    //idxes[53] = narrow_triangle1(phase_idx, b, 7, 6, 5, 4, 3);
    idxes[50] = min(max_surround - 1, calc_surround(b, black));
    idxes[51] = min(max_surround - 1, calc_surround(b, white));
}

inline void convert_idx(string str){
    int i, j;
    unsigned long long bk = 0, wt = 0;
    char elem;
    board b;
    b.n = 0;
    b.parity = 0;
    for (i = 0; i < hw; ++i){
        for (j = 0; j < hw; ++j){
            elem = str[i * hw + j];
            if (elem != '.'){
                bk |= (unsigned long long)(elem == '0') << (i * hw + j);
                wt |= (unsigned long long)(elem == '1') << (i * hw + j);
                ++b.n;
            }
        }
    }
    for (i = 0; i < b_idx_num; ++i){
        b.b[i] = n_line - 1;
        for (j = 0; j < idx_n_cell[i]; ++j){
            if (1 & (bk >> global_place[i][j]))
                b.b[i] -= pow3[hw_m1 - j] * 2;
            else if (1 & (wt >> global_place[i][j]))
                b.b[i] -= pow3[hw_m1 - j];
        }
    }
    int ai_player = (str[65] == '0' ? 0 : 1);
    int phase_idx = calc_phase_idx(&b);
    int idxes[52];
    calc_idx(phase_idx, &b, idxes);
    cout << phase_idx << " " << ai_player << " ";
    for (i = 0; i < 52; ++i)
        cout << idxes[i] << " ";
    b.p = ai_player;
    cout << max(0, min(max_canput * 2 - 1, calc_canput(&b) + max_canput)) << " ";
    string score;
    istringstream iss(str);
    for (i = 0; i < 3; ++i)
        iss >> score;
    cout << score << endl;
}

#define start_file 373
#define n_files 1000

int main(){
    board_init();
    init_evaluation_base();

    int t = 0;

    for (int i = start_file; i < n_files; ++i){
        cerr << "=";
        ostringstream sout;
        sout << setfill('0') << setw(7) << i;
        string file_name = sout.str();
        ifstream ifs("data/records3/" + file_name + ".txt");
        if (ifs.fail()){
            cerr << "evaluation file not exist" << endl;
            exit(1);
        }
        string line;
        while (getline(ifs, line)){
            ++t;
            convert_idx(line);
        }
        if (i % 10 == 9)
            cerr << endl;
    }
    cerr << t << endl;

    return 0;

}