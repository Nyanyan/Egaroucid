#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <fstream>
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif
#include "new_util/board.hpp"

using namespace std;

#define N_PATTERNS 12
#ifndef N_SYMMETRY_PATTERNS
    #define N_SYMMETRY_PATTERNS 46
#endif
#define N_RAW_PARAMS 46

#define MAX_EVALUATE_IDX 59049
#define MAX_PATTERN_CELLS 10

#define PNO 0

#define P30 1
#define P31 3
#define P32 9
#define P33 27
#define P34 81
#define P35 243
#define P36 729
#define P37 2187
#define P38 6561
#define P39 19683
#define P310 59049
#define P31m 2
#define P32m 8
#define P33m 26
#define P34m 80
#define P35m 242
#define P36m 728
#define P37m 2186
#define P38m 6560
#define P39m 19682
#define P310m 59048

#define P40 1
#define P41 4
#define P42 16
#define P43 64
#define P44 256
#define P45 1024
#define P46 4096
#define P47 16384
#define P48 65536

#define COORD_A1 63
#define COORD_B1 62
#define COORD_C1 61
#define COORD_D1 60
#define COORD_E1 59
#define COORD_F1 58
#define COORD_G1 57
#define COORD_H1 56

#define COORD_A2 55
#define COORD_B2 54
#define COORD_C2 53
#define COORD_D2 52
#define COORD_E2 51
#define COORD_F2 50
#define COORD_G2 49
#define COORD_H2 48

#define COORD_A3 47
#define COORD_B3 46
#define COORD_C3 45
#define COORD_D3 44
#define COORD_E3 43
#define COORD_F3 42
#define COORD_G3 41
#define COORD_H3 40

#define COORD_A4 39
#define COORD_B4 38
#define COORD_C4 37
#define COORD_D4 36
#define COORD_E4 35
#define COORD_F4 34
#define COORD_G4 33
#define COORD_H4 32

#define COORD_A5 31
#define COORD_B5 30
#define COORD_C5 29
#define COORD_D5 28
#define COORD_E5 27
#define COORD_F5 26
#define COORD_G5 25
#define COORD_H5 24

#define COORD_A6 23
#define COORD_B6 22
#define COORD_C6 21
#define COORD_D6 20
#define COORD_E6 19
#define COORD_F6 18
#define COORD_G6 17
#define COORD_H6 16

#define COORD_A7 15
#define COORD_B7 14
#define COORD_C7 13
#define COORD_D7 12
#define COORD_E7 11
#define COORD_F7 10
#define COORD_G7 9
#define COORD_H7 8

#define COORD_A8 7
#define COORD_B8 6
#define COORD_C8 5
#define COORD_D8 4
#define COORD_E8 3
#define COORD_F8 2
#define COORD_G8 1
#define COORD_H8 0

#define COORD_NO 64

#define N_LAST_MOVES 7

struct Feature_to_coord{
    int n_cells;
    uint_fast8_t cells[MAX_PATTERN_CELLS];
};

constexpr Feature_to_coord feature_to_coord[N_SYMMETRY_PATTERNS] = {
    // 0 line2
    {8, {COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_G2, COORD_H2, COORD_NO, COORD_NO}}, // 0
    {8, {COORD_B1, COORD_B2, COORD_B3, COORD_B4, COORD_B5, COORD_B6, COORD_B7, COORD_B8, COORD_NO, COORD_NO}}, // 1
    {8, {COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_F7, COORD_G7, COORD_H7, COORD_NO, COORD_NO}}, // 2
    {8, {COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_G7, COORD_G8, COORD_NO, COORD_NO}}, // 3

    // 1 line3
    {8, {COORD_A3, COORD_B3, COORD_C3, COORD_D3, COORD_E3, COORD_F3, COORD_G3, COORD_H3, COORD_NO, COORD_NO}}, // 4
    {8, {COORD_C1, COORD_C2, COORD_C3, COORD_C4, COORD_C5, COORD_C6, COORD_C7, COORD_C8, COORD_NO, COORD_NO}}, // 5
    {8, {COORD_A6, COORD_B6, COORD_C6, COORD_D6, COORD_E6, COORD_F6, COORD_G6, COORD_H6, COORD_NO, COORD_NO}}, // 6
    {8, {COORD_F1, COORD_F2, COORD_F3, COORD_F4, COORD_F5, COORD_F6, COORD_F7, COORD_F8, COORD_NO, COORD_NO}}, // 7

    // 2 line4
    {8, {COORD_A4, COORD_B4, COORD_C4, COORD_D4, COORD_E4, COORD_F4, COORD_G4, COORD_H4, COORD_NO, COORD_NO}}, // 8
    {8, {COORD_D1, COORD_D2, COORD_D3, COORD_D4, COORD_D5, COORD_D6, COORD_D7, COORD_D8, COORD_NO, COORD_NO}}, // 9
    {8, {COORD_A5, COORD_B5, COORD_C5, COORD_D5, COORD_E5, COORD_F5, COORD_G5, COORD_H5, COORD_NO, COORD_NO}}, // 10
    {8, {COORD_E1, COORD_E2, COORD_E3, COORD_E4, COORD_E5, COORD_E6, COORD_E7, COORD_E8, COORD_NO, COORD_NO}}, // 11

    // 3 diag5
    {5, {COORD_D1, COORD_E2, COORD_F3, COORD_G4, COORD_H5, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 12
    {5, {COORD_E1, COORD_D2, COORD_C3, COORD_B4, COORD_A5, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 13
    {5, {COORD_A4, COORD_B5, COORD_C6, COORD_D7, COORD_E8, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 14
    {5, {COORD_H4, COORD_G5, COORD_F6, COORD_E7, COORD_D8, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 15

    // 4 diag6
    {6, {COORD_C1, COORD_D2, COORD_E3, COORD_F4, COORD_G5, COORD_H6, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 16
    {6, {COORD_F1, COORD_E2, COORD_D3, COORD_C4, COORD_B5, COORD_A6, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 17
    {6, {COORD_A3, COORD_B4, COORD_C5, COORD_D6, COORD_E7, COORD_F8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 18
    {6, {COORD_H3, COORD_G4, COORD_F5, COORD_E6, COORD_D7, COORD_C8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 19

    // 5 diag7
    {7, {COORD_B1, COORD_C2, COORD_D3, COORD_E4, COORD_F5, COORD_G6, COORD_H7, COORD_NO, COORD_NO, COORD_NO}}, // 20
    {7, {COORD_G1, COORD_F2, COORD_E3, COORD_D4, COORD_C5, COORD_B6, COORD_A7, COORD_NO, COORD_NO, COORD_NO}}, // 21
    {7, {COORD_A2, COORD_B3, COORD_C4, COORD_D5, COORD_E6, COORD_F7, COORD_G8, COORD_NO, COORD_NO, COORD_NO}}, // 22
    {7, {COORD_H2, COORD_G3, COORD_F4, COORD_E5, COORD_D6, COORD_C7, COORD_B8, COORD_NO, COORD_NO, COORD_NO}}, // 23

    // 6 diag8
    {8, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_E5, COORD_F6, COORD_G7, COORD_H8, COORD_NO, COORD_NO}}, // 24
    {8, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_D5, COORD_C6, COORD_B7, COORD_A8, COORD_NO, COORD_NO}}, // 25

    // 7 edge + 2x
    {10, {COORD_B2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_G2}}, // 26
    {10, {COORD_B2, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B7}}, // 27
    {10, {COORD_B7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_G7}}, // 28
    {10, {COORD_G2, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7}}, // 29

    // 8 corner + block
    {10, {COORD_A1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_H1, COORD_C2, COORD_D2, COORD_E2, COORD_F2}}, // 30
    {10, {COORD_A1, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A8, COORD_B3, COORD_B4, COORD_B5, COORD_B6}}, // 31
    {10, {COORD_A8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_H8, COORD_C7, COORD_D7, COORD_E7, COORD_F7}}, // 32
    {10, {COORD_H1, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H8, COORD_G3, COORD_G4, COORD_G5, COORD_G6}}, // 33

    // 9 corner9
    {9, {COORD_A1, COORD_B1, COORD_C1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_C3, COORD_NO}}, // 34
    {9, {COORD_H1, COORD_G1, COORD_F1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_F3, COORD_NO}}, // 35
    {9, {COORD_A8, COORD_B8, COORD_C8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_C6, COORD_NO}}, // 36
    {9, {COORD_H8, COORD_G8, COORD_F8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_F6, COORD_NO}}, // 37

    // 10 triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_A4}}, // 38
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_H4}}, // 39
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_A5}}, // 40
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_H5}}, // 41

    // 11 cross
    {10, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_B1, COORD_C2, COORD_D3, COORD_A2, COORD_B3, COORD_C4}}, // 42
    {10, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_G1, COORD_F2, COORD_E3, COORD_H2, COORD_G3, COORD_F4}}, // 43
    {10, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_B8, COORD_C7, COORD_D6, COORD_A7, COORD_B6, COORD_C5}}, // 44
    {10, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_G8, COORD_F7, COORD_E6, COORD_H7, COORD_G6, COORD_F5}}  // 45
};

constexpr uint_fast16_t pow3[11] = {1, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int P3){
    return b_arr[p0] * P33 + b_arr[p1] * P32 + b_arr[p2] * P31 + b_arr[P3];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4){
    return b_arr[p0] * P34 + b_arr[p1] * P33 + b_arr[p2] * P32 + b_arr[P3] * P31 + b_arr[P4];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5){
    return b_arr[p0] * P35 + b_arr[p1] * P34 + b_arr[p2] * P33 + b_arr[P3] * P32 + b_arr[P4] * P31 + b_arr[p5];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6){
    return b_arr[p0] * P36 + b_arr[p1] * P35 + b_arr[p2] * P34 + b_arr[P3] * P33 + b_arr[P4] * P32 + b_arr[p5] * P31 + b_arr[p6];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6, const int p7){
    return b_arr[p0] * P37 + b_arr[p1] * P36 + b_arr[p2] * P35 + b_arr[P3] * P34 + b_arr[P4] * P33 + b_arr[p5] * P32 + b_arr[p6] * P31 + b_arr[p7];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6, const int p7, const int p8){
    return b_arr[p0] * P38 + b_arr[p1] * P37 + b_arr[p2] * P36 + b_arr[P3] * P35 + b_arr[P4] * P34 + b_arr[p5] * P33 + b_arr[p6] * P32 + b_arr[p7] * P31 + b_arr[p8];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6, const int p7, const int p8, const int p9){
    return b_arr[p0] * P39 + b_arr[p1] * P38 + b_arr[p2] * P37 + b_arr[P3] * P36 + b_arr[P4] * P35 + b_arr[p5] * P34 + b_arr[p6] * P33 + b_arr[p7] * P32 + b_arr[p8] * P31 + b_arr[p9];
}

inline void calc_idx(int phase_idx, Board *b, int idxes[]){
    uint_fast8_t b_arr[HW2];
    b->translate_to_arr_player_rev(b_arr);
    int i;
    uint64_t player_mobility = calc_legal(b->player, b->opponent);
    uint64_t opponent_mobility = calc_legal(b->opponent, b->player);
    for (i = 0; i < N_SYMMETRY_PATTERNS; ++i){
        if (feature_to_coord[i].n_cells == 10)
            idxes[i] = pick_pattern(phase_idx, -1, b_arr, feature_to_coord[i].cells[0], feature_to_coord[i].cells[1], feature_to_coord[i].cells[2], feature_to_coord[i].cells[3], feature_to_coord[i].cells[4], feature_to_coord[i].cells[5], feature_to_coord[i].cells[6], feature_to_coord[i].cells[7], feature_to_coord[i].cells[8], feature_to_coord[i].cells[9]);
        else if (feature_to_coord[i].n_cells == 9)
            idxes[i] = pick_pattern(phase_idx, -1, b_arr, feature_to_coord[i].cells[0], feature_to_coord[i].cells[1], feature_to_coord[i].cells[2], feature_to_coord[i].cells[3], feature_to_coord[i].cells[4], feature_to_coord[i].cells[5], feature_to_coord[i].cells[6], feature_to_coord[i].cells[7], feature_to_coord[i].cells[8]);
        else if (feature_to_coord[i].n_cells == 8)
            idxes[i] = pick_pattern(phase_idx, -1, b_arr, feature_to_coord[i].cells[0], feature_to_coord[i].cells[1], feature_to_coord[i].cells[2], feature_to_coord[i].cells[3], feature_to_coord[i].cells[4], feature_to_coord[i].cells[5], feature_to_coord[i].cells[6], feature_to_coord[i].cells[7]);
        else if (feature_to_coord[i].n_cells == 7)
            idxes[i] = pick_pattern(phase_idx, -1, b_arr, feature_to_coord[i].cells[0], feature_to_coord[i].cells[1], feature_to_coord[i].cells[2], feature_to_coord[i].cells[3], feature_to_coord[i].cells[4], feature_to_coord[i].cells[5], feature_to_coord[i].cells[6]);
        else if (feature_to_coord[i].n_cells == 6)
            idxes[i] = pick_pattern(phase_idx, -1, b_arr, feature_to_coord[i].cells[0], feature_to_coord[i].cells[1], feature_to_coord[i].cells[2], feature_to_coord[i].cells[3], feature_to_coord[i].cells[4], feature_to_coord[i].cells[5]);
        else if (feature_to_coord[i].n_cells == 5)
            idxes[i] = pick_pattern(phase_idx, -1, b_arr, feature_to_coord[i].cells[0], feature_to_coord[i].cells[1], feature_to_coord[i].cells[2], feature_to_coord[i].cells[3], feature_to_coord[i].cells[4]);
        else if (feature_to_coord[i].n_cells == 4)
            idxes[i] = pick_pattern(phase_idx, -1, b_arr, feature_to_coord[i].cells[0], feature_to_coord[i].cells[1], feature_to_coord[i].cells[2], feature_to_coord[i].cells[3]);
        else
            cerr << "err" << endl;
    }
}

inline void convert_idx(string str, ofstream *fout){
    int i, j;
    unsigned long long bk = 0, wt = 0;
    char elem;
    Board b;
    b.n = 0;
    b.parity = 0;
    for (i = 0; i < HW; ++i){
        for (j = 0; j < HW; ++j){
            elem = str[i * HW + j];
            if (elem != '.'){
                bk |= (unsigned long long)(elem == '0') << (HW2_M1 - (i * HW + j));
                wt |= (unsigned long long)(elem == '1') << (HW2_M1 - (i * HW + j));
                ++b.n;
            }
        }
    }
    int ai_player, score;
    ai_player = (str[65] == '0' ? 0 : 1);
    if (ai_player == 0){
        b.player = bk;
        b.opponent = wt;
    } else{
        b.player = wt;
        b.opponent = bk;
    }
    int moves[N_LAST_MOVES];
    for (i = 0; i < N_LAST_MOVES; ++i)
        moves[i] = str[67 + i] - '!';
    int idxes[N_RAW_PARAMS];
    calc_idx(0, &b, idxes);
    int n_stones = pop_count_ull(b.player | b.opponent);
    fout->write((char*)&ai_player, 4);
    for (i = 0; i < N_RAW_PARAMS; ++i)
        fout->write((char*)&idxes[i], 4);
    for (i = 0; i < N_LAST_MOVES; ++i)
        fout->write((char*)&moves[i], 4);
}

int main(int argc, char *argv[]){
    board_init();

    int t = 0;

    int start_file = atoi(argv[2]);
    int n_files = atoi(argv[3]);

    ofstream fout;
    fout.open(argv[4], ios::out|ios::binary|ios::trunc);
    if (!fout){
        cerr << "can't open" << endl;
        return 1;
    }

    for (int i = start_file; i < n_files; ++i){
        cerr << "=";
        ostringstream sout;
        sout << setfill('0') << setw(7) << i;
        string file_name = sout.str();
        ifstream ifs("data/" + file_name + ".txt");
        if (ifs.fail()){
            cerr << "evaluation file not exist" << endl;
            return 1;
        }
        string line;
        while (getline(ifs, line)){
            ++t;
            convert_idx(line, &fout);
        }
        if (i % 20 == 19)
            cerr << endl;
    }
    cerr << t << endl;
    return 0;

}