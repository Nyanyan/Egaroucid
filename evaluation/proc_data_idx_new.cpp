#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include "new_util/board.hpp"

using namespace std;

#define max_surround 100
#define max_canput 50
#define max_stability 65
#define max_stone_num 65

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

#define P41 4
#define P42 16
#define P43 64
#define P44 256
#define P45 1024
#define P46 4096
#define P47 16384
#define P48 65536

constexpr uint_fast16_t pow3[11] = {1, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};
uint64_t stability_edge_arr[N_8BIT][N_8BIT][2];

inline int calc_phase_idx(const Board *b){
    return (b->n - 4) / PHASE_N_STONES;
}

inline void probably_move_line(int p, int o, int place, int *np, int *no){
    int i, j;
    *np = p | (1 << place);
    for (i = place - 1; i >= 0 && (1 & (o >> i)); --i);
    if (1 & (p >> i)){
        for (j = place - 1; j > i; --j)
            *np ^= 1 << j;
    }
    for (i = place + 1; i < HW && (1 & (o >> i)); ++i);
    if (1 & (p >> i)){
        for (j = place + 1; j < i; ++j)
            *np ^= 1 << j;
    }
    *no = o & ~(*np);
}

int calc_stability_line(int b, int w, int ob, int ow){
    int i, nb, nw, res = 0b11111111;
    res &= b & ob;
    res &= w & ow;
    for (i = 0; i < HW; ++i){
        if ((1 & (b >> i)) == 0 && (1 & (w >> i)) == 0){
            probably_move_line(b, w, i, &nb, &nw);
            res &= calc_stability_line(nb, nw, ob, ow);
            probably_move_line(w, b, i, &nw, &nb);
            res &= calc_stability_line(nb, nw, ob, ow);
        }
    }
    return res;
}

inline void init_evaluation_base() {
    int idx, place, b, w, stab;
    for (b = 0; b < N_8BIT; ++b) {
        for (w = b; w < N_8BIT; ++w){
            stab = calc_stability_line(b, w, b, w);
            stability_edge_arr[b][w][0] = 0;
            stability_edge_arr[b][w][1] = 0;
            for (place = 0; place < HW; ++place){
                if (1 & (stab >> place)){
                    stability_edge_arr[b][w][0] |= 1ULL << place;
                    stability_edge_arr[b][w][1] |= 1ULL << (place * HW);
                }
            }
            stability_edge_arr[w][b][0] = stability_edge_arr[b][w][0];
            stability_edge_arr[w][b][1] = stability_edge_arr[b][w][1];
        }
    }
}

inline uint64_t calc_surround_part(const uint64_t player, const int dr){
    return (player << dr | player >> dr);
}

inline int calc_surround(const uint64_t player, const uint64_t empties){
    return pop_count_ull(empties & (
        calc_surround_part(player & 0b0111111001111110011111100111111001111110011111100111111001111110ULL, 1) | 
        calc_surround_part(player & 0b0000000011111111111111111111111111111111111111111111111100000000ULL, HW) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, HW_M1) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, HW_P1)
    ));
}

inline void calc_stability(Board *b, int *stab0, int *stab1){
    uint64_t full_h, full_v, full_d7, full_d9;
    uint64_t edge_stability = 0, player_stability = 0, opponent_stability = 0, n_stability;
    uint64_t h, v, d7, d9;
    const uint64_t player_mask = b->player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    const uint64_t opponent_mask = b->opponent & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    int pl, op;
    pl = b->player & 0b11111111;
    op = b->opponent & 0b11111111;
    edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = (b->player >> 56) & 0b11111111;
    op = (b->opponent >> 56) & 0b11111111;
    edge_stability |= stability_edge_arr[pl][op][0];
    pl = join_v_line(b->player, 0);
    op = join_v_line(b->opponent, 0);
    edge_stability |= stability_edge_arr[pl][op][1] << 7;
    pl = join_v_line(b->player, 7);
    op = join_v_line(b->opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][1];
    b->full_stability(&full_h, &full_v, &full_d7, &full_d9);

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

    *stab0 = pop_count_ull(player_stability);
    *stab1 = pop_count_ull(opponent_stability);
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

inline int create_canput_line_h(uint64_t b, uint64_t w, int t){
    return (((w >> (HW * t)) & 0b11111111) << HW) | ((b >> (HW * t)) & 0b11111111);
}

inline int create_canput_line_v(uint64_t b, uint64_t w, int t){
    return (join_v_line(w, t) << HW) | join_v_line(b, t);
}


inline void calc_idx(int phase_idx, Board *b, int idxes[]){
    uint_fast8_t b_arr[HW2];
    b->translate_to_arr_player(b_arr);
    idxes[0] = pick_pattern(phase_idx, 0, b_arr, 8, 9, 10, 11, 12, 13, 14, 15);
    idxes[1] = pick_pattern(phase_idx, 0, b_arr, 1, 9, 17, 25, 33, 41, 49, 57);
    idxes[2] = pick_pattern(phase_idx, 0, b_arr, 48, 49, 50, 51, 52, 53, 54, 55);
    idxes[3] = pick_pattern(phase_idx, 0, b_arr, 6, 14, 22, 30, 38, 46, 54, 62);
    idxes[4] = pick_pattern(phase_idx, 1, b_arr, 16, 17, 18, 19, 20, 21, 22, 23);
    idxes[5] = pick_pattern(phase_idx, 1, b_arr, 2, 10, 18, 26, 34, 42, 50, 58);
    idxes[6] = pick_pattern(phase_idx, 1, b_arr, 40, 41, 42, 43, 44, 45, 46, 47);
    idxes[7] = pick_pattern(phase_idx, 1, b_arr, 5, 13, 21, 29, 37, 45, 53, 61);
    idxes[8] = pick_pattern(phase_idx, 2, b_arr, 24, 25, 26, 27, 28, 29, 30, 31);
    idxes[9] = pick_pattern(phase_idx, 2, b_arr, 3, 11, 19, 27, 35, 43, 51, 59);
    idxes[10] = pick_pattern(phase_idx, 2, b_arr, 32, 33, 34, 35, 36, 37, 38, 39);
    idxes[11] = pick_pattern(phase_idx, 2, b_arr, 4, 12, 20, 28, 36, 44, 52, 60);
    idxes[12] = pick_pattern(phase_idx, 3, b_arr, 3, 12, 21, 30, 39);
    idxes[13] = pick_pattern(phase_idx, 3, b_arr, 4, 11, 18, 25, 32);
    idxes[14] = pick_pattern(phase_idx, 3, b_arr, 24, 33, 42, 51, 60);
    idxes[15] = pick_pattern(phase_idx, 3, b_arr, 59, 52, 45, 38, 31);
    idxes[16] = pick_pattern(phase_idx, 4, b_arr, 2, 11, 20, 29, 38, 47);
    idxes[17] = pick_pattern(phase_idx, 4, b_arr, 5, 12, 19, 26, 33, 40);
    idxes[18] = pick_pattern(phase_idx, 4, b_arr, 16, 25, 34, 43, 52, 61);
    idxes[19] = pick_pattern(phase_idx, 4, b_arr, 58, 51, 44, 37, 30, 23);
    idxes[20] = pick_pattern(phase_idx, 5, b_arr, 1, 10, 19, 28, 37, 46, 55);
    idxes[21] = pick_pattern(phase_idx, 5, b_arr, 6, 13, 20, 27, 34, 41, 48);
    idxes[22] = pick_pattern(phase_idx, 5, b_arr, 8, 17, 26, 35, 44, 53, 62);
    idxes[23] = pick_pattern(phase_idx, 5, b_arr, 57, 50, 43, 36, 29, 22, 15);
    idxes[24] = pick_pattern(phase_idx, 6, b_arr, 0, 9, 18, 27, 36, 45, 54, 63);
    idxes[25] = pick_pattern(phase_idx, 6, b_arr, 7, 14, 21, 28, 35, 42, 49, 56);
    idxes[26] = pick_pattern(phase_idx, 7, b_arr, 9, 0, 1, 2, 3, 4, 5, 6, 7, 14);
    idxes[27] = pick_pattern(phase_idx, 7, b_arr, 9, 0, 8, 16, 24, 32, 40, 48, 56, 49);
    idxes[28] = pick_pattern(phase_idx, 7, b_arr, 49, 56, 57, 58, 59, 60, 61, 62, 63, 54);
    idxes[29] = pick_pattern(phase_idx, 7, b_arr, 54, 63, 55, 47, 39, 31, 23, 15, 7);
    idxes[30] = pick_pattern(phase_idx, 8, b_arr, 0, 1, 2, 3, 8, 9, 10, 16, 17, 24);
    idxes[31] = pick_pattern(phase_idx, 8, b_arr, 7, 6, 5, 4, 15, 14, 13, 23, 22, 31);
    idxes[32] = pick_pattern(phase_idx, 8, b_arr, 63, 62, 61, 60, 55, 54, 53, 47, 46, 39);
    idxes[33] = pick_pattern(phase_idx, 8, b_arr, 56, 57, 58, 59, 48, 49, 50, 40, 41, 32);
    idxes[34] = pick_pattern(phase_idx, 9, b_arr, 0, 2, 3, 4, 5, 7, 10, 11, 12, 13);
    idxes[35] = pick_pattern(phase_idx, 9, b_arr, 0, 16, 24, 32, 40, 56, 17, 25, 33, 41);
    idxes[36] = pick_pattern(phase_idx, 9, b_arr, 56, 58, 59, 60, 61, 63, 50, 51, 52, 53);
    idxes[37] = pick_pattern(phase_idx, 9, b_arr, 7, 23, 31, 39, 47, 63, 22, 30, 38, 46);
    idxes[38] = pick_pattern(phase_idx, 10, b_arr, 0, 9, 18, 27, 1, 10, 19, 8, 17, 26);
    idxes[39] = pick_pattern(phase_idx, 10, b_arr, 7, 14, 21, 28, 6, 13, 20, 15, 22, 29);
    idxes[40] = pick_pattern(phase_idx, 10, b_arr, 56, 49, 42, 35, 57, 50, 43, 48, 41, 34);
    idxes[41] = pick_pattern(phase_idx, 10, b_arr, 63, 54, 45, 36, 62, 53, 44, 55, 46, 37);
    idxes[42] = pick_pattern(phase_idx, 11, b_arr, 0, 1, 2, 8, 9, 10, 16, 17, 18);
    idxes[43] = pick_pattern(phase_idx, 11, b_arr, 7, 6, 5, 15, 14, 13, 23, 22, 21);
    idxes[44] = pick_pattern(phase_idx, 11, b_arr, 56, 57, 58, 48, 49, 50, 40, 41, 42);
    idxes[45] = pick_pattern(phase_idx, 11, b_arr, 63, 62, 61, 55, 54, 53, 47, 46, 45);
    idxes[46] = pick_pattern(phase_idx, 12, b_arr, 10, 0, 1, 2, 3, 4, 5, 6, 7, 13);
    idxes[47] = pick_pattern(phase_idx, 12, b_arr, 17, 0, 8, 16, 24, 32, 40, 48, 56, 41);
    idxes[48] = pick_pattern(phase_idx, 12, b_arr, 50, 56, 57, 58, 59, 60, 61, 62, 63, 53);
    idxes[49] = pick_pattern(phase_idx, 12, b_arr, 46, 63, 55, 47, 39, 31, 23, 15, 7, 22);
    idxes[50] = pick_pattern(phase_idx, 13, b_arr, 0, 1, 2, 3, 4, 8, 9, 16, 24, 32);
    idxes[51] = pick_pattern(phase_idx, 13, b_arr, 7, 6, 5, 4, 3, 15, 14, 23, 31, 39);
    idxes[52] = pick_pattern(phase_idx, 13, b_arr, 63, 62, 61, 60, 59, 55, 54, 47, 39, 31);
    idxes[53] = pick_pattern(phase_idx, 13, b_arr, 56, 57, 58, 59, 60, 48, 49, 40, 32, 24);
    idxes[54] = pick_pattern(phase_idx, 14, b_arr, 0, 1, 8, 9, 10, 11, 17, 18, 25, 27);
    idxes[55] = pick_pattern(phase_idx, 14, b_arr, 7, 6, 15, 14, 13, 12, 22, 21, 30, 28);
    idxes[56] = pick_pattern(phase_idx, 14, b_arr, 56, 57, 48, 49, 50, 51, 41, 42, 33, 35);
    idxes[57] = pick_pattern(phase_idx, 14, b_arr, 63, 62, 55, 54, 53, 52, 46, 45, 38, 36);
    idxes[58] = pick_pattern(phase_idx, 15, b_arr, 0, 1, 8, 9, 10, 11, 12, 17, 25, 33);
    idxes[59] = pick_pattern(phase_idx, 15, b_arr, 7, 6, 15, 14, 13, 12, 11, 22, 30, 38);
    idxes[60] = pick_pattern(phase_idx, 15, b_arr, 56, 57, 48, 49, 50, 51, 52, 41, 33, 25);
    idxes[61] = pick_pattern(phase_idx, 15, b_arr, 63, 62, 55, 54, 53, 52, 51, 46, 38, 30);
    idxes[62] = min(max_surround - 1, calc_surround(b->player, ~(b->player | b->opponent)));
    idxes[63] = min(max_surround - 1, calc_surround(b->opponent, ~(b->player | b->opponent)));
    uint64_t player_mobility = calc_legal(b->player, b->opponent);
    uint64_t opponent_mobility = calc_legal(b->opponent, b->player);
    idxes[64] = pop_count_ull(player_mobility);
    idxes[65] = pop_count_ull(opponent_mobility);
    calc_stability(b, &idxes[66], &idxes[67]);
    idxes[68] = pop_count_ull(b->player);
    idxes[69] = pop_count_ull(b->opponent);
    idxes[70] = create_canput_line_h(player_mobility, opponent_mobility, 0);
    idxes[71] = create_canput_line_h(player_mobility, opponent_mobility, 7);
    idxes[72] = create_canput_line_v(player_mobility, opponent_mobility, 0);
    idxes[73] = create_canput_line_v(player_mobility, opponent_mobility, 7);
    idxes[74] = create_canput_line_h(player_mobility, opponent_mobility, 1);
    idxes[75] = create_canput_line_h(player_mobility, opponent_mobility, 6);
    idxes[76] = create_canput_line_v(player_mobility, opponent_mobility, 1);
    idxes[77] = create_canput_line_v(player_mobility, opponent_mobility, 6);
    idxes[78] = create_canput_line_h(player_mobility, opponent_mobility, 2);
    idxes[79] = create_canput_line_h(player_mobility, opponent_mobility, 5);
    idxes[80] = create_canput_line_v(player_mobility, opponent_mobility, 2);
    idxes[81] = create_canput_line_v(player_mobility, opponent_mobility, 5);
    idxes[82] = create_canput_line_h(player_mobility, opponent_mobility, 3);
    idxes[83] = create_canput_line_h(player_mobility, opponent_mobility, 4);
    idxes[84] = create_canput_line_v(player_mobility, opponent_mobility, 3);
    idxes[85] = create_canput_line_v(player_mobility, opponent_mobility, 4);
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
                bk |= (unsigned long long)(elem == '0') << (i * HW + j);
                wt |= (unsigned long long)(elem == '1') << (i * HW + j);
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
    score = stoi(str.substr(67));
    if (ai_player == 1)
        score = -score;
    //b.print();
    int idxes[86];
    calc_idx(0, &b, idxes);
    /*
    cout << idxes[68] + idxes[69] << " " << ai_player << " ";
    for (i = 0; i < 86; ++i)
        cout << idxes[i] << " ";
    cout << score << endl;
    */
    int n_stones = idxes[68] + idxes[69];
    fout->write((char*)&n_stones, 4);
    fout->write((char*)&ai_player, 4);
    for (i = 0; i < 86; ++i)
        fout->write((char*)&idxes[i], 4);
    fout->write((char*)&score, 4);
}

int main(int argc, char *argv[]){
    board_init();
    init_evaluation_base();

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
        ifstream ifs("data/" + string(argv[1]) + "/" + file_name + ".txt");
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