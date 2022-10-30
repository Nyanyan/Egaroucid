#include <iostream>

using namespace std;

#define N_PATTERNS 12
#define N_SYMMETRY_PATTERNS 45
#define MAX_PATTERN_CELLS 10
#define MAX_CELL_PATTERNS 13
#define MAX_SURROUND 100
#define MAX_CANPUT 50
//#define MAX_STABILITY 65
#define MAX_STONE_NUM 65
#define N_CANPUT_PATTERNS 4
#define MAX_EVALUATE_IDX 59049
#define N_PHASES 30

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

constexpr uint_fast16_t pow3[11] = {1, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};

int16_t pattern_arr[2][N_PHASES][N_PATTERNS][MAX_EVALUATE_IDX];
int16_t eval_sur0_sur1_arr[N_PHASES][MAX_SURROUND][MAX_SURROUND];
int16_t eval_canput0_canput1_arr[N_PHASES][MAX_CANPUT][MAX_CANPUT];
int16_t eval_num0_num1_arr[N_PHASES][MAX_STONE_NUM][MAX_STONE_NUM];
int16_t eval_canput_pattern[N_PHASES][N_CANPUT_PATTERNS][P48];

inline bool init_evaluation_calc(const char* file){
    cerr << file << endl;
    FILE* fp;
    #ifdef _WIN64
        if (fopen_s(&fp, file, "rb") != 0){
            cerr << "can't open eval.egev" << endl;
            return false;
        }
    #else
        fp = fopen("resources/eval.egev", "rb");
        if (fp == NULL){
            cerr << "can't open eval.egev" << endl;
            return false;
        }
    #endif
    int phase_idx, pattern_idx;
    constexpr int pattern_sizes[N_PATTERNS] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 9};
    for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        //cerr << "evaluation function " << phase_idx * 100 / N_PHASES << " % initialized" << endl;
        for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx){
            if (fread(pattern_arr[0][phase_idx][pattern_idx], 2, pow3[pattern_sizes[pattern_idx]], fp) < pow3[pattern_sizes[pattern_idx]]){
                cerr << "eval.egev broken" << endl;
                fclose(fp);
                return false;
            }
        }
        if (fread(eval_sur0_sur1_arr[phase_idx], 2, MAX_SURROUND * MAX_SURROUND, fp) < MAX_SURROUND * MAX_SURROUND){
            cerr << "eval.egev broken" << endl;
            fclose(fp);
            return false;
        }
        if (fread(eval_canput0_canput1_arr[phase_idx], 2, MAX_CANPUT * MAX_CANPUT, fp) < MAX_CANPUT * MAX_CANPUT){
            cerr << "eval.egev broken" << endl;
            fclose(fp);
            return false;
        }
        /*
        if (fread(eval_stab0_stab1_arr[phase_idx], 2, MAX_STABILITY * MAX_STABILITY, fp) < MAX_STABILITY * MAX_STABILITY){
            cerr << "eval.egev broken" << endl;
            fclose(fp);
            return false;
        }
        */
        if (fread(eval_num0_num1_arr[phase_idx], 2, MAX_STONE_NUM * MAX_STONE_NUM, fp) < MAX_STONE_NUM * MAX_STONE_NUM){
            cerr << "eval.egev broken" << endl;
            fclose(fp);
            return false;
        }
        if (fread(eval_canput_pattern[phase_idx], 2, N_CANPUT_PATTERNS * P48, fp) < N_CANPUT_PATTERNS * P48){
            cerr << "eval.egev broken" << endl;
            fclose(fp);
            return false;
        }
    }
    cerr << "evaluation function initialized" << endl;
    return true;
}

int main(){
    init_evaluation_calc("resources/eval.egev");
    cout << "#define N_PATTERNS 12\n#define MAX_EVALUATE_IDX 59049\n#define MAX_SURROUND 100\n#define MAX_CANPUT 50\n#define MAX_STONE_NUM 65\n#define N_CANPUT_PATTERNS 4\n#define P48 65536" << endl;

    cout << "int16_t pattern_arr[2][N_PHASES][N_PATTERNS][MAX_EVALUATE_IDX]={{" << endl;
    int i, j, k;
    for (i = 0; i < N_PHASES; ++i){
        cout << "{" << endl;
        for (j = 0; j < N_PATTERNS; ++j){
            cout << "{" << endl;
            for (k = 0; k < MAX_EVALUATE_IDX; ++k){
                cout << pattern_arr[0][i][j][k];
                if (k == MAX_EVALUATE_IDX - 1)
                    cout << endl;
                else
                    cout << ",";
            }
            cout << "}";
            if (j == N_PATTERNS - 1)
                cout << endl;
            else
                cout << ",";
        }
        cout << "}";
        if (i == N_PHASES - 1)
            cout << endl;
        else
            cout << ",";
    }
    cout << "}};" << endl;

    cout << "constexpr int16_t eval_sur0_sur1_arr[N_PHASES][MAX_SURROUND][MAX_SURROUND]={" << endl;
    for (i = 0; i < N_PHASES; ++i){
        cout << "{" << endl;
        for (j = 0; j < MAX_SURROUND; ++j){
            cout << "{" << endl;
            for (k = 0; k < MAX_SURROUND; ++k){
                cout << eval_sur0_sur1_arr[i][j][k];
                if (k == MAX_SURROUND - 1)
                    cout << endl;
                else
                    cout << ",";
            }
            cout << "}";
            if (j == MAX_SURROUND - 1)
                cout << endl;
            else
                cout << ",";
        }
        cout << "}";
        if (i == N_PHASES - 1)
            cout << endl;
        else
            cout << ",";
    }
    cout << "};" << endl;

    cout << "constexpr int16_t eval_canput0_canput1_arr[N_PHASES][MAX_CANPUT][MAX_CANPUT]={" << endl;
    for (i = 0; i < N_PHASES; ++i){
        cout << "{" << endl;
        for (j = 0; j < MAX_CANPUT; ++j){
            cout << "{" << endl;
            for (k = 0; k < MAX_CANPUT; ++k){
                cout << eval_canput0_canput1_arr[i][j][k];
                if (k == MAX_CANPUT - 1)
                    cout << endl;
                else
                    cout << ",";
            }
            cout << "}";
            if (j == MAX_CANPUT - 1)
                cout << endl;
            else
                cout << ",";
        }
        cout << "}";
        if (i == N_PHASES - 1)
            cout << endl;
        else
            cout << ",";
    }
    cout << "};" << endl;

    cout << "constexpr int16_t eval_num0_num1_arr[N_PHASES][MAX_STONE_NUM][MAX_STONE_NUM]={" << endl;
    for (i = 0; i < N_PHASES; ++i){
        cout << "{" << endl;
        for (j = 0; j < MAX_STONE_NUM; ++j){
            cout << "{" << endl;
            for (k = 0; k < MAX_STONE_NUM; ++k){
                cout << eval_num0_num1_arr[i][j][k];
                if (k == MAX_STONE_NUM - 1)
                    cout << endl;
                else
                    cout << ",";
            }
            cout << "}";
            if (j == MAX_STONE_NUM - 1)
                cout << endl;
            else
                cout << ",";
        }
        cout << "}";
        if (i == N_PHASES - 1)
            cout << endl;
        else
            cout << ",";
    }
    cout << "};" << endl;

    cout << "constexpr int16_t eval_canput_pattern[N_PHASES][N_CANPUT_PATTERNS][P48]={" << endl;
    for (i = 0; i < N_PHASES; ++i){
        cout << "{" << endl;
        for (j = 0; j < N_CANPUT_PATTERNS; ++j){
            cout << "{" << endl;
            for (k = 0; k < P48; ++k){
                cout << eval_canput_pattern[i][j][k];
                if (k == P48 - 1)
                    cout << endl;
                else
                    cout << ",";
            }
            cout << "}";
            if (j == N_CANPUT_PATTERNS - 1)
                cout << endl;
            else
                cout << ",";
        }
        cout << "}";
        if (i == N_PHASES - 1)
            cout << endl;
        else
            cout << ",";
    }
    cout << "};" << endl;
}