#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <ios>
#include <iomanip>

#define ADJ_N_EVAL (16 + 3 + 4)
#define ADJ_MAX_EVALUATE_IDX 65536

#define ADJ_MAX_SURROUND 64
#define ADJ_MAX_CANPUT 35
#define ADJ_MAX_STONE_NUM 65
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

/*
    @brief 4 ^ N definition
*/
#define P40 1
#define P41 4
#define P42 16
#define P43 64
#define P44 256
#define P45 1024
#define P46 4096
#define P47 16384
#define P48 65536

#define N_PHASES 30

constexpr size_t adj_eval_sizes[ADJ_N_EVAL] = {
    P38, P38, P38, P35, P36, P37, P38, 
    P310, P310, P310, P310, P39, P310, P310, P310, P310, 
    ADJ_MAX_SURROUND * ADJ_MAX_SURROUND, 
    ADJ_MAX_CANPUT * ADJ_MAX_CANPUT, 
    ADJ_MAX_STONE_NUM * ADJ_MAX_STONE_NUM, 
    P44 * P44, P44 * P44, P44 * P44, P44 * P44
};

int16_t eval_arr[N_PHASES][ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];

constexpr int switch_idxes[ADJ_N_EVAL] = {
    0, 5, 7, 8, // for move ordering
    1, 2, 3, 4, 
    6, 9, 10, 11, 
    12, 13, 14, 15, 
    16, 
    17, 
    18, 
    19, 20, 21, 22
};

inline bool file_open(FILE **fp, const char *file, const char *mode){
    #ifdef _WIN64
        return fopen_s(fp, file, mode) == 0;
    #elif _WIN32
        return fopen_s(fp, file, mode) == 0;
    #else
        *fp = fopen(file, mode);
        return *fp != NULL;
    #endif
}

void import_eval(const char* file){
    FILE* fp;
    if (!file_open(&fp, file, "rb")){
        std::cerr << "[ERROR] [FATAL] can't open " << file << std::endl;
        return;
    }
    for (int phase = 0; phase < N_PHASES; ++phase){
        for (int i = 0; i < ADJ_N_EVAL; ++i){
            if (fread(eval_arr[phase][i], 2, adj_eval_sizes[i], fp) < adj_eval_sizes[i]){
                std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
                fclose(fp);
                return;
            }
        }
    }
}

void output_eval(std::string file){
    std::ofstream fout;
    fout.open(file, std::ios::out|std::ios::binary|std::ios::trunc);
    if (!fout){
        std::cerr << "can't open " << file << std::endl;
        return;
    }
    for (int phase = 0; phase < N_PHASES; ++phase){
        for (int i = 0; i < ADJ_N_EVAL; ++i){
            int j = switch_idxes[i];
            for (int k = 0; k < (int)adj_eval_sizes[j]; ++k){
                fout.write((char*)&(eval_arr[phase][j][k]), 2);
            }
        }
    }
    fout.close();
}

int main(int argc, char* argv[]){
    if (argc < 3){
        std::cerr << "input [in_file] [out_file]" << std::endl;
        return 1;
    }
    std::string in_file = std::string(argv[1]);
    std::string out_file = std::string(argv[2]);
    
    import_eval(in_file.c_str());
    output_eval(out_file);
    return 0;
}