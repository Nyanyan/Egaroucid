#include <iostream>
#include <fstream>

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

#define N_PARAM_PER_PHASE 793168

int main(){
    FILE* fp;
    if (!file_open(&fp, "eval_old.egev", "rb")){
        std::cerr << "[ERROR] [FATAL] can't open eval" << std::endl;
        return false;
    }
    short params[N_PARAM_PER_PHASE];
    for (int i = 0; i < 15; ++i){
        fread(params, 2, N_PARAM_PER_PHASE, fp);
        std::ofstream ofs;
        std::string file_name = std::to_string(i) + ".txt";
        ofs.open(file_name, std::ios::out);
        for (int j = 0; j < N_PARAM_PER_PHASE; ++j){
            ofs << params[j] << std::endl;
        }
        ofs.close();
    }
    return 0;
}