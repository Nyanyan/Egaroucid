#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include "evaluation_definition.hpp"

inline void adj_convert_idx(std::string str, std::ofstream *fout){
    int i, j;
    uint64_t bk = 0, wt = 0;
    char elem;
    Board b;
    for (i = 0; i < HW; ++i){
        for (j = 0; j < HW; ++j){
            elem = str[i * HW + j];
            if (elem != '.'){
                bk |= (uint64_t)(elem == '0') << (i * HW + j);
                wt |= (uint64_t)(elem == '1') << (i * HW + j);
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
    uint16_t idxes[ADJ_N_FEATURES];
    adj_calc_features(&b, idxes);
    int n_stones = pop_count_ull(b.player | b.opponent);
    fout->write((char*)&n_stones, 2);
    fout->write((char*)&ai_player, 2);
    fout->write((char*)idxes, 2 * ADJ_N_FEATURES);
    fout->write((char*)&score, 2);
}

int main(int argc, char *argv[]){
    if (argc < 5){
        std::cerr << "input [input dir] [start file no] [n files] [output file]" << std::endl;
        return 1;
    }

    int t = 0;

    int start_file = atoi(argv[2]);
    int n_files = atoi(argv[3]);

    std::ofstream fout;
    fout.open(argv[4], std::ios::out|std::ios::binary|std::ios::trunc);
    if (!fout){
        std::cerr << "can't open" << std::endl;
        return 1;
    }

    for (int i = start_file; i < n_files; ++i){
        std::cerr << "=";
        std::ostringstream sout;
        sout << std::setfill('0') << std::setw(7) << i;
        std::string file_name = sout.str();
        std::ifstream ifs(std::string(argv[1]) + "/" + file_name + ".txt");
        if (ifs.fail()){
            std::cerr << "evaluation file not exist" << std::endl;
            return 1;
        }
        std::string line;
        while (getline(ifs, line)){
            ++t;
            adj_convert_idx(line, &fout);
        }
        if (i % 20 == 19)
            std::cerr << std::endl;
    }
    std::cerr << t << std::endl;
    return 0;

}