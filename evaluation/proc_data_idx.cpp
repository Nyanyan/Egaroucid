#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include "evaluation_definition.hpp"

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
    int idxes[N_FEATURES];
    calc_features(&b, idxes);
    int n_stones = pop_count_ull(b.player | b.opponent);
    fout->write((char*)&n_stones, 4);
    fout->write((char*)&ai_player, 4);
    for (i = 0; i < N_FEATURES; ++i)
        fout->write((char*)&idxes[i], 4);
    fout->write((char*)&score, 4);
}

int main(int argc, char *argv[]){
    if (argc < 5){
        std::cerr << "input [input dir] [start file no] [n files] [output file]" << std::endl;
        return 1;
    }
    board_init();

    int t = 0;

    int start_file = atoi(argv[2]);
    int n_files = atoi(argv[3]);

    ofstream fout;
    fout.open(argv[4], ios::out|ios::binary|ios::trunc);
    if (!fout){
        std::cerr << "can't open" << std::endl;
        return 1;
    }

    for (int i = start_file; i < n_files; ++i){
        std::cerr << "=";
        ostringstream sout;
        sout << setfill('0') << setw(7) << i;
        string file_name = sout.str();
        ifstream ifs("data/" + string(argv[1]) + "/" + file_name + ".txt");
        if (ifs.fail()){
            std::cerr << "evaluation file not exist" << std::endl;
            return 1;
        }
        string line;
        while (getline(ifs, line)){
            ++t;
            convert_idx(line, &fout);
        }
        if (i % 20 == 19)
            std::cerr << std::endl;
    }
    std::cerr << t << std::endl;
    return 0;

}