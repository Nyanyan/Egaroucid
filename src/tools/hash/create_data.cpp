#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <unordered_set>
#include "./../../engine/board.hpp"

using namespace std;

#define USE_N_DISCS 53

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
    if (pop_count_ull(b.player | b.opponent) <= USE_N_DISCS){
        const uint16_t *p = (uint16_t*)&b.player;
        const uint16_t *o = (uint16_t*)&b.opponent;
        for (int i = 0; i < 4; ++i)
            fout->write((char*)&p[i], 2);
        for (int i = 0; i < 4; ++i)
            fout->write((char*)&o[i], 2);
    }
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

    unordered_set<string> boards;
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
            if (boards.find(line) == boards.end()){
                ++t;
                convert_idx(line, &fout);
                boards.emplace(line);
            }
        }
        if (i % 20 == 19)
            cerr << endl;
    }
    cerr << t << endl;
    return 0;

}