#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <unordered_set>
#include "new_util/board.hpp"

using namespace std;

#define MAX_N_LINE 4
#define POPULATION 1024
#define N_DATA 10000
#define INF 100000.0

constexpr uint64_t diagonal_lines[22] = {
    0x0000000000010204ULL, 0x0000000001020408ULL, 0x0000000102040810ULL, 0x0000010204081020ULL, 0x0001020408102040ULL, 
    0x0102040810204080ULL, 
    0x0204081020408000ULL, 0x0408102040800000ULL, 0x0810204080000000ULL, 0x1020408000000000ULL, 0x2040800000000000ULL, 
    0x0000000000804020ULL, 0x0000000080402010ULL, 0x0000008040201008ULL, 0x0000804020100804ULL, 0x0080402010080402ULL,
    0x8040201008040201ULL, 
    0x4020100804020100ULL, 0x2010080402010000ULL, 0x1008040201000000ULL, 0x0804020100000000ULL, 0x0402010000000000ULL
};

int n_use_cell;
int n_use_line;

struct Gene{
    uint64_t cell;
    uint64_t line[MAX_N_LINE];
    double score;
};

Gene genes[POPULATION];

struct Datum{
    Board board;
    double score;
};

Datum data[N_DATA];

struct Feature{
    uint64_t cell_player;
    uint64_t cell_opponent;
    bool line_player[MAX_N_LINE];
    bool line_opponent[MAX_N_LINE];
    bool line_empty[MAX_N_LINE];
};

void init(){
    int i, j, k, line, offset;
    bool overlap;
    for (i = 0; i < POPULATION; ++i){
        genes[i].score = INF;
        genes[i].cell = 0ULL;
        while (pop_count_ull(genes[i].cell) != n_use_cell)
            genes[i].cell = myrand_ull();
        for (j = 0; j < n_use_line; ++j){
            line = myrandrange(0, 38);
            if (line < 8)
                genes[i].line[j] = 0xFF00000000000000ULL >> (line * HW); // horizontal
            else if (line < 16)
                genes[i].line[j] = 0x8080808080808080ULL >> (line - 8); // vertical
            else
                genes[i].line[j] = diagonal_lines[line - 16];
            overlap = false;
            for (k = 0; k < j; ++k)
                overlap |= (genes[i].line[j] == genes[i].line[k]);
            if (overlap)
                --j;
        }
    }
}

Datum convert_data(string str){
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
    Datum res = {b, (double)(score + 64) / 128.0};
    return res;
}

void input_data(string dir, int start_file, int end_file){
    int i, t = 0;
    for (i = start_file; i < end_file && t < N_DATA; ++i){
        cerr << "=";
        ostringstream sout;
        sout << setfill('0') << setw(7) << i;
        string file_name = sout.str();
        ifstream ifs("data/" + dir + "/" + file_name + ".txt");
        if (ifs.fail()){
            cerr << "evaluation file not exist" << endl;
            return;
        }
        string line;
        while (getline(ifs, line)){
            data[t++] = convert_data(line);
            if (t >= N_DATA)
                break;
        }
        if (i % 20 == 19)
            cerr << endl;
    }
    cerr << endl << t << " data" << endl;
}

void scoring(Gene *gene){
    unordered_set<Feature> features;
    int i, j;
    Feature feature;
    for (i = 0; i < N_DATA; ++i){
        feature.cell_player = data[i].board.player & gene->cell;
        feature.cell_opponent = data[i].board.opponent & gene->cell;
        for (j = 0; j < n_use_line; ++j){
            feature.line_player[j] = data[i].board.player
        }
    }
}

void scoring_all(){

}

int main(int argc, char *argv[]){
    n_use_cell = atoi(argv[1]);
    n_use_line = atoi(argv[2]);
    cerr << "cell: " << n_use_cell << " line: " << n_use_line << " population: " << POPULATION << endl;

    init();
    cerr << "initialized" << endl;

    input_data("records3", 0, 10);

    scoring_all();

    return 0;
}