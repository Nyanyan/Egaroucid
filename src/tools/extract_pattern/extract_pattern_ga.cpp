#include <iostream>
#include <vector>
#include<algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif
#include "./../../engine/board.hpp"


#define N_CELLS_IN_PATTERN 9
#define N_VARIATION_IN_PATTERN 19683
#define N_GENES 1000
#define N_DATA 10000000
#define MIN_N_DISCS 24
#define GA_SCORE_UNDEFINED -1.0
#define POW2_2N_CELLS_IN_PATTERN 262144
#define N_SHOW_RESULT 30

struct Gene{
    uint64_t pattern;
    double score;
};

struct Datum{
    Board board;
    int score;
};

Gene genes[N_GENES];

std::vector<Datum> data;

std::vector<int> duplication[POW2_2N_CELLS_IN_PATTERN];

void get_data(std::string file){
    FILE* fp;
    if (fopen_s(&fp, file.c_str(), "rb") != 0) {
        std::cerr << "can't open " << file << std::endl;
        return;
    }
    Board board;
    char player, policy, score;
    while (data.size() < N_DATA){
        if (fread(&(board.player), 8, 1, fp) < 1)
            break;
        fread(&(board.opponent), 8, 1, fp);
        fread(&player, 1, 1, fp);
        fread(&policy, 1, 1, fp);
        fread(&score, 1, 1, fp);
        if (board.n_discs() >= MIN_N_DISCS){
            Datum datum;
            datum.board = board;
            datum.score = score;
            data.emplace_back(datum);
        }
    }
}

void scoring(Gene *gene){
    for (int i = 0; i < POW2_2N_CELLS_IN_PATTERN; ++i)
        duplication[i].clear();
    for (Datum &datum: data){
        int gathered_idx = _pext_u64(datum.board.player, gene->pattern) << N_CELLS_IN_PATTERN;
        gathered_idx |= _pext_u64(datum.board.opponent, gene->pattern);
        duplication[gathered_idx].emplace_back(datum.score);
    }
    int n_duplication = 0;
    double res = 0.0;
    for (int i = 0; i < POW2_2N_CELLS_IN_PATTERN; ++i){
        if (duplication[i].size() >= 2){
            ++n_duplication;
            double avg = 0.0;
            for (int &score: duplication[i]){
                avg += score;
            }
            avg /= duplication[i].size();
            double var = 0.0;
            for (int &score: duplication[i]){
                var += (avg - score) * (avg - score);
            }
            var /= duplication[i].size();
            res += var;
        }
    }
    res /= n_duplication;
    gene->score = res;
}

void init_ga(){
    for (int i = 0; i < N_GENES; ++i){
        if ((i & 0b11111) == 0)
            std::cerr << '\r' << ((double)i / N_GENES);
        genes[i].pattern = 0;
        while (pop_count_ull(genes[i].pattern) < N_CELLS_IN_PATTERN)
            genes[i].pattern |= 1ULL << myrandrange(0, 64);
        scoring(&genes[i]);
    }
    std::cerr << std::endl;
}

void ga(){
    int parent0 = myrandrange(0, N_GENES);
    int parent1 = parent0;
    while (parent1 == parent0)
        parent1 = myrandrange(0, N_GENES);
    Gene new_genes[4];
    new_genes[0] = genes[parent0];
    new_genes[1] = genes[parent1];
    new_genes[2].score = GA_SCORE_UNDEFINED;
    new_genes[3].score = GA_SCORE_UNDEFINED;
    new_genes[2].pattern = 0;
    new_genes[3].pattern = 0;
    int pattern_bits[2][N_CELLS_IN_PATTERN];
    int idx0 = 0, idx1 = 0;
    for (int i = 0; i < HW2; ++i){
        if (genes[parent0].pattern & (1ULL << i))
            pattern_bits[0][idx0++] = i;
        if (genes[parent1].pattern & (1ULL << i))
            pattern_bits[1][idx1++] = i;
    }
    int idx_div = myrandrange(1, N_CELLS_IN_PATTERN - 1);
    for (int i = 0; i < idx_div; ++i){
        new_genes[2].pattern |= 1ULL << pattern_bits[0][i];
        new_genes[3].pattern |= 1ULL << pattern_bits[1][i];
    }
    for (int i = idx_div; i < N_CELLS_IN_PATTERN; ++i){
        new_genes[2].pattern |= 1ULL << pattern_bits[1][i];
        new_genes[3].pattern |= 1ULL << pattern_bits[0][i];
    }
    scoring(&new_genes[2]);
    scoring(&new_genes[3]);
    double score_1st = 100000000.0, score_2nd = 100000000.0;
    int idx_1st = -1, idx_2nd = -1;
    for (int i = 0; i < 4; ++i){
        if (new_genes[i].score < score_1st){
            score_2nd = score_1st;
            score_1st = new_genes[i].score;
            idx_2nd = idx_1st;
            idx_1st = i;
        } else if (new_genes[i].score < score_2nd){
            score_2nd = new_genes[i].score;
            idx_2nd = i;
        }
    }
    //std::cerr << score_1st << " " << score_2nd << " " << idx_1st << " " << idx_2nd << std::endl;
    genes[parent0] = new_genes[idx_1st];
    genes[parent1] = new_genes[idx_2nd];
}

bool comp_gene(Gene &a, Gene &b){
    return a.score < b.score;
}

std::string ga_idx_to_coord(int cell){
    int x = cell % HW;
    int y = cell / HW;
    char res[] = "XX";
    res[0] = 'A' + x;
    res[1] = '1' + y;
    return std::string(res);
}

void output_result(){
    std::sort(genes, genes + N_GENES, comp_gene);
    std::cerr << genes[0].score << std::endl;
    for (int i = 0; i < N_SHOW_RESULT; ++i){
        std::cout << genes[i].score << " ";
        int n_shown = 0;
        for (int j = 0; j < HW2; ++j){
            if (1 & (genes[i].pattern >> j)){
                std::cout << "COORD_" + ga_idx_to_coord(j);
                ++n_shown;
                if (n_shown == N_CELLS_IN_PATTERN){
                    std::cout << std::endl;
                } else{
                    std::cout << ", ";
                }
            }
        }
    }
}

int main(){
    std::cerr << "getting data..." << std::endl;
    get_data("./../../../train_data/board_data/records19/0.dat");
    std::cerr << "initializing..." << std::endl;
    init_ga();
    std::cerr << "start!" << std::endl;
    output_result();
    uint64_t strt = tim();
    uint64_t interval_strt = strt;
    uint64_t t = 0;
    while (true){
        ga();
        if (tim() - interval_strt >= 10000){
            interval_strt = tim();
            std::cout << t << " " << tim() - strt << std::endl;
            std::cerr << t << " " << tim() - strt << " ";
            output_result();
        }
        ++t;
    }
    return 0;
}
