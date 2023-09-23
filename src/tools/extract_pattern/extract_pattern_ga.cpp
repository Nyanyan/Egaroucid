#include <iostream>
#include <vector>
#include<algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif
#include "./../../engine/board.hpp"


#define N_CELLS_IN_PATTERN 10
#define N_VARIATION_IN_PATTERN 59049
#define POW2_2N_CELLS_IN_PATTERN 1048576
#define N_GENES 2000
#define N_DATA 10000000
#define MIN_N_DISCS 24
#define MAX_N_DISCS 54
#define N_SHOW_RESULT 40
#define DUPLICATE_THRESHOLD 20

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
    uint64_t player, opponent;
    char p, policy, score;
    while (data.size() < N_DATA){
        if (fread(&player, 8, 1, fp) < 1)
            break;
        fread(&opponent, 8, 1, fp);
        fread(&p, 1, 1, fp);
        fread(&policy, 1, 1, fp);
        fread(&score, 1, 1, fp);
        if (pop_count_ull(player | opponent) >= MIN_N_DISCS && pop_count_ull(player | opponent) <= MAX_N_DISCS){
            int rotate_rnd = myrandrange(0, 8);
            switch(rotate_rnd){
                case 0:
                    break;
                case 1:
                    player = black_line_mirror(player);
                    opponent = black_line_mirror(opponent);
                    break;
                case 2:
                    player = white_line_mirror(player);
                    opponent = white_line_mirror(opponent);
                    break;
                case 3:
                    player = rotate_180(player);
                    opponent = rotate_180(opponent);
                    break;
                case 4:
                    player = horizontal_mirror(player);
                    opponent = horizontal_mirror(opponent);
                    break;
                case 5:
                    player = vertical_mirror(player);
                    opponent = vertical_mirror(opponent);
                    break;
                case 6:
                    player = vertical_mirror(player);
                    opponent = vertical_mirror(opponent);
                    player = black_line_mirror(player);
                    opponent = black_line_mirror(opponent);
                    break;
                case 7:
                    player = horizontal_mirror(player);
                    opponent = horizontal_mirror(opponent);
                    player = black_line_mirror(player);
                    opponent = black_line_mirror(opponent);
                    break;
                default:
                    break;

            }
            Datum datum;
            datum.board.player = player;
            datum.board.opponent = opponent;
            datum.score = score;
            data.emplace_back(datum);
        }
    }
}

int get_gathered_idx(uint64_t player, uint64_t opponent, uint64_t pattern){
    int gathered_idx = _pext_u64(player, pattern) << N_CELLS_IN_PATTERN;
    gathered_idx |= _pext_u64(opponent, pattern);
    return gathered_idx;
}

void scoring(Gene *gene){
    for (int i = 0; i < POW2_2N_CELLS_IN_PATTERN; ++i)
        duplication[i].clear();
    for (Datum &datum: data){
        uint64_t player = datum.board.player;
        uint64_t opponent = datum.board.opponent;
        int gathered_idx = get_gathered_idx(player, opponent, gene->pattern);
        duplication[gathered_idx].emplace_back(datum.score);

        /*
        
        player = black_line_mirror(player);
        opponent = black_line_mirror(opponent);
        gathered_idx = get_gathered_idx(player, opponent, gene->pattern);
        duplication[gathered_idx].emplace_back(datum.score);

        player = white_line_mirror(player);
        opponent = white_line_mirror(opponent);
        gathered_idx = get_gathered_idx(player, opponent, gene->pattern);
        duplication[gathered_idx].emplace_back(datum.score);

        player = black_line_mirror(player);
        opponent = black_line_mirror(opponent);
        gathered_idx = get_gathered_idx(player, opponent, gene->pattern);
        duplication[gathered_idx].emplace_back(datum.score);

        player = horizontal_mirror(player);
        opponent = horizontal_mirror(opponent);
        gathered_idx = get_gathered_idx(player, opponent, gene->pattern);
        duplication[gathered_idx].emplace_back(datum.score);

        player = white_line_mirror(player);
        opponent = white_line_mirror(opponent);
        gathered_idx = get_gathered_idx(player, opponent, gene->pattern);
        duplication[gathered_idx].emplace_back(datum.score);

        player = black_line_mirror(player);
        opponent = black_line_mirror(opponent);
        gathered_idx = get_gathered_idx(player, opponent, gene->pattern);
        duplication[gathered_idx].emplace_back(datum.score);

        player = white_line_mirror(player);
        opponent = white_line_mirror(opponent);
        gathered_idx = get_gathered_idx(player, opponent, gene->pattern);
        duplication[gathered_idx].emplace_back(datum.score);

        */

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

uint64_t generate_random_pattern(){
    uint64_t res = 0;
    while (pop_count_ull(res) < N_CELLS_IN_PATTERN)
        res |= 1ULL << myrandrange(0, 64);
    return res;
}

void init_ga(){
    for (int i = 0; i < N_GENES; ++i){
        if ((i & 0b11111) == 0)
            std::cerr << '\r' << ((double)i / N_GENES);
        genes[i].pattern = generate_random_pattern();
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
    int pattern_bits[2][N_CELLS_IN_PATTERN];
    int idx0 = 0, idx1 = 0;
    for (int i = 0; i < HW2; ++i){
        if (genes[parent0].pattern & (1ULL << i))
            pattern_bits[0][idx0++] = i;
        if (genes[parent1].pattern & (1ULL << i))
            pattern_bits[1][idx1++] = i;
    }
    bool dups[2][N_CELLS_IN_PATTERN];
    for (int i = 0; i < N_CELLS_IN_PATTERN; ++i){
        dups[0][i] = false;
        dups[1][i] = false;
        for (int j = 0; j < N_CELLS_IN_PATTERN; ++j){
            dups[0][i] |= pattern_bits[0][i] == pattern_bits[1][j];
            dups[1][i] |= pattern_bits[1][i] == pattern_bits[0][j];
        }
    }
    for (int i = 0; i < N_CELLS_IN_PATTERN; ++i){
        if (myrandom() < 0.5){
            int rand_idx = myrandrange(0, N_CELLS_IN_PATTERN);
            if (!dups[0][i] && !dups[1][rand_idx])
                std::swap(pattern_bits[0][i], pattern_bits[1][rand_idx]);
        }
    }
    new_genes[2].pattern = 0;
    new_genes[3].pattern = 0;
    for (int i = 0; i < N_CELLS_IN_PATTERN; ++i){
        new_genes[2].pattern |= 1ULL << pattern_bits[0][i];
        new_genes[3].pattern |= 1ULL << pattern_bits[1][i];
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

void delete_same_patterns(){
    std::unordered_set<uint64_t> patterns;
    for (int i = 0; i < N_GENES; ++i){
        if (patterns.find(genes[i].pattern) != patterns.end()){
            while (patterns.find(genes[i].pattern) != patterns.end())
                genes[i].pattern = generate_random_pattern();
            scoring(&genes[i]);
        }
        patterns.emplace(genes[i].pattern);
    }
}

void output_result(uint64_t t, uint64_t strt){
    std::sort(genes, genes + N_GENES, comp_gene);
    std::cerr << t << " " << tim() - strt << " " << genes[0].score << std::endl;
    std::ofstream ofs;
    ofs.open("log.txt", std::ios_base::app);
    ofs << t << " " << tim() - strt << std::endl;
    int n_cell_shown[HW2];
    for (int i = 0; i < HW2; ++i)
        n_cell_shown[i] = 0;
    int n_res_shown = 0;
    for (int i = 0; i < N_GENES && n_res_shown < N_SHOW_RESULT; ++i){
        int cell_duplicate = 0;
        for (int j = 0; j < HW2; ++j){
            if (1 & (genes[i].pattern >> j))
                cell_duplicate += n_cell_shown[j];
        }
        if (cell_duplicate <= DUPLICATE_THRESHOLD * n_res_shown){
            ofs << genes[i].score << " ";
            int n_shown = 0;
            for (int j = 0; j < HW2; ++j){
                if (1 & (genes[i].pattern >> j)){
                    ++n_cell_shown[j];
                    ++n_cell_shown[(7 - j / HW) * HW + (j % HW)]; // vertical
                    ++n_cell_shown[(j / HW) * HW + (7 - j % HW)]; // horizontal
                    ++n_cell_shown[HW2_M1 - j]; // 180
                    if (j / HW != j % HW){
                        ++n_cell_shown[(j / HW) + (j % HW) * HW]; // white
                        ++n_cell_shown[(7 - j / HW) + (j % HW) * HW]; // white + h
                        ++n_cell_shown[(j / HW) + (7 - j % HW) * HW]; // white + v
                        ++n_cell_shown[(7 - j / HW) + (7 - j % HW) * HW]; // white + 180
                    }
                    ofs << "COORD_" + ga_idx_to_coord(j);
                    if (++n_shown < N_CELLS_IN_PATTERN)
                        ofs << ", ";
                }
            }
            ofs << std::endl;
            ++n_res_shown;
        }
    }
    ofs.close();
}

int main(){
    std::cerr << "getting data..." << std::endl;
    get_data("./../../../train_data/board_data/records19/0.dat");
    std::cerr << "initializing..." << std::endl;
    init_ga();
    std::cerr << "start!" << std::endl;
    uint64_t strt = tim();
    uint64_t t = 0;
    while (true){
        ga();
        if ((t++ & 0b1111111) == 0){
            delete_same_patterns();
            output_result(t, strt);
        }
    }
    return 0;
}
