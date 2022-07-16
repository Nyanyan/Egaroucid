#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <omp.h> 
#include "new_util/board.hpp"

using namespace std;

#define MAX_N_LINE 6
#define POPULATION 4096
#define N_DATA 1000000
//#define POPULATION 1024
//#define N_DATA 100000
#define SCORING_SIZE_THRESHOLD 5
#define HASH_SIZE 1048576
#define HASH_MASK 1048575

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
int n_possible_state;

struct Gene{
    uint64_t cell;
    uint64_t line[MAX_N_LINE];
    double score;

    uint64_t hash(){
        uint64_t hash = 
            hash_rand_player[0][0b1111111111111111 & cell] ^ 
            hash_rand_player[1][0b1111111111111111 & (cell >> 16)] ^ 
            hash_rand_player[2][0b1111111111111111 & (cell >> 32)] ^ 
            hash_rand_player[3][0b1111111111111111 & (cell >> 48)];
        for (int i = 0; i < n_use_line; ++i){
            hash ^= 
                hash_rand_opponent[0][0b1111111111111111 & line[i]] ^ 
                hash_rand_opponent[1][0b1111111111111111 & (line[i] >> 16)] ^ 
                hash_rand_opponent[2][0b1111111111111111 & (line[i] >> 32)] ^ 
                hash_rand_opponent[3][0b1111111111111111 & (line[i] >> 48)];
        }
        return hash;
    }
};

vector<Gene> genes;

struct Datum{
    Board board;
    double score;
};

Datum data[N_DATA];

vector<double> scoring_hash_table[HASH_SIZE];

struct Feature{
    uint64_t cell_player;
    uint64_t cell_opponent;
    bool line_player[MAX_N_LINE];
    bool line_opponent[MAX_N_LINE];
    bool line_empty[MAX_N_LINE];

    bool operator==(const Feature& other) const {
        bool res = false;
        if (cell_player == other.cell_player && cell_opponent == other.cell_opponent){
            res = true;
            for (int i = 0; i < n_use_line; ++i){
                res &= (line_player[i] == other.line_player[i]);
                res &= (line_opponent[i] == other.line_opponent[i]);
                res &= (line_empty[i] == other.line_empty[i]);
            }
        }
        return res;
    };

    bool operator<(const Feature& other) {
        if (cell_player < other.cell_player)
            return true;
        if (cell_opponent < other.cell_opponent)
            return true;
        for (int i = 0; i < n_use_line; ++i){
            if (!line_player[i] && other.line_player[i])
                return true;
            if (!line_opponent[i] && other.line_opponent[i])
                return true;
            if (!line_empty[i] && other.line_empty[i])
                return true;
        }
        return false;
    };

    uint64_t hash(){
        uint64_t hash = 
            hash_rand_player[0][0b1111111111111111 & cell_player] ^ 
            hash_rand_player[1][0b1111111111111111 & (cell_player >> 16)] ^ 
            hash_rand_player[2][0b1111111111111111 & (cell_player >> 32)] ^ 
            hash_rand_player[3][0b1111111111111111 & (cell_player >> 48)] ^ 
            hash_rand_opponent[0][0b1111111111111111 & cell_opponent] ^ 
            hash_rand_opponent[1][0b1111111111111111 & (cell_opponent >> 16)] ^ 
            hash_rand_opponent[2][0b1111111111111111 & (cell_opponent >> 32)] ^ 
            hash_rand_opponent[3][0b1111111111111111 & (cell_opponent >> 48)];
        int shift = 0;
        for (int i = 0; i < n_use_line; ++i){
            hash ^= (uint64_t)line_player[i] << (shift++);
            hash ^= (uint64_t)line_opponent[i] << (shift++);
            hash ^= (uint64_t)line_empty[i] << (shift++);
        }
        return hash;
    }
};

struct Feature_hash {
    size_t operator()(const Feature& x) const noexcept {
        size_t hash = 
            hash_rand_player[0][0b1111111111111111 & x.cell_player] ^ 
            hash_rand_player[1][0b1111111111111111 & (x.cell_player >> 16)] ^ 
            hash_rand_player[2][0b1111111111111111 & (x.cell_player >> 32)] ^ 
            hash_rand_player[3][0b1111111111111111 & (x.cell_player >> 48)] ^ 
            hash_rand_opponent[0][0b1111111111111111 & x.cell_opponent] ^ 
            hash_rand_opponent[1][0b1111111111111111 & (x.cell_opponent >> 16)] ^ 
            hash_rand_opponent[2][0b1111111111111111 & (x.cell_opponent >> 32)] ^ 
            hash_rand_opponent[3][0b1111111111111111 & (x.cell_opponent >> 48)];
        return hash;
    };
};

void init(){
    int i, j, k, line, offset;
    bool overlap;
    int genes_strt = (int)genes.size();
    for (i = genes_strt; i < POPULATION; ++i){
        cerr << "\r" << i;
        genes.emplace_back(Gene{});
        genes[i].score = INF;
        genes[i].cell = 0ULL;
        while (pop_count_ull(genes[i].cell) != n_use_cell){
            genes[i].cell = myrand_ull();
            //genes[i].cell &= 0x00000000FFFFFFFFULL;
        }
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
    cerr << endl;
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

double calc_sd(vector<double> &lst){
    double avg = 0.0;
    for (const double &elem: lst)
        avg += elem / lst.size();
    double distribution = 0.0;
    for (const double &elem: lst)
        distribution += (avg - elem) * (avg - elem) / lst.size();
    return sqrt(distribution);
}

void scoring(Gene *gene){
    int i, j;
    Feature feature;
    for (i = 0; i < HASH_SIZE; ++i){
        scoring_hash_table[i].clear();
        scoring_hash_table[i].shrink_to_fit();
    }
    for (i = 0; i < N_DATA; ++i){
        feature.cell_player = data[i].board.player & gene->cell;
        feature.cell_opponent = data[i].board.opponent & gene->cell;
        for (j = 0; j < n_use_line; ++j){
            feature.line_player[j] = (data[i].board.player & ~gene->cell & gene->line[j]) > 0;
            feature.line_opponent[j] = (data[i].board.opponent & ~gene->cell & gene->line[j]) > 0;
            feature.line_empty[j] = (~(data[i].board.player | data[i].board.opponent) & ~gene->cell & gene->line[j]) > 0;
        }
        scoring_hash_table[feature.hash() & HASH_MASK].emplace_back(data[i].score);
    }
    int n_appear_state = 0;
    int n_all_appear_state = 0;
    double avg_sd = 0.0;
    for (i = 0; i < HASH_SIZE; ++i){
        if (scoring_hash_table[i].size() >= SCORING_SIZE_THRESHOLD){
            ++n_appear_state;
            avg_sd += 1.0 - calc_sd(scoring_hash_table[i]);
        }
        if (scoring_hash_table[i].size())
            ++n_all_appear_state;
    }
    avg_sd /= n_appear_state;
    double appear_score = min(0.5, (double)n_all_appear_state / n_possible_state) * 2.0;
    gene->score = appear_score * avg_sd;
    //gene->score = avg_sd;
    //cerr << n_all_appear_state << " " << appear_score << " " << avg_sd << " " << gene->score << endl;
}

pair<int, double> scoring_all(){
    int i;
    double max_score = -INF;
    int max_idx = -1;
    //#pragma omp parallel for
    for (i = 0; i < POPULATION; ++i){
        scoring(&genes[i]);
        cerr << "\r" << i << " " << genes[i].score;
        if (max_score < genes[i].score){
            max_score = genes[i].score;
            max_idx = i;
        }
    }
    cerr << endl;
    return make_pair(max_idx, max_score);
}

void mate(Gene *parent0, Gene *parent1, Gene *child0, Gene *child1){
    uint64_t cell0 = parent0->cell;
    uint64_t cell1 = parent1->cell;
    int i;
    int threshold = myrandrange(0, n_use_cell);
    child0->cell = 0;
    child1->cell = 0;
    for (i = 0; i < n_use_cell; ++i){
        if (i < threshold){
            child0->cell |= cell0 & (-cell0);
            child1->cell |= cell1 & (-cell1);
        } else{
            child0->cell |= cell1 & (-cell1);
            child1->cell |= cell0 & (-cell0);
        }
        cell0 &= (cell0 - 1);
        cell1 &= (cell1 - 1);
    }
    //cerr << pop_count_ull(child0->cell) << " " << pop_count_ull(child1->cell) << endl;
    threshold = myrandrange(0, n_use_line);
    for (i = 0; i < n_use_line; ++i){
        if (i < threshold){
            child0->line[i] = parent0->line[i];
            child1->line[i] = parent1->line[i];
        } else{
            child0->line[i] = parent1->line[i];
            child1->line[i] = parent0->line[i];
        }
    }
}

bool cmp_gene(Gene &x, Gene &y){
    return x.score > y.score;
}

bool ga(pair<int, double> &best){
    int parent0_idx = myrandrange(0, POPULATION);
    int parent1_idx = myrandrange(0, POPULATION);
    while (parent0_idx == parent1_idx)
        parent1_idx = myrandrange(0, POPULATION);
    Gene child0, child1;
    mate(&genes[parent0_idx], &genes[parent1_idx], &child0, &child1);
    scoring(&child0);
    scoring(&child1);
    vector<Gene> lst = {genes[parent0_idx], genes[parent1_idx], child0, child1};
    sort(lst.begin(), lst.end(), cmp_gene);
    //cerr << lst[0].score << " " << lst[1].score << endl;
    genes[parent0_idx] = lst[0];
    genes[parent1_idx] = lst[1];
    bool res = lst[0].score > best.second;
    if (res)
        best = make_pair(parent0_idx, lst[0].score);
    return res;
}

vector<Gene> best_n_genes(int n){
    sort(genes.begin(), genes.end(), cmp_gene);
    unordered_set<uint64_t> hashes;
    uint64_t hash;
    uint64_t t = 0;
    vector<Gene> res;
    for (int i = 0; i < POPULATION && t < n; ++i){
        hash = genes[i].hash();
        if (hashes.find(hash) == hashes.end()){
            res.emplace_back(genes[i]);
            hashes.emplace(hash);
            ++t;
        }
    }
    return res;
}

int main(int argc, char *argv[]){
    board_init();
    n_use_cell = atoi(argv[1]);
    n_use_line = atoi(argv[2]);
    uint64_t tl = (uint64_t)atoi(argv[3]) * 60 * 1000; // minutes
    n_possible_state = pow(3, n_use_cell) * pow(8, n_use_line);
    cerr << "cell: " << n_use_cell << " line: " << n_use_line << " population: " << POPULATION << " possible_state: " << n_possible_state << " tl: " << argv[3] << " minutes" << endl;

    cerr << "initializing" << endl;
    init();
    cerr << "initialized" << endl;

    input_data("records15", 0, 150);

    cerr << "scoring..." << endl;
    pair<int, double> idx_score = scoring_all();
    cerr << "done" << endl;
    //cerr << idx_score.first << " " << idx_score.second << endl;
    //cerr << genes[idx_score.first].cell << " " << genes[idx_score.first].line[0] << " " << genes[idx_score.first].line[1] << " " << genes[idx_score.first].line[2] << " " << genes[idx_score.first].line[3] << endl;

    uint64_t strt = tim();
    uint64_t last = 0ULL;
    uint64_t t = 0;
    uint64_t updated = 0;
    while (tim() - strt < tl){
        ++t;
        if (t - updated > 10000){
            cerr << endl << "updating genes" << endl;
            genes = best_n_genes(POPULATION / 100);
            uint64_t init_strt = tim();
            init();
            idx_score = scoring_all();
            strt += tim() - init_strt;
            updated = t;
        }
        if (ga(idx_score)){
            updated = t;
            cerr << endl << "best score" << endl;
        }
        if (tim() - strt - last > 1000){
            cerr << '\r' << t << " " << (double)(tim() - strt) / tl << "       " << idx_score.second << "                    ";
            last = tim() - strt;
        }
    }
    cerr << endl;
    cerr << t << " times" << endl;

    vector<Gene> best_genes = best_n_genes(100);
    for (Gene &gene: best_genes){
        cout << gene.score << " ";
        cout << gene.cell << " ";
        for (int j = 0; j < n_use_line; ++j)
            cout << gene.line[j] << " ";
        cout << endl;
    }

    return 0;
}