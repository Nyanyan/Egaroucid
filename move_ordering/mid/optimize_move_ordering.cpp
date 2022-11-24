#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>

using namespace std;

#define N_WEIGHT 6
#define N_MOVE_ORDERING_DEPTH 7
#define N_COMMON_DATA 4

constexpr int cell_weight[64] = {
    18,  4,  16, 12, 12, 16,  4, 18,
     4,  2,   6,  8,  8,  6,  2,  4,
    16,  6,  14, 10, 10, 14,  6, 16,
    12,  8,  10,  0,  0, 10,  8, 12,
    12,  8,  10,  0,  0, 10,  8, 12,
    16,  6,  14, 10, 10, 14,  6, 16,
     4,  2,   6,  8,  8,  6,  2,  4,
    18,  4,  16, 12, 12, 16,  4, 18
};

inline uint64_t tim(){
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

mt19937 raw_myrandom(tim());

inline double myrandom(){
    return (double)raw_myrandom() / mt19937::max();
}

inline int32_t myrandrange(int32_t s, int32_t e){
    return s +(int)((e - s) * myrandom());
}

inline uint32_t myrand_uint(){
    return (uint32_t)raw_myrandom();
}

double start_temp = 0.001;
double end_temp =   0.00001;

double temperature_x(double x){
    return pow(start_temp, 1 - x) * pow(end_temp, x);
    //return start_temp + (end_temp - start_temp) * x;
}

double calc_temperature(uint64_t strt, uint64_t now, uint64_t tl){
    return temperature_x((double)(now - strt) / tl);
}

double prob(double p_score, double n_score, uint64_t strt, uint64_t now, uint64_t tl){
    double dis = p_score - n_score;
    if (dis >= 0)
        return 1.0;
    return exp(dis / calc_temperature(strt, now, tl));
}

#define INF 10000000
int move_ordering_weights[N_WEIGHT];
constexpr int default_weights[N_WEIGHT] = {1, -24, -18, 15, -93, -28};
constexpr int max_weights[N_WEIGHT] = {INF, -1, -1, INF, -1, -1};
constexpr int min_weights[N_WEIGHT] = {1, -INF, -INF, 1, -INF, -INF};

struct Datum{
    int cell;
    int common_data[N_COMMON_DATA];
    int eval_data[N_MOVE_ORDERING_DEPTH];
};

struct Board_datum{
    int n_discs;
    int policy;
    vector<Datum> moves;
};

vector<Board_datum> data;

void init(){
    for (int j = 0; j < N_WEIGHT; ++j){
        move_ordering_weights[j] = default_weights[j];
    }
}

void import_data(){
    ifstream ifs("data/records15_with_eval/0000000_small.txt");
    if (!ifs) {
        cerr << "can't open" << endl;
        return;
    }
    string line;
    int i, idx;
    int t = 0;
    while (getline(ifs, line)){
        //cerr << line << endl;
        Board_datum board_datum;
        board_datum.n_discs = 0;
        for (i = 0; i < 64; ++i)
            board_datum.n_discs += (int)(line[i] != '.');
        board_datum.policy = line[64] - '!';
        idx = 67;
        bool flag = false;
        while (idx < line.size()){
            Datum datum;
            datum.cell = line[idx++] - '!';
            flag |= board_datum.policy == datum.cell;
            datum.common_data[0] = cell_weight[datum.cell];
            for (i = 0; i < N_COMMON_DATA - 1; ++i)
                datum.common_data[i + 1] = line[idx++] - '!';
            ++idx;
            for (i = 0; i < N_MOVE_ORDERING_DEPTH; ++i)
                datum.eval_data[i] = (line[idx++] - '!') * 2 - 64;
            board_datum.moves.emplace_back(datum);
        }
        //cerr << flag << endl;
        data.emplace_back(board_datum);
        ++t;
    }
    cerr << "\r" << t << " data" << endl;
}

int calc_move_score(Datum m, int depth){ // n_discs: former
    int res = 0;
    for (int i = 0; i < N_COMMON_DATA; ++i){
        res += move_ordering_weights[i] * m.common_data[i];
    }
    if (depth >= 0)
        res += (move_ordering_weights[N_COMMON_DATA] + depth * move_ordering_weights[N_COMMON_DATA + 1]) * m.eval_data[depth];
    return res;
}

bool cmp_score(pair<int, int>& a, pair<int, int>& b){
    return a.first > b.first;
}

double scoring_part(double *mean_rank){
    double score = 0.0;
    int t = 0;
    *mean_rank = 0.0;
    for (Board_datum &datum: data){
        for (int depth = -1; depth < N_MOVE_ORDERING_DEPTH; ++depth){
            ++t;
            int max_score = -1000000000;
            int min_score = 1000000000;
            int move_score;
            int policy_score = 0;
            vector<int> scores;
            for (Datum &m: datum.moves){
                move_score = calc_move_score(m, depth);
                max_score = max(max_score, move_score);
                min_score = min(min_score, move_score);
                if (m.cell == datum.policy)
                    policy_score = move_score;
                scores.emplace_back(move_score);
            }
            //cerr << max_score << " " << policy_score << " " << min_score << endl;
            score += 0.01 * ((double)(max_score - policy_score) / max(1.0, (double)(max_score - min_score)));
            int upper_count = 0;
            for (int s: scores){
                if (s > policy_score)
                    ++upper_count;
            }
            score += 1.0 * ((double)upper_count / (double)(scores.size()));
            *mean_rank += upper_count;
        }
    }
    *mean_rank /= t;
    return score / t;
}

void anneal(uint64_t tl){
    double mean_rank;
    double score = scoring_part(&mean_rank);
    cerr << score << " " << mean_rank << endl;
    uint64_t strt = tim();
    int idx, delta;
    double n_score;
    while (tim() - strt < tl){
        idx = myrandrange(0, N_WEIGHT);
        delta = myrandrange(-2, 3);
        if (delta == 0)
            continue;
        move_ordering_weights[idx] += delta;
        if (max_weights[idx] < move_ordering_weights[idx]){
            move_ordering_weights[idx] -= delta;
            continue;
        }
        if (min_weights[idx] > move_ordering_weights[idx]){
            move_ordering_weights[idx] -= delta;
            continue;
        }
        n_score = scoring_part(&mean_rank);
        if (prob(score, n_score, strt, tim(), tl) >= myrandom()){
            score = n_score;
            cerr << "\r" << score << " " << mean_rank << "                     ";
        } else{
            move_ordering_weights[idx] -= delta;
        }
    }
    cerr << "\r" << score << " " << mean_rank << "                     " << endl;
}

int main(){
    init();
    import_data();
    uint64_t tl = 120 * 1000;
    anneal(tl);
    for (int k = 0; k < N_WEIGHT; ++k){
        cerr << move_ordering_weights[k] << " ";
    }
    cerr << endl;
}