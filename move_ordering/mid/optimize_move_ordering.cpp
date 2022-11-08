#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>

using namespace std;

#define N_WEIGHT 6
#define N_MOVE_ORDERING_DEPTH 14
#define N_MOVE_ORDERING_PHASES 6
#define N_MOVE_ORDERING_PHASE_DISCS 10
#define N_COMMON_DATA 4

inline int get_phase(int n_discs){
    return (n_discs - 4) / N_MOVE_ORDERING_PHASE_DISCS;
}

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

double start_temp = 0.04;
double end_temp =   0.00001;

double temperature_x(double x){
    return pow(start_temp, 1 - x) * pow(end_temp, x);
    //return start_temp + (end_temp - start_temp) * x;
}

double calc_temperature(uint64_t strt, uint64_t now, uint64_t tl){
    return temperature_x((double)(now - strt) / tl);
}

double prob(double p_score, double n_score, uint64_t strt, uint64_t now, uint64_t tl){
    double dis = n_score - p_score;
    if (dis >= 0)
        return 1.0;
    return exp(dis / calc_temperature(strt, now, tl));
}

#define INF 10000000
int move_ordering_weights[N_MOVE_ORDERING_PHASES][N_WEIGHT];
constexpr int default_weights[N_WEIGHT] = {-20, 10, -20, 10, -100, -10};
constexpr int max_weights[N_WEIGHT] = {0, INF, 0, INF, 0, 0};
constexpr int min_weights[N_WEIGHT] = {-INF, 0, -INF, 0, -INF, -INF};

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
    for (int k = 0; k < N_MOVE_ORDERING_PHASES; ++k){
        for (int j = 0; j < N_WEIGHT; ++j){
            move_ordering_weights[k][j] = default_weights[j];
        }
    }
}

void import_data(){
    ifstream ifs("data/records16_with_evals/0000000.txt");
    if (!ifs) {
        cerr << "can't open" << endl;
        return;
    }
    string line;
    int i, idx;
    int t = 0;
    while (getline(ifs, line)){
        Board_datum board_datum;
        board_datum.n_discs = 0;
        for (i = 0; i < 64; ++i)
            board_datum.n_discs += (int)(line[i] != '.');
        board_datum.policy = line[64] - '!';
        idx = 67;
        while (idx < line.size() - 4){
            Datum datum;
            datum.cell = line[idx++] - '!';
            for (i = 0; i < N_COMMON_DATA; ++i)
                datum.common_data[i] = line[idx++] - '!';
            ++idx;
            for (i = 0; i < N_MOVE_ORDERING_DEPTH; ++i)
                datum.eval_data[i] = (line[idx++] - '!') * 2 - 64;
            board_datum.moves.emplace_back(datum);
            idx += 2;
        }
        data.emplace_back(board_datum);
        ++t;
    }
    cerr << "\r" << t << " data" << endl;
}

int calc_move_score(Datum m, int phase, int depth){ // n_discs: former
    int res = 0;
    for (int i = 0; i < N_COMMON_DATA; ++i){
        if (i == 1)
            continue;
        res += move_ordering_weights[phase][i] * m.common_data[i];
    }
    res += (move_ordering_weights[phase][N_COMMON_DATA] + depth * move_ordering_weights[phase][N_COMMON_DATA + 1]) * m.eval_data[depth];
    return res;
}

bool cmp_score(pair<int, int>& a, pair<int, int>& b){
    return a.first > b.first;
}

double scoring_part(int phase){
    double score = 0.0;
    vector<pair<int, int>> scores;
    int t = 0;
    for (Board_datum &datum: data){
        if (get_phase(datum.n_discs) == phase){
            for (int depth = 0; depth < N_MOVE_ORDERING_DEPTH; ++depth){
                ++t;
                scores.clear();
                for (Datum &m: datum.moves){
                    scores.emplace_back(make_pair(calc_move_score(m, phase, depth), m.cell));
                }
                sort(scores.begin(), scores.end(), cmp_score);
                for (int i = 0; i < (int)datum.moves.size(); ++i){
                    if (datum.policy == scores[i].second){
                        score += 1.0 - (1.0 + i) / datum.moves.size();
                        break;
                    }
                }
            }
        }
    }
    return score / t;
}

void anneal(int phase, uint64_t tl){
    cerr << phase << " " << endl;
    double score = scoring_part(phase);
    cerr << score << endl;
    uint64_t strt = tim();
    int idx, delta;
    double n_score;
    while (tim() - strt < tl){
        idx = myrandrange(0, N_WEIGHT);
        //if (idx == 2 || idx == 3)
        //    continue;
        if (idx == 1)
            continue;
        delta = myrandrange(-2, 3);
        if (delta == 0)
            continue;
        move_ordering_weights[phase][idx] += delta;
        if (max_weights[idx] < move_ordering_weights[phase][idx]){
            move_ordering_weights[phase][idx] -= delta;
            continue;
        }
        if (min_weights[idx] > move_ordering_weights[phase][idx]){
            move_ordering_weights[phase][idx] -= delta;
            continue;
        }
        n_score = scoring_part(phase);
        if (prob(score, n_score, strt, tim(), tl) >= myrandom()){
            score = n_score;
            cerr << "\r" << score << "                     ";
        } else{
            move_ordering_weights[phase][idx] -= delta;
        }
    }
    cerr << "\r" << score << "                     " << endl;
}

int main(){
    init();
    import_data();
    uint64_t tl = 15 * 1000;
    for (int i = 0; i < N_MOVE_ORDERING_PHASES; ++i){
        //for (int j = 0; j < N_MOVE_ORDERING_DEPTH; ++j){
        anneal(i, tl);
        for (int k = 0; k < N_WEIGHT; ++k){
            //if (k == 2 || k == 3)
            //    continue;
            if (k == 1)
                continue;
            cerr << move_ordering_weights[i][k] << " ";
        }
        cerr << endl;
        //}
    }
    for (int i = 0; i < N_MOVE_ORDERING_PHASES; ++i){
        cout << "{";
        //for (int j = 0; j < N_MOVE_ORDERING_DEPTH; ++j){
        //    cout << "{";
        for (int k = 0; k < N_WEIGHT; ++k){
            //if (k == 2 || k == 3)
            //    continue;
            if (k == 1)
                continue;
            cout << move_ordering_weights[i][k] << ", ";
        }
        //cout << "}, ";
        //}
        cout << "}, " << endl;
    }
}