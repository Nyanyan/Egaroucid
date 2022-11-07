#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>

using namespace std;

#define N_WEIGHT 5
#define N_MOVE_ORDERING_DEPTH 12
#define N_MOVE_ORDERING_PHASES 6
#define N_MOVE_ORDERING_PHASE_DISCS 10

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

double start_temp = 0.1;
double end_temp =   0.0001;

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


int move_ordering_weights[N_MOVE_ORDERING_PHASES][N_MOVE_ORDERING_DEPTH][N_WEIGHT];

struct Datum{
    int cell;
    int common_data[N_WEIGHT - 1];
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
        for (int i = 0; i < N_MOVE_ORDERING_DEPTH; ++i){
            for (int j = 0; j < N_WEIGHT; ++j){
                move_ordering_weights[k][i][j] = 0;
            }
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
            for (i = 0; i < N_WEIGHT - 1; ++i)
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
    for (int i = 0; i < N_WEIGHT - 1 - 2; ++i)
        res += move_ordering_weights[phase][depth][i] * m.common_data[i];
    res += move_ordering_weights[phase][depth][N_WEIGHT - 1] * m.eval_data[depth];
    return res;
}

bool cmp_score(pair<int, int>& a, pair<int, int>& b){
    return a.first > b.first;
}

double scoring_part(int phase, int depth){
    double score = 0.0;
    vector<pair<int, int>> scores;
    int t = 0;
    for (Board_datum &datum: data){
        if (get_phase(datum.n_discs) == phase){
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
    return score / t;
}

void anneal(int phase, int depth, uint64_t tl){
    cerr << phase << " " << depth << endl;
    double score = scoring_part(phase, depth);
    cerr << score << endl;
    uint64_t strt = tim();
    int idx, delta;
    double n_score;
    while (tim() - strt < tl){
        idx = myrandrange(0, N_WEIGHT);
        if (idx == 2 || idx == 3)
            continue;
        delta = myrandrange(-4, 5);
        move_ordering_weights[phase][depth][idx] += delta;
        n_score = scoring_part(phase, depth);
        if (prob(score, n_score, strt, tim(), tl) >= myrandom()){
            score = n_score;
            cerr << "\r" << score << "                     ";
        } else{
            move_ordering_weights[phase][depth][idx] -= delta;
        }
    }
    cerr << "\r" << score << "                     " << endl;
}

int main(){
    init();
    import_data();
    uint64_t tl = 1 * 1000;
    for (int i = 0; i < N_MOVE_ORDERING_PHASES; ++i){
        for (int j = 0; j < N_MOVE_ORDERING_DEPTH; ++j){
            anneal(i, j, tl);
        }
    }
    for (int i = 0; i < N_MOVE_ORDERING_PHASES; ++i){
        cout << "{";
        for (int j = 0; j < N_MOVE_ORDERING_DEPTH; ++j){
            cout << "{";
            for (int k = 0; k < N_WEIGHT; ++k){
                if (k == 2 || k == 3)
                    continue;
                cout << move_ordering_weights[i][j][k] << ", ";
            }
            cout << "}, ";
        }
        cout << "}, " << endl;
    }
}