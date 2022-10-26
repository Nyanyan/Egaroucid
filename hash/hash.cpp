#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <math.h>
#include <unordered_set>
#include "new_util/board.hpp"

using namespace std;

unsigned long long hour = 0;
unsigned long long minute = 3;
unsigned long long second = 0;

#define N_HASH1 8
#define N_HASH2 65536
#define N_DATA (26461501 + 100)

#define HASH_SIZE 16777216
#define HASH_MASK 16777215
#define HASH_N_BIT 23

double start_temp = 0.00001;
double end_temp =   0.00000001;

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


uint32_t hash_rand[N_HASH1][N_HASH2];
uint16_t data[N_DATA][N_HASH1];
int data_hash[N_DATA];
vector<int> hash_idxes[N_HASH1][N_HASH2];
int appear_hash[HASH_SIZE];
int hash_variety = 0;
int n_data;

inline int pop_count_uint(uint32_t x){
    x = (x & 0x55555555) + ((x & 0xAAAAAAAA) >> 1);
    x = (x & 0x33333333) + ((x & 0xCCCCCCCC) >> 2);
    return (x & 0x0F0F0F0F) + ((x & 0xF0F0F0F0) >> 4);
}

void init(){
    for (int i = 0; i < HASH_SIZE; ++i)
        appear_hash[i] = 0;
}

int create_hash_rand(){
    return myrand_uint_rev() & HASH_MASK;
}

void initialize_param(){
    int i, j;
    for (i = 0; i < N_HASH1; ++i){
        for (j = 0; j < N_HASH2; ++j){
            hash_rand[i][j] = create_hash_rand();
        }
    }
}

void input_param(char *argv[]){
    FILE* fp;
    if (fopen_s(&fp, argv[4], "rb") != 0) {
        cerr << "can't open " << argv[4] << endl;
        return;
    }
    for (int i = 0; i < N_HASH1; ++i)
        fread(hash_rand[i], 4, N_HASH2, fp);
}

void input_test_data(int argc, char *argv[]){
    int t = 0, i;
    FILE* fp;
    for (int file_idx = 5; file_idx < argc; ++file_idx){
        cerr << argv[file_idx] << endl;
        if (fopen_s(&fp, argv[file_idx], "rb") != 0) {
            cerr << "can't open " << argv[file_idx] << endl;
            continue;
        }
        while (t < N_DATA - 10){
            if ((t & 0b1111111111111111) == 0b1111111111111111)
                cerr << '\r' << t;
            if (fread(data[t], 2, 8, fp) < 1)
                break;
            for (i = 0; i < N_HASH1; ++i)
                hash_idxes[i][data[t][i]].emplace_back(t);
            ++t;
        }
    }
    n_data = t;
    cerr << "\r" << n_data << " data" << endl;
}

void output_param(){
    ofstream fout;
    fout.open("learned_data/hash.eghs", ios::out|ios::binary|ios::trunc);
    if (!fout){
        cerr << "can't open hash.eghs" << endl;
        return;
    }
    int i, j;
    for (i = 0; i < N_HASH1; ++i){
        for (j = 0; j < N_HASH2; ++j){
            fout.write((char*)&hash_rand[i][j], 4);
        }
    }
    fout.close();
}

int calc_hash(int idx){
    int res = 0;
    for (int i = 0; i < N_HASH1; ++i)
        res ^= hash_rand[i][data[idx][i]];
    return res;
}

double calc_score(){
    unordered_set<int> hashes;
    int i, j, hash;
    for (i = 0; i < n_data; ++i){
        hash = calc_hash(i);
        data_hash[i] = hash;
        hashes.emplace(hash);
        ++appear_hash[hash];
    }
    hash_variety = (int)hashes.size();
    return (double)hashes.size() / n_data;
}

double calc_score_diff(int hi, int hj){
    int hash;
    for (int i: hash_idxes[hi][hj]){
        --appear_hash[data_hash[i]];
        if (appear_hash[data_hash[i]] == 0)
            --hash_variety;
        hash = calc_hash(i);
        data_hash[i] = hash;
        ++appear_hash[hash];
        if (appear_hash[hash] == 1)
            ++hash_variety;
    }
    return (double)hash_variety / n_data;
}

void anneal(uint64_t tl){
    double f_score = calc_score(), n_score;
    cerr << f_score << "                         ";
    int hi, hj, swap_idx;
    uint64_t strt = tim();
    while(tim() - strt < tl){
        hi = myrandrange(0, N_HASH1);
        hj = myrandrange(0, N_HASH2);
        if (hash_idxes[hi][hj].size()){
            swap_idx = myrandrange(0, HASH_N_BIT);
            hash_rand[hi][hj] ^= 1 << swap_idx;
            n_score = calc_score_diff(hi, hj);
            //if (prob(f_score, n_score, strt, tim(), tl) >= myrandom()){
            if (f_score <= n_score){
                f_score = n_score;
                cerr << "\r" << ((tim() - strt) * 1000 / tl) << " " << f_score << "                         ";
            } else{
                hash_rand[hi][hj] ^= 1 << swap_idx;
                n_score = calc_score_diff(hi, hj);
                if (f_score != n_score)
                    cerr << f_score << n_score << endl;
            }
        }
    }
    cerr << "\r" << f_score << "                       " << endl;
}

int main(int argc, char *argv[]){
    hour = atoi(argv[1]);
    minute = atoi(argv[2]);
    second = atoi(argv[3]);
    int i, j;

    minute += hour * 60;
    second += minute * 60;

    cerr << second << " sec" << endl;

    board_init();
    //initialize_param();
    input_param(argv);
    input_test_data(argc, argv);

    anneal(second * 1000);

    output_param();

    return 0;
}