#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <math.h>
#include <unordered_set>
#include <iomanip>
#include "./../../engine/board.hpp"

using namespace std;

unsigned long long hour = 0;
unsigned long long minute = 3;
unsigned long long second = 0;

#define N_HASH1 8
#define N_HASH2 65536
#define N_DATA 50000000

#define HASH_MAX_SIZE 134217728

int HASH_SIZE, HASH_MASK, HASH_N_BIT;

double start_temp = 0.00001;
double end_temp =   0.00000000000001;

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


uint32_t hash_rand[N_HASH1][N_HASH2];
uint16_t data[N_DATA][N_HASH1];
int data_hash[N_DATA];
vector<int> hash_idxes[N_HASH1][N_HASH2];
int appear_hash[HASH_MAX_SIZE];
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
            if (fread(data[t], 2, N_HASH1, fp) < 1)
                break;
            for (i = 0; i < N_HASH1; ++i)
                hash_idxes[i][data[t][i]].emplace_back(t);
            ++t;
        }
    }
    n_data = t;
    cerr << "\r" << n_data << " data" << endl;
}

void output_param(uint64_t second, double score){
    ofstream fout;
    fout.open("learned_data/hash" + to_string(HASH_N_BIT) + "_" + to_string(second) + "sec_" + to_string(score) + ".eghs", ios::out|ios::binary|ios::trunc);
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

double calc_final_score(){
    uint64_t res = 0;
    for (int i = 0; i < HASH_SIZE; ++i){
        if (appear_hash[i])
            res += (appear_hash[i] - 1) * (appear_hash[i] - 1);
    }
    return (double)res / hash_variety;
}

/*
double calc_final_score(){
    return 1.0 - (double)hash_variety / HASH_SIZE;
}
*/

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
    return calc_final_score();
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
    return calc_final_score();
    //return (double)hash_variety / n_data;
}

double anneal(uint64_t tl){
    double f_score = calc_score(), n_score;
    cerr << setprecision(15) << f_score << endl;
    cerr << setprecision(15) << f_score << "                         ";
    int hi, hj, swap_idx;
    uint64_t strt = tim();
    while(tim() - strt < tl){
        hi = myrandrange(0, N_HASH1);
        hj = myrandrange(0, N_HASH2);
        if (hash_idxes[hi][hj].size()){
            swap_idx = myrandrange(0, HASH_N_BIT);
            hash_rand[hi][hj] ^= 1 << swap_idx;
            n_score = calc_score_diff(hi, hj);
            if (prob(f_score, n_score, strt, tim(), tl) >= myrandom()){
            //if (f_score >= n_score){
                f_score = n_score;
                cerr << "\r" << ((tim() - strt) * 1000 / tl) << " " << setprecision(15) << f_score << "                         ";
            } else{
                hash_rand[hi][hj] ^= 1 << swap_idx;
                n_score = calc_score_diff(hi, hj);
                if (f_score != n_score)
                    cerr << "ERROR" << setprecision(15) << f_score << " " << setprecision(15) << n_score << endl;
            }
        }
    }
    cerr << "\r" << setprecision(15) << f_score << "                       " << endl;
    return f_score;
}

int main(int argc, char *argv[]){
    HASH_N_BIT = atoi(argv[1]);
    HASH_SIZE = 1 << HASH_N_BIT;
    HASH_MASK = HASH_SIZE - 1;
    hour = atoi(argv[2]);
    minute = atoi(argv[3]);
    second = atoi(argv[4]);
    int i, j;

    minute += hour * 60;
    second += minute * 60;

    cerr << second << " sec" << endl;
    cerr << "level " << HASH_N_BIT << " " << "size " << HASH_SIZE << endl;

    board_init();
    initialize_param();
    //input_param(argv);
    input_test_data(argc, argv);

    double score = anneal(second * 1000);

    output_param(second, score);

    return 0;
}