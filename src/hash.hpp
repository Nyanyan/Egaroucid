/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include "common.hpp"

using namespace std;

#define N_HASH_LEVEL 30

uint32_t hash_rand_player[4][65536];
uint32_t hash_rand_opponent[4][65536];

constexpr size_t hash_sizes[N_HASH_LEVEL] = {
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
    4194304,
    8388608,
    16777216,
    33554432,
    67108864,
    134217728,
    268435456,
    536870912
};

void hash_init_rand(int hash_level){
    int i, j;
    for (i = 0; i < 4; ++i){
        for (j = 0; j < 65536; ++j){
            hash_rand_player[i][j] = 0;
            while (pop_count_uint(hash_rand_player[i][j]) < 4)
                hash_rand_player[i][j] = myrand_uint_rev() & (hash_sizes[hash_level] - 1);
            hash_rand_opponent[i][j] = 0;
            while (pop_count_uint(hash_rand_opponent[i][j]) < 4)
                hash_rand_opponent[i][j] = myrand_uint_rev() & (hash_sizes[hash_level] - 1);
        }
    }
    cerr << "hash initialized randomly" << endl;
}

bool hash_init(int hash_level){
    FILE* fp;
    if (fopen_s(&fp, ("resources/hash" + to_string(hash_level) + ".eghs").c_str(), "rb") != 0) {
        cerr << "can't open hash" + to_string(hash_level) + ".eghs" << endl;
        //board_init_rand();
        return false;
    } else{
        for (int i = 0; i < 4; ++i){
            if (fread(hash_rand_player[i], 4, 65536, fp) < 65536){
                cerr << "hash" + to_string(hash_level) + ".eghs broken" << endl;
                return false;
            }
        }
        for (int i = 0; i < 4; ++i){
            if (fread(hash_rand_opponent[i], 4, 65536, fp) < 65536){
                cerr << "hash" + to_string(hash_level) + ".eghs broken" << endl;
                return false;
            }
        }
    }
    cerr << "hash initialized" << endl;
    return true;
}