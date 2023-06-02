/*
    Egaroucid Project

    @file hash.hpp
        Hash manager
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include "common.hpp"

/*
    @brief definition of maximum hash level
*/
#define N_HASH_LEVEL 30
#define MIN_HASH_LEVEL 23
#define DEFAULT_HASH_LEVEL 25
#define MAX_HASH_LEVEL 27

int global_hash_level = DEFAULT_HASH_LEVEL;

/*
    @brief array for calculating hash code
*/
uint32_t hash_rand_player[4][65536];
uint32_t hash_rand_opponent[4][65536];

/*
    @brief definition of hash sizes

    2 ^ hash_level will be tha size of hash
*/
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

/*
    @brief initialize hash array randomly

    @param hash_level           hash level
*/
void hash_init_rand(int hash_level){
    int i, j;
    for (i = 0; i < 4; ++i){
        for (j = 0; j < 65536; ++j){
            hash_rand_player[i][j] = 0;
            while (pop_count_uint(hash_rand_player[i][j]) < hash_level / 6)
                hash_rand_player[i][j] = myrand_uint_rev() & (hash_sizes[hash_level] - 1);
            hash_rand_opponent[i][j] = 0;
            while (pop_count_uint(hash_rand_opponent[i][j]) < hash_level / 6)
                hash_rand_opponent[i][j] = myrand_uint_rev() & (hash_sizes[hash_level] - 1);
        }
    }
}

/*
    @brief initialize hash array from optimized data

    @param hash_level           hash level
*/
bool hash_init(int hash_level){
    FILE* fp;
    if (!file_open(&fp, ("resources/hash/hash" + std::to_string(hash_level) + ".eghs").c_str(), "rb")){
        std::cerr << "[ERROR] can't open hash" + std::to_string(hash_level) + ".eghs" << std::endl;
        return false;
    }
    for (int i = 0; i < 4; ++i){
        if (fread(hash_rand_player[i], 4, 65536, fp) < 65536){
            std::cerr << "[ERROR] hash" + std::to_string(hash_level) + ".eghs broken" << std::endl;
            return false;
        }
    }
    for (int i = 0; i < 4; ++i){
        if (fread(hash_rand_opponent[i], 4, 65536, fp) < 65536){
            std::cerr << "[ERROR] hash" + std::to_string(hash_level) + ".eghs broken" << std::endl;
            return false;
        }
    }
    return true;
}

/*
    @brief initialize hash array from optimized data

    @param hash_level           hash level
    @param binary_path          path to binary
*/
bool hash_init(int hash_level, std::string binary_path){
    FILE* fp;
    if (!file_open(&fp, (binary_path + "resources/hash/hash" + std::to_string(hash_level) + ".eghs").c_str(), "rb")){
        std::cerr << "[ERROR] can't open hash" + std::to_string(hash_level) + ".eghs" << std::endl;
        return false;
    }
    for (int i = 0; i < 4; ++i){
        if (fread(hash_rand_player[i], 4, 65536, fp) < 65536){
            std::cerr << "[ERROR] hash" + std::to_string(hash_level) + ".eghs broken" << std::endl;
            return false;
        }
    }
    for (int i = 0; i < 4; ++i){
        if (fread(hash_rand_opponent[i], 4, 65536, fp) < 65536){
            std::cerr << "[ERROR] hash" + std::to_string(hash_level) + ".eghs broken" << std::endl;
            return false;
        }
    }
    return true;
}