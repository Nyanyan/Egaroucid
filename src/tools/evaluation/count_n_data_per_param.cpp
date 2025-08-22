#include <iostream>
#include "evaluation_definition.hpp"

struct Adj_Data {
    uint16_t features[ADJ_N_FEATURES];
    double score;
};

#define ADJ_MAX_N_DATA 200000000
#define ADJ_MAX_N_FILES 200



void adj_init_arr(int eval_size, int *host_rev_idx_arr, int *host_n_appear_arr) {
    for (int i = 0; i < eval_size; ++i) {
        host_n_appear_arr[i] = 0;
    }
    int strt_idx = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        for (int j = 0; j < adj_eval_sizes[i]; ++j) {
            host_rev_idx_arr[strt_idx + j] = strt_idx + adj_calc_rev_idx(i, j);
        }
        strt_idx += adj_eval_sizes[i];
    }
}


int adj_import_data(int n_files, char* files[], Adj_Data* host_train_data, int *host_rev_idx_arr, int64_t n_max_data) {
    int n_data = 0;
    FILE* fp;
    int16_t n_discs, score, player;
    Adj_Data data;
    double score_avg = 0.0;
    int start_idx_arr[ADJ_N_FEATURES];
    int start_idx = 0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i){
        if (i > 0){
            if (adj_feature_to_eval_idx[i] > adj_feature_to_eval_idx[i - 1]){
                start_idx += adj_eval_sizes[adj_feature_to_eval_idx[i - 1]];
            }
        }
        start_idx_arr[i] = start_idx;
    }
    for (int file_idx = 0; file_idx < n_files; ++file_idx) {
        std::cerr << files[file_idx] << std::endl;
        if (fopen_s(&fp, files[file_idx], "rb") != 0) {
            std::cerr << "can't open " << files[file_idx] << std::endl;
            continue;
        }
        while (n_data < ADJ_MAX_N_DATA && n_data < n_max_data) {
            if (fread(&n_discs, 2, 1, fp) < 1)
                break;
            fread(&player, 2, 1, fp);
            fread(host_train_data[n_data].features, 2, ADJ_N_FEATURES, fp);
            fread(&score, 2, 1, fp);
            // host_train_data[n_data].score = (double)score * ADJ_STEP;
            if ((n_data & 0xffff) == 0xffff)
                std::cerr << '\r' << n_data;
            score_avg += score;
            ++n_data;
        }
        fclose(fp);
        std::cerr << '\r' << n_data << std::endl;
    }
    score_avg /= n_data;
    std::cerr << std::endl;
    //std::cerr << n_data << " data loaded" << std::endl;
    std::cerr << "score avg " << score_avg << std::endl;
    return n_data;
}


int main(int argc, char* argv[]) {
    std::cerr << EVAL_DEFINITION_NAME << std::endl;
    std::cerr << EVAL_DEFINITION_DESCRIPTION << std::endl;
    if (argc < 2) {
        std::cerr << "input [train_data...]" << std::endl;
        return 1;
    }
    const int N_FILES_BEFORE_TRAIN_DATA = 1;
    if (argc - N_FILES_BEFORE_TRAIN_DATA >= ADJ_MAX_N_FILES) {
        std::cerr << "too many train files" << std::endl;
        return 1;
    }
    char* train_files[ADJ_MAX_N_FILES];
    int n_train_data_file = argc - N_FILES_BEFORE_TRAIN_DATA;
    for (int i = 0; i < n_train_data_file; ++i)
        train_files[i] = argv[i + N_FILES_BEFORE_TRAIN_DATA];

    int eval_size = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i){
        eval_size += adj_eval_sizes[i];
    }
    std::cerr << "eval_size " << eval_size << std::endl;
    int *host_rev_idx_arr = (int*)malloc(sizeof(int) * eval_size); // reversed index
    Adj_Data* host_train_data = (Adj_Data*)malloc(sizeof(Adj_Data) * ADJ_MAX_N_DATA); // train data
    int *host_n_appear_arr = (int*)malloc(sizeof(int) * eval_size);
    adj_init_arr(eval_size, host_rev_idx_arr, host_n_appear_arr);
    int n_all_data = adj_import_data(n_train_data_file, train_files, host_train_data, host_rev_idx_arr, ADJ_MAX_N_DATA);

    // calculate n_appear of train data
    int host_start_idx_arr[ADJ_N_FEATURES];
    int start_idx = 0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i){
        if (i > 0){
            if (adj_feature_to_eval_idx[i] > adj_feature_to_eval_idx[i - 1]){
                start_idx += adj_eval_sizes[adj_feature_to_eval_idx[i - 1]];
            }
        }
        host_start_idx_arr[i] = start_idx;
    }
    std::cerr << "n_data " << n_all_data << std::endl;
    for (int data_idx = 0; data_idx < n_all_data; ++data_idx){
        for (int i = 0; i < ADJ_N_FEATURES; ++i){
            #if ADJ_CELL_WEIGHT
                if (host_train_data[data_idx].features[i] < 10){
                    ++host_n_appear_arr[host_train_data[data_idx].features[i]];
                    ++host_n_appear_arr[host_rev_idx_arr[host_train_data[data_idx].features[i]]];
                } else if (host_train_data[data_idx].features[i] < 20){
                    ++host_n_appear_arr[host_train_data[data_idx].features[i] - 10];
                    ++host_n_appear_arr[host_rev_idx_arr[host_train_data[data_idx].features[i] - 10]];
                }
            #else
                // std::cerr << data_idx << " " << i << " " << host_start_idx_arr[i] << " " << (int)host_train_data[data_idx].features[i] << std::endl;
                ++host_n_appear_arr[host_start_idx_arr[i] + (int)host_train_data[data_idx].features[i]];
                int rev_idx = host_rev_idx_arr[host_start_idx_arr[i] + (int)host_train_data[data_idx].features[i]];
                //if (rev_idx != start_idx_arr[i] + (int)host_train_data[data_idx].features[i])
                ++host_n_appear_arr[rev_idx];
            #endif
        }
    }
    for (int i = 0; i < eval_size; ++i) {
        std::cout << host_n_appear_arr[i] << std::endl;
    }
}