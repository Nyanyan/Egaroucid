#include <iostream>
#include "evaluation_definition.hpp"

#define ADJ_MAX_N_FILES 64
#if ADJ_CELL_WEIGHT
    #define ADJ_MAX_N_DATA 1000000
#else
    #define ADJ_MAX_N_DATA 100000000
#endif

void adj_init_arr(int eval_size, int *host_rev_idx_arr, int *host_n_appear_arr) {
    for (int i = 0; i < eval_size; ++i){
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

/*
    @brief import train data
*/
int adj_import_train_data(int n_files, char* files[], int *host_rev_idx_arr, int *host_n_appear_arr, int *score_n_appear_arr) {
    int n_data = 0;
    FILE* fp;
    int16_t n_discs, score, player;
    uint16_t features[ADJ_N_FEATURES];
    float score_avg = 0.0;
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
        while (n_data < ADJ_MAX_N_DATA) {
            if (fread(&n_discs, 2, 1, fp) < 1)
                break;
            fread(&player, 2, 1, fp);
            fread(features, 2, ADJ_N_FEATURES, fp);
            for (int i = 0; i < ADJ_N_FEATURES; ++i){
                #if ADJ_CELL_WEIGHT
                    if (features[i] < 10){
                        ++host_n_appear_arr[features[i]];
                        ++host_n_appear_arr[host_rev_idx_arr[features[i]]];
                    } else if (features[i] < 20){
                        ++host_n_appear_arr[features[i] - 10];
                        ++host_n_appear_arr[host_rev_idx_arr[features[i] - 10]];
                    }
                #else
                    ++host_n_appear_arr[start_idx_arr[i] + (int)features[i]];
                    ++host_n_appear_arr[host_rev_idx_arr[start_idx_arr[i] + (int)features[i]]];
                #endif
            }
            fread(&score, 2, 1, fp);
            ++score_n_appear_arr[score + 64];
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
    std::cerr << n_data << " data loaded" << std::endl;
    std::cerr << "score avg " << score_avg << std::endl;
    return n_data;
}

int main(int argc, char* argv[]){
    std::cerr << EVAL_DEFINITION_NAME << std::endl;
    std::cerr << EVAL_DEFINITION_DESCRIPTION << std::endl;
    if (argc < 2) {
        std::cerr << "input [test_data...]" << std::endl;
        return 1;
    }

    int eval_size = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i){
        eval_size += adj_eval_sizes[i];
    }
    char* test_files[ADJ_MAX_N_FILES];
    for (int i = 1; i < argc; ++i)
        test_files[i - 1] = argv[i];
    int *host_rev_idx_arr = (int*)malloc(sizeof(int) * eval_size); // reversed index
    int *host_n_appear_arr = (int*)malloc(sizeof(int) * eval_size);
    int score_n_appear_arr[129];
    for (int i = 0; i < 129; ++i){
        score_n_appear_arr[i] = 0;
    }
    adj_init_arr(eval_size, host_rev_idx_arr, host_n_appear_arr);
    int n_data = adj_import_train_data(argc - 1, test_files, host_rev_idx_arr, host_n_appear_arr, score_n_appear_arr);

    std::cout << n_data << std::endl;
    for (int i = 0; i < 129; ++i){
        std::cout << "score " << i - HW2 << " " << score_n_appear_arr[i] << std::endl;
    }

    int start_idx = 0;
    for (int pattern = 0; pattern < ADJ_N_EVAL; ++pattern){
        int max_appear = 0;
        for (int i = start_idx; i < start_idx + adj_eval_sizes[pattern]; ++i){
            max_appear = std::max(max_appear, host_n_appear_arr[i]);
        }
        int *histgram_arr = (int*)malloc(sizeof(int) * (max_appear + 1));
        for (int n_appear = 0; n_appear <= max_appear; ++n_appear){
            histgram_arr[n_appear] = 0;
        }
        for (int i = start_idx; i < start_idx + adj_eval_sizes[pattern]; ++i){
            ++histgram_arr[host_n_appear_arr[i]];
        }
        /*
        uint64_t sum_n_appear = 0;
        for (int n_appear = 1; n_appear <= max_appear; ++n_appear){
            sum_n_appear += histgram_arr[n_appear];
        }
        uint64_t sub_sum_n_appear = 0;
        for (int n_appear = 1; n_appear <= max_appear; ++n_appear){
            sub_sum_n_appear += histgram_arr[n_appear];
            if (sub_sum_n_appear >= sum_n_appear * 0.5){
                std::cout << "pattern(short) " << pattern << " " << n_appear << std::endl;
                break;
            }
        }
        */
        for (int n_appear = 0; n_appear <= max_appear; ++n_appear){
            if (histgram_arr[n_appear])
                std::cout << "pattern " << pattern << " " << n_appear << " " << histgram_arr[n_appear] << std::endl;
        }
        free(histgram_arr);
    }

    free(host_rev_idx_arr);
    free(host_n_appear_arr);
}
