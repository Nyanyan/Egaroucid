#include <iostream>
#include <fstream>
#include "evaluation_definition.hpp"

#define ADJ_MAX_N_FILES 64
//#define ADJ_MAX_N_DATA 20000000

struct Adj_Data {
    int features[ADJ_N_FEATURES];
    float score;
};

int16_t eval_arr[ADJ_N_PHASES][ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];

int initialize_eval_arr(char *in_file){
    FILE* fp;
    if (!file_open(&fp, in_file, "rb")){
        std::cerr << "[ERROR] [FATAL] can't open eval " << in_file << std::endl;
        return false;
    }
    for (int phase_idx = 0; phase_idx < ADJ_N_PHASES; ++phase_idx){
        for (int feature_idx = 0; feature_idx < ADJ_N_EVAL; ++feature_idx){
            if (fread(eval_arr[phase_idx][feature_idx], 2, adj_eval_sizes[feature_idx], fp) < adj_eval_sizes[feature_idx]){
                std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
                fclose(fp);
                return 1;
            }
        }
    }
    fclose(fp);
    return 0;
}

int import_data(int n_files, char* files[], std::vector<Adj_Data> &data) {
    int n_data = 0;
    FILE* fp;
    int16_t n_discs, score, player;
    uint16_t raw_features[ADJ_N_FEATURES];
    float score_avg = 0.0;
    Adj_Data datum;
    for (int file_idx = 0; file_idx < n_files; ++file_idx) {
        std::cerr << files[file_idx] << std::endl;
        if (fopen_s(&fp, files[file_idx], "rb") != 0) {
            std::cerr << "can't open " << files[file_idx] << std::endl;
            continue;
        }
        //while (n_data < ADJ_MAX_N_DATA) {
        while (true) {
            if (fread(&n_discs, 2, 1, fp) < 1)
                break;
            fread(&player, 2, 1, fp);
            fread(raw_features, 2, ADJ_N_FEATURES, fp);
            fread(&score, 2, 1, fp);
            for (int i = 0; i < ADJ_N_FEATURES; ++i){
                datum.features[i] = raw_features[i];
            }
            datum.score = (float)score;
            data.emplace_back(datum);
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

void test_loss(int phase, int n_data, std::vector<Adj_Data> &data, float *mse, float *mae){
    *mse = 0.0;
    *mae = 0.0;
    for (int i = 0; i < n_data; ++i){
        int score = 0;
        for (int j = 0; j < ADJ_N_FEATURES; ++j){
            score += eval_arr[phase][adj_feature_to_eval_idx[j]][data[i].features[j]];
        }
        score += score >= 0 ? ADJ_STEP_2 : -ADJ_STEP_2;
        score /= ADJ_STEP;
        //score = std::clamp(score, -SCORE_MAX, SCORE_MAX);
        if (score < -SCORE_MAX)
            score = -SCORE_MAX;
        if (score > SCORE_MAX)
            score = SCORE_MAX;
        float abs_error = fabs(data[i].score - score);
        *mse += abs_error * abs_error;
        *mae += abs_error;
    }
    *mse /= n_data;
    *mae /= n_data;
}

std::vector<Adj_Data> global_data;

int main(int argc, char *argv[]){
    if (argc < 4){
        std::cerr << "input [eval_file] [phase] [test_files]" << std::endl;
        return 1;
    }
    char* in_file = argv[1];
    int phase = atoi(argv[2]);
    char* test_files[ADJ_MAX_N_FILES];
    for (int i = 3; i < argc; ++i)
        test_files[i - 3] = argv[i];
    int n_files = argc - 3;
    
    if (initialize_eval_arr(in_file)){
        return 1;
    }

    int n_data = import_data(n_files, test_files, global_data);

    float mse, mae;
    test_loss(phase, n_data, global_data, &mse, &mae);
    std::cerr << "phase " << phase << " n_data " << n_data << " mse " << mse << " mae " << mae << std::endl;
    std::cout << "phase " << phase << " n_data " << n_data << " mse " << mse << " mae " << mae << std::endl;

    return 0;
}