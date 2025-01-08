#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "evaluation_definition.hpp"

struct Datum {
    int16_t n;
    int16_t player_short;
    uint16_t idxes[ADJ_N_FEATURES];
    int16_t score_short;
};

int main(int argc, char *argv[]){
    std::cerr << EVAL_DEFINITION_NAME << std::endl;
    std::cerr << EVAL_DEFINITION_DESCRIPTION << std::endl;
    if (argc < 9){
        std::cerr << "input [input dir] [start file no] [n files] [output file] [phase] [use_n_moves_min] [use_n_moves_max] [min_n_data]" << std::endl;
        return 1;
    }

    evaluation_definition_init();

    int t = 0;

    int start_file = atoi(argv[2]);
    int n_files = atoi(argv[3]);
    int phase = atoi(argv[5]);
    int use_n_moves_min = atoi(argv[6]);
    int use_n_moves_max = atoi(argv[7]);
    int min_n_data = atoi(argv[8]);

    std::ofstream fout;
    fout.open(argv[4], std::ios::out|std::ios::binary|std::ios::trunc);
    if (!fout){
        std::cerr << "can't open output file " << argv[4] << std::endl;
        return 1;
    }

    bool data_not_available = 
        phase * ADJ_N_PHASE_DISCS > use_n_moves_max || 
        (phase + 1) * ADJ_N_PHASE_DISCS <= use_n_moves_min;
    if (data_not_available){
        std::cerr << "data not available at phase " << phase << std::endl;
        return 0;
    }

    Board board;
    int8_t player, score, policy;
    int16_t player_short, score_short, n;
    uint16_t idxes[ADJ_N_FEATURES];
    FILE* fp;
    std::string file;
    std::vector<Datum> data_memo;
    for (int i = start_file; i < n_files; ++i){
        std::cerr << "=";
        file = std::string(argv[1]) + "/" + std::to_string(i) + ".dat";
        if (fopen_s(&fp, file.c_str(), "rb") != 0) {
            std::cerr << "can't open data " << file << std::endl;
            continue;
        }
        while (true) {
            //if (repeated && t >= min_n_data) {
            //    break;
            //}
            if (fread(&(board.player), 8, 1, fp) < 1)
                break;
            fread(&(board.opponent), 8, 1, fp);
            fread(&player, 1, 1, fp);
            fread(&policy, 1, 1, fp);
            fread(&score, 1, 1, fp);
            n = pop_count_ull(board.player | board.opponent);
            if (calc_phase(&board, player) == phase && n - 4 >= use_n_moves_min && n - 4 <= use_n_moves_max){
                #ifdef ADJ_MIN_N_DISCS
                if (ADJ_MIN_N_DISCS <= n && n <= ADJ_MAX_N_DISCS){
                #endif
                    player_short = player;
                    score_short = score;
                    adj_calc_features(&board, idxes);
                    fout.write((char*)&n, 2);
                    fout.write((char*)&player_short, 2);
                    fout.write((char*)idxes, 2 * ADJ_N_FEATURES);
                    fout.write((char*)&score_short, 2);
                    if (t < min_n_data) {
                        Datum datum;
                        datum.n = n;
                        datum.player_short = player_short;
                        for (int j = 0; j < ADJ_N_FEATURES; ++j) {
                            datum.idxes[j] = idxes[j];
                        }
                        datum.score_short = score_short;
                        data_memo.emplace_back(datum);
                    }
                    ++t;
                #ifdef ADJ_MIN_N_DISCS
                }
                #endif
            }
        }
        if (i % 20 == 19)
            std::cerr << std::endl;
    }
    while (t < min_n_data) {
        for (Datum &datum: data_memo) {
            fout.write((char*)&datum.n, 2);
            fout.write((char*)&datum.player_short, 2);
            fout.write((char*)datum.idxes, 2 * ADJ_N_FEATURES);
            fout.write((char*)&datum.score_short, 2);
            ++t;
        }
    }
    fout.close();
    std::cerr << t << std::endl;
    return 0;

}