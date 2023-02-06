#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include "evaluation_definition.hpp"

int main(int argc, char *argv[]){
    if (argc < 6){
        std::cerr << "input [input dir] [start file no] [n files] [output file] [n_discs]" << std::endl;
        return 1;
    }

    evaluation_definition_init();

    int t = 0;

    int start_file = atoi(argv[2]);
    int n_files = atoi(argv[3]);
    int n_discs = atoi(argv[5]);

    std::ofstream fout;
    fout.open(argv[4], std::ios::out|std::ios::binary|std::ios::trunc);
    if (!fout){
        std::cerr << "can't open" << std::endl;
        return 1;
    }

    Board board;
    int8_t player, score, policy;
    int16_t player_short, score_short, n;
    uint16_t idxes[ADJ_N_FEATURES];
    FILE* fp;
    std::string file;
    for (int i = start_file; i < n_files; ++i){
        std::cerr << "=";
        file = std::string(argv[1]) + "/" + std::to_string(i) + ".dat";
        if (fopen_s(&fp, file.c_str(), "rb") != 0) {
            std::cerr << "can't open " << file << std::endl;
            continue;
        }
        while (true){
            if (fread(&(board.player), 8, 1, fp) < 1)
                break;
            fread(&(board.opponent), 8, 1, fp);
            fread(&player, 1, 1, fp);
            fread(&policy, 1, 1, fp);
            fread(&score, 1, 1, fp);
            n = pop_count_ull(board.player | board.opponent);
            if (n == n_discs){
                player_short = player;
                score_short = score;
                adj_calc_features(&board, idxes);
                fout.write((char*)&n, 2);
                fout.write((char*)&player_short, 2);
                fout.write((char*)idxes, 2 * ADJ_N_FEATURES);
                fout.write((char*)&score_short, 2);
                ++t;
            }
        }
        if (i % 20 == 19)
            std::cerr << std::endl;
    }
    std::cerr << t << std::endl;
    return 0;

}