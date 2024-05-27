#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "evaluation_definition.hpp"

#define MAX_N_DATA 5000000

struct Datum{
    int16_t n;
    int16_t player;
    uint16_t idxes[ADJ_N_FEATURES];
    int16_t score;
};

int main(int argc, char *argv[]){
    std::cerr << EVAL_DEFINITION_NAME << std::endl;
    std::cerr << EVAL_DEFINITION_DESCRIPTION << std::endl;
    
    evaluation_definition_init();

    int16_t n, player_short, score_short;
    uint16_t idxes[ADJ_N_FEATURES];
    for (int phase = 0; phase < 60; ++phase){
        std::string in_file = "./../../../train_data/bin_data/20240223_1/" + std::to_string(phase) + "/27.dat";
        std::string out_file = "./../../../train_data/bin_data/20240223_1/" + std::to_string(phase) + "/64.dat";
        std::cerr << in_file << " " << out_file << std::endl;
        FILE* fp;
        if (fopen_s(&fp, in_file.c_str(), "rb") != 0) {
            std::cerr << "can't open data " << in_file << std::endl;
            return 1;
        }
        std::ofstream fout;
        fout.open(out_file, std::ios::out|std::ios::binary|std::ios::trunc);
        if (!fout){
            std::cerr << "can't open output file " << out_file << std::endl;
            return 1;
        }
        uint64_t n_data = 0;
        while (n_data < MAX_N_DATA){
            if (fread(&n, 2, 1, fp) < 1){
                break;
            }
            fread(&player_short, 2, 1, fp);
            fread(idxes, 2, ADJ_N_FEATURES, fp);
            fread(&score_short, 2, 1, fp);

            fout.write((char*)&n, 2);
            fout.write((char*)&player_short, 2);
            fout.write((char*)idxes, 2 * ADJ_N_FEATURES);
            fout.write((char*)&score_short, 2);
            ++n_data;
        }
        fclose(fp);
        fout.close();
    }
    return 0;

}
