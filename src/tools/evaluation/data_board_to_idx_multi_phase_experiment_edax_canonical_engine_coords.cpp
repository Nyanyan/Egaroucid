/*
    Egaroucid Project

    @file data_board_to_idx_multi_phase_experiment_edax_canonical_engine_coords.cpp
        Board-data converter for the isolated Edax-linear experiment with
        Edax-style reflection sharing and engine-coordinate feature IDs
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "evaluation_definition_experiment_edax_linear.hpp"

struct Datum {
    int16_t n;
    int16_t player_short;
    uint16_t idxes[ADJ_N_FEATURES];
    int16_t score_short;
};

constexpr int EDAX_SYM_S10[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
constexpr int EDAX_SYM_C10[10] = {9, 8, 7, 6, 4, 5, 3, 2, 1, 0};
constexpr int EDAX_SYM_C9[9] = {0, 2, 1, 4, 3, 5, 7, 6, 8};

inline uint16_t edax_player_feature_lsd(const int sym[], int n_digits, int idx) {
    int res = 0;
    for (int i = 0; i < n_digits; ++i) {
        res += ((idx / adj_pow3[sym[i]]) % 3) * adj_pow3[i];
    }
    return (uint16_t)res;
}

inline uint16_t edax_canonical_raw_idx(int eval_idx, int idx) {
    int transformed = idx;
    switch (eval_idx) {
        case 0:
            transformed = edax_player_feature_lsd(EDAX_SYM_C9, 9, idx);
            break;
        case 1:
            transformed = edax_player_feature_lsd(EDAX_SYM_C10, 10, idx);
            break;
        case 2:
        case 3:
            transformed = edax_player_feature_lsd(EDAX_SYM_S10, 10, idx);
            break;
        case 4:
        case 5:
        case 6:
        case 7:
            transformed = edax_player_feature_lsd(EDAX_SYM_S10 + 2, 8, idx);
            break;
        case 8:
            transformed = edax_player_feature_lsd(EDAX_SYM_S10 + 3, 7, idx);
            break;
        case 9:
            transformed = edax_player_feature_lsd(EDAX_SYM_S10 + 4, 6, idx);
            break;
        case 10:
            transformed = edax_player_feature_lsd(EDAX_SYM_S10 + 5, 5, idx);
            break;
        case 11:
            transformed = edax_player_feature_lsd(EDAX_SYM_S10 + 6, 4, idx);
            break;
        default:
            transformed = idx;
            break;
    }
    return (uint16_t)std::min(idx, transformed);
}

inline void edax_calc_canonical_features(Board *board, uint16_t res[]) {
    uint_fast8_t b_arr[HW2];
    board->translate_to_arr_player(b_arr);
    for (int feature = 0; feature < ADJ_N_FEATURES; ++feature) {
        int raw_idx = 0;
        for (int i = 0; i < adj_feature_to_coord[feature].n_cells; ++i) {
            raw_idx *= 3;
            raw_idx += b_arr[HW2 - 1 - adj_feature_to_coord[feature].cells[i]];
        }
        int eval_idx = adj_feature_to_eval_idx[feature];
        res[feature] = edax_canonical_raw_idx(eval_idx, raw_idx);
    }
}

inline void write_datum(std::ofstream &fout, const Datum &datum) {
    fout.write((char*)&datum.n, 2);
    fout.write((char*)&datum.player_short, 2);
    fout.write((char*)datum.idxes, 2 * ADJ_N_FEATURES);
    fout.write((char*)&datum.score_short, 2);
}

int main(int argc, char *argv[]) {
    std::cerr << EVAL_DEFINITION_NAME << "_canonical_engine_coords" << std::endl;
    std::cerr << EVAL_DEFINITION_DESCRIPTION << " with Edax reflection sharing and engine-coordinate feature IDs" << std::endl;
    if (argc < 10) {
        std::cerr << "input [input dir] [start file no] [end file no exclusive] [output root dir] [output file name] [max_n_data_per_phase] [use_n_moves_min] [use_n_moves_max] [min_n_data_per_phase]" << std::endl;
        return 1;
    }

    evaluation_definition_init();

    const std::string input_dir = argv[1];
    const int start_file = std::atoi(argv[2]);
    const int end_file = std::atoi(argv[3]);
    const std::filesystem::path output_root_dir = argv[4];
    const std::string output_file_name = argv[5];
    const int max_n_data_per_phase = std::atoi(argv[6]);
    const int use_n_moves_min = std::atoi(argv[7]);
    const int use_n_moves_max = std::atoi(argv[8]);
    const int min_n_data_per_phase = std::atoi(argv[9]);

    std::array<std::ofstream, ADJ_N_PHASES> fout;
    std::array<int, ADJ_N_PHASES> n_data{};
    std::array<std::vector<Datum>, ADJ_N_PHASES> data_memo;

    for (int phase = 0; phase < ADJ_N_PHASES; ++phase) {
        std::filesystem::path phase_dir = output_root_dir / std::to_string(phase);
        std::filesystem::create_directories(phase_dir);
        std::filesystem::path output_file = phase_dir / output_file_name;
        fout[phase].open(output_file, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!fout[phase]) {
            std::cerr << "can't open output file " << output_file << std::endl;
            return 1;
        }
    }

    auto all_phase_full = [&]() {
        if (max_n_data_per_phase <= 0) {
            return false;
        }
        return std::all_of(n_data.begin(), n_data.end(), [&](int n) {
            return n >= max_n_data_per_phase;
        });
    };

    Board board;
    int8_t player, score, policy;
    uint16_t idxes[ADJ_N_FEATURES];
    FILE *fp;

    for (int file_idx = start_file; file_idx < end_file; ++file_idx) {
        if (all_phase_full()) {
            break;
        }
        std::string file = input_dir + "/" + std::to_string(file_idx) + ".dat";
        if (fopen_s(&fp, file.c_str(), "rb") != 0) {
            std::cerr << "can't open data " << file << std::endl;
            continue;
        }
        std::cerr << "reading " << file << std::endl;
        while (!all_phase_full()) {
            if (fread(&(board.player), 8, 1, fp) < 1) {
                break;
            }
            fread(&(board.opponent), 8, 1, fp);
            fread(&player, 1, 1, fp);
            fread(&policy, 1, 1, fp);
            fread(&score, 1, 1, fp);

            int16_t n = (int16_t)pop_count_ull(board.player | board.opponent);
            int phase = calc_phase(&board, player);
            if (phase < 0 || phase >= ADJ_N_PHASES) {
                continue;
            }
            if (max_n_data_per_phase > 0 && n_data[phase] >= max_n_data_per_phase) {
                continue;
            }
            if (n - 4 < use_n_moves_min || n - 4 > use_n_moves_max) {
                continue;
            }

            edax_calc_canonical_features(&board, idxes);

            Datum datum;
            datum.n = n;
            datum.player_short = player;
            for (int i = 0; i < ADJ_N_FEATURES; ++i) {
                datum.idxes[i] = idxes[i];
            }
            datum.score_short = score;
            write_datum(fout[phase], datum);
            if ((int)data_memo[phase].size() < min_n_data_per_phase) {
                data_memo[phase].emplace_back(datum);
            }
            ++n_data[phase];
        }
        fclose(fp);
    }

    for (int phase = 0; phase < ADJ_N_PHASES; ++phase) {
        if (n_data[phase] == 0 && min_n_data_per_phase > 0) {
            std::cerr << "phase " << phase << " has no data to repeat" << std::endl;
            continue;
        }
        int memo_idx = 0;
        while (n_data[phase] < min_n_data_per_phase && !data_memo[phase].empty()) {
            write_datum(fout[phase], data_memo[phase][memo_idx]);
            ++n_data[phase];
            memo_idx = (memo_idx + 1) % data_memo[phase].size();
        }
    }

    for (int phase = 0; phase < ADJ_N_PHASES; ++phase) {
        fout[phase].close();
        std::cerr << "phase " << phase << " " << n_data[phase] << std::endl;
    }

    return 0;
}
