/*
    Egaroucid Project

    @file eval_positions.cpp
        Write the production pattern evaluation of NNUE training records, before integer rounding
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later

    Usage: eval_positions <eval.egev2> <eval_move_ordering_end.egev> <in.bin> <out.f32> [threads]
    in.bin: records of convert_data.cpp (20 bytes); out.f32: one float per record, in discs for the
    side to move (the value mid_evaluate rounds to an integer)
*/

#include <cstdio>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "./../../engine/evaluate.hpp"
#include "./../../engine/search.hpp"

#pragma pack(push, 1)
struct Record {
    uint64_t player;
    uint64_t opponent;
    int8_t score;
    uint8_t empties;
    uint8_t source;
    uint8_t random;
};
#pragma pack(pop)

static float raw_evaluate(Board *board) {
    Eval_search eval;
    calc_eval_features(board, &eval);
    const int phase_idx = (board->n_discs() - 4) / PHASE_N_DISCS;
    const int num0 = pop_count_ull(board->player);
    const int res = calc_pattern(phase_idx, &eval.features[eval.feature_idx]) + eval_num_arr[phase_idx][num0];
    return (float)res / STEP;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "usage: eval_positions <eval.egev2> <eval_move_ordering_end.egev> <in.bin> <out.f32> [threads]" << std::endl;
        return 1;
    }
    eval_move_ordering_auto_requested = false;  // the pattern value needs no ordering models
    if (!evaluate_init(std::string(argv[1]), std::string(argv[2]), false)) return 1;
    std::FILE *in = std::fopen(argv[3], "rb");
    std::FILE *outf = std::fopen(argv[4], "wb");
    if (!in || !outf) { std::cerr << "cannot open " << argv[3] << " or " << argv[4] << std::endl; return 1; }
    const int n_threads = argc >= 6 ? std::atoi(argv[5]) : 8;
    const size_t chunk = 1 << 22;
    std::vector<Record> recs(chunk);
    std::vector<float> out(chunk);
    size_t total = 0;
    while (true) {
        const size_t n = std::fread(recs.data(), sizeof(Record), chunk, in);
        if (n == 0) break;
        std::vector<std::thread> ths;
        for (int t = 0; t < n_threads; ++t) {
            ths.emplace_back([&, t, n]() {
                for (size_t i = t; i < n; i += n_threads) {
                    Board b;
                    b.player = recs[i].player;
                    b.opponent = recs[i].opponent;
                    out[i] = raw_evaluate(&b);
                }
            });
        }
        for (auto &th : ths) th.join();
        std::fwrite(out.data(), sizeof(float), n, outf);
        total += n;
    }
    std::fclose(in);
    std::fclose(outf);
    std::cerr << total << " positions evaluated" << std::endl;
    return 0;
}
