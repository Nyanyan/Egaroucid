/*
    Egaroucid Project

    @file nnue_check.cpp
        Check the engine's NNUE against the test vectors written by export_nnue.py, and time it
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later

    Usage: nnue_check <weights.egnn> <weights.vectors.txt>
    vector line: player(hex) opponent(hex) pattern_raw expected_total expected_score idx0 ... idx63
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "./../../engine/setting.hpp"
#include "./../../engine/nnue/nnue_eval.hpp"

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: nnue_check <weights.egnn> <weights.vectors.txt>" << std::endl;
        return 1;
    }
    if (!nnue::init(argv[1], true)) return 1;
    std::ifstream ifs(argv[2]);
    std::string line;
    struct Vec { uint64_t p, o; int raw; int64_t total; int score; int idx[NNUE_N_INSTANCES]; };
    std::vector<Vec> vecs;
    while (std::getline(ifs, line)) {
        std::istringstream ss(line);
        Vec v;
        std::string ph, oh;
        ss >> ph >> oh >> v.raw >> v.total >> v.score;
        v.p = std::stoull(ph, nullptr, 16);
        v.o = std::stoull(oh, nullptr, 16);
        for (int i = 0; i < NNUE_N_INSTANCES; ++i) ss >> v.idx[i];
        vecs.push_back(v);
    }
    int bad_idx = 0, bad_total = 0, bad_scalar = 0, bad_score = 0;
    for (const Vec &v : vecs) {
        int idx[NNUE_N_INSTANCES];
        nnue::compute_indices(v.p, v.o, idx);
        for (int i = 0; i < NNUE_N_INSTANCES; ++i) {
            if (idx[i] != v.idx[i]) {
                if (bad_idx < 5) std::cerr << "index mismatch instance " << i << " got " << idx[i] << " want " << v.idx[i] << std::endl;
                ++bad_idx;
            }
        }
        const int empties = 64 - __builtin_popcountll(v.p | v.o);
        const int64_t total = nnue::evaluate_total(v.p, v.o, empties, v.raw);
        if (total != v.total) {
            if (bad_total < 5) std::cerr << "total mismatch got " << total << " want " << v.total << std::endl;
            ++bad_total;
        }
        if (nnue::score_of_total(total) != v.score) ++bad_score;
        if (nnue::evaluate_total_scalar(v.idx, v.p, empties, v.raw) != nnue::evaluate_total_idx(v.idx, v.p, empties, v.raw)) ++bad_scalar;
    }
    std::cout << "vectors " << vecs.size() << " index mismatches " << bad_idx << " total mismatches " << bad_total
              << " score mismatches " << bad_score << " scalar/simd mismatches " << bad_scalar << std::endl;
    int64_t sum = 0;
    const int reps = 200;
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; ++r) {
        for (const Vec &v : vecs) sum += nnue::evaluate_total(v.p, v.o, 64 - __builtin_popcountll(v.p | v.o), v.raw);
    }
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::high_resolution_clock::now() - t0).count() / (reps * vecs.size());
    std::cout << "evaluate_total " << ns << " ns per call (hot loop over the vectors, checksum " << sum << ")" << std::endl;
    auto timeit = [&](const char *name, auto fn) {
        int64_t acc = 0;
        const auto s0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; ++r) for (const Vec &v : vecs) acc += fn(v);
        const double t = std::chrono::duration<double, std::nano>(std::chrono::high_resolution_clock::now() - s0).count() / (reps * vecs.size());
        std::cout << "  " << name << " " << t << " ns (checksum " << acc << ")" << std::endl;
    };
    timeit("indices", [&](const Vec &v) { int idx[NNUE_N_INSTANCES]; nnue::compute_indices(v.p, v.o, idx); return (int64_t)idx[7] + idx[63]; });
    timeit("network + linear (given indices)", [&](const Vec &v) { return nnue::evaluate_total_idx(v.idx, v.p, 64 - __builtin_popcountll(v.p | v.o), v.raw); });
    timeit("linear (given indices)", [&](const Vec &v) {
        const int16_t *lin = nnue::weights.lin + (size_t)nnue::bucket_of(64 - __builtin_popcountll(v.p | v.o)) * NNUE_N_ROWS;
        int32_t l = 0; for (int i = 0; i < NNUE_N_INSTANCES; ++i) l += lin[v.idx[i]]; return (int64_t)l; });
    return (bad_idx || bad_total) ? 1 : 0;
}
