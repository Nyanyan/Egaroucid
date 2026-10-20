/*
    Egaroucid Project

    @file uhead_check.cpp
        Check the engine's uncertainty head (dual network) against the test vectors written by
        export_uncertainty_head.py, and time the shared pass against the plain evaluation
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later

    Usage: uhead_check <weights.egnn> <head.egnh> <head.vectors.txt> [<separate uncertainty network.egnn>]
    (the last one only for the timing comparison)
    vector line: player(hex) opponent(hex) pattern_raw expected_total expected_u
*/

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "./../../engine/setting.hpp"
#include "./../../engine/nnue/nnue_eval.hpp"

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "usage: uhead_check <weights.egnn> <head.egnh> <head.vectors.txt>" << std::endl;
        return 1;
    }
    if (!nnue::init(argv[1], true)) return 1;
    if (!nnue::init_uhead(argv[2], argv[1], 0, true)) return 1;
    std::ifstream ifs(argv[3]);
    std::string line;
    struct Vec { uint64_t p, o; int raw; int64_t total; double u; int idx[NNUE_N_INSTANCES]; };
    std::vector<Vec> vecs;
    while (std::getline(ifs, line)) {
        std::istringstream ss(line);
        Vec v;
        std::string ph, oh;
        if (!(ss >> ph >> oh >> v.raw >> v.total >> v.u)) continue;
        v.p = std::stoull(ph, nullptr, 16);
        v.o = std::stoull(oh, nullptr, 16);
        nnue::compute_indices(v.p, v.o, v.idx);
        vecs.push_back(v);
    }
    int bad_total = 0, bad_u = 0;
    double max_du = 0.0;
    for (const Vec &v : vecs) {
        const int empties = 64 - __builtin_popcountll(v.p | v.o);
        int64_t total;
        double u;
        if (!nnue::evaluate_total_and_u_idx(v.idx, v.p, empties, v.raw, &total, &u)) {
            std::cerr << "the shared pass does not support this network" << std::endl;
            return 1;
        }
        if (total != v.total || total != nnue::evaluate_total_idx(v.idx, v.p, empties, v.raw)) ++bad_total;
        max_du = std::max(max_du, std::fabs(u - v.u));
        if (std::fabs(u - v.u) > 2e-2) ++bad_u;  // the head runs on int16 weights
    }
    std::cout << "vectors " << vecs.size() << " total mismatches " << bad_total << " u mismatches (> 0.02) " << bad_u
              << " max |u - expected| " << max_du << std::endl;
    const int reps = 200;
    auto timeit = [&](const char *name, auto fn) {
        double acc = 0;
        const auto s0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; ++r) for (const Vec &v : vecs) acc += fn(v);
        const double t = std::chrono::duration<double, std::nano>(std::chrono::high_resolution_clock::now() - s0).count() / (reps * vecs.size());
        std::cout << "  " << name << " " << t << " ns (checksum " << acc << ")" << std::endl;
    };
    timeit("evaluation only", [&](const Vec &v) { return (double)nnue::evaluate_total_idx(v.idx, v.p, 64 - __builtin_popcountll(v.p | v.o), v.raw); });
    if (argc >= 5 && nnue::init_weights(nnue::uncertainty_weights, argv[4], false)) {
        timeit("separate uncertainty network", [&](const Vec &v) {
            return (double)nnue::evaluate_total_idx(v.idx, v.p, 64 - __builtin_popcountll(v.p | v.o), 0, nnue::uncertainty_weights);
        });
        timeit("evaluation + separate uncertainty network", [&](const Vec &v) {
            const int empties = 64 - __builtin_popcountll(v.p | v.o);
            return (double)nnue::evaluate_total_idx(v.idx, v.p, empties, v.raw) + nnue::evaluate_total_idx(v.idx, v.p, empties, 0, nnue::uncertainty_weights);
        });
    }
    timeit("evaluation and u (shared pass)", [&](const Vec &v) {
        int64_t total;
        double u;
        nnue::evaluate_total_and_u_idx(v.idx, v.p, 64 - __builtin_popcountll(v.p | v.o), v.raw, &total, &u);
        return (double)total + u;
    });
    return (bad_total || bad_u) ? 1 : 0;
}
