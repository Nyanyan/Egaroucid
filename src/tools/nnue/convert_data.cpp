/*
    Egaroucid Project

    @file convert_data.cpp
        Convert NNUE training data (board+score text or f5d6 transcripts) to fixed-size binary records
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later

    Record (16 + 4 bytes, little endian):
        uint64 player    side to move, bit 63 = a1 ... bit 0 = h8 (Egaroucid's convention)
        uint64 opponent
        int8   score     final disc difference (or search value) for the side to move
        uint8  empties
        uint8  source    0: board+score text, 1: transcript
        uint8  random    number of random opening moves of the game (transcripts), 0 otherwise

    Usage:
        convert_data boards <input .txt> ... -o <out dir>
            lines "<64 chars: X = side to move, O = opponent, - = empty> <score>"
        convert_data transcripts <root dir> ... -o <out dir> [-target <positions per empties>] [-threads <n>]
            <root dir>/<random moves>/<file>.txt, one f5d6 game per line; positions before the random
            moves end are dropped, positions are thinned per empties count so that each count gets
            about <target> positions (default 6000000) out of 1M games per random-move folder
*/

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

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
static_assert(sizeof(Record) == 20);

static inline int cell_bit(int x, int y) { return 63 - (y * 8 + x); }

static uint64_t flips(uint64_t me, uint64_t op, int x, int y) {
    const int b = cell_bit(x, y);
    if (((me | op) >> b) & 1) return 0;
    static const int dirs[8][2] = {{-1,-1},{0,-1},{1,-1},{-1,0},{1,0},{-1,1},{0,1},{1,1}};
    uint64_t res = 0;
    for (auto &d : dirs) {
        int cx = x + d[0], cy = y + d[1];
        uint64_t line = 0;
        while (0 <= cx && cx < 8 && 0 <= cy && cy < 8 && ((op >> cell_bit(cx, cy)) & 1)) {
            line |= 1ULL << cell_bit(cx, cy);
            cx += d[0]; cy += d[1];
        }
        if (line && 0 <= cx && cx < 8 && 0 <= cy && cy < 8 && ((me >> cell_bit(cx, cy)) & 1)) res |= line;
    }
    return res;
}

static bool has_legal(uint64_t me, uint64_t op) {
    for (int y = 0; y < 8; ++y) for (int x = 0; x < 8; ++x) if (flips(me, op, x, y)) return true;
    return false;
}

static uint32_t hash32(uint64_t a, uint64_t b) {
    uint64_t h = a * 0x9E3779B97F4A7C15ULL ^ (b + 0x632BE59BD9B4E019ULL + (a << 6) + (a >> 2));
    h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 29;
    return (uint32_t)(h >> 32);
}

static int convert_boards(const std::vector<std::string> &inputs, const fs::path &out) {
    for (const auto &in : inputs) {
        std::ifstream ifs(in);
        if (!ifs) { std::cerr << "cannot open " << in << std::endl; return 1; }
        fs::path dst = out / ("boards_" + fs::path(in).stem().string() + ".bin");
        std::ofstream ofs(dst, std::ios::binary);
        std::string line;
        uint64_t n = 0, bad = 0;
        while (std::getline(ifs, line)) {
            if (line.size() < 66) { ++bad; continue; }
            Record r{0, 0, 0, 0, 0, 0};
            int empties = 0;
            for (int i = 0; i < 64; ++i) {
                const char c = line[i];
                if (c == 'X') r.player |= 1ULL << (63 - i);
                else if (c == 'O') r.opponent |= 1ULL << (63 - i);
                else ++empties;
            }
            const int score = std::stoi(line.substr(65));
            if (score < -64 || score > 64) { ++bad; continue; }
            r.score = (int8_t)score;
            r.empties = (uint8_t)empties;
            ofs.write(reinterpret_cast<const char*>(&r), sizeof(r));
            ++n;
        }
        std::cerr << in << ": " << n << " records, " << bad << " skipped" << std::endl;
    }
    return 0;
}

struct Job { fs::path file; int random; };

static int convert_transcripts(const std::vector<std::string> &roots, const fs::path &out, double target, int n_threads) {
    std::vector<Job> jobs;
    for (const auto &root : roots) for (const auto &d : fs::directory_iterator(root)) {
        if (!d.is_directory()) continue;
        int random;
        try { random = std::stoi(d.path().filename().string()); } catch (...) { continue; }
        for (const auto &f : fs::directory_iterator(d.path())) {
            if (f.path().extension() == ".txt") jobs.push_back({f.path(), random});
        }
    }
    std::sort(jobs.begin(), jobs.end(), [](const Job &a, const Job &b) { return a.file < b.file; });
    // Positions with e empties exist in the games whose random phase ended by then: about one folder
    // (1M games) per random-move count r <= 60 - e. Keep probability brings each e to about `target`.
    std::vector<int> folders;
    for (const auto &j : jobs) folders.push_back(j.random);
    std::sort(folders.begin(), folders.end());
    folders.erase(std::unique(folders.begin(), folders.end()), folders.end());
    double keep[61];
    for (int e = 0; e <= 60; ++e) {
        double games = 0;
        for (int r : folders) if (r <= 60 - e) games += 1e6;
        keep[e] = games > 0 ? std::min(1.0, target / games) : 0.0;
    }
    std::atomic<size_t> next{0};
    std::atomic<uint64_t> total{0}, games_total{0}, bad_total{0};
    std::mutex log_mutex;
    auto worker = [&]() {
        while (true) {
            const size_t k = next++;
            if (k >= jobs.size()) break;
            const Job &job = jobs[k];
            std::ifstream ifs(job.file);
            fs::path dst = out / ("tr_" + std::to_string(job.random) + "_" + job.file.stem().string() + ".bin");
            std::ofstream ofs(dst, std::ios::binary);
            std::string line;
            std::vector<Record> game;
            uint64_t n = 0, g = 0, bad = 0;
            while (std::getline(ifs, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
                if (line.size() % 2) { ++bad; continue; }
                uint64_t black = 0x0000000810000000ULL, white = 0x0000001008000000ULL;  // d5 e4 black, d4 e5 white
                bool black_to_move = true;
                game.clear();
                std::vector<int> movers;
                bool ok = true;
                for (size_t i = 0; i < line.size(); i += 2) {
                    const int x = line[i] - 'a', y = line[i + 1] - '1';
                    if (x < 0 || x > 7 || y < 0 || y > 7) { ok = false; break; }
                    uint64_t me = black_to_move ? black : white, op = black_to_move ? white : black;
                    if (!has_legal(me, op)) {
                        black_to_move = !black_to_move;
                        std::swap(me, op);
                    }
                    const int ply = (int)(i / 2);
                    if (ply >= job.random) {
                        const int empties = 60 - ply;
                        Record r{me, op, 0, (uint8_t)empties, 1, (uint8_t)job.random};
                        if (empties >= 1 && (hash32(me, op) & 0xFFFFFF) < keep[empties] * 16777216.0) {
                            game.push_back(r);
                            movers.push_back(black_to_move ? 0 : 1);
                        }
                    }
                    const uint64_t f = flips(me, op, x, y);
                    if (!f) { ok = false; break; }
                    me |= f | (1ULL << cell_bit(x, y));
                    op &= ~f;
                    if (black_to_move) { black = me; white = op; } else { white = me; black = op; }
                    black_to_move = !black_to_move;
                }
                if (!ok) { ++bad; continue; }
                const int nb = __builtin_popcountll(black), nw = __builtin_popcountll(white);
                const int empty = 64 - nb - nw;
                int diff = nb - nw;
                if (diff > 0) diff += empty; else if (diff < 0) diff -= empty;  // empties go to the winner
                for (size_t i = 0; i < game.size(); ++i) {
                    game[i].score = (int8_t)(movers[i] == 0 ? diff : -diff);
                }
                ofs.write(reinterpret_cast<const char*>(game.data()), game.size() * sizeof(Record));
                n += game.size();
                ++g;
            }
            total += n; games_total += g; bad_total += bad;
            if (k % 100 == 0) {
                std::lock_guard<std::mutex> lk(log_mutex);
                std::cerr << "[" << k + 1 << "/" << jobs.size() << "] " << job.file.string() << " games " << g
                          << " records " << n << " bad " << bad << std::endl;
            }
        }
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) threads.emplace_back(worker);
    for (auto &t : threads) t.join();
    std::cerr << "files " << jobs.size() << " games " << games_total << " records " << total << " bad " << bad_total << std::endl;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { std::cerr << "usage: convert_data boards|transcripts ... -o <out dir>" << std::endl; return 1; }
    const std::string mode = argv[1];
    std::vector<std::string> inputs;
    fs::path out;
    double target = 6e6;
    int threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) out = argv[++i];
        else if (a == "-target" && i + 1 < argc) target = std::stod(argv[++i]);
        else if (a == "-threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else inputs.push_back(a);
    }
    if (out.empty() || inputs.empty()) { std::cerr << "need inputs and -o <out dir>" << std::endl; return 1; }
    fs::create_directories(out);
    if (mode == "boards") return convert_boards(inputs, out);
    if (mode == "transcripts") {
        return convert_transcripts(inputs, out, target, threads);
    }
    std::cerr << "unknown mode " << mode << std::endl;
    return 1;
}
