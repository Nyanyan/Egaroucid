/*
    Egaroucid Project

    @file eval_record_diff.cpp
        Replay transcripts and compare static evaluation files
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "./../../../engine/evaluate.hpp"

struct PositionSample {
    Board board;
    int record_idx;
    int ply;
    int phase;
    std::string prefix;
};

struct RecordSummary {
    int record_idx;
    std::string record;
    int n_positions;
    int final_score_black;
};

struct ScoredSample {
    int abs_delta;
    int delta;
    int abs_raw_delta;
    int raw_delta;
    int normal_eval;
    int candidate_eval;
    int normal_raw_eval;
    int candidate_raw_eval;
    int record_idx;
    int ply;
    int phase;
    std::string prefix;
};

struct EvalScoreSet {
    std::vector<int> rounded;
    std::vector<int> raw;
};

bool is_coord_token(const std::string &token) {
    if (token.size() < 2 || token.size() % 2 != 0) {
        return false;
    }
    for (size_t i = 0; i < token.size(); i += 2) {
        const char x = (char)(token[i] | 0x20);
        const char y = token[i + 1];
        if (x == 'p' && y == 's') {
            continue;
        }
        if (x < 'a' || x > 'h' || y < '1' || y > '8') {
            return false;
        }
    }
    return true;
}

std::string extract_record_from_line(const std::string &line) {
    std::istringstream iss(line);
    std::string token;
    while (iss >> token) {
        if (token == "record") {
            std::string record;
            if (iss >> record && is_coord_token(record)) {
                return record;
            }
            return "";
        }
    }
    std::string trimmed;
    for (char c: line) {
        if (!std::isspace((unsigned char)c)) {
            trimmed.push_back(c);
        }
    }
    if (is_coord_token(trimmed)) {
        return trimmed;
    }
    return "";
}

bool replay_record(
    const std::string &record,
    const int record_idx,
    std::vector<PositionSample> *positions,
    RecordSummary *summary
) {
    Board board;
    board.reset();
    int side_to_move = BLACK;
    std::string prefix;
    Flip flip;
    const int start_positions = (int)positions->size();
    for (int i = 0; i < (int)record.size(); i += 2) {
        if (board.get_legal() == 0) {
            board.pass();
            side_to_move ^= 1;
        }
        const char x_char = (char)(record[i] | 0x20);
        const char y_char = record[i + 1];
        if (x_char == 'p' && y_char == 's') {
            if (board.get_legal() != 0) {
                std::cerr << "[WARN] explicit pass with legal moves skipped record " << record_idx << " ply " << (i / 2) << "\n";
                return false;
            }
            board.pass();
            side_to_move ^= 1;
            prefix += "ps";
            continue;
        }
        const int x = (int)(x_char - 'a');
        const int y = (int)(y_char - '1');
        if (x < 0 || x >= HW || y < 0 || y >= HW) {
            std::cerr << "[WARN] invalid coord skipped record " << record_idx << " ply " << (i / 2) << "\n";
            return false;
        }
        PositionSample sample;
        sample.board = board.copy();
        sample.record_idx = record_idx;
        sample.ply = i / 2;
        sample.phase = std::min(N_PHASES - 1, (sample.board.n_discs() - 4) / PHASE_N_DISCS);
        sample.prefix = prefix;
        positions->emplace_back(sample);

        const int policy = HW2_M1 - (y * HW + x);
        calc_flip(&flip, &board, policy);
        if (flip.flip == 0ULL) {
            std::cerr << "[WARN] illegal move skipped record " << record_idx << " ply " << (i / 2) << " coord " << x_char << y_char << "\n";
            positions->resize(start_positions);
            return false;
        }
        board.move_board(&flip);
        side_to_move ^= 1;
        prefix.push_back(x_char);
        prefix.push_back(y_char);
    }
    if (board.get_legal() == 0) {
        board.pass();
        side_to_move ^= 1;
    }
    int final_score_black = board.score_player();
    if (side_to_move != BLACK) {
        final_score_black = -final_score_black;
    }
    summary->record_idx = record_idx;
    summary->record = record;
    summary->n_positions = (int)positions->size() - start_positions;
    summary->final_score_black = final_score_black;
    return true;
}

bool load_records(
    const std::string &records_file,
    std::vector<PositionSample> *positions,
    std::vector<RecordSummary> *summaries
) {
    std::ifstream in(records_file);
    if (!in) {
        std::cerr << "[ERROR] can't open records file " << records_file << "\n";
        return false;
    }
    std::string line;
    int line_idx = 0;
    int record_idx = 0;
    while (std::getline(in, line)) {
        ++line_idx;
        const std::string record = extract_record_from_line(line);
        if (record.empty()) {
            continue;
        }
        RecordSummary summary;
        if (replay_record(record, record_idx, positions, &summary)) {
            summaries->emplace_back(summary);
            ++record_idx;
        } else {
            std::cerr << "[WARN] skipped line " << line_idx << "\n";
        }
    }
    return !summaries->empty();
}

int mid_evaluate_raw_local(Board *board) {
    Search search(board);
#if USE_SIMD_EVALUATION
    calc_eval_features(&(search.board), &(search.eval));
    const int phase_idx = search.phase();
    const int num0 = pop_count_ull(search.board.player);
    return calc_pattern(phase_idx, &search.eval.features[search.eval.feature_idx])
        + eval_num_arr[phase_idx][num0]
        + eval_fm_calc(phase_idx, &search.eval.features[search.eval.feature_idx]);
#else
    calc_eval_features(board, &search.eval);
    const int phase_idx = search.phase();
    const int num0 = pop_count_ull(search.board.player);
    return calc_pattern(phase_idx, &search.eval)
        + eval_num_arr[phase_idx][num0]
        + eval_fm_calc(phase_idx, &search.eval);
#endif
}

EvalScoreSet evaluate_positions(
    const std::string &eval_file,
    const std::string &mo_file,
    const std::vector<PositionSample> &positions
) {
    if (!evaluate_init(eval_file, mo_file, false)) {
        return {};
    }
    EvalScoreSet scores;
    scores.rounded.reserve(positions.size());
    scores.raw.reserve(positions.size());
    for (const PositionSample &sample: positions) {
        Board board = sample.board.copy();
        scores.rounded.emplace_back(mid_evaluate(&board));
        board = sample.board.copy();
        scores.raw.emplace_back(mid_evaluate_raw_local(&board));
    }
    return scores;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr
            << "usage: eval_record_diff [records.txt] [normal.egev2] [candidate.egevfm] [mo.egev] [top_n=20]\n"
            << "records.txt may contain bare transcripts or battle_fixed_time.py output lines with 'record ...'\n";
        return 1;
    }
    const std::string records_file = argv[1];
    const std::string normal_eval_file = argv[2];
    const std::string candidate_eval_file = argv[3];
    const std::string mo_file = argv[4];
    const int top_n = argc >= 6 ? std::max(0, std::atoi(argv[5])) : 20;

    bit_init();
    mobility_init();
    flip_init();

    std::vector<PositionSample> positions;
    std::vector<RecordSummary> summaries;
    if (!load_records(records_file, &positions, &summaries)) {
        std::cerr << "[ERROR] no valid records loaded\n";
        return 1;
    }

    const EvalScoreSet normal_scores = evaluate_positions(normal_eval_file, mo_file, positions);
    if (normal_scores.rounded.size() != positions.size() || normal_scores.raw.size() != positions.size()) {
        std::cerr << "[ERROR] normal evaluation failed\n";
        return 1;
    }
    const EvalScoreSet candidate_scores = evaluate_positions(candidate_eval_file, mo_file, positions);
    if (candidate_scores.rounded.size() != positions.size() || candidate_scores.raw.size() != positions.size()) {
        std::cerr << "[ERROR] candidate evaluation failed\n";
        return 1;
    }

    std::array<int64_t, N_PHASES> phase_sum_delta = {};
    std::array<int64_t, N_PHASES> phase_sum_abs_delta = {};
    std::array<int64_t, N_PHASES> phase_sum_raw_delta = {};
    std::array<int64_t, N_PHASES> phase_sum_abs_raw_delta = {};
    std::array<int, N_PHASES> phase_counts = {};
    std::vector<ScoredSample> top_samples;
    top_samples.reserve(positions.size());

    int64_t sum_delta = 0;
    int64_t sum_abs_delta = 0;
    int64_t sum_sq_delta = 0;
    int64_t sum_raw_delta = 0;
    int64_t sum_abs_raw_delta = 0;
    int64_t sum_sq_raw_delta = 0;
    int max_abs_delta = 0;
    int max_abs_raw_delta = 0;
    for (size_t i = 0; i < positions.size(); ++i) {
        const int delta = candidate_scores.rounded[i] - normal_scores.rounded[i];
        const int raw_delta = candidate_scores.raw[i] - normal_scores.raw[i];
        const int abs_delta = std::abs(delta);
        const int abs_raw_delta = std::abs(raw_delta);
        sum_delta += delta;
        sum_abs_delta += abs_delta;
        sum_sq_delta += (int64_t)delta * delta;
        sum_raw_delta += raw_delta;
        sum_abs_raw_delta += abs_raw_delta;
        sum_sq_raw_delta += (int64_t)raw_delta * raw_delta;
        max_abs_delta = std::max(max_abs_delta, abs_delta);
        max_abs_raw_delta = std::max(max_abs_raw_delta, abs_raw_delta);
        const int phase = positions[i].phase;
        phase_sum_delta[phase] += delta;
        phase_sum_abs_delta[phase] += abs_delta;
        phase_sum_raw_delta[phase] += raw_delta;
        phase_sum_abs_raw_delta[phase] += abs_raw_delta;
        ++phase_counts[phase];
        top_samples.push_back({
            abs_delta,
            delta,
            abs_raw_delta,
            raw_delta,
            normal_scores.rounded[i],
            candidate_scores.rounded[i],
            normal_scores.raw[i],
            candidate_scores.raw[i],
            positions[i].record_idx,
            positions[i].ply,
            phase,
            positions[i].prefix
        });
    }

    std::sort(top_samples.begin(), top_samples.end(), [](const ScoredSample &a, const ScoredSample &b) {
        if (a.abs_raw_delta != b.abs_raw_delta) {
            return a.abs_raw_delta > b.abs_raw_delta;
        }
        if (a.abs_delta != b.abs_delta) {
            return a.abs_delta > b.abs_delta;
        }
        if (a.record_idx != b.record_idx) {
            return a.record_idx < b.record_idx;
        }
        return a.ply < b.ply;
    });

    const double n = (double)positions.size();
    const double mean_delta = (double)sum_delta / n;
    const double mean_abs_delta = (double)sum_abs_delta / n;
    const double rms_delta = std::sqrt((double)sum_sq_delta / n);
    const double mean_raw_delta = (double)sum_raw_delta / n;
    const double mean_abs_raw_delta = (double)sum_abs_raw_delta / n;
    const double rms_raw_delta = std::sqrt((double)sum_sq_raw_delta / n);

    std::cout << "records " << summaries.size() << "\n";
    std::cout << "positions " << positions.size() << "\n";
    std::cout << "normal_eval " << normal_eval_file << "\n";
    std::cout << "candidate_eval " << candidate_eval_file << "\n";
    std::cout << "mean_delta " << mean_delta << "\n";
    std::cout << "mean_abs_delta " << mean_abs_delta << "\n";
    std::cout << "rms_delta " << rms_delta << "\n";
    std::cout << "max_abs_delta " << max_abs_delta << "\n";
    std::cout << "mean_raw_delta " << mean_raw_delta << "\n";
    std::cout << "mean_abs_raw_delta " << mean_abs_raw_delta << "\n";
    std::cout << "rms_raw_delta " << rms_raw_delta << "\n";
    std::cout << "max_abs_raw_delta " << max_abs_raw_delta << "\n";

    std::cout << "phase_summary phase count mean_delta mean_abs_delta mean_raw_delta mean_abs_raw_delta\n";
    for (int phase = 0; phase < N_PHASES; ++phase) {
        if (phase_counts[phase] == 0) {
            continue;
        }
        std::cout
            << "phase_summary " << phase
            << " " << phase_counts[phase]
            << " " << ((double)phase_sum_delta[phase] / phase_counts[phase])
            << " " << ((double)phase_sum_abs_delta[phase] / phase_counts[phase])
            << " " << ((double)phase_sum_raw_delta[phase] / phase_counts[phase])
            << " " << ((double)phase_sum_abs_raw_delta[phase] / phase_counts[phase])
            << "\n";
    }

    std::cout << "record_summary record_idx n_positions final_score_black record\n";
    for (const RecordSummary &summary: summaries) {
        std::cout
            << "record_summary " << summary.record_idx
            << " " << summary.n_positions
            << " " << summary.final_score_black
            << " " << summary.record
            << "\n";
    }

    std::cout << "top_abs_delta abs_raw_delta raw_delta abs_delta delta normal candidate normal_raw candidate_raw record_idx ply phase prefix\n";
    for (int i = 0; i < std::min<int>(top_n, (int)top_samples.size()); ++i) {
        const ScoredSample &sample = top_samples[i];
        std::cout
            << "top_abs_delta " << sample.abs_raw_delta
            << " " << sample.raw_delta
            << " " << sample.abs_delta
            << " " << sample.delta
            << " " << sample.normal_eval
            << " " << sample.candidate_eval
            << " " << sample.normal_raw_eval
            << " " << sample.candidate_raw_eval
            << " " << sample.record_idx
            << " " << sample.ply
            << " " << sample.phase
            << " " << (sample.prefix.empty() ? "-" : sample.prefix)
            << "\n";
    }

    return 0;
}
