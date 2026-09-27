/*
    Egaroucid Project

    @file xot.hpp
        XOT opening identification
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "util.hpp"

constexpr int XOT_START_N_DISCS = 12;
constexpr int XOT_OPENING_N_MOVES = XOT_START_N_DISCS - 4;

inline std::vector<std::string>& xot_opening_transcripts() {
    static std::vector<std::string> transcripts;
    return transcripts;
}

inline std::vector<std::string>& xot_opening_sequence_keys() {
    static std::vector<std::string> keys;
    return keys;
}

inline std::string normalize_xot_opening_moves(const std::vector<int>& moves) {
    if ((int)moves.size() != XOT_OPENING_N_MOVES) {
        return "";
    }
    std::string normalized;
    for (int symmetry_idx = 0; symmetry_idx < 8; ++symmetry_idx) {
        std::string candidate;
        candidate.reserve(moves.size() * 2);
        for (int move : moves) {
            if (!is_valid_policy(move)) {
                return "";
            }
            candidate += idx_to_coord(convert_coord_to_representative_board(move, symmetry_idx));
        }
        if (normalized.empty() || candidate < normalized) {
            normalized = candidate;
        }
    }
    return normalized;
}

inline bool is_xot_opening_moves(const std::vector<int>& moves) {
    const std::string key = normalize_xot_opening_moves(moves);
    const std::vector<std::string>& keys = xot_opening_sequence_keys();
    return !key.empty() && std::binary_search(keys.begin(), keys.end(), key);
}

inline std::string normalize_xot_transcript_line(std::string line) {
    const size_t comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
        line.erase(comment_pos);
    }
    line.erase(std::remove_if(line.begin(), line.end(), [](unsigned char c) {
        return std::isspace(c);
    }), line.end());
    return line;
}

inline bool xot_transcript_to_board(const std::string& transcript, Board* board, std::vector<int>* moves = nullptr) {
    Board local_board;
    local_board.reset();
    Flip flip;
    if (moves != nullptr) {
        moves->clear();
    }
    for (size_t i = 0; i + 1 < transcript.size() && local_board.n_discs() < XOT_START_N_DISCS; i += 2) {
        if (!is_coord_like_chars(transcript[i], transcript[i + 1])) {
            return false;
        }
        const int coord = get_coord_from_chars(transcript[i], transcript[i + 1]);
        if ((local_board.get_legal() & (1ULL << coord)) == 0) {
            return false;
        }
        if (moves != nullptr) {
            moves->emplace_back(coord);
        }
        calc_flip(&flip, &local_board, coord);
        local_board.move_board(&flip);
        if (!local_board.is_end() && local_board.get_legal() == 0ULL) {
            local_board.pass();
        }
    }
    if (local_board.n_discs() != XOT_START_N_DISCS) {
        if (moves != nullptr) {
            moves->clear();
        }
        return false;
    }
    *board = local_board;
    return true;
}

inline std::vector<int> get_random_xot_moves() {
    const std::vector<std::string>& transcripts = xot_opening_transcripts();
    if (transcripts.empty()) {
        return {};
    }
    const std::string& transcript = transcripts[myrandrange(0, (int)transcripts.size())];
    Board board;
    std::vector<int> moves;
    if (!xot_transcript_to_board(transcript, &board, &moves)) {
        return {};
    }
    return moves;
}

inline bool xot_init(const std::string& xot_dir, bool show_log = true) {
    std::string file_path = xot_dir;
    if (!file_path.empty() && file_path.back() != '/' && file_path.back() != '\\') {
        file_path += "/";
    }
    file_path += "openingslarge.txt";

    std::ifstream ifs(file_path);
    std::vector<std::string>& transcripts = xot_opening_transcripts();
    std::vector<std::string>& sequence_keys = xot_opening_sequence_keys();
    transcripts.clear();
    sequence_keys.clear();
    if (!ifs) {
        if (show_log) {
            std::cerr << "[WARNING] XOT openings file not found: " << file_path << std::endl;
        }
        return false;
    }

    std::string line;
    int loaded = 0;
    int skipped = 0;
    while (std::getline(ifs, line)) {
        const std::string transcript = normalize_xot_transcript_line(line);
        if (transcript.empty()) {
            continue;
        }
        Board board;
        std::vector<int> moves;
        if (xot_transcript_to_board(transcript, &board, &moves)) {
            transcripts.emplace_back(transcript);
            sequence_keys.emplace_back(normalize_xot_opening_moves(moves));
            ++loaded;
        } else {
            ++skipped;
        }
    }

    std::sort(sequence_keys.begin(), sequence_keys.end());
    sequence_keys.erase(std::unique(sequence_keys.begin(), sequence_keys.end()), sequence_keys.end());
    if (show_log) {
        std::cerr << "loaded XOT openings " << sequence_keys.size() << " unique sequences from "
            << loaded << " lines";
        if (skipped) {
            std::cerr << " (" << skipped << " skipped)";
        }
        std::cerr << std::endl;
    }
    return !sequence_keys.empty();
}
