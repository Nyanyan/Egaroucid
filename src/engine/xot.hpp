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

struct XotBoardKey {
    uint64_t player;
    uint64_t opponent;
};

inline bool xot_board_key_less(const XotBoardKey& lhs, const XotBoardKey& rhs) {
    return lhs.player < rhs.player || (lhs.player == rhs.player && lhs.opponent < rhs.opponent);
}

inline bool xot_board_key_equal(const XotBoardKey& lhs, const XotBoardKey& rhs) {
    return lhs.player == rhs.player && lhs.opponent == rhs.opponent;
}

inline std::vector<XotBoardKey>& xot_board_keys() {
    static std::vector<XotBoardKey> keys;
    return keys;
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

inline bool xot_transcript_to_board_key(const std::string& transcript, XotBoardKey* key) {
    Board board;
    board.reset();
    Flip flip;
    for (size_t i = 0; i + 1 < transcript.size() && board.n_discs() < XOT_START_N_DISCS; i += 2) {
        if (!is_coord_like_chars(transcript[i], transcript[i + 1])) {
            return false;
        }
        const int coord = get_coord_from_chars(transcript[i], transcript[i + 1]);
        if ((board.get_legal() & (1ULL << coord)) == 0) {
            return false;
        }
        calc_flip(&flip, &board, coord);
        board.move_board(&flip);
        if (!board.is_end() && board.get_legal() == 0ULL) {
            board.pass();
        }
    }
    if (board.n_discs() != XOT_START_N_DISCS) {
        return false;
    }
    board = representative_board(board);
    key->player = board.player;
    key->opponent = board.opponent;
    return true;
}

inline bool xot_init(const std::string& xot_dir, bool show_log = true) {
    std::string file_path = xot_dir;
    if (!file_path.empty() && file_path.back() != '/' && file_path.back() != '\\') {
        file_path += "/";
    }
    file_path += "openingslarge.txt";

    std::ifstream ifs(file_path);
    std::vector<XotBoardKey>& keys = xot_board_keys();
    keys.clear();
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
        XotBoardKey key{};
        if (xot_transcript_to_board_key(transcript, &key)) {
            keys.emplace_back(key);
            ++loaded;
        } else {
            ++skipped;
        }
    }

    std::sort(keys.begin(), keys.end(), xot_board_key_less);
    keys.erase(std::unique(keys.begin(), keys.end(), xot_board_key_equal), keys.end());
    if (show_log) {
        std::cerr << "loaded XOT openings " << keys.size() << " unique positions from "
            << loaded << " lines";
        if (skipped) {
            std::cerr << " (" << skipped << " skipped)";
        }
        std::cerr << std::endl;
    }
    return !keys.empty();
}

inline bool is_xot_board_key(Board board) {
    const std::vector<XotBoardKey>& keys = xot_board_keys();
    if (keys.empty()) {
        return false;
    }
    board = representative_board(board);
    const XotBoardKey target{ board.player, board.opponent };
    const auto it = std::lower_bound(keys.begin(), keys.end(), target, xot_board_key_less);
    return it != keys.end() && xot_board_key_equal(*it, target);
}
