/*
    Egaroucid Project

    @file contest_book.hpp
        Lightweight opening database for contest starts
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "board.hpp"
#include "common.hpp"
#include "search.hpp"

#define CONTEST_BOOK_EXTENSION ".egcb"

struct Contest_book_hash {
    size_t operator()(const Board &board) const {
        size_t h0 = std::hash<uint64_t>{}(board.player);
        size_t h1 = std::hash<uint64_t>{}(board.opponent);
        return h0 ^ (h1 + 0x9e3779b97f4a7c15ULL + (h0 << 6) + (h0 >> 2));
    }
};

struct Contest_book_move {
    int policy;
    int value;

    Contest_book_move()
        : policy(MOVE_UNDEFINED), value(SCORE_UNDEFINED) {}

    Contest_book_move(int policy_, int value_)
        : policy(policy_), value(value_) {}
};

struct Contest_book_entry {
    int value;
    std::vector<Contest_book_move> moves;

    Contest_book_entry()
        : value(SCORE_UNDEFINED) {}
};

inline std::string contest_book_sanitize_name(std::string name) {
    std::string res;
    res.reserve(name.size());
    for (char c: name) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) || c == '-' || c == '_') {
            res += c;
        } else {
            res += '_';
        }
    }
    return res;
}

inline std::filesystem::path contest_book_path_for_start(const std::string &dir, const std::string &initial_board) {
    return std::filesystem::path(dir) / (contest_book_sanitize_name(initial_board) + CONTEST_BOOK_EXTENSION);
}

class Contest_book {
    private:
        bool loaded;
        std::string source_file;
        std::unordered_map<Board, Contest_book_entry, Contest_book_hash> entries;

        bool parse_line(const std::string &line) {
            std::istringstream iss(line);
            std::string board_cells;
            std::string side;
            int value;
            if (!(iss >> board_cells >> side >> value)) {
                return false;
            }
            if (board_cells.size() != HW2 || side.size() != 1) {
                return false;
            }
            Board board(board_cells + " " + side);
            Contest_book_entry entry;
            entry.value = value;

            std::string move_token;
            while (iss >> move_token) {
                size_t sep = move_token.find(':');
                if (sep == std::string::npos || sep < 2) {
                    continue;
                }
                int policy = get_coord_from_chars(move_token[0], move_token[1]);
                int score = SCORE_UNDEFINED;
                try {
                    score = std::stoi(move_token.substr(sep + 1));
                } catch (const std::exception&) {
                    continue;
                }
                if (is_valid_policy(policy)) {
                    entry.moves.emplace_back(policy, score);
                }
            }
            if (entry.moves.empty()) {
                return false;
            }
            entries[board] = entry;
            return true;
        }

    public:
        Contest_book()
            : loaded(false) {}

        void clear() {
            loaded = false;
            source_file.clear();
            entries.clear();
        }

        bool init(const std::string &file, bool show_log) {
            clear();
            std::ifstream ifs(file);
            if (!ifs) {
                if (show_log) {
                    std::cerr << "contest book not found: " << file << std::endl;
                }
                return false;
            }
            std::string line;
            uint64_t n_loaded = 0;
            while (std::getline(ifs, line)) {
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                if (parse_line(line)) {
                    ++n_loaded;
                }
            }
            loaded = n_loaded > 0;
            source_file = file;
            if (show_log) {
                std::cerr << "contest book loaded " << n_loaded << " boards from " << file << std::endl;
            }
            return loaded;
        }

        bool is_loaded() const {
            return loaded;
        }

        uint64_t size() const {
            return entries.size();
        }

        std::string source() const {
            return source_file;
        }

        bool get(const Board &board, Contest_book_entry *entry) const {
            if (!loaded) {
                return false;
            }
            auto it = entries.find(board);
            if (it == entries.end()) {
                return false;
            }
            *entry = it->second;
            return true;
        }

        bool get_search_result(const Board &board, Search_result *result) const {
            Contest_book_entry entry;
            if (!get(board, &entry)) {
                return false;
            }
            uint64_t legal = board.get_legal();
            int best_policy = MOVE_UNDEFINED;
            int best_value = -SCORE_MAX;
            for (const Contest_book_move &move: entry.moves) {
                if (is_valid_policy(move.policy) && (legal & (1ULL << move.policy)) && move.value > best_value) {
                    best_policy = move.policy;
                    best_value = move.value;
                }
            }
            if (!is_valid_policy(best_policy)) {
                return false;
            }
            result->policy = best_policy;
            result->value = entry.value;
            result->depth = SEARCH_BOOK;
            result->time = 0;
            result->nodes = 0;
            result->clog_time = 0;
            result->clog_nodes = 0;
            result->nps = 0;
            result->is_end_search = false;
            result->probability = -1;
            return true;
        }
};
