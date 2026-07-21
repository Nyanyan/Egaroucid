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
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>
#include "board.hpp"
#include "common.hpp"
#include "search.hpp"
#include "util.hpp"

#define CONTEST_BOOK_EXTENSION ".egcb"
#define CONTEST_ROOT_TABLE_FILENAME "contest_root_table.egcb"
constexpr int CONTEST_ROOT_TABLE_DEFAULT_N_DISCS = 14;

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

inline std::filesystem::path contest_book_path_for_filename(
    const std::filesystem::path &root,
    const std::string &filename,
    bool *found
) {
    std::filesystem::path exact_path = root / filename;
    std::error_code ec;
    if (std::filesystem::exists(exact_path, ec)) {
        *found = true;
        return exact_path;
    }
    if (!std::filesystem::is_directory(root, ec)) {
        *found = false;
        return exact_path;
    }
    std::string prefixed_suffix = "_" + filename;
    for (const std::filesystem::directory_entry &entry: std::filesystem::directory_iterator(root, ec)) {
        if (ec) {
            break;
        }
        std::string candidate = entry.path().filename().string();
        if (candidate.size() >= prefixed_suffix.size() && candidate.ends_with(prefixed_suffix)) {
            *found = true;
            return entry.path();
        }
    }
    *found = false;
    return exact_path;
}

inline bool contest_book_representative_start(const std::string &initial_board, std::string *representative_start) {
    std::string compact = initial_board;
    compact.erase(std::remove_if(compact.begin(), compact.end(), [](unsigned char c) {
        return std::isspace(c);
    }), compact.end());
    if (compact.size() != HW2 + 1) {
        return false;
    }
    int player_to_move;
    if (is_black_like_char(compact[HW2])) {
        player_to_move = BLACK;
    } else if (is_white_like_char(compact[HW2])) {
        player_to_move = WHITE;
    } else {
        return false;
    }
    Board board;
    if (!board.from_str(initial_board)) {
        return false;
    }
    *representative_start = representative_board(board).to_str(player_to_move);
    return true;
}

inline std::filesystem::path contest_book_path_for_start(const std::string &dir, const std::string &initial_board) {
    std::filesystem::path root(dir);
    std::string filename = contest_book_sanitize_name(initial_board) + CONTEST_BOOK_EXTENSION;
    std::filesystem::path exact_path = root / filename;
    std::error_code ec;
    if (std::filesystem::exists(exact_path, ec)) {
        return exact_path;
    }

    bool found = false;
    std::string representative_start;
    if (!contest_book_representative_start(initial_board, &representative_start)) {
        return contest_book_path_for_filename(root, filename, &found);
    }
    std::string representative_filename = contest_book_sanitize_name(representative_start) + CONTEST_BOOK_EXTENSION;
    if (representative_filename != filename) {
        std::filesystem::path representative_path = contest_book_path_for_filename(
            root,
            representative_filename,
            &found
        );
        if (found) {
            return representative_path;
        }
    }
    return contest_book_path_for_filename(root, filename, &found);
}

class Contest_book {
    private:
        bool loaded;
        std::string source_file;
        std::unordered_map<Board, Contest_book_entry, Contest_book_hash> entries;

        bool parse_line(const std::string &line, bool *duplicate_representative) {
            *duplicate_representative = false;
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
            int symmetry_idx;
            Board representative = representative_board(board, &symmetry_idx);
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
                    policy = convert_coord_to_representative_board(policy, symmetry_idx);
                    entry.moves.emplace_back(policy, score);
                }
            }
            if (entry.moves.empty()) {
                return false;
            }
            if (entries.find(representative) != entries.end()) {
                *duplicate_representative = true;
                return false;
            }
            entries.emplace(representative, std::move(entry));
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
            uint64_t n_duplicate_representatives = 0;
            while (std::getline(ifs, line)) {
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                bool duplicate_representative = false;
                if (parse_line(line, &duplicate_representative)) {
                    ++n_loaded;
                } else if (duplicate_representative) {
                    ++n_duplicate_representatives;
                }
            }
            if (n_duplicate_representatives > 0) {
                if (show_log) {
                    std::cerr << "[WARNING] contest book invalid: " << n_duplicate_representatives
                              << " duplicate representative board line(s) in " << file << std::endl;
                }
                clear();
                return false;
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

        bool all_entries_have_n_discs(int n_discs) const {
            for (const auto &entry: entries) {
                if (entry.first.n_discs() != n_discs) {
                    return false;
                }
            }
            return true;
        }

        std::string source() const {
            return source_file;
        }

        bool get(const Board &board, Contest_book_entry *entry) const {
            if (!loaded) {
                return false;
            }
            int symmetry_idx;
            Board representative = representative_board(board, &symmetry_idx);
            auto it = entries.find(representative);
            if (it == entries.end()) {
                return false;
            }
            *entry = it->second;
            for (Contest_book_move &move: entry->moves) {
                if (is_valid_policy(move.policy)) {
                    move.policy = convert_coord_from_representative_board(move.policy, symmetry_idx);
                }
            }
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

/*
    A contest-root table deliberately reuses Contest_book's canonical-board
    lookup and policy conversion, but has a separate, strict file header.  It
    is a single table shared by every GGS start rather than one deep DAG per
    start.  Individual contest books remain preferred where they exist.
*/
class Contest_root_table {
    private:
        Contest_book book;
        bool loaded;
        int n_discs;
        uint64_t expected_entries;
        std::string source_file;

        static bool parse_header_int(const std::string &line, const std::string &prefix, int *value) {
            if (!line.starts_with(prefix)) {
                return false;
            }
            try {
                size_t consumed = 0;
                const int parsed = std::stoi(line.substr(prefix.size()), &consumed);
                if (consumed != line.size() - prefix.size()) {
                    return false;
                }
                *value = parsed;
            } catch (const std::exception&) {
                return false;
            }
            return true;
        }

        static bool parse_header_uint64(const std::string &line, const std::string &prefix, uint64_t *value) {
            if (!line.starts_with(prefix)) {
                return false;
            }
            const std::string value_text = line.substr(prefix.size());
            if (value_text.empty() || value_text[0] == '-' || value_text[0] == '+') {
                return false;
            }
            try {
                size_t consumed = 0;
                const uint64_t parsed = std::stoull(value_text, &consumed);
                if (consumed != value_text.size()) {
                    return false;
                }
                *value = parsed;
            } catch (const std::exception&) {
                return false;
            }
            return true;
        }

    public:
        Contest_root_table()
            : loaded(false), n_discs(CONTEST_ROOT_TABLE_DEFAULT_N_DISCS), expected_entries(0) {}

        void clear() {
            book.clear();
            loaded = false;
            n_discs = CONTEST_ROOT_TABLE_DEFAULT_N_DISCS;
            expected_entries = 0;
            source_file.clear();
        }

        bool init(const std::string &dir, bool show_log) {
            clear();
            const std::filesystem::path path = std::filesystem::path(dir) / CONTEST_ROOT_TABLE_FILENAME;
            std::ifstream ifs(path);
            if (!ifs) {
                if (show_log) {
                    std::cerr << "contest root table not found: " << path.string() << std::endl;
                }
                return false;
            }
            std::string format;
            std::string root_discs;
            std::string entries;
            if (
                !std::getline(ifs, format) ||
                !std::getline(ifs, root_discs) ||
                !std::getline(ifs, entries) ||
                format != "# contest_root_table_v1" ||
                !parse_header_int(root_discs, "# root_discs ", &n_discs) ||
                !parse_header_uint64(entries, "# entries ", &expected_entries) ||
                n_discs < 4 || n_discs > HW2 ||
                expected_entries == 0
            ) {
                if (show_log) {
                    std::cerr << "[WARNING] contest root table invalid header: " << path.string() << std::endl;
                }
                clear();
                return false;
            }
            if (!book.init(path.string(), show_log)) {
                clear();
                return false;
            }
            if (book.size() != expected_entries) {
                if (show_log) {
                    std::cerr << "[WARNING] contest root table entry count mismatch: expected "
                              << expected_entries << ", loaded " << book.size() << " from " << path.string() << std::endl;
                }
                clear();
                return false;
            }
            if (!book.all_entries_have_n_discs(n_discs)) {
                if (show_log) {
                    std::cerr << "[WARNING] contest root table contains a non-root row: "
                              << path.string() << std::endl;
                }
                clear();
                return false;
            }
            loaded = true;
            source_file = path.string();
            if (show_log) {
                std::cerr << "contest root table loaded " << expected_entries
                          << " roots at " << n_discs << " discs from " << source_file << std::endl;
            }
            return true;
        }

        bool is_loaded() const {
            return loaded;
        }

        uint64_t size() const {
            return book.size();
        }

        int root_n_discs() const {
            return n_discs;
        }

        std::string source() const {
            return source_file;
        }

        bool get(const Board &board, Contest_book_entry *entry) const {
            return loaded && board.n_discs() == n_discs && book.get(board, entry);
        }

        bool get_search_result(const Board &board, Search_result *result) const {
            return loaded && board.n_discs() == n_discs && book.get_search_result(board, result);
        }
};
