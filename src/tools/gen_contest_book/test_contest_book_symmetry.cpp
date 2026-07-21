/*
    Egaroucid Project

    Contest-book symmetry regression test.

    Example:
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 test_contest_book_symmetry.cpp -o test_contest_book_symmetry.exe
*/

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "../../engine/engine_all.hpp"
#include "../../engine/contest_book.hpp"

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class Temporary_directory {
    private:
        std::filesystem::path path_;

    public:
        Temporary_directory() {
            uint64_t suffix = static_cast<uint64_t>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count()
            );
            path_ = std::filesystem::temp_directory_path() /
                ("egaroucid_contest_book_symmetry_" + std::to_string(suffix));
            require(std::filesystem::create_directory(path_), "could not create temporary directory");
        }

        ~Temporary_directory() {
            std::error_code ec;
            std::filesystem::remove_all(path_, ec);
        }

        const std::filesystem::path &path() const {
            return path_;
        }
};

std::vector<Board> symmetry_boards(Board board) {
    Board black_line = board;
    black_line.board_black_line_mirror();
    Board vertical = board.get_vertical_mirror();
    Board black_line_vertical = black_line.get_vertical_mirror();
    Board horizontal = board;
    horizontal.board_horizontal_mirror();
    Board black_line_horizontal = black_line;
    black_line_horizontal.board_horizontal_mirror();
    Board horizontal_vertical = horizontal;
    horizontal_vertical.board_vertical_mirror();
    Board black_line_horizontal_vertical = black_line_horizontal;
    black_line_horizontal_vertical.board_vertical_mirror();
    return {
        board,
        black_line,
        vertical,
        black_line_vertical,
        horizontal,
        black_line_horizontal,
        horizontal_vertical,
        black_line_horizontal_vertical,
    };
}

int first_legal_policy(const Board &board) {
    uint64_t legal = board.get_legal();
    for (int policy = 0; policy < HW2; ++policy) {
        if (legal & (1ULL << policy)) {
            return policy;
        }
    }
    return MOVE_UNDEFINED;
}

void require_eight_distinct_symmetries(const std::vector<Board> &boards, const std::string &label) {
    require(boards.size() == 8, label + " did not produce eight transformations");
    std::unordered_set<Board, Contest_book_hash> unique(boards.begin(), boards.end());
    require(unique.size() == 8, label + " test board is not asymmetric");
}

void check_lookup(
    const Contest_book &book,
    const Board &board,
    const Board &expected_representative,
    int representative_policy,
    int expected_value,
    const std::string &label
) {
    int symmetry_idx;
    Board representative = representative_board(board, &symmetry_idx);
    require(representative == expected_representative, label + " has an unexpected representative");
    int expected_policy = convert_coord_from_representative_board(representative_policy, symmetry_idx);

    Contest_book_entry entry;
    require(book.get(board, &entry), label + " was not found");
    require(entry.value == expected_value, label + " returned a different value");
    require(entry.moves.size() == 1, label + " returned a different move count");
    require(entry.moves[0].policy == expected_policy, label + " returned an untransformed policy");
    require(entry.moves[0].value == expected_value, label + " returned a different move value");
    require(board.get_legal() & (1ULL << entry.moves[0].policy), label + " returned an illegal policy");

    Search_result result;
    require(book.get_search_result(board, &result), label + " did not return a search result");
    require(result.policy == expected_policy, label + " search result policy differs from get()");
    require(result.value == expected_value, label + " search result value differs from get()");
}

} // namespace

int main() {
    try {
        bit_init();
        mobility_init();
        flip_init();

        Board root("------------------XXXO----OOXO-----OOX----O-OO------------------ X");
        Board root_representative = representative_board(root);
        require(
            root_representative.to_str(BLACK) ==
                "------------------O-OO-----OOX----OOXO----XXXO------------------ X",
            "root representative differs from records321_14_random_setup"
        );
        std::vector<Board> root_symmetries = symmetry_boards(root_representative);
        require_eight_distinct_symmetries(root_symmetries, "root");

        int root_policy = first_legal_policy(root_representative);
        require(is_valid_policy(root_policy), "root has no legal move");

        Board child = root_representative;
        Flip flip;
        calc_flip(&flip, &child, root_policy);
        child.move_board(&flip);
        int child_policy = first_legal_policy(child);
        require(is_valid_policy(child_policy), "child has no legal move");

        int child_to_representative_idx;
        Board child_representative = representative_board(child, &child_to_representative_idx);
        int child_representative_policy = convert_coord_to_representative_board(
            child_policy,
            child_to_representative_idx
        );
        std::vector<Board> child_symmetries = symmetry_boards(child_representative);
        require_eight_distinct_symmetries(child_symmetries, "child");

        // Store the child in a non-representative orientation. This exercises
        // policy conversion during parsing as well as during lookup.
        Board stored_child = child_symmetries[3];
        int stored_child_to_representative_idx;
        require(
            representative_board(stored_child, &stored_child_to_representative_idx) == child_representative,
            "stored child representative differs"
        );
        int stored_child_policy = convert_coord_from_representative_board(
            child_representative_policy,
            stored_child_to_representative_idx
        );
        require(
            stored_child.get_legal() & (1ULL << stored_child_policy),
            "stored child policy is illegal"
        );

        Temporary_directory temporary_directory;
        std::string canonical_start = root_representative.to_str(BLACK);
        std::string filename = "0000123_" + contest_book_sanitize_name(canonical_start) + CONTEST_BOOK_EXTENSION;
        std::filesystem::path book_path = temporary_directory.path() / filename;
        {
            std::ofstream ofs(book_path);
            require(static_cast<bool>(ofs), "could not create test book");
            ofs << "# contest_book_v1\n";
            ofs << "# initial " << canonical_start << '\n';
            ofs << root_representative.to_str() << " 8 " << idx_to_coord(root_policy) << ":8\n";
            ofs << stored_child.to_str() << " -4 " << idx_to_coord(stored_child_policy) << ":-4\n";
        }

        // Canonical orientation keeps the pre-symmetry path behavior.
        require(
            contest_book_path_for_start(temporary_directory.path().string(), canonical_start) == book_path,
            "canonical start did not resolve its prefixed book"
        );
        for (size_t i = 0; i < root_symmetries.size(); ++i) {
            require(
                contest_book_path_for_start(
                    temporary_directory.path().string(),
                    root_symmetries[i].to_str(BLACK)
                ) == book_path,
                "root symmetry " + std::to_string(i) + " did not resolve the canonical book"
            );
        }

        Contest_book book;
        require(book.init(book_path.string(), false), "test book did not load");
        require(book.size() == 2, "test book did not register two representative boards");

        for (size_t i = 0; i < root_symmetries.size(); ++i) {
            check_lookup(
                book,
                root_symmetries[i],
                root_representative,
                root_policy,
                8,
                "root symmetry " + std::to_string(i)
            );
        }
        for (size_t i = 0; i < child_symmetries.size(); ++i) {
            check_lookup(
                book,
                child_symmetries[i],
                child_representative,
                child_representative_policy,
                -4,
                "child symmetry " + std::to_string(i)
            );
        }

        Board duplicate_orientation = root_symmetries[1];
        int duplicate_to_representative_idx;
        require(
            representative_board(duplicate_orientation, &duplicate_to_representative_idx) == root_representative,
            "duplicate test orientation has a different representative"
        );
        int duplicate_policy = convert_coord_from_representative_board(
            root_policy,
            duplicate_to_representative_idx
        );
        std::filesystem::path duplicate_path = temporary_directory.path() / "duplicate.egcb";
        {
            std::ofstream ofs(duplicate_path);
            require(static_cast<bool>(ofs), "could not create duplicate test book");
            ofs << "# contest_book_v1\n";
            ofs << root_representative.to_str() << " 8 " << idx_to_coord(root_policy) << ":8\n";
            ofs << duplicate_orientation.to_str() << " 6 " << idx_to_coord(duplicate_policy) << ":6\n";
        }

        Contest_book duplicate_book;
        std::ostringstream duplicate_log;
        std::streambuf *previous_cerr = std::cerr.rdbuf(duplicate_log.rdbuf());
        bool duplicate_loaded = duplicate_book.init(duplicate_path.string(), true);
        std::cerr.rdbuf(previous_cerr);
        require(!duplicate_loaded, "book with duplicate representatives was accepted");
        require(duplicate_book.size() == 0, "invalid duplicate book retained entries");
        require(
            duplicate_log.str().find("1 duplicate representative board line(s)") != std::string::npos,
            "duplicate warning did not report its count"
        );

        std::cout << "contest book symmetry test passed" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "contest book symmetry test failed: " << error.what() << std::endl;
        return 1;
    }
}
