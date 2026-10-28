/*
    Egaroucid Project

    @file test_xot.cpp
        XOT generation regression test (#649)
    @license GPL-3.0-or-later

    Run from the repository root:
    g++ -O1 -std=c++20 -march=native -pthread src/tools/diagnostics/test_xot.cpp -o test_xot.exe
    ./test_xot.exe [XOT resource directory]
    Add -DHAS_NO_AVX2 to test the Generic build.
*/

#include <stdexcept>
// Evaluation headers must precede util.hpp/search.hpp in the engine include order.
#include "../../engine/evaluate.hpp"
#include "../../engine/xot.hpp"

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Board replay_xot_moves(const std::vector<int>& moves) {
    require(moves.size() == XOT_START_N_DISCS - 4, "XOT must contain eight moves");
    Board board;
    board.reset();
    for (int move : moves) {
        require(0 <= move && move < HW2, "Move coordinate is out of range");
        require((board.get_legal() & (1ULL << move)) != 0, "Generated XOT contains an illegal move");
        Flip flip;
        calc_flip(&flip, &board, move);
        board.move_board(&flip);
        if (!board.is_end() && board.get_legal() == 0ULL) {
            board.pass();
        }
    }
    require(board.n_discs() == XOT_START_N_DISCS, "Generated XOT must have twelve discs");
    require(is_xot_board_key(board), "Generated board must still be recognized as XOT");
    return representative_board(board);
}

int main(int argc, char** argv) {
    try {
        bit_init();
        mobility_init();
        flip_init();
        raw_myrandom.seed(649);
        const std::string directory = argc > 1 ? argv[1] : "bin/resources/xot";
        require(xot_init(directory, false), "Could not load XOT resources");
        const auto transcripts = xot_opening_transcripts();
        Board initial;
        initial.reset();
        const uint64_t legal_first_moves = initial.get_legal();

        uint64_t first_moves = 0;
        for (int sample = 0; sample < 1000; ++sample) {
            const auto moves = get_random_xot_moves();
            replay_xot_moves(moves);
            first_moves |= 1ULL << moves.front();
        }
        require(first_moves == legal_first_moves, "XOT generation must cover all four legal first moves");
        require(xot_opening_transcripts() == transcripts, "Generation must not change the source transcripts");

        // Exercise every opening in all four orientations with a reproducible random stream.
        size_t generated = 0;
        for (const auto& transcript : transcripts) {
            Board original;
            require(xot_transcript_to_board(transcript, &original), "Invalid source transcript");
            original = representative_board(original);
            xot_opening_transcripts() = { transcript };
            first_moves = 0;
            for (int sample = 0; sample < 128 && first_moves != legal_first_moves; ++sample) {
                const auto moves = get_random_xot_moves();
                const Board board = replay_xot_moves(moves);
                require(board.player == original.player && board.opponent == original.opponent,
                    "Orientation must preserve the selected opening");
                first_moves |= 1ULL << moves.front();
                ++generated;
            }
            require(first_moves == legal_first_moves, "Each opening must support all four legal first moves");
        }

        xot_opening_transcripts().clear();
        require(get_random_xot_moves().empty(), "Empty XOT data must return no moves");
        xot_opening_transcripts() = { "f5" };
        require(get_random_xot_moves().empty(), "Invalid XOT data must return no moves");

        std::cout << "PASS: " << transcripts.size() << " openings cover all four first moves; "
            << generated << " generated sequences preserve legality and the selected XOT position; "
            << "empty and invalid data handled\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
