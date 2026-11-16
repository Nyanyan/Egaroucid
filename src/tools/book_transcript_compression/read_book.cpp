// GPL-3.0-or-later
#include "book_reader.hpp"

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: read_book <book.bin> [transcript] [--export-reference <reference.bin>]\n";
            return 1;
        }
        bit_init();
        mobility_init();
        flip_init();
        last_flip_init();
        const auto book = transcript_book::load_book(argv[1]);
        std::string transcript, reference;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--export-reference" && i + 1 < argc) reference = argv[++i];
            else if (transcript.empty() && argv[i][0] != '-') transcript = argv[i];
            else throw std::runtime_error("Invalid argument");
        }
        if (!reference.empty()) {
            std::ofstream file(reference, std::ios::binary);
            transcript_book::require(bool(file), "Cannot open reference output");
            for (const auto& entry : book.policies) {
                for (uint64_t value : {entry.first.first, entry.first.second, entry.second}) {
                    for (int i = 0; i < 8; ++i) file.put(static_cast<char>((value >> (8 * i)) & 255));
                }
            }
            file.close();
            transcript_book::require(bool(file), "Cannot write reference output");
        }
        Board board;
        board.reset();
        transcript_book::require(transcript.size() % 2 == 0, "Transcript must contain coordinate pairs");
        for (size_t i = 0; i < transcript.size(); i += 2) {
            const char x = transcript[i], y = transcript[i + 1];
            transcript_book::require(x >= 'a' && x <= 'h' && y >= '1' && y <= '8', "Invalid coordinate");
            if (!board.get_legal()) board.pass();
            const int cell = 63 - ((y - '1') * 8 + x - 'a');
            transcript_book::require((board.get_legal() & (1ULL << cell)) != 0, "Illegal transcript move");
            Flip flip;
            calc_flip(&flip, &board, cell);
            board.move_board(&flip);
        }
        if (!board.get_legal()) board.pass();
        const uint64_t moves = book.lookup(board);
        std::cout << "positions: " << book.policies.size() << "\nmoves:";
        for (int cell = 63; cell >= 0; --cell) {
            if (moves & (1ULL << cell)) {
                std::cout << ' ' << char('a' + (63 - cell) % 8) << char('1' + (63 - cell) / 8);
            }
        }
        if (!moves) std::cout << " (none)";
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
