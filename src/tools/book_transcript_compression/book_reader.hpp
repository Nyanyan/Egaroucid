// GPL-3.0-or-later
#pragma once
#include <array>
#include <bit>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../../engine/board.hpp"

namespace transcript_book {

using Key = std::pair<uint64_t, uint64_t>;
using Bytes = std::vector<uint8_t>;

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

inline uint64_t transform(uint64_t value, int symmetry) {
    if (symmetry & 1) value = black_line_mirror(value);
    if (symmetry & 2) value = vertical_mirror(value);
    if (symmetry & 4) value = horizontal_mirror(value);
    return value;
}

inline Key canonical(const Board& board, int* symmetry = nullptr) {
    Key best;
    for (int s = 0; s < 8; ++s) {
        Key candidate{transform(board.player, s), transform(board.opponent, s)};
        if (s == 0 || candidate < best) {
            best = candidate;
            if (symmetry) *symmetry = s;
        }
    }
    return best;
}

class BitDecoder {
    const Bytes& bytes;
    uint64_t bit_count, position = 0;
    bool arithmetic;
    uint32_t low = 0, high = 0xffffffffU, value = 0;
    std::array<std::array<uint32_t, 2>, 24> probabilities;

    int read() {
        const uint64_t index = position++;
        return index < bit_count ? (bytes[20 + index / 8] >> (7 - index % 8)) & 1 : 0;
    }

public:
    BitDecoder(const Bytes& bytes_, uint32_t bits, bool arithmetic_)
        : bytes(bytes_), bit_count(bits), arithmetic(arithmetic_) {
        for (auto& p : probabilities) p = {1, 1};
        if (arithmetic) {
            for (int i = 0; i < 32; ++i) value = (value << 1) | read();
        }
    }

    int bit(int context = -1) {
        if (!arithmetic) {
            require(position < bit_count, "Truncated raw bitstream");
            return read();
        }
        const auto p = context < 0 ? std::array<uint32_t, 2>{1, 1} : probabilities[context];
        const uint32_t split = low + (uint64_t(high) - low + 1) * p[0] / (p[0] + p[1]) - 1;
        const int result = value > split;
        if (result) low = split + 1;
        else high = split;
        while (true) {
            if (high < 0x80000000U) {
            } else if (low >= 0x80000000U) {
                low -= 0x80000000U;
                high -= 0x80000000U;
                value -= 0x80000000U;
            } else if (low >= 0x40000000U && high < 0xc0000000U) {
                low -= 0x40000000U;
                high -= 0x40000000U;
                value -= 0x40000000U;
            } else break;
            low <<= 1;
            high = (high << 1) | 1;
            value = (value << 1) | read();
        }
        if (context >= 0) {
            auto& counts = probabilities[context];
            ++counts[result];
            if (counts[0] + counts[1] >= 4096) {
                for (auto& count : counts) count = (count + 1) / 2;
            }
        }
        return result;
    }

    uint64_t uniform(uint64_t total) {
        require(total > 0, "Invalid combination count");
        int width = 0;
        for (uint64_t t = total; t > 1; t >>= 1) ++width;
        const uint64_t cutoff = (1ULL << (width + 1)) - total;
        uint64_t result = 0;
        for (int i = 0; i < width; ++i) result = (result << 1) | bit();
        return result < cutoff ? result : ((result << 1) | bit()) - cutoff;
    }
};

struct Book {
    std::map<Key, uint64_t> policies;

    uint64_t lookup(const Board& board) const {
        int symmetry;
        const auto found = policies.find(canonical(board, &symmetry));
        if (found == policies.end()) return 0;
        uint64_t moves = found->second;
        if (symmetry & 4) moves = horizontal_mirror(moves);
        if (symmetry & 2) moves = vertical_mirror(moves);
        if (symmetry & 1) moves = black_line_mirror(moves);
        return moves & board.get_legal();
    }
};

inline Book load_book(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "Cannot open book file");
    const Bytes bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    require(!file.bad(), "Cannot read book file");
    require(bytes.size() >= 20, "Truncated book header");
    require(bytes[0] == 'B' && bytes[1] == 'T' && bytes[2] == 'B' && bytes[3] == 'C'
            && bytes[4] == 1 && (bytes[5] & ~7) == 0, "Unsupported book format");
    auto u32 = [&](size_t offset) {
        uint32_t result = 0;
        for (int i = 0; i < 4; ++i) result |= uint32_t(bytes[offset + i]) << (8 * i);
        return result;
    };
    const uint32_t nodes = u32(8), positions = u32(12), bit_count = u32(16);
    require(bytes.size() == 20 + (uint64_t(bit_count) + 7) / 8, "Incorrect book payload length");
    if (bit_count % 8) {
        require((bytes.back() & ((1U << (8 - bit_count % 8)) - 1)) == 0, "Nonzero padding bits");
    }
    const bool arithmetic = bytes[5] & 1, dfs = bytes[5] & 2, parent_context = bytes[5] & 4;
    BitDecoder decoder(bytes, bit_count, arithmetic);
    uint64_t comb[34][34]{};
    for (int n = 0; n <= 33; ++n) {
        comb[n][0] = 1;
        for (int k = 1; k <= n; ++k) {
            comb[n][k] = n ? comb[n - 1][k - 1] + comb[n - 1][k] : 0;
        }
    }
    Board initial;
    initial.reset();
    std::deque<std::pair<Key, int>> pending{{canonical(initial), 0}};
    Book book;
    for (uint32_t i = 0; i < nodes; ++i) {
        require(!pending.empty(), "Book tree ended before node count");
        const auto current = dfs ? pending.back() : pending.front();
        if (dfs) pending.pop_back();
        else pending.pop_front();
        Board board(current.first.first, current.first.second);
        std::vector<int> legal;
        const uint64_t legal_mask = board.get_legal();
        for (int cell = 0; cell < 64; ++cell) {
            if (legal_mask & (1ULL << cell)) legal.push_back(cell);
        }
        const int n = static_cast<int>(legal.size());
        require(n > 0 && n <= 33, "Invalid book node mobility");
        const int stage = std::min(3, (board.n_discs() - 4) / 12);
        int selected_count = 1;
        for (int k = 1; k < n; ++k) {
            if (!decoder.bit(stage * 3 + std::min(k - 1, 2))) break;
            ++selected_count;
        }
        uint64_t rank = decoder.uniform(comb[n][selected_count]);
        std::vector<int> selected;
        int limit = n - 1;
        for (int k = selected_count; k > 0; --k) {
            while (limit >= k - 1 && comb[limit][k] > rank) --limit;
            require(limit >= k - 1, "Invalid legal move rank");
            selected.push_back(limit);
            rank -= comb[limit][k];
            --limit;
        }
        require(rank == 0, "Invalid combination rank");
        uint64_t policy = 0;
        for (auto it = selected.rbegin(); it != selected.rend(); ++it) {
            const int cell = legal[*it];
            const int registered = decoder.bit(parent_context ? 12 + 2 * stage + current.second : 12 + stage);
            const bool follow = registered ? decoder.bit((parent_context ? 20 : 16) + stage) : true;
            if (registered) policy |= 1ULL << cell;
            if (follow) {
                Flip flip;
                calc_flip(&flip, &board, cell);
                Board child = board.move_copy(&flip);
                if (!child.get_legal()) {
                    Board passed = child;
                    passed.pass();
                    if (passed.get_legal()) child = passed;
                }
                pending.emplace_back(canonical(child), registered);
            }
        }
        if (policy) require(book.policies.emplace(current.first, policy).second, "Duplicate policy board");
    }
    require(pending.empty(), "Book tree exceeds node count");
    require(book.policies.size() == positions, "Incorrect book position count");
    return book;
}

} // namespace transcript_book
