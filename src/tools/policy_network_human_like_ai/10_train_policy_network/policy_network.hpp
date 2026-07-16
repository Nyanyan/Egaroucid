/*
    Egaroucid Project

    @file policy_network.hpp
        Lightweight C++ inference helper for the policy network.
        Inputs are player-to-move / opponent bitboards.

    Related issue: #613
*/

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace egaroucid_policy {

constexpr int HW = 8;
constexpr int HW2 = 64;
constexpr int HW2_M1 = 63;
constexpr int BLACK = 0;
constexpr int WHITE = 1;

inline bool is_black_like_char(char c) {
    return c == 'B' || c == 'b' || c == 'X' || c == 'x' || c == '0' || c == '*';
}

inline bool is_white_like_char(char c) {
    return c == 'W' || c == 'w' || c == 'O' || c == 'o' || c == '1';
}

inline bool is_vacant_like_char(char c) {
    return c == '-' || c == '.';
}

inline int policy_from_coord(char file, char rank) {
    file = static_cast<char>(file | 0x20);
    if (file < 'a' || file > 'h' || rank < '1' || rank > '8') {
        return -1;
    }
    const int x = file - 'a';
    const int y = rank - '1';
    return HW2_M1 - (y * HW + x);
}

inline std::string coord_from_policy(int policy) {
    if (policy < 0 || policy >= HW2) {
        return "??";
    }
    const int y = HW - 1 - policy / HW;
    const int x = HW - 1 - policy % HW;
    std::string res;
    res += static_cast<char>('a' + x);
    res += static_cast<char>('1' + y);
    return res;
}

inline uint64_t bit_from_policy(int policy) {
    return 1ULL << policy;
}

inline int popcount64(uint64_t x) {
    int res = 0;
    while (x) {
        x &= x - 1;
        ++res;
    }
    return res;
}

struct BoardState {
    uint64_t black = 0;
    uint64_t white = 0;
    int side_to_move = BLACK;
};

inline bool parse_board_65(const std::string &text, BoardState *board) {
    std::string s;
    s.reserve(text.size());
    for (char c : text) {
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
            s += c;
        }
    }
    if (s.size() != HW2 && s.size() != HW2 + 1) {
        return false;
    }
    board->black = 0;
    board->white = 0;
    for (int i = 0; i < HW2; ++i) {
        const uint64_t bit = 1ULL << (HW2_M1 - i);
        if (is_black_like_char(s[i])) {
            board->black |= bit;
        } else if (is_white_like_char(s[i])) {
            board->white |= bit;
        } else if (!is_vacant_like_char(s[i])) {
            return false;
        }
    }
    if (board->black & board->white) {
        return false;
    }
    if (s.size() == HW2 + 1) {
        if (is_black_like_char(s[HW2])) {
            board->side_to_move = BLACK;
        } else if (is_white_like_char(s[HW2])) {
            board->side_to_move = WHITE;
        } else {
            return false;
        }
    } else {
        board->side_to_move = popcount64(board->black) == popcount64(board->white) ? BLACK : WHITE;
    }
    return true;
}

inline uint64_t calc_flips(uint64_t player, uint64_t opponent, int policy) {
    if (policy < 0 || policy >= HW2) {
        return 0;
    }
    const uint64_t move_bit = bit_from_policy(policy);
    if ((player | opponent) & move_bit) {
        return 0;
    }
    const int pos = HW2_M1 - policy;
    const int x0 = pos % HW;
    const int y0 = pos / HW;
    static constexpr int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    static constexpr int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    uint64_t flips = 0;
    for (int dir = 0; dir < 8; ++dir) {
        int x = x0 + dx[dir];
        int y = y0 + dy[dir];
        uint64_t line = 0;
        while (0 <= x && x < HW && 0 <= y && y < HW) {
            const int p = HW2_M1 - (y * HW + x);
            const uint64_t bit = bit_from_policy(p);
            if (opponent & bit) {
                line |= bit;
            } else {
                if ((player & bit) && line != 0) {
                    flips |= line;
                }
                break;
            }
            x += dx[dir];
            y += dy[dir];
        }
    }
    return flips;
}

inline uint64_t legal_moves(uint64_t player, uint64_t opponent) {
    uint64_t res = 0;
    const uint64_t occupied = player | opponent;
    for (int policy = 0; policy < HW2; ++policy) {
        const uint64_t bit = bit_from_policy(policy);
        if ((occupied & bit) == 0 && calc_flips(player, opponent, policy) != 0) {
            res |= bit;
        }
    }
    return res;
}

inline bool apply_policy(BoardState *board, int policy) {
    uint64_t &player = board->side_to_move == BLACK ? board->black : board->white;
    uint64_t &opponent = board->side_to_move == BLACK ? board->white : board->black;
    const uint64_t flips = calc_flips(player, opponent, policy);
    if (flips == 0) {
        return false;
    }
    player ^= flips | bit_from_policy(policy);
    opponent ^= flips;
    board->side_to_move ^= 1;
    return true;
}

inline bool apply_pass_if_needed(BoardState *board) {
    const uint64_t player = board->side_to_move == BLACK ? board->black : board->white;
    const uint64_t opponent = board->side_to_move == BLACK ? board->white : board->black;
    if (legal_moves(player, opponent) != 0) {
        return true;
    }
    board->side_to_move ^= 1;
    const uint64_t next_player = board->side_to_move == BLACK ? board->black : board->white;
    const uint64_t next_opponent = board->side_to_move == BLACK ? board->white : board->black;
    return legal_moves(next_player, next_opponent) != 0;
}

inline bool play_transcript(const std::string &transcript, BoardState *board, std::string *error = nullptr) {
    board->black = (1ULL << policy_from_coord('e', '4')) | (1ULL << policy_from_coord('d', '5'));
    board->white = (1ULL << policy_from_coord('d', '4')) | (1ULL << policy_from_coord('e', '5'));
    board->side_to_move = BLACK;

    std::string s;
    s.reserve(transcript.size());
    for (char c : transcript) {
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
            s += c;
        }
    }
    if (s.size() % 2 != 0) {
        if (error) {
            *error = "transcript length must be even";
        }
        return false;
    }
    for (size_t i = 0; i < s.size(); i += 2) {
        if (!apply_pass_if_needed(board)) {
            if (error) {
                *error = "game is already over";
            }
            return false;
        }
        const int policy = policy_from_coord(s[i], s[i + 1]);
        if (policy < 0) {
            if (error) {
                *error = "invalid coordinate in transcript";
            }
            return false;
        }
        if (!apply_policy(board, policy)) {
            if (error) {
                *error = "illegal move in transcript: " + coord_from_policy(policy);
            }
            return false;
        }
    }
    apply_pass_if_needed(board);
    return true;
}

class PolicyNetwork {
  public:
    struct Layer {
        uint32_t input_dim = 0;
        uint32_t output_dim = 0;
        uint32_t activation = 0;
        float alpha = 0.0F;
        std::vector<float> weights;
        std::vector<float> bias;
    };

    bool load(const std::string &path, std::string *error = nullptr) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            if (error) {
                *error = "cannot open weights file";
            }
            return false;
        }
        char magic[16];
        f.read(magic, sizeof(magic));
        const char expected[16] = {'E', 'G', 'R', '_', 'P', 'O', 'L', 'I', 'C', 'Y', '_', 'V', '1', '\0', '\0', '\0'};
        if (!f || std::memcmp(magic, expected, sizeof(magic)) != 0) {
            if (error) {
                *error = "invalid weights magic";
            }
            return false;
        }
        uint32_t version = 0;
        uint32_t n_layers = 0;
        uint32_t input_size = 0;
        uint32_t output_size = 0;
        f.read(reinterpret_cast<char *>(&version), sizeof(version));
        f.read(reinterpret_cast<char *>(&n_layers), sizeof(n_layers));
        f.read(reinterpret_cast<char *>(&input_size), sizeof(input_size));
        f.read(reinterpret_cast<char *>(&output_size), sizeof(output_size));
        if (!f || version != 1 || input_size != 128 || output_size != 64 || n_layers == 0) {
            if (error) {
                *error = "unsupported weights header";
            }
            return false;
        }
        layers_.clear();
        layers_.resize(n_layers);
        uint32_t expected_input = input_size;
        for (Layer &layer : layers_) {
            f.read(reinterpret_cast<char *>(&layer.input_dim), sizeof(layer.input_dim));
            f.read(reinterpret_cast<char *>(&layer.output_dim), sizeof(layer.output_dim));
            f.read(reinterpret_cast<char *>(&layer.activation), sizeof(layer.activation));
            f.read(reinterpret_cast<char *>(&layer.alpha), sizeof(layer.alpha));
            if (!f || layer.input_dim != expected_input || layer.output_dim == 0) {
                if (error) {
                    *error = "invalid layer shape";
                }
                return false;
            }
            const size_t n_weights = static_cast<size_t>(layer.input_dim) * layer.output_dim;
            layer.weights.resize(n_weights);
            layer.bias.resize(layer.output_dim);
            f.read(reinterpret_cast<char *>(layer.weights.data()), static_cast<std::streamsize>(n_weights * sizeof(float)));
            f.read(reinterpret_cast<char *>(layer.bias.data()), static_cast<std::streamsize>(layer.bias.size() * sizeof(float)));
            if (!f) {
                if (error) {
                    *error = "truncated weights file";
                }
                return false;
            }
            expected_input = layer.output_dim;
        }
        if (layers_.back().output_dim != 64) {
            if (error) {
                *error = "output layer is not 64-wide";
            }
            return false;
        }
        return true;
    }

    std::array<float, 64> predict(uint64_t player, uint64_t opponent) const {
        if (layers_.empty()) {
            throw std::runtime_error("PolicyNetwork::predict called before load");
        }
        std::vector<float> current(128);
        for (int i = 0; i < 64; ++i) {
            current[i] = static_cast<float>((player >> (63 - i)) & 1ULL);
            current[64 + i] = static_cast<float>((opponent >> (63 - i)) & 1ULL);
        }
        std::vector<float> next;
        for (const Layer &layer : layers_) {
            next.assign(layer.output_dim, 0.0F);
            for (uint32_t j = 0; j < layer.output_dim; ++j) {
                float sum = layer.bias[j];
                for (uint32_t i = 0; i < layer.input_dim; ++i) {
                    sum += current[i] * layer.weights[static_cast<size_t>(i) * layer.output_dim + j];
                }
                if (layer.activation == 1 && sum < 0.0F) {
                    sum *= layer.alpha;
                }
                next[j] = sum;
            }
            current.swap(next);
        }

        std::array<float, 64> prob{};
        const float max_logit = *std::max_element(current.begin(), current.end());
        float sum_exp = 0.0F;
        for (int i = 0; i < 64; ++i) {
            prob[i] = std::exp(current[i] - max_logit);
            sum_exp += prob[i];
        }
        if (sum_exp > 0.0F) {
            for (float &v : prob) {
                v /= sum_exp;
            }
        }
        return prob;
    }

  private:
    std::vector<Layer> layers_;
};

inline std::vector<std::pair<int, float>> top_k(const std::array<float, 64> &prob, int k) {
    std::vector<std::pair<int, float>> entries;
    entries.reserve(64);
    for (int i = 0; i < 64; ++i) {
        entries.emplace_back(i, prob[i]);
    }
    if (k < 0 || k > 64) {
        k = 64;
    }
    std::partial_sort(
        entries.begin(),
        entries.begin() + k,
        entries.end(),
        [](const auto &a, const auto &b) { return a.second > b.second; });
    entries.resize(k);
    return entries;
}

}  // namespace egaroucid_policy
