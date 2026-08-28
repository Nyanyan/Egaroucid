/*
    Focused regression tests for the cheap stability-cutoff feasibility guard.

    Build and run from bin/:
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \
            ../src/tools/search/test_stability_cutoff_guard.cpp \
            -o test_stability_cutoff_guard.exe
*/

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../engine/engine_all.hpp"

namespace {

struct Coverage {
    uint64_t positions = 0;
    uint64_t nws_cases = 0;
    uint64_t nws_threshold_pass = 0;
    uint64_t nws_guard_true = 0;
    uint64_t nws_guard_false = 0;
    uint64_t nws_guard_equal = 0;
    uint64_t full_cases = 0;
    uint64_t full_threshold_pass = 0;
    uint64_t full_guard_true = 0;
    uint64_t full_guard_false = 0;
    uint64_t full_guard_equal = 0;
};

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int stability_cut_reference(Search *search, int *alpha, int *beta) {
    if (*beta >= stability_threshold[search->n_discs]) {
        const int n_beta = HW2 - 2 * pop_count_ull(
            calc_stability(search->board.opponent, search->board.player)
        );
        if (n_beta <= *alpha) {
            return n_beta;
        }
        if (n_beta < *beta) {
            *beta = n_beta;
        }
    }
    return SCORE_UNDEFINED;
}

int stability_cut_nws_reference(Search *search, int alpha) {
    if (alpha >= stability_threshold_nws[search->n_discs]) {
        const int n_beta = HW2 - 2 * pop_count_ull(
            calc_stability(search->board.opponent, search->board.player)
        );
        if (n_beta <= alpha) {
            return n_beta;
        }
    }
    return SCORE_UNDEFINED;
}

uint64_t next_random(uint64_t *state) {
    uint64_t value = *state;
    value ^= value << 7;
    value ^= value >> 9;
    value ^= value << 8;
    *state = value;
    return value;
}

uint_fast8_t select_legal_move(uint64_t legal, uint64_t random_value) {
    int index = static_cast<int>(random_value % pop_count_ull(legal));
    while (index-- > 0) {
        legal &= legal - 1;
    }
    return first_bit(&legal);
}

std::vector<Board> generate_reachable_positions() {
    constexpr int n_games = 192;
    std::vector<Board> positions;
    positions.reserve(n_games * 55);
    uint64_t random_state = 0x9E3779B97F4A7C15ULL;

    for (int game = 0; game < n_games; ++game) {
        Board board;
        board.reset();
        bool passed = false;
        while (board.n_discs() < HW2) {
            positions.emplace_back(board.copy());
            uint64_t legal = board.get_legal();
            if (legal == 0) {
                if (passed) {
                    break;
                }
                board.pass();
                passed = true;
                continue;
            }
            passed = false;
            const uint_fast8_t move = select_legal_move(
                legal,
                next_random(&random_state) + static_cast<uint64_t>(game)
            );
            Flip flip;
            calc_flip(&flip, &board, move);
            require(flip.flip != 0, "generated move must flip at least one disc");
            board.move_board(&flip);
        }
    }

    std::sort(
        positions.begin(),
        positions.end(),
        [](const Board &left, const Board &right) {
            return std::pair(left.player, left.opponent) <
                std::pair(right.player, right.opponent);
        }
    );
    positions.erase(
        std::unique(
            positions.begin(),
            positions.end(),
            [](const Board &left, const Board &right) {
                return left.player == right.player && left.opponent == right.opponent;
            }
        ),
        positions.end()
    );
    return positions;
}

void add_bound(std::vector<int> *values, int value, int minimum, int maximum) {
    if (minimum <= value && value <= maximum) {
        values->emplace_back(value);
    }
}

void deduplicate(std::vector<int> *values) {
    std::sort(values->begin(), values->end());
    values->erase(std::unique(values->begin(), values->end()), values->end());
}

std::string case_description(
    const char *kind,
    uint64_t position,
    int n_discs,
    int alpha,
    int beta
) {
    return std::string(kind) + " position=" + std::to_string(position) +
        " n_discs=" + std::to_string(n_discs) +
        " alpha=" + std::to_string(alpha) +
        " beta=" + std::to_string(beta);
}

void test_nws_position(Search *search, uint64_t position, Coverage *coverage) {
    const int threshold = stability_threshold_nws[search->n_discs];
    const int cheap_upper = HW2 - 2 * pop_count_ull(search->board.opponent);
    const int exact_stability_upper = HW2 - 2 * pop_count_ull(
        calc_stability(search->board.opponent, search->board.player)
    );
    std::vector<int> alphas{-64, -63, -32, -1, 0, 1, 31, 32, 62, 63};
    for (int center: {threshold, cheap_upper, exact_stability_upper}) {
        add_bound(&alphas, center - 1, -64, 63);
        add_bound(&alphas, center, -64, 63);
        add_bound(&alphas, center + 1, -64, 63);
    }
    deduplicate(&alphas);

    for (const int alpha: alphas) {
        ++coverage->nws_cases;
        const bool threshold_pass = alpha >= threshold;
        const bool guard = cheap_upper <= alpha;
        if (threshold_pass) {
            ++coverage->nws_threshold_pass;
            if (guard) {
                ++coverage->nws_guard_true;
            } else {
                ++coverage->nws_guard_false;
            }
            if (cheap_upper == alpha) {
                ++coverage->nws_guard_equal;
            }
        }

        const int reference = stability_cut_nws_reference(search, alpha);
        const int guarded = stability_cut_nws(search, alpha);
        const std::string description = case_description(
            "NWS", position, search->n_discs, alpha, alpha + 1
        );
        require(reference == guarded, description + " return mismatch");
        if (!threshold_pass || !guard) {
            require(
                reference == SCORE_UNDEFINED,
                description + " rejected guard changed the reference result"
            );
        }
    }
}

void test_full_window_position(Search *search, uint64_t position, Coverage *coverage) {
    const int threshold = stability_threshold[search->n_discs];
    const int cheap_upper = HW2 - 2 * pop_count_ull(search->board.opponent);
    const int exact_stability_upper = HW2 - 2 * pop_count_ull(
        calc_stability(search->board.opponent, search->board.player)
    );
    std::vector<int> betas{-63, -32, -1, 0, 1, 32, 63, 64};
    for (int center: {threshold, cheap_upper, exact_stability_upper}) {
        add_bound(&betas, center - 1, -63, 64);
        add_bound(&betas, center, -63, 64);
        add_bound(&betas, center + 1, -63, 64);
    }
    deduplicate(&betas);

    for (const int beta: betas) {
        std::vector<int> alphas{-64, -63, -32, -1, 0, 1, 31, 32, 63};
        for (int center: {beta, cheap_upper, exact_stability_upper}) {
            add_bound(&alphas, center - 2, -64, 63);
            add_bound(&alphas, center - 1, -64, 63);
            add_bound(&alphas, center, -64, 63);
        }
        deduplicate(&alphas);

        for (const int alpha: alphas) {
            if (alpha >= beta) {
                continue;
            }
            ++coverage->full_cases;
            const bool threshold_pass = beta >= threshold;
            const bool guard = cheap_upper < beta;
            if (threshold_pass) {
                ++coverage->full_threshold_pass;
                if (guard) {
                    ++coverage->full_guard_true;
                } else {
                    ++coverage->full_guard_false;
                }
                if (cheap_upper == beta) {
                    ++coverage->full_guard_equal;
                }
            }

            int reference_alpha = alpha;
            int reference_beta = beta;
            int guarded_alpha = alpha;
            int guarded_beta = beta;
            const int reference = stability_cut_reference(
                search, &reference_alpha, &reference_beta
            );
            const int guarded = stability_cut(
                search, &guarded_alpha, &guarded_beta
            );
            const std::string description = case_description(
                "full", position, search->n_discs, alpha, beta
            );
            require(reference == guarded, description + " return mismatch");
            require(
                reference_alpha == guarded_alpha,
                description + " alpha update mismatch"
            );
            require(
                reference_beta == guarded_beta,
                description + " beta update mismatch"
            );
            if (!threshold_pass || !guard) {
                require(
                    reference == SCORE_UNDEFINED,
                    description + " rejected guard changed the reference result"
                );
                require(
                    reference_alpha == alpha && reference_beta == beta,
                    description + " rejected guard changed the reference window"
                );
            }
        }
    }
}

void test_reachable_positions() {
    bit_init();
    mobility_init();
    flip_init();
    stability_init();

    const std::vector<Board> positions = generate_reachable_positions();
    require(positions.size() >= 5000, "too few distinct reachable positions");

    Coverage coverage;
    for (std::size_t index = 0; index < positions.size(); ++index) {
        Search search;
        search.board = positions[index].copy();
        search.root_n_discs = positions[index].n_discs();
        search.n_discs = search.root_n_discs;
        require(search.n_discs < HW2, "terminal position entered stability cutoff test");
        ++coverage.positions;
        test_nws_position(&search, index, &coverage);
        test_full_window_position(&search, index, &coverage);
    }

    require(coverage.nws_cases > 50000, "too few NWS boundary cases");
    require(coverage.full_cases > 250000, "too few full-window boundary cases");
    require(coverage.nws_guard_true > 0, "NWS guard true path was not covered");
    require(coverage.nws_guard_false > 0, "NWS guard false path was not covered");
    require(coverage.nws_guard_equal > 0, "NWS <= equality boundary was not covered");
    require(coverage.full_guard_true > 0, "full-window guard true path was not covered");
    require(coverage.full_guard_false > 0, "full-window guard false path was not covered");
    require(coverage.full_guard_equal > 0, "full-window < equality boundary was not covered");

    std::cout
        << "positions=" << coverage.positions
        << " nws_cases=" << coverage.nws_cases
        << " nws_threshold_pass=" << coverage.nws_threshold_pass
        << " nws_guard_true=" << coverage.nws_guard_true
        << " nws_guard_false=" << coverage.nws_guard_false
        << " nws_guard_equal=" << coverage.nws_guard_equal
        << " full_cases=" << coverage.full_cases
        << " full_threshold_pass=" << coverage.full_threshold_pass
        << " full_guard_true=" << coverage.full_guard_true
        << " full_guard_false=" << coverage.full_guard_false
        << " full_guard_equal=" << coverage.full_guard_equal
        << '\n';
}

} // namespace

int main() {
    try {
        test_reachable_positions();
    } catch (const std::exception &error) {
        std::cerr << "Stability cutoff guard test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "Stability cutoff guard tests passed" << std::endl;
    return 0;
}
