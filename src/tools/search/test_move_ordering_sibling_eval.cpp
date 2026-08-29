/*
    Regression test for SIMD sibling-shared move-ordering evaluation.

    Build and run from bin/ so evaluation resources are available:
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \\
            ../src/tools/search/test_move_ordering_sibling_eval.cpp \\
            -o test_move_ordering_sibling_eval.exe
        ./test_move_ordering_sibling_eval.exe
*/

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "../../engine/ai.hpp"

namespace {

uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

uint64_t next_random() {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1DULL;
}

bool same_flip_value(const Flip_value &a, const Flip_value &b) {
    return a.flip.pos == b.flip.pos &&
        a.flip.flip == b.flip.flip &&
        a.value == b.value &&
        a.static_eval == b.static_eval &&
        a.n_legal == b.n_legal;
}

bool same_visible_search_state(const Search &a, const Search &b) {
    if (a.board != b.board ||
        a.root_n_discs != b.root_n_discs ||
        a.n_discs != b.n_discs ||
        a.parity != b.parity ||
        a.eval.feature_idx != b.eval.feature_idx ||
        std::memcmp(&a.eval, &b.eval, sizeof(Eval_search)) != 0) {
        return false;
    }
#if USE_KILLER_MOVE_MO
    if (std::memcmp(a.move_history, b.move_history, sizeof(a.move_history)) != 0) {
        return false;
    }
#endif
    return true;
}

template <typename MoveList>
void reference_move_list_evaluate_nws(
    Search *search,
    MoveList &move_list,
    const int canput,
    uint_fast8_t moves[],
    const int depth,
    const int alpha,
    const bool is_end_search,
    bool *searching
) {
    if (canput <= 1) {
        return;
    }
    const int eval_alpha = -std::min(SCORE_MAX, alpha + MOVE_ORDERING_NWS_VALUE_OFFSET_BETA);
    const int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA);
    int eval_depth = is_end_search ? (depth >> 4) : (depth >> 3);
    if (use_root_move_ordering_extension(search, canput, is_end_search)) {
        eval_depth = std::max(eval_depth, depth >> 2);
    }
    for (int i = 0; i < canput; ++i) {
        if (move_list[i].flip.flip) {
            if (move_list[i].flip.pos == moves[0]) {
                move_list[i].value = W_1ST_MOVE;
            } else if (move_list[i].flip.pos == moves[1]) {
                move_list[i].value = W_2ND_MOVE;
            } else {
                move_evaluate_nws(
                    search,
                    &move_list[i],
                    eval_alpha,
                    eval_beta,
                    eval_depth,
                    !is_end_search,
                    !is_end_search,
                    searching
                );
            }
        }
    }
}

bool compare_array_api(
    const Search &parent,
    const std::vector<Flip_value> &source,
    uint_fast8_t moves[],
    uint64_t *child_checks
) {
    Flip_value reference[MAX_N_BRANCHES];
    Flip_value candidate[MAX_N_BRANCHES];
    const int n = static_cast<int>(source.size());
    for (int i = 0; i < n; ++i) {
        reference[i] = source[i];
        candidate[i] = source[i];
    }
    Search reference_search = parent;
    Search candidate_search = parent;
    bool reference_searching = true;
    bool candidate_searching = true;
    reference_move_list_evaluate_nws(
        &reference_search, reference, n, moves, 0, 0, false, &reference_searching);
    move_list_evaluate_nws(
        &candidate_search, candidate, n, moves, 0, 0, false, &candidate_searching);
    if (reference_searching != candidate_searching ||
        !same_visible_search_state(reference_search, candidate_search)) {
        return false;
    }
    for (int i = 0; i < n; ++i) {
        if (!same_flip_value(reference[i], candidate[i])) {
            return false;
        }
        ++*child_checks;
    }
    return true;
}

bool compare_vector_api(
    const Search &parent,
    const std::vector<Flip_value> &source,
    uint_fast8_t moves[],
    uint64_t *child_checks
) {
    std::vector<Flip_value> reference = source;
    std::vector<Flip_value> candidate = source;
    Search reference_search = parent;
    Search candidate_search = parent;
    bool reference_searching = true;
    bool candidate_searching = true;
    reference_move_list_evaluate_nws(
        &reference_search,
        reference,
        static_cast<int>(reference.size()),
        moves,
        0,
        0,
        false,
        &reference_searching
    );
    move_list_evaluate_nws(
        &candidate_search, candidate, moves, 0, 0, false, &candidate_searching);
    if (reference_searching != candidate_searching ||
        !same_visible_search_state(reference_search, candidate_search)) {
        return false;
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!same_flip_value(reference[i], candidate[i])) {
            return false;
        }
        ++*child_checks;
    }
    return true;
}

bool compare_direct_child(const Search &parent, const Flip &flip) {
    Search reference = parent;
    Search candidate = parent;
    Eval_features sibling_base;
    eval_prepare_move_sibling_base(&candidate.eval, &candidate.board, &sibling_base);
    reference.move(&flip);
    candidate.move_with_sibling_eval(&flip, &sibling_base);
    if (!same_visible_search_state(reference, candidate) ||
        mid_evaluate_diff(&reference) != mid_evaluate_diff(&candidate)) {
        return false;
    }
    reference.undo(&flip);
    candidate.undo(&flip);
    return same_visible_search_state(reference, candidate);
}

std::vector<Flip_value> make_move_list(Board *board, uint64_t legal) {
    std::vector<Flip_value> result;
    result.reserve(pop_count_ull(legal));
    for (int cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        Flip_value move;
        calc_flip_value(&move, board, cell);
        result.emplace_back(move);
    }
    return result;
}

} // namespace

int main(int argc, char **argv) {
#if !USE_SIMD
    std::cerr << "This driver requires the SIMD evaluation path.\n";
    return 2;
#else
    const std::string resource_dir = argc >= 2 ? argv[1] : "./resources";
    const int n_games = argc >= 3 ? std::max(1, std::stoi(argv[2])) : 256;
    std::cerr << "driver: engine table initialization\n";
    bit_init();
    mobility_init();
    flip_init();
    move_ordering_init();
    if (!evaluate_init(
            resource_dir + "/eval.egev2",
            resource_dir + "/eval_move_ordering_end.egev",
            false)) {
        std::cerr << "evaluation initialization failed\n";
        return 2;
    }
    std::cerr << "driver: comparing " << n_games << " deterministic games\n";

    uint64_t positions = 0;
    uint64_t direct_child_checks = 0;
    uint64_t array_child_checks = 0;
    uint64_t vector_child_checks = 0;
    uint64_t fast_eligible_positions = 0;
    for (int game = 0; game < n_games; ++game) {
        Board initial;
        initial.reset();
        Search trajectory(&initial, MPC_74_LEVEL, false, false);
        bool previously_passed = false;
        for (int ply = 0; ply < 60; ++ply) {
            // Eval_search stores incremental midgame states only through the
            // 63-disc position.  Production handles the final move in its
            // dedicated last-move search without eval_move().
            if (trajectory.n_discs >= HW2_M1) {
                break;
            }
            uint64_t legal = trajectory.board.get_legal();
            if (legal == 0) {
                if (previously_passed) {
                    break;
                }
                trajectory.pass();
                previously_passed = true;
                continue;
            }
            previously_passed = false;
            const std::vector<Flip_value> move_list = make_move_list(&trajectory.board, legal);
            const int n_moves = static_cast<int>(move_list.size());
            uint_fast8_t tt_moves[2] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
            if (n_moves >= 6 && ((game + ply) & 1)) {
                tt_moves[0] = move_list[0].flip.pos;
                tt_moves[1] = move_list[1].flip.pos;
            }
            int n_real_eval = n_moves;
            if (tt_moves[0] != MOVE_UNDEFINED) {
                n_real_eval -= 2;
            }
            if (n_real_eval >= 4) {
                ++fast_eligible_positions;
            }
            for (const Flip_value &move : move_list) {
                if (!compare_direct_child(trajectory, move.flip)) {
                    std::cerr << "direct child mismatch at game=" << game
                              << " ply=" << ply
                              << " move=" << static_cast<int>(move.flip.pos) << '\n';
                    return 1;
                }
                ++direct_child_checks;
            }
            if (!compare_array_api(trajectory, move_list, tt_moves, &array_child_checks)) {
                std::cerr << "array API mismatch at game=" << game << " ply=" << ply << '\n';
                return 1;
            }
            if (!compare_vector_api(trajectory, move_list, tt_moves, &vector_child_checks)) {
                std::cerr << "vector API mismatch at game=" << game << " ply=" << ply << '\n';
                return 1;
            }
            ++positions;

            const int selected = static_cast<int>(next_random() % move_list.size());
            trajectory.move(&move_list[selected].flip);
        }
        if ((game & 31) == 31) {
            std::cerr << "driver: completed games=" << (game + 1) << '\n';
        }
    }
    std::cout << "OK positions=" << positions
              << " fast_eligible=" << fast_eligible_positions
              << " direct_children=" << direct_child_checks
              << " array_children=" << array_child_checks
              << " vector_children=" << vector_child_checks << '\n';
    std::cerr << "driver: complete\n";
    return 0;
#endif
}
