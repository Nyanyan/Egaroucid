/*
    Focused regression tests for linked NWS cancellation contexts.

    Build and run from bin/ so that the evaluation resources are available:
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \
            ../src/tools/search/test_search_cancellation_context.cpp \
            -o test_search_cancellation_context.exe
*/

#define EGAROUCID_TEST_MIDSEARCH_NWS_CANCELLATION

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../../engine/engine_all.hpp"

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_root_context() {
    alignas(std::atomic_ref<bool>::required_alignment) bool root_flag = true;
    const Search_cancellation_context root{&root_flag, nullptr};

    require(root.current == &root_flag, "root exposes its current flag");
    require(root.parent == nullptr, "root has no parent");
    require(is_searching(root), "true root is active");

    search_cancellation_store(&root_flag, false);
    require(!is_searching(root), "false root cancels its context");
}

void test_leaf_context() {
    alignas(std::atomic_ref<bool>::required_alignment) bool root_flag = true;
    alignas(std::atomic_ref<bool>::required_alignment) bool leaf_flag = true;
    const Search_cancellation_context root{&root_flag, nullptr};
    const Search_cancellation_context leaf{&leaf_flag, &root};

    require(is_searching(leaf), "true root and leaf are active");
    search_cancellation_store(&root_flag, false);
    require(!is_searching(leaf), "false root cancels a true leaf");
    search_cancellation_store(&root_flag, true);
    search_cancellation_store(&leaf_flag, false);
    require(!is_searching(leaf), "false leaf cancels a true root");
    require(is_searching(root), "leaf cancellation does not change root");
}

void test_three_level_context() {
    alignas(std::atomic_ref<bool>::required_alignment) bool root_flag = true;
    alignas(std::atomic_ref<bool>::required_alignment) bool middle_flag = true;
    alignas(std::atomic_ref<bool>::required_alignment) bool leaf_flag = true;
    const Search_cancellation_context root{&root_flag, nullptr};
    const Search_cancellation_context middle{&middle_flag, &root};
    const Search_cancellation_context leaf{&leaf_flag, &middle};

    require(is_searching(leaf), "three true flags are active");
    search_cancellation_store(&middle_flag, false);
    require(!is_searching(leaf), "middle flag cancels a three-level leaf");
    require(is_searching(root), "middle cancellation leaves root active");
    search_cancellation_store(&middle_flag, true);
    search_cancellation_store(&root_flag, false);
    require(!is_searching(leaf), "root flag reaches a three-level leaf");
}

bool initialize_engine() {
    thread_pool.resize(0);
    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    endsearch_init();
#if USE_MPC_PRE_CALCULATION
    mpc_init();
#endif
    move_ordering_init();
    if (!transposition_table.resize(10)) {
        return false;
    }
    hash_init_rand(10);
    global_hash_level = 10;
#if USE_CRC32C_HASH
    global_hash_bit_mask = (1U << global_hash_level) - 1;
#endif
    stability_init();
    return evaluate_init(
        "./resources/eval.egev2",
        "./resources/eval_move_ordering_end.egev",
        false
    );
}

bool *ancestor_to_cancel = nullptr;
bool *expected_simple_flag = nullptr;
int simple_children_completed = 0;
bool simple_received_current_flag = true;

void cancel_ancestor_after_first_simple_child(bool *searching) {
    ++simple_children_completed;
    simple_received_current_flag &= searching == expected_simple_flag;
    if (simple_children_completed == 1) {
        search_cancellation_store(ancestor_to_cancel, false);
    }
}

void test_simple_nws_uses_only_current_flag() {
    Board board;
    board.reset();
    const int n_root_children = pop_count_ull(board.get_legal());
    require(n_root_children > 1, "simple NWS test position has multiple moves");

    transposition_table.init();
    global_searching = true;
    alignas(std::atomic_ref<bool>::required_alignment) bool root_flag = true;
    alignas(std::atomic_ref<bool>::required_alignment) bool current_flag = true;
    const Search_cancellation_context root{&root_flag, nullptr};
    const Search_cancellation_context leaf{&current_flag, &root};
    Search search(&board, MPC_100_LEVEL, false, false);
    const uint64_t player_before = search.board.player;
    const uint64_t opponent_before = search.board.opponent;
    const int n_discs_before = search.n_discs;
    const uint_fast8_t parity_before = search.parity;
    const uint_fast8_t eval_feature_idx_before = search.eval.feature_idx;
    const int evaluation_before = mid_evaluate_diff(&search);

    ancestor_to_cancel = &root_flag;
    expected_simple_flag = &current_flag;
    simple_children_completed = 0;
    simple_received_current_flag = true;
    midsearch_nws_after_eval1_test_hook = cancel_ancestor_after_first_simple_child;
    const int value = nega_alpha_ordering_nws(
        &search,
        SCORE_MAX - 1,
        2,
        Nws_node_hint::no_static_eval(),
        board.get_legal(),
        false,
        leaf
    );
    midsearch_nws_after_eval1_test_hook = nullptr;

    require(!search_cancellation_load(&root_flag), "hook cancels the ancestor flag");
    require(search_cancellation_load(&current_flag), "hook leaves current flag active");
    require(!is_searching(leaf), "full context observes cancelled ancestor afterwards");
    require(simple_received_current_flag, "simple NWS receives context.current");
    require(
        simple_children_completed == n_root_children,
        "simple NWS continues after ancestor-only cancellation"
    );
    require(value != SCORE_UNDEFINED, "simple NWS returns its completed value");
    require(search.board.player == player_before, "simple NWS restores player bitboard");
    require(search.board.opponent == opponent_before, "simple NWS restores opponent bitboard");
    require(search.n_discs == n_discs_before, "simple NWS restores disc count");
    require(search.parity == parity_before, "simple NWS restores parity");
    require(search.eval.feature_idx == eval_feature_idx_before, "simple NWS restores evaluation stack");
    require(mid_evaluate_diff(&search) == evaluation_before, "simple NWS restores evaluation value");

    ancestor_to_cancel = nullptr;
    expected_simple_flag = nullptr;
}

} // namespace

int main() {
    try {
        test_root_context();
        test_leaf_context();
        test_three_level_context();
        require(initialize_engine(), "engine initialization");
        test_simple_nws_uses_only_current_flag();
    } catch (const std::exception &error) {
        midsearch_nws_after_eval1_test_hook = nullptr;
        ancestor_to_cancel = nullptr;
        expected_simple_flag = nullptr;
        global_searching = true;
        std::cerr << "Search cancellation context test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "Search cancellation context tests passed" << std::endl;
    return 0;
}
