/*
    Focused regression tests for transposition-table storage lifetime.

    Build and run both storage modes:
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \
            -DIS_GGS_TOURNAMENT -DTT_USE_STACK=1 \
            -DEGAROUCID_TEST_TRANSPOSITION_TABLE_STACK_LEVEL=8 \
            ../src/tools/search/test_transposition_table_storage.cpp \
            -o test_transposition_table_storage_stack.exe
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \
            -DIS_GGS_TOURNAMENT -DTT_USE_STACK=0 \
            ../src/tools/search/test_transposition_table_storage.cpp \
            -o test_transposition_table_storage_heap.exe
*/

#define EGAROUCID_TEST_TRANSPOSITION_TABLE_STORAGE

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../../engine/engine_all.hpp"

namespace {

constexpr int TEST_HASH_LEVEL = 10;
constexpr uint32_t TEST_HASH_SIZE = 1U << TEST_HASH_LEVEL;

#if TT_USE_STACK
static_assert(
    TRANSPOSITION_TABLE_STACK_SIZE < TEST_HASH_SIZE,
    "the stack-mode test must exercise both storage regions"
);
#endif

void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Search make_search(const Board &board) {
    Search search;
    search.board = board;
    search.root_n_discs = pop_count_ull(board.player | board.opponent);
    search.n_discs = search.root_n_discs;
    search.mpc_level = MPC_100_LEVEL;
    return search;
}

void require_initial_node(Hash_node *node, const std::string &label) {
    require(node->board.player == 0ULL, label + ": player initialized");
    require(node->board.opponent == 0ULL, label + ": opponent initialized");
    require(node->data.get_level() == 0, label + ": level initialized");
    require(node->data.get_depth() == 0, label + ": depth initialized");
    require(node->data.get_mpc_level() == 0, label + ": MPC level initialized");
    require(node->data.get_importance() == 0, label + ": importance initialized");

    int lower = 0;
    int upper = 0;
    node->data.get_bounds(&lower, &upper);
    require(lower == -SCORE_MAX, label + ": lower bound initialized");
    require(upper == SCORE_MAX, label + ": upper bound initialized");

    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {0, 0};
    node->data.get_moves(moves);
    require(moves[0] == MOVE_UNDEFINED, label + ": first move initialized");
    require(moves[1] == MOVE_UNDEFINED, label + ": second move initialized");

    require(node->lock.try_lock(), label + ": lock initialized unlocked");
    node->lock.unlock();
}

void test_poisoned_raw_storage_construction() {
    constexpr size_t n_nodes = 17;
    void *storage = std::malloc(sizeof(Hash_node) * n_nodes);
    require(storage != nullptr, "raw test storage allocation");
    std::memset(storage, 0xFF, sizeof(Hash_node) * n_nodes);

    Hash_node *nodes = static_cast<Hash_node*>(storage);
    const uint64_t constructed_before = transposition_table_test_constructed_nodes.load();
    init_transposition_table<true>(nodes, 0, n_nodes);
    require(
        transposition_table_test_constructed_nodes.load() == constructed_before + n_nodes,
        "raw fresh-storage traversal constructs every node exactly once"
    );
    for (size_t i = 0; i < n_nodes; ++i) {
        require_initial_node(nodes + i, "raw node " + std::to_string(i));
    }

    // Hash_node is required to remain trivially destructible, so ending the
    // storage duration needs no additional full-array destruction traversal.
    std::free(storage);
}

void require_exact_entry(
    Search *search,
    const uint32_t hash,
    const int depth,
    const int value,
    const int move,
    const std::string &label
) {
    int lower = -SCORE_MAX;
    int upper = SCORE_MAX;
    require(
        transposition_table.get_bounds(search, hash, depth, &lower, &upper),
        label + ": bounds found"
    );
    require(lower == value && upper == value, label + ": exact value retained");

    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
    require(
        transposition_table.get_moves_any_level(&search->board, hash, moves),
        label + ": move found"
    );
    require(moves[0] == move, label + ": best move retained");
}

void test_resize_register_probe_delete_repeat() {
    const bool auto_reset_before = transposition_table_auto_reset_importance;
    transposition_table_auto_reset_importance = false;

    const Board boards[] = {
        Board{1ULL << 0, 1ULL << 1},
        Board{1ULL << 2, 1ULL << 3},
        Board{1ULL << 4, 1ULL << 5}
    };
    const int depths[] = {30, 20, 10};
    const int values[] = {8, -4, 2};
    const int moves[] = {10, 20, 30};
#if TT_USE_STACK
    // A three-entry collision chain beginning here crosses from the static
    // prefix into the heap suffix.
    const uint32_t last_primary_hash = TRANSPOSITION_TABLE_STACK_SIZE - 1;
#else
    const uint32_t last_primary_hash = TEST_HASH_SIZE - 1;
#endif

    for (int round = 0; round < 4; ++round) {
        const uint64_t constructed_before_resize = transposition_table_test_constructed_nodes.load();
        require(transposition_table.resize(TEST_HASH_LEVEL), "hash10 resize round " + std::to_string(round));
#if TT_USE_STACK
        const uint64_t expected_constructed_after_resize = constructed_before_resize
            + TEST_HASH_SIZE + TRANSPOSITION_TABLE_N_LOOP - 1
            - TRANSPOSITION_TABLE_STACK_SIZE;
#else
        const uint64_t expected_constructed_after_resize =
            constructed_before_resize + TEST_HASH_SIZE + TRANSPOSITION_TABLE_N_LOOP - 1;
#endif
        require(
            transposition_table_test_constructed_nodes.load() == expected_constructed_after_resize,
            "resize constructs only fresh heap nodes"
        );

        Search searches[] = {
            make_search(boards[0]),
            make_search(boards[1]),
            make_search(boards[2])
        };
        for (int i = 0; i < 3; ++i) {
            transposition_table.reg(
                &searches[i],
                last_primary_hash,
                depths[i],
                -SCORE_MAX,
                SCORE_MAX,
                values[i],
                moves[i]
            );
        }
        for (int i = 0; i < 3; ++i) {
            require_exact_entry(
                &searches[i],
                last_primary_hash,
                depths[i],
                values[i],
                moves[i],
                "probe entry " + std::to_string(i) + " round " + std::to_string(round)
            );
        }

        transposition_table.del(&searches[1].board, last_primary_hash);
        int lower = -SCORE_MAX;
        int upper = SCORE_MAX;
        require(
            !transposition_table.get_bounds(
                &searches[1], last_primary_hash, depths[1], &lower, &upper
            ),
            "deleted middle probe is absent"
        );
        require_exact_entry(
            &searches[0], last_primary_hash, depths[0], values[0], moves[0],
            "first probe survives middle delete"
        );
        require_exact_entry(
            &searches[2], last_primary_hash, depths[2], values[2], moves[2],
            "guard probe survives middle delete"
        );

        transposition_table.init();
        require(
            transposition_table_test_constructed_nodes.load() == expected_constructed_after_resize,
            "normal init does not reconstruct live nodes"
        );
        for (int i = 0; i < 3; ++i) {
            lower = -SCORE_MAX;
            upper = SCORE_MAX;
            require(
                !transposition_table.get_bounds(
                    &searches[i], last_primary_hash, depths[i], &lower, &upper
                ),
                "normal init clears entry " + std::to_string(i)
            );
        }
    }

    transposition_table_auto_reset_importance = auto_reset_before;
}

void test_parallel_fresh_heap_construction() {
    const uint64_t constructed_before = transposition_table_test_constructed_nodes.load();
    thread_pool.resize(3);
    require(transposition_table.resize(TEST_HASH_LEVEL), "parallel hash10 resize");
    thread_pool.resize(0);
#if TT_USE_STACK
    const uint64_t expected_constructed = constructed_before
        + TEST_HASH_SIZE + TRANSPOSITION_TABLE_N_LOOP - 1
        - TRANSPOSITION_TABLE_STACK_SIZE;
#else
    const uint64_t expected_constructed =
        constructed_before + TEST_HASH_SIZE + TRANSPOSITION_TABLE_N_LOOP - 1;
#endif
    require(
        transposition_table_test_constructed_nodes.load() == expected_constructed,
        "parallel resize constructs each fresh heap node exactly once"
    );
    transposition_table.init();
    require(
        transposition_table_test_constructed_nodes.load() == expected_constructed,
        "post-parallel normal init does not reconstruct nodes"
    );
}

void test_concurrent_disjoint_access() {
    require(transposition_table.resize(TEST_HASH_LEVEL), "concurrent hash10 resize");
    const bool auto_reset_before = transposition_table_auto_reset_importance;
    transposition_table_auto_reset_importance = false;

    constexpr int n_threads = 4;
    constexpr int iterations = 2000;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {
        threads.emplace_back([thread_idx, &failures]() {
            const Board board{
                1ULL << (thread_idx * 2),
                1ULL << (thread_idx * 2 + 1)
            };
            Search search = make_search(board);
            const uint32_t hash = 64 + thread_idx * 16;
            const int value = thread_idx * 2 - 3;
            for (int i = 0; i < iterations; ++i) {
                transposition_table.reg(
                    &search, hash, 12, -SCORE_MAX, SCORE_MAX, value, 8 + thread_idx
                );
                int lower = -SCORE_MAX;
                int upper = SCORE_MAX;
                if (
                    !transposition_table.get_bounds(&search, hash, 12, &lower, &upper) ||
                    lower != value || upper != value
                ) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    for (std::thread &thread: threads) {
        thread.join();
    }
    require(failures.load(std::memory_order_relaxed) == 0, "concurrent disjoint TT access");

    transposition_table_auto_reset_importance = auto_reset_before;
}

} // namespace

int main() {
    try {
        thread_pool.resize(0);
        test_poisoned_raw_storage_construction();
        test_resize_register_probe_delete_repeat();
        test_parallel_fresh_heap_construction();
        test_concurrent_disjoint_access();
    } catch (const std::exception &error) {
        std::cerr << "Transposition-table storage test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "Transposition-table storage tests passed (TT_USE_STACK="
              << TT_USE_STACK << ")\n";
    return 0;
}
