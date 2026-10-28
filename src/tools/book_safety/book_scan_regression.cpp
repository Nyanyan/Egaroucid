#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <vector>
#include "engine/engine_all.hpp"

using Candidate = std::tuple<int, int, int, uint64_t, uint64_t, int, int>;
using Candidates = std::vector<Candidate>;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void set_threads(int n) {
    thread_pool.resize(n - 1);
    while (thread_pool.get_n_idle() != n - 1) std::this_thread::yield();
}

Candidate describe(const Book_deviate_todo_elem &e) {
    return {-e.remaining_error, e.board.n_discs(), e.player, e.board.player, e.board.opponent,
            e.max_error_per_move, e.max_leaf_error};
}

Candidates describe(const std::vector<Book_deviate_todo_elem> &list) {
    Candidates result;
    for (const auto &e: list) result.push_back(describe(e));
    return result;
}

void make_opening_book() {
    book.delete_all();
    Board root; root.reset();
    std::vector<Board> frontier{representative_board(root)};
    for (int depth = 0; depth < 7; ++depth) {
        std::unordered_set<Board, Book_hash> next;
        for (Board b: frontier) {
            book.change(b, 0, 1);
            uint64_t legal = b.get_legal();
            for (int cell = 0; cell < 64; ++cell) if ((legal >> cell) & 1) {
                Flip flip; calc_flip(&flip, &b, cell);
                Board child = b; child.move_board(&flip);
                if (!child.get_legal() && !child.is_end()) child.pass();
                next.insert(representative_board(child));
            }
        }
        frontier.assign(next.begin(), next.end());
    }
    for (Board b: frontier) {
        book.change(b, 0, 1);
        uint64_t legal = b.get_legal();
        book.add_leaf(&b, 0, first_bit(&legal), 1);
    }
    require(book.size() == 12823 && frontier.size() == 10649, "opening fixture changed");
}

struct Result { Candidates black_best, white_best, merged, leaves; };

Result scan(int sum) {
    Board root; root.reset(); Board copy = root;
    int player = BLACK; bool learning = true;
    Book_deviate_todo_elem start;
    start.board = root; start.player = BLACK; start.max_error_per_move = 6;
    start.remaining_error = sum; start.max_leaf_error = 6;
    std::vector<std::vector<Book_deviate_todo_elem>> a(sum + 1), b(sum + 1);
    Result result;
    book.reset_seen();
    get_book_deviate_todo(start, BLACK, 8, a, tim(), &learning, &copy, &player, 1);
    book.reset_seen();
    get_book_deviate_todo(start, WHITE, 8, b, tim(), &learning, &copy, &player, 1);
    for (const auto &bucket: a) {
        auto values = describe(bucket);
        require(std::is_sorted(values.begin(), values.end()), "black-best candidate priority is not stable");
        result.black_best.insert(result.black_best.end(), values.begin(), values.end());
    }
    for (const auto &bucket: b) {
        auto values = describe(bucket);
        require(std::is_sorted(values.begin(), values.end()), "white-best candidate priority is not stable");
        result.white_best.insert(result.white_best.end(), values.begin(), values.end());
    }
    book.reset_seen();
    result.merged = describe(merge_book_deviate_todo(a, b));
    book.reset_seen();
    std::vector<Book_deviate_todo_elem> leaves;
    get_book_recalculate_leaf_todo(start, 8, 2, leaves, tim(), &learning, &copy, &player, false);
    result.leaves = describe(leaves);
    require(std::is_sorted(result.leaves.begin(), result.leaves.end()), "leaf candidate priority is not stable");
    book.reset_seen();
    return result;
}

int main() {
    try {
        bit_init(); mobility_init(); flip_init();
        make_opening_book();
        for (bool varied_scores: {false, true}) {
            if (varied_scores) book.for_each_board_snapshot([](Board b) {
                book.change(b, int(book_hash_board(b.player, b.opponent) % 3) - 1, 1);
                return true;
            });
            const int sum = varied_scores ? 6 : 0;
            set_threads(1);
            const Result expected = scan(sum);
            if (!varied_scores) {
                require(expected.merged.size() == 10649, "expansion candidates were lost or repeated");
                require(expected.leaves.size() == book.size(), "leaf candidates were lost or repeated");
            }
            for (int threads: {1, 8, 32}) for (int repeat = 0; repeat < 3; ++repeat) {
                set_threads(threads);
                Result actual = scan(sum);
                require(actual.black_best == expected.black_best && actual.white_best == expected.white_best,
                        "candidate order or metadata depends on scheduling");
                require(actual.merged == expected.merged && actual.leaves == expected.leaves,
                        "merged or leaf candidate order/metadata depends on scheduling");
            }
            std::cout << "PASS: ordered expansion/leaf candidates, scores=" << varied_scores
                      << " expanded=" << expected.merged.size() << " leaves=" << expected.leaves.size()
                      << " (1/8/32 threads, 3 repeats)\n";
        }
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
