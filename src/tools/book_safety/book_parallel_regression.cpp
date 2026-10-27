#include <algorithm>
#include <random>
#include <stdexcept>
#include <tuple>
#include "engine/evaluate.hpp"
#include "engine/book_table.hpp"
#include "checked_book_table.hpp"
#define Board_table Checked_book_table
#include "engine/engine_all.hpp"
#undef Board_table

using Snapshot = std::vector<std::tuple<uint64_t, uint64_t, int, uint32_t, bool>>;

Snapshot snapshot(Book& target) {
    Snapshot result;
    target.for_each_board_snapshot([&](Board b) {
        Book_elem value = target.get(b);
        result.emplace_back(b.player, b.opponent, value.value, value.n_lines, value.seen);
        return true;
    });
    std::sort(result.begin(), result.end());
    return result;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void set_threads(int count) {
    thread_pool.resize(count - 1);
    while (thread_pool.get_n_idle() < count - 1) std::this_thread::yield();
}

int main() {
    try {
        bit_init(); mobility_init(); flip_init();
        std::mt19937_64 random(643);
        std::unordered_map<Board, std::unordered_set<Board, Book_hash>, Book_hash> parents;
        book.delete_all();
        for (int game = 0; game < 180; ++game) {
            Board b; b.reset();
            while (b.check_pass()) {
                Board canonical = representative_board(b);
                const int value = int(book_hash_board(canonical.player, canonical.opponent) % 41) - 20;
                book.change(b, value, 1);
                uint64_t legal = b.get_legal();
                int nth = random() % pop_count_ull(legal), cell = 0;
                for (; cell < 64; ++cell) if ((legal >> cell) & 1) { if (nth-- == 0) break; }
                Flip flip; calc_flip(&flip, &b, cell); b.move_board(&flip);
                if (!b.get_legal() && !b.is_end()) b.pass();
                parents[representative_board(b)].insert(canonical);
            }
            book.change(b, b.score_player(), 60);
        }
        // Enumerate short openings as well: different move orders deliberately converge,
        // so coverage of shared children does not depend on rare random-game transpositions.
        Board initial; initial.reset();
        std::vector<Board> frontier{initial};
        for (int depth = 0; depth < 7; ++depth) {
            std::unordered_set<Board, Book_hash> next;
            for (Board b : frontier) {
                book.change(b, int(book_hash_board(b.player, b.opponent) % 41) - 20, 1);
                uint64_t legal = b.get_legal();
                for (int cell = 0; cell < 64; ++cell) if ((legal >> cell) & 1) {
                    Flip flip; calc_flip(&flip, &b, cell);
                    Board child = b; child.move_board(&flip);
                    if (!child.get_legal() && !child.is_end()) child.pass();
                    child = representative_board(child);
                    parents[child].insert(b);
                    next.insert(child);
                }
            }
            frontier.assign(next.begin(), next.end());
        }
        for (Board b : frontier) book.change(b, int(book_hash_board(b.player, b.opponent) % 41) - 20, 1);
        int transpositions = 0;
        for (const auto& entry : parents) if (entry.second.size() > 1) ++transpositions;
        require(transpositions > 20, "fixture has too few shared children reached by different parents");
        std::cout << "fixture boards=" << book.size() << " transposed_children=" << transpositions << '\n';

        Board root; root.reset(); bool stop = false;
        set_threads(1);
        book.reset_seen();
        const int max_discs = book.flag_reachable_boards(root, true, &stop);
        const Snapshot reachable = snapshot(book);
        book.reset_seen();
        std::unordered_set<Board, Book_hash> expected_pass_boards;
        book.get_pass_boards(root, expected_pass_boards, false);
        const Snapshot pass_seen = snapshot(book);

        bool race_free = true;
        for (int threads : {4, 8}) {
            set_threads(threads);
            for (int repeat = 0; repeat < 3; ++repeat) {
                book.reset_seen();
                book_access_probe::conflicts = 0;
                book_access_probe::enabled = true;
                const int actual_max = book.flag_reachable_boards(root, true, &stop);
                book_access_probe::enabled = false;
                std::cout << "reachable threads=" << threads << " repeat=" << repeat << " conflicts=" << book_access_probe::conflicts << '\n';
                race_free &= book_access_probe::conflicts == 0;
                require(actual_max == max_discs && snapshot(book) == reachable, "reachable traversal changed its result");

                book.reset_seen();
                std::unordered_set<Board, Book_hash> actual_pass_boards;
                book_access_probe::conflicts = 0;
                book_access_probe::enabled = true;
                book.get_pass_boards(root, actual_pass_boards, false);
                book_access_probe::enabled = false;
                std::cout << "pass boards threads=" << threads << " repeat=" << repeat << " conflicts=" << book_access_probe::conflicts << '\n';
                race_free &= book_access_probe::conflicts == 0;
                require(actual_pass_boards == expected_pass_boards && snapshot(book) == pass_seen, "pass traversal changed its result");
            }
        }
        static_assert(sizeof(Book_elem) == 12, "book element grew");
        static_assert(std::is_trivially_copyable<Book_elem>::value, "book element lost trivial copying");
        require(race_free, "overlapping shared-child accesses detected");
        std::cout << "PASS: shared-child accesses and serial/parallel traversal results\n";
    } catch (const std::exception& e) {
        book_access_probe::enabled = false;
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
