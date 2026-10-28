// Exercise real table rebuilds with deterministic memory snapshots and malloc failures.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>
#include "engine/board.hpp"
#include "engine/book_memory.hpp"

static size_t allocation_limit = SIZE_MAX;
static std::vector<size_t> allocations;
static void *test_malloc(size_t bytes) {
    allocations.push_back(bytes);
    return bytes > allocation_limit ? nullptr : std::malloc(bytes);
}
// All dependencies are included above: expose only the table's diagnostics.
#define private public
#define malloc test_malloc
#include "engine/book_table.hpp"
#undef malloc
#undef private

using Table = Board_table<uint64_t>;
using book_table_memory::Memory_status;
constexpr uint64_t GiB = 1ULL << 30;
static Memory_status memory;
static int queries = 0;
static Memory_status query_memory() { ++queries; return memory; }

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
static void reset_probe(Memory_status status, size_t limit = SIZE_MAX) {
    memory = status; queries = 0; allocation_limit = limit; allocations.clear();
}
static void check_values(Table &table, const std::vector<Board> &keys, int count) {
    require(table.size() == size_t(count), "board count changed");
    for (int i = 0; i < count; ++i) {
        uint64_t value;
        require(table.get(keys[i], &value) && value == uint64_t(i), "stored value changed");
    }
    require(book_table_memory::pending_rebuild_bytes.load() == 0, "memory reservation leaked");
}

int main() {
    try {
        const Memory_status ample{true, 8 * GiB, 16 * GiB};
        const Memory_status low{true, 0, 16 * GiB};
        const Memory_status unknown{};
        Table table;
        std::vector<Board> keys;
        // One real segment can cross the boundary without allocating a 67M-board book.
        for (uint64_t i = 0; keys.size() < 32776; ++i) {
            Board board{i, i ^ 0x647645643ULL};
            if (Table::segment_idx(book_hash_board(board.player, board.opponent)) == 0) keys.push_back(board);
        }
        for (int i = 0; i < 64; ++i) table.set(keys[i], i);
        auto &segment = table.segments[0];
        const auto normal = Table::n_groups_for(32768);
        const auto compact = Table::n_groups_for(32768, true);
        require(normal > compact, "fixture does not distinguish policies");

        reset_probe(low);
        require(Table::rebuild_segment(segment, 32767, query_memory), "small rebuild failed");
        require(queries == 0 && segment.n_groups == Table::n_groups_for(32767), "small segment queried memory or used 80%");
        check_values(table, keys, 64);
        reset_probe(low, 0);
        require(!Table::rebuild_segment(segment, 32767, query_memory) && queries == 0 && allocations.size() == 1,
            "small segment changed its allocation-failure behavior");
        check_values(table, keys, 64);
        std::cout << "PASS: small segment keeps 65% without a memory query\n";

        reset_probe(ample);
        require(Table::rebuild_segment(segment, 32768, query_memory), "normal rebuild failed");
        require(queries == 1 && segment.n_groups == normal && allocations.size() == 1, "large segment did not use 65% with ample memory");
        check_values(table, keys, 64);
        std::cout << "PASS: large segment keeps 65% with ample memory\n";

        reset_probe(low);
        require(Table::rebuild_segment(segment, 32768, query_memory), "compact rebuild failed");
        require(queries == 1 && segment.n_groups == compact && allocations == std::vector<size_t>{size_t(Table::array_bytes(compact))}, "low memory did not choose 80% directly");
        check_values(table, keys, 64);

        const auto required = Table::array_bytes(normal);
        reset_probe(Memory_status{true, GiB + required - 1, 16 * GiB});
        require(Table::rebuild_segment(segment, 32768, query_memory) && segment.n_groups == compact, "temporary destination or headroom was not counted");
        reset_probe(Memory_status{true, GiB + required, 16 * GiB});
        require(Table::rebuild_segment(segment, 32768, query_memory) && segment.n_groups == normal, "sufficient boundary budget rejected");
        reset_probe(Memory_status{true, GiB / 10 + required, GiB});
        require(Table::rebuild_segment(segment, 32768, query_memory) && segment.n_groups == normal, "small-machine headroom is not 10%");
        check_values(table, keys, 64);
        std::cout << "PASS: 80% only under pressure; temporary memory/headroom boundaries\n";

        reset_probe(unknown);
        require(Table::rebuild_segment(segment, 32768, query_memory) && segment.n_groups == normal, "unknown memory status did not try 65%");
        reset_probe(ample, size_t(Table::array_bytes(compact)));
        require(Table::rebuild_segment(segment, 32768, query_memory), "80% allocation fallback failed");
        require(segment.n_groups == compact && allocations == std::vector<size_t>{size_t(required), size_t(Table::array_bytes(compact))}, "allocation fallback did not try 65%, then 80%");
        check_values(table, keys, 64);
        std::cout << "PASS: failed 65% allocation retries 80%; unknown status preserves usual policy\n";

        auto *old_keys = segment.keys;
        const auto old_groups = segment.n_groups;
        reset_probe(ample, 0);
        require(!Table::rebuild_segment(segment, 32768, query_memory), "two failed allocations reported success");
        require(allocations.size() == 2 && segment.keys == old_keys && segment.n_groups == old_groups, "failed rebuild changed original segment");
        check_values(table, keys, 64);
        reset_probe(low, 0);
        require(!Table::rebuild_segment(segment, 32768, query_memory) && allocations.size() == 1, "low-memory failure retried a larger buffer");
        check_values(table, keys, 64);
        std::cout << "PASS: failed rebuilds preserve every value and release reservations\n";

        reset_probe(ample);
        {
            book_table_memory::Reservation other(4 * GiB);
            memory.available_bytes = 4 * GiB;
            require(Table::rebuild_segment(segment, 32768, query_memory) && segment.n_groups == compact, "concurrent rebuild budget ignored");
            require(book_table_memory::pending_rebuild_bytes.load() == 4 * GiB, "another reservation changed");
        }
        reset_probe(ample);
        require(Table::rebuild_segment(segment, 32768, query_memory) && segment.n_groups == normal, "80% policy stuck after memory recovered");
        check_values(table, keys, 64);

        std::atomic<int> ready{0}; std::atomic<bool> finish{false};
        std::vector<std::thread> workers;
        for (int i = 0; i < 8; ++i) workers.emplace_back([&]() {
            book_table_memory::Reservation reservation(GiB);
            ready.fetch_add(1);
            while (!finish.load()) std::this_thread::yield();
        });
        while (ready.load() != 8) std::this_thread::yield();
        const bool all_reserved = book_table_memory::pending_rebuild_bytes.load() == 8 * GiB;
        finish = true;
        for (auto &worker : workers) worker.join();
        require(all_reserved && book_table_memory::pending_rebuild_bytes.load() == 0, "parallel memory accounting failed");
        std::cout << "PASS: concurrent rebuild accounting and recovery to 65%\n";

        // Use the public insertion path to check bad_alloc and lock recovery too.
        allocation_limit = SIZE_MAX;
        table.clear();
        const uint32_t full_groups = (32768 + 6) / 7;
        require(Table::resize_segment(segment, full_groups), "fixture allocation failed");
        const int count = full_groups * 7;
        for (int i = 0; i < count; ++i) table.set(keys[i], i);
        reset_probe(ample, 0);
        bool caught = false;
        try { table.set(keys[count], count); } catch (const std::bad_alloc&) { caught = true; }
        require(caught, "insertion did not report allocation failure");
        check_values(table, keys, count);
        allocation_limit = SIZE_MAX;
        table.set(keys[count], count);
        check_values(table, keys, count + 1);
        std::cout << "PASS: public insertion survives allocation failure and unlocks its segment\n";

        const auto live = book_table_memory::query_status();
#if defined(_WIN32)
        require(live.known && live.total_physical_bytes > 0 && live.available_bytes <= live.total_physical_bytes, "Windows memory query failed");
#endif
        std::cout << "memory_query " << live.known << ' ' << live.available_bytes << ' ' << live.total_physical_bytes << '\n';
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
