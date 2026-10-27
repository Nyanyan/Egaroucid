// Test-only wrapper: observe overlapping accesses to the same stored element.
// The production table and its actual segment locks are used unchanged.
#pragma once
#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace book_access_probe {
    struct Active { int readers = 0, writers = 0; };
    inline std::mutex mutex;
    inline std::unordered_map<const void*, Active> active;
    inline std::atomic<bool> enabled{false};
    inline std::atomic<int> conflicts{0};

    struct Access {
        const void* address;
        bool write, observing;
        Access(const void* address_, bool write_) : address(address_), write(write_), observing(enabled.load()) {
            if (!observing) return;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto& state = active[address];
                if (state.writers || (write && state.readers)) ++conflicts;
                if (write) ++state.writers; else ++state.readers;
            }
            // Make transposed-parent tasks overlap reliably without changing book values.
            for (int i = 0; i < 8; ++i) std::this_thread::yield();
        }
        ~Access() {
            if (!observing) return;
            std::lock_guard<std::mutex> lock(mutex);
            auto& state = active[address];
            if (write) --state.writers; else --state.readers;
        }
    };
}

template<class Value>
class Checked_book_table : public Board_table<Value> {
    using Base = Board_table<Value>;
public:
    bool get(const Board& key, Value* result) const {
        return get(key, book_hash_board(key.player, key.opponent), result);
    }
    bool get(const Board& key, uint64_t hash, Value* result) const {
        return const_cast<Checked_book_table*>(this)->Base::modify(key, hash, [&](Value& value) {
            book_access_probe::Access access(&value, false);
            *result = value;
        });
    }
    bool get_unlocked(const Board& key, uint64_t hash, Value* result) const {
        return const_cast<Checked_book_table*>(this)->Base::modify_unlocked(key, hash, [&](Value& value) {
            book_access_probe::Access access(&value, false);
            *result = value;
        });
    }
    template<class F>
    bool modify(const Board& key, F&& function) {
        return modify(key, book_hash_board(key.player, key.opponent), std::forward<F>(function));
    }
    template<class F>
    bool modify(const Board& key, uint64_t hash, F&& function) {
        return Base::modify(key, hash, [&](Value& value) {
            book_access_probe::Access access(&value, true);
            function(value);
        });
    }
    template<class F>
    bool modify_unlocked(const Board& key, uint64_t hash, F&& function) {
        return Base::modify_unlocked(key, hash, [&](Value& value) {
            book_access_probe::Access access(&value, true);
            function(value);
        });
    }
};
