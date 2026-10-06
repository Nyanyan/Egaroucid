/*
    Egaroucid Project

    @file book_table.hpp
        Compact hash table for book (board -> value)
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <new>
#include <thread>
#include <vector>
#include <algorithm>
#include "board.hpp"

#if defined(__GNUC__) || defined(__clang__)
    #define BOOK_TABLE_PREFETCH(addr) __builtin_prefetch((const void*)(addr))
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    #include <xmmintrin.h>
    #define BOOK_TABLE_PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
    #define BOOK_TABLE_PREFETCH(addr)
#endif

/*
    @brief Hash function for book

    All 64 bits are well mixed: the book table uses the upper bits to select a segment,
    the middle bits to select a group, and the lower 7 bits as a tag.

    @param player               a bitboard representing player
    @param opponent             a bitboard representing opponent
    @return hash code
*/
inline uint64_t book_hash_board(uint64_t player, uint64_t opponent) {
    uint64_t h = player * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 32;
    h = (h ^ opponent) * 0xD6E8FEB86659FD93ULL;
    h ^= h >> 32;
    h *= 0xD6E8FEB86659FD93ULL;
    h ^= h >> 32;
    return h;
}

/*
    @brief Hash function for book

    @param board                board
    @return hash code
*/
struct Book_hash {
    size_t operator()(const Board &board) const {
        return (size_t)book_hash_board(board.player, board.opponent);
    }
};

/*
    @brief lock for a segment of the book table

    Critical sections are very short, so spin first and yield only when it takes long.
*/
class Book_table_lock {
    private:
        std::atomic<bool> locked;

    public:
        Book_table_lock()
            : locked(false) {}

        inline void lock() {
            int n_spin = 0;
            while (locked.exchange(true, std::memory_order_acquire)) {
                do {
                    if (++n_spin >= 128) {
                        n_spin = 0;
                        std::this_thread::yield();
                    }
                } while (locked.load(std::memory_order_relaxed));
            }
        }

        inline void unlock() {
            locked.store(false, std::memory_order_release);
        }
};

/*
    @brief Hash table with a board as a key

    Open addressing table split into independent segments.
        - one segment is one memory block: keys (16 bytes), values, and control bytes (1 byte) for each slot,
          so that memory per registered board is about (17 + sizeof(Value)) / load factor bytes
        - slots are grouped by 8; a group is searched with one 64-bit operation on its control bytes
        - each segment grows by itself, so that neither a long pause nor a large temporary memory is needed
        - each segment has its own lock; get / set / modify / erase are thread safe
        - for_each functions take no lock: do not insert / erase from other threads while iterating
        - if the memory for a new board can't be allocated, upsert / set / reserve throw std::bad_alloc
          (as std::unordered_map does); the table is not changed and no lock is left locked

    Value must be trivially copyable.
*/
template<class Value>
class Board_table {
    public:
        static constexpr int N_SEGMENT_BITS = 11;
        static constexpr int N_SEGMENTS = 1 << N_SEGMENT_BITS;

    private:
        static constexpr uint32_t GROUP_SIZE = 8;
        static constexpr uint32_t MAX_USED_PER_GROUP = 7; // max load factor 7/8
        static constexpr uint32_t NOT_FOUND = 0xFFFFFFFFU;
        static constexpr uint8_t CTRL_EMPTY = 0x80;
        static constexpr uint8_t CTRL_DELETED = 0xFE;
        static constexpr uint64_t CTRL_LSBS = 0x0101010101010101ULL;
        static constexpr uint64_t CTRL_MSBS = 0x8080808080808080ULL;
        // load factor just after a segment is resized by itself:
        // about 0.65 for a small segment (few rebuilds), about 0.80 for a large segment.
        // A book is loaded at the maximum load factor, so every segment is resized soon after boards are added (book learning).
        // With 0.65 the memory of the whole book grows by 1/3 at that time (112M boards + 2%: 3,352 MiB -> 4,468 MiB,
        // with 0.80: 3,652 MiB). In a big book this memory matters more than the time of the rebuilds
        // (3 times as many element copies per registered board) and of the search in a fuller table.
        static constexpr uint32_t RESIZE_TARGET_NUMERATOR = 100;
        static constexpr uint32_t RESIZE_TARGET_DENOMINATOR = 65 * GROUP_SIZE;
        static constexpr uint32_t RESIZE_TARGET_DENOMINATOR_LARGE = 80 * GROUP_SIZE;
        static constexpr uint32_t LARGE_SEGMENT_N_BOARDS = 32768; // 1 / 2048 of a book of 67M boards

        // The lock is written at every access, and the others are only read in searching.
        // They are placed on different cache lines, so that searching threads do not disturb each other.
        struct alignas(64) Segment_lock {
            Book_table_lock lock;
        };

        struct alignas(64) Segment {
            Book_table_lock *lock;
            uint32_t n_groups;      // capacity is n_groups * GROUP_SIZE
            uint32_t n_used;        // number of registered boards
            uint32_t n_deleted;     // number of deleted marks
            Board *keys;            // head of the memory block
            Value *values;
            uint8_t *ctrl;          // 0x00-0x7F: registered (tag), CTRL_EMPTY, CTRL_DELETED

            Segment()
                : lock(nullptr), n_groups(0), n_used(0), n_deleted(0), keys(nullptr), values(nullptr), ctrl(nullptr) {}
        };

        Segment *segments;
        Segment_lock *segment_locks;
        std::atomic<int64_t> n_elems;

        struct Segment_guard {
            const Segment &segment;
            Segment_guard(const Segment &segment_)
                : segment(segment_) {
                segment.lock->lock();
            }
            ~Segment_guard() {
                segment.lock->unlock();
            }
        };

        static inline uint32_t segment_idx(uint64_t hash) {
            return (uint32_t)(hash >> (64 - N_SEGMENT_BITS));
        }

        static inline uint8_t tag_of(uint64_t hash) {
            return (uint8_t)(hash & 0x7F);
        }

        static inline uint32_t home_group(uint64_t hash, uint32_t n_groups) {
            return (uint32_t)(((uint64_t)(uint32_t)(hash >> 7) * (uint64_t)n_groups) >> 32);
        }

        static inline uint64_t load_ctrl(const uint8_t *ctrl, uint32_t group) {
            uint64_t res;
            memcpy(&res, ctrl + (size_t)group * GROUP_SIZE, sizeof(uint64_t));
            return res;
        }

        // bytes equal to tag (may include false positives on upper bytes, keys are compared anyway)
        static inline uint64_t match_tag(uint64_t ctrl, uint8_t tag) {
            uint64_t x = ctrl ^ (CTRL_LSBS * tag);
            return (x - CTRL_LSBS) & ~x & CTRL_MSBS;
        }

        static inline uint64_t match_empty(uint64_t ctrl) {
            return ctrl & ~(ctrl << 6) & CTRL_MSBS;
        }

        static inline uint64_t match_empty_or_deleted(uint64_t ctrl) {
            return ctrl & CTRL_MSBS;
        }

        // control bytes are read as a little endian 64-bit integer
        static inline uint32_t first_match_idx(uint64_t match) {
            return (uint32_t)ctz(match) >> 3;
        }

        static inline bool allocate_arrays(uint32_t n_groups, Board **keys, Value **values, uint8_t **ctrl) {
            size_t capacity = (size_t)n_groups * GROUP_SIZE;
            void *block = malloc(capacity * (sizeof(Board) + sizeof(Value) + 1));
            if (block == nullptr) {
                return false;
            }
            *keys = (Board*)block;
            *values = (Value*)((uint8_t*)block + capacity * sizeof(Board));
            *ctrl = (uint8_t*)block + capacity * (sizeof(Board) + sizeof(Value));
            memset(*ctrl, CTRL_EMPTY, capacity);
            return true;
        }

        static inline uint32_t find_slot(const Segment &segment, const Board &key, uint64_t hash) {
            if (segment.n_groups == 0) {
                return NOT_FOUND;
            }
            const uint8_t tag = tag_of(hash);
            uint32_t group = home_group(hash, segment.n_groups);
            for (;;) { // every segment has at least 1 empty slot
                uint64_t ctrl = load_ctrl(segment.ctrl, group);
                uint64_t match = match_tag(ctrl, tag);
                while (match) {
                    uint32_t slot = group * GROUP_SIZE + first_match_idx(match);
                    if (segment.keys[slot] == key) {
                        return slot;
                    }
                    match &= match - 1;
                }
                if (match_empty(ctrl)) {
                    return NOT_FOUND;
                }
                if (++group == segment.n_groups) {
                    group = 0;
                }
            }
        }

        // the key must not be registered, and the segment must have room
        static inline uint32_t insert_new(Board *keys, Value *values, uint8_t *ctrl_arr, uint32_t n_groups, const Board &key, const Value &value, uint64_t hash) {
            uint32_t group = home_group(hash, n_groups);
            for (;;) {
                uint64_t available = match_empty_or_deleted(load_ctrl(ctrl_arr, group));
                if (available) {
                    uint32_t slot = group * GROUP_SIZE + first_match_idx(available);
                    keys[slot] = key;
                    values[slot] = value;
                    ctrl_arr[slot] = tag_of(hash);
                    return slot;
                }
                if (++group == n_groups) {
                    group = 0;
                }
            }
        }

        // rebuild a segment with new capacity (deleted marks are removed)
        // returns false if the memory can't be allocated (the segment is not changed)
        static inline bool resize_segment(Segment &segment, uint32_t new_n_groups) {
            if (new_n_groups == 0) {
                free(segment.keys);
                segment.keys = nullptr;
                segment.values = nullptr;
                segment.ctrl = nullptr;
                segment.n_groups = 0;
                segment.n_deleted = 0;
                return true;
            }
            Board *new_keys;
            Value *new_values;
            uint8_t *new_ctrl;
            if (!allocate_arrays(new_n_groups, &new_keys, &new_values, &new_ctrl)) {
                return false;
            }
            const uint32_t capacity = segment.n_groups * GROUP_SIZE;
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                if (segment.ctrl[slot] < CTRL_EMPTY) {
                    const Board &key = segment.keys[slot];
                    insert_new(new_keys, new_values, new_ctrl, new_n_groups, key, segment.values[slot], book_hash_board(key.player, key.opponent));
                }
            }
            free(segment.keys);
            segment.keys = new_keys;
            segment.values = new_values;
            segment.ctrl = new_ctrl;
            segment.n_groups = new_n_groups;
            segment.n_deleted = 0;
            return true;
        }

        static inline uint32_t n_groups_for(uint32_t n_used) {
            const uint32_t denominator = n_used < LARGE_SEGMENT_N_BOARDS ? RESIZE_TARGET_DENOMINATOR : RESIZE_TARGET_DENOMINATOR_LARGE;
            return (uint32_t)(((uint64_t)n_used * RESIZE_TARGET_NUMERATOR + denominator - 1) / denominator);
        }

        // find the slot of the key, or register the key with init_value
        static inline uint32_t find_or_insert_slot(Segment &segment, const Board &key, const Value &init_value, uint64_t hash, bool *inserted) {
            uint32_t insert_slot = NOT_FOUND;
            if (segment.n_groups) {
                const uint8_t tag = tag_of(hash);
                uint32_t group = home_group(hash, segment.n_groups);
                for (;;) {
                    uint64_t ctrl = load_ctrl(segment.ctrl, group);
                    uint64_t match = match_tag(ctrl, tag);
                    while (match) {
                        uint32_t slot = group * GROUP_SIZE + first_match_idx(match);
                        if (segment.keys[slot] == key) {
                            *inserted = false;
                            return slot;
                        }
                        match &= match - 1;
                    }
                    if (insert_slot == NOT_FOUND) {
                        uint64_t available = match_empty_or_deleted(ctrl);
                        if (available) {
                            insert_slot = group * GROUP_SIZE + first_match_idx(available);
                        }
                    }
                    if (match_empty(ctrl)) {
                        break;
                    }
                    if (++group == segment.n_groups) {
                        group = 0;
                    }
                }
            }
            *inserted = true;
            if (insert_slot != NOT_FOUND && segment.ctrl[insert_slot] == CTRL_DELETED) { // reuse a deleted slot
                segment.keys[insert_slot] = key;
                segment.values[insert_slot] = init_value;
                segment.ctrl[insert_slot] = tag_of(hash);
                --segment.n_deleted;
                ++segment.n_used;
                return insert_slot;
            }
            if (segment.n_used + segment.n_deleted + 1 > segment.n_groups * MAX_USED_PER_GROUP) { // no room
                if (!resize_segment(segment, n_groups_for(segment.n_used + 1))) {
                    // Out of memory: the segment is not changed, and the lock is released by the guard of the caller.
                    // (std::exit() must not be called here. It runs the destructors of the global objects with this segment
                    //  locked: the thread pool then waits for its threads, which wait for this lock, and the process never ends.)
                    throw std::bad_alloc();
                }
                insert_slot = insert_new(segment.keys, segment.values, segment.ctrl, segment.n_groups, key, init_value, hash);
            } else {
                segment.keys[insert_slot] = key;
                segment.values[insert_slot] = init_value;
                segment.ctrl[insert_slot] = tag_of(hash);
            }
            ++segment.n_used;
            return insert_slot;
        }

    public:
        Board_table()
            : n_elems(0) {
            segments = new Segment[N_SEGMENTS];
            segment_locks = new Segment_lock[N_SEGMENTS];
            for (int i = 0; i < N_SEGMENTS; ++i) {
                segments[i].lock = &segment_locks[i].lock;
            }
        }

        ~Board_table() {
            for (int i = 0; i < N_SEGMENTS; ++i) {
                free(segments[i].keys);
            }
            delete[] segments;
            delete[] segment_locks;
        }

        Board_table(const Board_table&) = delete;
        Board_table& operator=(const Board_table&) = delete;

        /*
            @brief get the number of registered boards
        */
        inline size_t size() const {
            return (size_t)n_elems.load(std::memory_order_relaxed);
        }

        /*
            @brief get registered value

            @param key                  board
            @param res                  the value is copied here if the board is registered
            @return registered?
        */
        inline bool get(const Board &key, Value *res) const {
            return get(key, book_hash_board(key.player, key.opponent), res);
        }

        /*
            @brief load the memory to be read by get() into the cache

            When several boards are searched, call this for all boards first.
            The table is much larger than the cache with a big book, and this hides most of the memory latency.

            @param hash                 book_hash_board() of the board
        */
        inline void prefetch(uint64_t hash) const {
            const Segment &segment = segments[segment_idx(hash)];
            Segment_guard guard(segment);
            if (segment.n_groups) {
                BOOK_TABLE_PREFETCH(segment.ctrl + (size_t)home_group(hash, segment.n_groups) * GROUP_SIZE);
            }
        }

        /*
            @brief load the memory of the board itself into the cache

            Call this after prefetch() and before get() / modify(), when several boards are searched.

            @param hash                 book_hash_board() of the board
        */
        inline void prefetch_entry(uint64_t hash) const {
            const Segment &segment = segments[segment_idx(hash)];
            Segment_guard guard(segment);
            if (segment.n_groups) {
                uint32_t group = home_group(hash, segment.n_groups);
                uint64_t match = match_tag(load_ctrl(segment.ctrl, group), tag_of(hash));
                if (match) {
                    uint32_t slot = group * GROUP_SIZE + first_match_idx(match);
                    BOOK_TABLE_PREFETCH(segment.keys + slot);
                    BOOK_TABLE_PREFETCH(segment.values + slot);
                }
            }
        }

        /*
            @brief functions without lock

            Faster when many threads search at the same time (a lock is a write to shared memory).
            Use them only while no thread registers / deletes boards.
            Different threads may modify different boards.
        */
        inline void prefetch_unlocked(uint64_t hash) const {
            const Segment &segment = segments[segment_idx(hash)];
            if (segment.n_groups) {
                BOOK_TABLE_PREFETCH(segment.ctrl + (size_t)home_group(hash, segment.n_groups) * GROUP_SIZE);
            }
        }

        inline void prefetch_entry_unlocked(uint64_t hash) const {
            const Segment &segment = segments[segment_idx(hash)];
            if (segment.n_groups) {
                uint32_t group = home_group(hash, segment.n_groups);
                uint64_t match = match_tag(load_ctrl(segment.ctrl, group), tag_of(hash));
                if (match) {
                    uint32_t slot = group * GROUP_SIZE + first_match_idx(match);
                    BOOK_TABLE_PREFETCH(segment.keys + slot);
                    BOOK_TABLE_PREFETCH(segment.values + slot);
                }
            }
        }

        inline bool get_unlocked(const Board &key, uint64_t hash, Value *res) const {
            const Segment &segment = segments[segment_idx(hash)];
            uint32_t slot = find_slot(segment, key, hash);
            if (slot == NOT_FOUND) {
                return false;
            }
            *res = segment.values[slot];
            return true;
        }

        template<class F>
        inline bool modify_unlocked(const Board &key, uint64_t hash, F &&func) {
            Segment &segment = segments[segment_idx(hash)];
            uint32_t slot = find_slot(segment, key, hash);
            if (slot == NOT_FOUND) {
                return false;
            }
            func(segment.values[slot]);
            return true;
        }

        /*
            @brief get registered value with precalculated hash

            @param key                  board
            @param hash                 book_hash_board() of the board
            @param res                  the value is copied here if the board is registered
            @return registered?
        */
        inline bool get(const Board &key, uint64_t hash, Value *res) const {
            const Segment &segment = segments[segment_idx(hash)];
            Segment_guard guard(segment);
            uint32_t slot = find_slot(segment, key, hash);
            if (slot == NOT_FOUND) {
                return false;
            }
            *res = segment.values[slot];
            return true;
        }

        /*
            @brief check if a board is registered
        */
        inline bool contains(const Board &key) const {
            const uint64_t hash = book_hash_board(key.player, key.opponent);
            const Segment &segment = segments[segment_idx(hash)];
            Segment_guard guard(segment);
            return find_slot(segment, key, hash) != NOT_FOUND;
        }

        /*
            @brief modify registered value

            @param key                  board
            @param func                 func(Value&) is called if the board is registered
            @return registered?
        */
        template<class F>
        inline bool modify(const Board &key, F &&func) {
            return modify(key, book_hash_board(key.player, key.opponent), func);
        }

        template<class F>
        inline bool modify(const Board &key, uint64_t hash, F &&func) {
            Segment &segment = segments[segment_idx(hash)];
            Segment_guard guard(segment);
            uint32_t slot = find_slot(segment, key, hash);
            if (slot == NOT_FOUND) {
                return false;
            }
            func(segment.values[slot]);
            return true;
        }

        /*
            @brief register a board, or modify registered value

            @param key                  board
            @param init_value           value to register if the board is not registered
            @param func                 func(Value&) is called if the board is already registered
            @param count                false: the caller counts new boards and calls add_size() afterwards
            @return newly registered?
        */
        template<class F>
        inline bool upsert(const Board &key, const Value &init_value, F &&func, bool count = true) {
            const uint64_t hash = book_hash_board(key.player, key.opponent);
            Segment &segment = segments[segment_idx(hash)];
            bool inserted;
            {
                Segment_guard guard(segment);
                uint32_t slot = find_or_insert_slot(segment, key, init_value, hash, &inserted);
                if (!inserted) {
                    func(segment.values[slot]);
                }
            }
            if (inserted && count) {
                n_elems.fetch_add(1, std::memory_order_relaxed);
            }
            return inserted;
        }

        /*
            @brief register a board, or overwrite registered value

            @return newly registered?
        */
        inline bool set(const Board &key, const Value &value) {
            return upsert(key, value, [&value](Value &registered) { registered = value; });
        }

        inline void add_size(int64_t n) {
            n_elems.fetch_add(n, std::memory_order_relaxed);
        }

        /*
            @brief delete a board

            @return deleted?
        */
        inline bool erase(const Board &key) {
            const uint64_t hash = book_hash_board(key.player, key.opponent);
            Segment &segment = segments[segment_idx(hash)];
            {
                Segment_guard guard(segment);
                uint32_t slot = find_slot(segment, key, hash);
                if (slot == NOT_FOUND) {
                    return false;
                }
                // a group that has an empty slot has never been full, so no search passes it
                if (match_empty(load_ctrl(segment.ctrl, slot / GROUP_SIZE))) {
                    segment.ctrl[slot] = CTRL_EMPTY;
                } else {
                    segment.ctrl[slot] = CTRL_DELETED;
                    ++segment.n_deleted;
                }
                --segment.n_used;
                if (segment.n_used == 0) {
                    resize_segment(segment, 0);
                } else if (segment.n_groups > 1 && segment.n_used * 5 < segment.n_groups * GROUP_SIZE) { // too sparse
                    resize_segment(segment, n_groups_for(segment.n_used)); // out of memory: the segment is kept as it is
                }
            }
            n_elems.fetch_sub(1, std::memory_order_relaxed);
            return true;
        }

        /*
            @brief delete all boards and release memory
        */
        inline void clear() {
            for (int i = 0; i < N_SEGMENTS; ++i) {
                Segment &segment = segments[i];
                Segment_guard guard(segment);
                resize_segment(segment, 0);
                segment.n_used = 0;
            }
            n_elems.store(0, std::memory_order_relaxed);
        }

        /*
            @brief prepare for registering boards

            @param n                    expected number of boards after registration
        */
        inline void reserve(size_t n) {
            double n_per_segment = (double)n / N_SEGMENTS;
            // boards are distributed to segments at random: add 3 sigma
            uint64_t n_expected = (uint64_t)(n_per_segment + 3.0 * sqrt(n_per_segment) + 1.0);
            uint64_t n_groups = (n_expected + MAX_USED_PER_GROUP - 1) / MAX_USED_PER_GROUP;
            if (n_groups >= (1ULL << 29)) {
                return; // too large, segments grow by themselves
            }
            for (int i = 0; i < N_SEGMENTS; ++i) {
                Segment &segment = segments[i];
                Segment_guard guard(segment);
                if (segment.n_groups < (uint32_t)n_groups && !resize_segment(segment, (uint32_t)n_groups)) {
                    throw std::bad_alloc(); // out of memory (the lock is released by the guard)
                }
            }
        }

        /*
            @brief release unused memory
        */
        inline void shrink_to_fit() {
            for (int i = 0; i < N_SEGMENTS; ++i) {
                Segment &segment = segments[i];
                Segment_guard guard(segment);
                uint32_t n_groups = (segment.n_used + MAX_USED_PER_GROUP - 1) / MAX_USED_PER_GROUP;
                if (segment.n_used % MAX_USED_PER_GROUP == 0 && segment.n_used) {
                    ++n_groups; // keep at least 1 empty slot
                }
                if (n_groups < segment.n_groups) {
                    resize_segment(segment, n_groups); // out of memory: the segment is kept as it is
                }
            }
        }

        /*
            @brief memory used by this table in bytes
        */
        inline size_t memory_usage() const {
            size_t res = sizeof(Segment) * N_SEGMENTS;
            for (int i = 0; i < N_SEGMENTS; ++i) {
                res += (size_t)segments[i].n_groups * GROUP_SIZE * (sizeof(Board) + sizeof(Value) + 1);
            }
            return res;
        }

        /*
            @brief call func(const Board&, Value&) for all boards in a segment

            No lock is taken: boards must not be registered / deleted while iterating.
            If func returns false, iteration stops.

            @return false if stopped
        */
        template<class F>
        inline bool for_each_in_segment_while(int segment_idx_, F &&func) {
            Segment &segment = segments[segment_idx_];
            const uint32_t capacity = segment.n_groups * GROUP_SIZE;
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                if (segment.ctrl[slot] < CTRL_EMPTY) {
                    if (!func((const Board&)segment.keys[slot], segment.values[slot])) {
                        return false;
                    }
                }
            }
            return true;
        }

        template<class F>
        inline void for_each_in_segment(int segment_idx_, F &&func) {
            for_each_in_segment_while(segment_idx_, [&func](const Board &key, Value &value) {
                func(key, value);
                return true;
            });
        }

        /*
            @brief call func(const Board&, Value&) for all boards

            No lock is taken: boards must not be registered / deleted while iterating.
        */
        template<class F>
        inline void for_each(F &&func) {
            for (int i = 0; i < N_SEGMENTS; ++i) {
                for_each_in_segment(i, func);
            }
        }

        /*
            @brief call func(const Board&, Value&) for all boards while func returns true

            @return false if stopped
        */
        template<class F>
        inline bool for_each_while(F &&func) {
            for (int i = 0; i < N_SEGMENTS; ++i) {
                if (!for_each_in_segment_while(i, func)) {
                    return false;
                }
            }
            return true;
        }

        /*
            @brief call func(const Board&, const Value&) for all boards in a segment with the segment locked

            Safe even if other threads register / delete boards.
            func must not access this table. If func returns false, iteration stops.

            @return false if stopped
        */
        template<class F>
        inline bool for_each_in_segment_locked_while(int segment_idx_, F &&func) const {
            const Segment &segment = segments[segment_idx_];
            Segment_guard guard(segment);
            const uint32_t capacity = segment.n_groups * GROUP_SIZE;
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                if (segment.ctrl[slot] < CTRL_EMPTY) {
                    if (!func((const Board&)segment.keys[slot], (const Value&)segment.values[slot])) {
                        return false;
                    }
                }
            }
            return true;
        }

        /*
            @brief copy all boards in a segment

            Used for iterating safely while the table is modified.
        */
        /*
            @brief copy the boards in a segment which satisfy pred(const Board&, const Value&)

            pred must not access this table (the segment is locked).
        */
        template<class F>
        inline void get_keys_in_segment_if(int segment_idx_, std::vector<Board> &res, F &&pred) const {
            res.clear();
            const Segment &segment = segments[segment_idx_];
            Segment_guard guard(segment);
            const uint32_t capacity = segment.n_groups * GROUP_SIZE;
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                if (segment.ctrl[slot] < CTRL_EMPTY && pred((const Board&)segment.keys[slot], (const Value&)segment.values[slot])) {
                    res.emplace_back(segment.keys[slot]);
                }
            }
        }

        inline void get_keys_in_segment(int segment_idx_, std::vector<Board> &res) const {
            res.clear();
            const Segment &segment = segments[segment_idx_];
            Segment_guard guard(segment);
            res.reserve(segment.n_used);
            const uint32_t capacity = segment.n_groups * GROUP_SIZE;
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                if (segment.ctrl[slot] < CTRL_EMPTY) {
                    res.emplace_back(segment.keys[slot]);
                }
            }
        }
};
