/*
    Egaroucid Project
    Four-way, full-board transposition table that shares endgame records among
    positions differing only in the colors of solid discs (Edax-style).
    SPDX-License-Identifier: GPL-3.0-or-later
*/
#pragma once
#include <algorithm>
#include <climits>
#include "stability.hpp"

// A solid disc has all four of its lines full. It can never flip and never
// takes part in a flip, so positions that differ only in the colors of solid
// discs have the same moves and, when the game ends on a full board, scores
// that differ by twice the solid discs moved between the players.
//
// Every record of a position lives in the bucket of its normalized board, in
// which the side to move gives its solid discs to the opponent. Endgame
// records (depth >= empties) are stored for the normalized board as its disc
// difference; a reader adds back twice its own solid discs. Shallower records
// keep the real board, because the evaluation function sees disc colors. As
// in Edax, a game ending with empty squares that changes the winner between
// two such positions makes the shift inexact.

inline uint64_t dogai_tt_solid_mask(uint64_t occupied) {
    const uint64_t h = full_stability_h(occupied);
    if (!h) return 0;
    const uint64_t v = full_stability_v(occupied);
    if (!(h & v)) return 0;
    uint64_t full_h, full_v, full_d7, full_d9;
    full_stability(occupied, &full_h, &full_v, &full_d7, &full_d9);
    return h & v & full_d7 & full_d9;
}

struct Dogai_tt_key {
    Board board;    // real position
    Board norm;     // the side to move's solid discs given to the opponent
    int ofs;        // real score = normalized score + ofs
    int empties;
    uint32_t hash;  // bucket of every record of this position
};

inline Dogai_tt_key dogai_tt_key_from_solid(const Board &board, uint64_t solid, uint32_t ordinary_hash) {
    Dogai_tt_key key;
    key.board = board;
    key.empties = HW2 - pop_count_ull(board.player | board.opponent);
    if (solid) {
        key.norm = Board(board.player ^ solid, board.opponent ^ solid);
        key.ofs = 2 * pop_count_ull(solid);
        key.hash = key.norm.hash();
    } else {
        key.norm = board;
        key.ofs = 0;
        key.hash = ordinary_hash;
    }
    return key;
}

inline Dogai_tt_key dogai_tt_key(const Board &board, uint32_t ordinary_hash) {
    return dogai_tt_key_from_solid(board, dogai_tt_solid_mask(board.player | board.opponent) & board.player, ordinary_hash);
}

// A node reaches the table several times (prefetch, probe, store) with the
// same board, so the solid discs are cached per number of discs.
inline Dogai_tt_key dogai_tt_key(const Search *search, uint32_t ordinary_hash) {
    auto &slot = search->dogai_tt_key_cache[search->n_discs & (N_DOGAI_TT_KEY_CACHE - 1)];
    if (slot.player != search->board.player || slot.opponent != search->board.opponent) {
        slot.player = search->board.player;
        slot.opponent = search->board.opponent;
        slot.solid = dogai_tt_solid_mask(slot.player | slot.opponent) & slot.player;
    }
    return dogai_tt_key_from_solid(search->board, slot.solid, ordinary_hash);
}

// Only the board prefilter is read without the bucket lock. Atomic board words
// make even a concurrent replacement well-defined; a hit is always rechecked
// under the lock before reading its bounds, depth, precision or moves.
struct alignas(32) Clustered_hash_node {
    std::atomic<uint64_t> player{0};
    std::atomic<uint64_t> opponent{0};
    Hash_data data;
    uint16_t generation = 0;
    Spinlock lock; // The first lane owns the lock for all four lanes.
};
static_assert(sizeof(Clustered_hash_node) == 32);
struct alignas(128) Clustered_hash_bucket {
    Clustered_hash_node lanes[4];
};
static_assert(sizeof(Clustered_hash_bucket) == 128);

class Transposition_table {
    Clustered_hash_bucket *table = nullptr;
    size_t bucket_mask = 0;
    int lane_count = 4;
    std::atomic<uint16_t> generation{1};

    struct Found {
        Hash_data *data = nullptr;
        int shift = 0;
    };

    Clustered_hash_bucket &bucket(uint32_t hash) {
        return table[(hash >> 2) & bucket_mask];
    }
    static bool same_board(const Clustered_hash_node &node, const Board &board) {
        return node.player.load(std::memory_order_relaxed) == board.player
            && node.opponent.load(std::memory_order_relaxed) == board.opponent;
    }
    static void set_board(Clustered_hash_node &node, const Board &board) {
        node.player.store(board.player, std::memory_order_relaxed);
        node.opponent.store(board.opponent, std::memory_order_relaxed);
    }
    // Whether the lane holds a record of this position, and the shift that
    // turns its bounds into this position's scores. Call under the lock.
    static bool own(Clustered_hash_node &node, const Dogai_tt_key &key, int *shift) {
        if (!node.data.get_importance()) return false;
        const bool end = node.data.get_depth() >= key.empties;
        if (end && same_board(node, key.norm)) {
            *shift = key.ofs;
            return true;
        }
        if (!end && same_board(node, key.board)) {
            *shift = 0;
            return true;
        }
        return false;
    }
    bool possible_hit(Clustered_hash_bucket &b, const Dogai_tt_key &key) {
        for (int i = 0; i < lane_count; ++i) {
            if (same_board(b.lanes[i], key.norm) || same_board(b.lanes[i], key.board)) return true;
        }
        return false;
    }
    // Select a whole record. Never combine bounds from different depths or
    // probability levels. An incomparable deeper/selective record may coexist
    // with a shallower/exact one, so a query must examine every matching lane.
    Found find(Clustered_hash_bucket &b, const Dogai_tt_key &key, int depth,
               uint_fast8_t mpc, bool nonexact) {
        Found best;
        for (int i = 0; i < lane_count; ++i) {
            auto &node = b.lanes[i];
            int shift;
            if (!own(node, key, &shift) || !node.data.has_usable_bounds(depth, mpc)) continue;
            if (!nonexact && node.data.get_window_width() != 0) continue;
            if (!best.data || node.data.get_level_no_importance() > best.data->get_level_no_importance()
                || (node.data.get_level_no_importance() == best.data->get_level_no_importance()
                    && node.data.get_window_width() < best.data->get_window_width())) {
                best.data = &node.data;
                best.shift = shift;
            }
        }
        return best;
    }
    static void get_shifted_bounds(const Found &found, int *lower, int *upper) {
        found.data->get_bounds(lower, upper);
        if (found.shift) {
            *lower = std::clamp(*lower + found.shift, -SCORE_MAX, SCORE_MAX);
            *upper = std::clamp(*upper + found.shift, -SCORE_MAX, SCORE_MAX);
        }
    }
    bool bounds(const Dogai_tt_key &key, int depth, uint_fast8_t mpc,
                bool nonexact, int *lower, int *upper) {
        auto &b = bucket(key.hash);
        if (!possible_hit(b, key)) return false;
        std::lock_guard lock(b.lanes[0].lock);
        const Found found = find(b, key, depth, mpc, nonexact);
        if (!found.data) return false;
        get_shifted_bounds(found, lower, upper);
        return true;
    }
    void store(const Search *search, uint32_t hash, int depth, int alpha,
               int beta, int value, int policy, bool overwrite_only) {
        if (!search->can_use_tt_eval_bounds()) return;
        const Dogai_tt_key key = dogai_tt_key(search, hash);
        const bool end = depth >= key.empties;
        const Board &target = end ? key.norm : key.board;
        if (end && key.ofs) {
            // The real score is at least ofs - 64 (the solid discs never
            // flip), so a weaker lower bound below that may be raised.
            alpha -= key.ofs;
            beta -= key.ofs;
            value = std::max(value - key.ofs, -SCORE_MAX);
        }
        auto &b = bucket(key.hash);
        std::lock_guard lock(b.lanes[0].lock);
        const uint16_t now = generation.load(std::memory_order_relaxed);
        Clustered_hash_node *matching = nullptr, *dominated = nullptr, *victim = nullptr;
        int victim_rank = INT_MAX;
        // First locate this position. Evicting the first weak collision before
        // checking the other lanes loses complementary bounds for the same key.
        for (int i = 0; i < lane_count; ++i) {
            auto &node = b.lanes[i];
            int shift;
            if (own(node, key, &shift)) {
                // The same depth implies the same kind of record (endgame or
                // not), so both bounds are in the same frame.
                if (node.data.get_depth() == depth && node.data.get_mpc_level() == search->mpc_level) {
                    if (overwrite_only) node.data.reg_new_level(depth, search->mpc_level, alpha, beta, value, policy);
                    else node.data.reg_same_level(alpha, beta, value, policy);
                    node.generation = now;
                    return;
                }
                if (node.data.has_usable_bounds(depth, search->mpc_level)) matching = &node;
                if (depth >= node.data.get_depth() && search->mpc_level >= node.data.get_mpc_level()) dominated = &node;
            }
            const int age = std::min<int>(uint16_t(now - node.generation), 255);
            const int rank = node.data.get_importance()
                ? int(node.data.get_level_no_importance()) - age * 2048 : INT_MIN;
            if (rank < victim_rank) { victim_rank = rank; victim = &node; }
        }
        if (matching && !overwrite_only) return;
        if (overwrite_only) {
            if (!matching) matching = dominated;
            if (!matching) {
                int shift;
                for (int i = 0; i < lane_count; ++i)
                    if (own(b.lanes[i], key, &shift)) matching = &b.lanes[i];
            }
            if (!matching) return;
            matching->data.reg_new_level(depth, search->mpc_level, alpha, beta, value, policy);
            set_board(*matching, target);
            matching->generation = now;
            return;
        }
        if (dominated) {
            // An endgame record replaces the position's own shallower record
            // and moves it to the normalized board.
            dominated->data.reg_new_level(depth, search->mpc_level, alpha, beta, value, policy);
            set_board(*dominated, target);
            dominated->generation = now;
        } else {
            victim->data.init();
            victim->data.reg_new_data(depth, search->mpc_level, alpha, beta, value, policy);
            victim->generation = now;
            set_board(*victim, target);
        }
    }
public:
    Transposition_table() = default;
    Transposition_table(const Transposition_table &) = delete;
    Transposition_table &operator=(const Transposition_table &) = delete;
    ~Transposition_table() { delete[] table; }

    bool resize(int hash_level) {
        const size_t entries = hash_sizes[hash_level];
        const size_t buckets = std::max<size_t>(1, entries / 4);
        auto *replacement = new (std::nothrow) Clustered_hash_bucket[buckets];
        if (!replacement) return false;
        delete[] table;
        table = replacement;
        bucket_mask = buckets - 1;
        lane_count = int(std::min<size_t>(entries, 4));
        init();
        return true;
    }
    bool set_size() { return resize(DEFAULT_HASH_LEVEL); }
    // Like the original init/resize, this is called only when searches stopped.
    void init() {
        if (!table) return;
        for (size_t i = 0; i <= bucket_mask; ++i) {
            for (auto &node : table[i].lanes) {
                node.player.store(0, std::memory_order_relaxed);
                node.opponent.store(0, std::memory_order_relaxed);
                node.data.init();
                node.generation = 0;
            }
        }
        generation.store(1, std::memory_order_relaxed);
    }
    // Aging is a single write at an iteration/search boundary, never a shared
    // registration counter or a concurrent full-table importance sweep.
    void reset_importance() { generation.fetch_add(1, std::memory_order_relaxed); }
    void reset_importance_new_thread(int) { reset_importance(); }
    void reg(const Search *s, uint32_t h, int d, int a, int b, int v, int p) { store(s, h, d, a, b, v, p, false); }
    void reg_overwrite(const Search *s, uint32_t h, int d, int a, int b, int v, int p) { store(s, h, d, a, b, v, p, true); }
    void get(const Search *s, uint32_t h, int d, int *l, int *u, uint_fast8_t moves[]) {
        const Dogai_tt_key key = dogai_tt_key(s, h);
        auto &b = bucket(key.hash);
        if (!possible_hit(b, key)) return;
        std::lock_guard lock(b.lanes[0].lock);
        if (const Found found = find(b, key, 0, 0, true); found.data) found.data->get_moves(moves);
        if (s->can_use_tt_eval_bounds()) {
            const Found found = find(b, key, d, s->mpc_level, s->can_use_tt_nonexact_bounds());
            if (found.data) get_shifted_bounds(found, l, u);
        }
    }
    bool get_bounds(const Search *s, uint32_t h, int d, int *l, int *u) {
        if (!s->can_use_tt_eval_bounds()) return false;
        return bounds(dogai_tt_key(s, h), d, s->mpc_level, s->can_use_tt_nonexact_bounds(), l, u);
    }
    bool get_bounds_any_level(const Search *s, uint32_t h, int *l, int *u) {
        if (!s->can_use_tt_eval_bounds()) return false;
        return bounds(dogai_tt_key(s, h), 0, 0, s->can_use_tt_nonexact_bounds(), l, u);
    }
    bool get_bounds_any_level(const Board *b, uint32_t h, int *l, int *u) { return bounds(dogai_tt_key(*b, h), 0, 0, true, l, u); }
    bool get_moves_any_level(const Board *board, uint32_t h, uint_fast8_t moves[]) {
        const Dogai_tt_key key = dogai_tt_key(*board, h);
        auto &b = bucket(key.hash);
        if (!possible_hit(b, key)) return false;
        std::lock_guard lock(b.lanes[0].lock);
        const Found found = find(b, key, 0, 0, true);
        if (!found.data) return false;
        found.data->get_moves(moves);
        return true;
    }
    void get_info(Board board, int *l, int *u, uint_fast8_t moves[], int *d, uint_fast8_t *mpc) {
        const Dogai_tt_key key = dogai_tt_key(board, board.hash());
        auto &b = bucket(key.hash);
        if (!possible_hit(b, key)) return;
        std::lock_guard lock(b.lanes[0].lock);
        if (const Found found = find(b, key, 0, 0, true); found.data) {
            get_shifted_bounds(found, l, u); found.data->get_moves(moves);
            *d = found.data->get_depth(); *mpc = found.data->get_mpc_level();
        }
    }
    // Also drops the shared endgame record of the normalized board.
    void del(const Board *board, uint32_t h) {
        const Dogai_tt_key key = dogai_tt_key(*board, h);
        auto &b = bucket(key.hash);
        std::lock_guard lock(b.lanes[0].lock);
        for (int i = 0; i < lane_count; ++i) {
            auto &node = b.lanes[i];
            if (same_board(node, key.board)
                || (same_board(node, key.norm) && node.data.get_depth() >= key.empties)) node.data.init();
        }
    }
    bool has_node(const Search *s, uint32_t h, int d) { int l, u; return get_bounds(s, h, d, &l, &u); }
    bool has_node_any_level(const Search *s, uint32_t h) {
        int l, u; return bounds(dogai_tt_key(s, h), 0, 0, true, &l, &u);
    }
    int has_node_any_level_cutoff(const Search *s, uint32_t h, int d, int alpha, int beta) {
        int l = -SCORE_MAX, u = SCORE_MAX;
        if (!has_node_any_level_get_bounds(s, h, d, &l, &u)) return TRANSPOSITION_TABLE_NOT_HAS_NODE;
        if (u <= alpha) return u;
        if (l >= beta) return l;
        return TRANSPOSITION_TABLE_HAS_NODE;
    }
    bool has_node_any_level_get_bounds(const Search *s, uint32_t h, int d, int *l, int *u) {
        if (!s->can_use_tt_eval_bounds()) return false;
        const Dogai_tt_key key = dogai_tt_key(s, h);
        auto &b = bucket(key.hash);
        if (!possible_hit(b, key)) return false;
        std::lock_guard lock(b.lanes[0].lock);
        if (!find(b, key, 0, 0, true).data) return false;
        if (const Found found = find(b, key, d, s->mpc_level, s->can_use_tt_nonexact_bounds()); found.data) get_shifted_bounds(found, l, u);
        return true;
    }
    void prefetch(uint32_t h) {
#if USE_SIMD
        _mm_prefetch(reinterpret_cast<const char *>(&bucket(h)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(&bucket(h)) + 64, _MM_HINT_T0);
#endif
    }
    void prefetch(const Search *s, uint32_t h) { prefetch(dogai_tt_key(s, h).hash); }
};
