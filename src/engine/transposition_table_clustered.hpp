/*
    Egaroucid Project
    Four-way, full-board transposition table.
    SPDX-License-Identifier: GPL-3.0-or-later
*/
#pragma once
#include <climits>

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

    Clustered_hash_bucket &bucket(uint32_t hash) {
        return table[(hash >> 2) & bucket_mask];
    }
    static bool same_board(const Clustered_hash_node &node, const Board &board) {
        return node.player.load(std::memory_order_relaxed) == board.player
            && node.opponent.load(std::memory_order_relaxed) == board.opponent;
    }
    bool possible_hit(Clustered_hash_bucket &b, const Board &board) {
        for (int i = 0; i < lane_count; ++i) if (same_board(b.lanes[i], board)) return true;
        return false;
    }
    // Select a whole record. Never combine bounds from different depths or
    // probability levels. An incomparable deeper/selective record may coexist
    // with a shallower/exact one, so a query must examine every matching lane.
    Hash_data *find(Clustered_hash_bucket &b, const Board &board, int depth,
                   uint_fast8_t mpc, bool nonexact) {
        Hash_data *best = nullptr;
        for (int i = 0; i < lane_count; ++i) {
            auto &node = b.lanes[i];
            if (!same_board(node, board) || !node.data.get_importance()
                || !node.data.has_usable_bounds(depth, mpc)) continue;
            if (!nonexact && node.data.get_window_width() != 0) continue;
            if (!best || node.data.get_level_no_importance() > best->get_level_no_importance()
                || (node.data.get_level_no_importance() == best->get_level_no_importance()
                    && node.data.get_window_width() < best->get_window_width())) best = &node.data;
        }
        return best;
    }
    bool bounds(const Board &board, uint32_t hash, int depth, uint_fast8_t mpc,
                bool nonexact, int *lower, int *upper) {
        auto &b = bucket(hash);
        if (!possible_hit(b, board)) return false;
        std::lock_guard lock(b.lanes[0].lock);
        auto *data = find(b, board, depth, mpc, nonexact);
        if (!data) return false;
        data->get_bounds(lower, upper);
        return true;
    }
    void store(const Search *search, uint32_t hash, int depth, int alpha,
               int beta, int value, int policy, bool overwrite_only) {
        auto &b = bucket(hash);
        std::lock_guard lock(b.lanes[0].lock);
        const uint16_t now = generation.load(std::memory_order_relaxed);
        Clustered_hash_node *matching = nullptr, *dominated = nullptr, *victim = nullptr;
        int victim_rank = INT_MAX;
        // First locate this board. Evicting the first weak collision before
        // checking the other lanes loses complementary bounds for the same key.
        for (int i = 0; i < lane_count; ++i) {
            auto &node = b.lanes[i];
            if (node.data.get_importance() && same_board(node, search->board)) {
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
                for (int i = 0; i < lane_count; ++i)
                    if (b.lanes[i].data.get_importance() && same_board(b.lanes[i], search->board)) matching = &b.lanes[i];
            }
            if (!matching) return;
            matching->data.reg_new_level(depth, search->mpc_level, alpha, beta, value, policy);
            matching->generation = now;
            return;
        }
        if (dominated) {
            dominated->data.reg_new_level(depth, search->mpc_level, alpha, beta, value, policy);
            dominated->generation = now;
        } else {
            victim->data.init();
            victim->data.reg_new_data(depth, search->mpc_level, alpha, beta, value, policy);
            victim->generation = now;
            victim->player.store(search->board.player, std::memory_order_relaxed);
            victim->opponent.store(search->board.opponent, std::memory_order_relaxed);
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
        auto &b = bucket(h);
        if (!possible_hit(b, s->board)) return;
        std::lock_guard lock(b.lanes[0].lock);
        if (auto *data = find(b, s->board, 0, 0, true)) data->get_moves(moves);
        if (auto *data = find(b, s->board, d, s->mpc_level, s->can_use_tt_nonexact_bounds())) data->get_bounds(l, u);
    }
    bool get_bounds(const Search *s, uint32_t h, int d, int *l, int *u) {
        return bounds(s->board, h, d, s->mpc_level, s->can_use_tt_nonexact_bounds(), l, u);
    }
    bool get_bounds_any_level(const Search *s, uint32_t h, int *l, int *u) {
        return bounds(s->board, h, 0, 0, s->can_use_tt_nonexact_bounds(), l, u);
    }
    bool get_bounds_any_level(const Board *b, uint32_t h, int *l, int *u) { return bounds(*b, h, 0, 0, true, l, u); }
    bool get_moves_any_level(const Board *board, uint32_t h, uint_fast8_t moves[]) {
        auto &b = bucket(h);
        if (!possible_hit(b, *board)) return false;
        std::lock_guard lock(b.lanes[0].lock);
        auto *data = find(b, *board, 0, 0, true);
        if (!data) return false;
        data->get_moves(moves);
        return true;
    }
    void get_info(Board board, int *l, int *u, uint_fast8_t moves[], int *d, uint_fast8_t *mpc) {
        auto &b = bucket(board.hash());
        if (!possible_hit(b, board)) return;
        std::lock_guard lock(b.lanes[0].lock);
        if (auto *data = find(b, board, 0, 0, true)) {
            data->get_bounds(l, u); data->get_moves(moves);
            *d = data->get_depth(); *mpc = data->get_mpc_level();
        }
    }
    void del(const Board *board, uint32_t h) {
        auto &b = bucket(h);
        std::lock_guard lock(b.lanes[0].lock);
        for (int i = 0; i < lane_count; ++i) if (same_board(b.lanes[i], *board)) b.lanes[i].data.init();
    }
    bool has_node(const Search *s, uint32_t h, int d) { int l, u; return get_bounds(s, h, d, &l, &u); }
    bool has_node_any_level(const Search *s, uint32_t h) {
        int l, u; return bounds(s->board, h, 0, 0, true, &l, &u);
    }
    int has_node_any_level_cutoff(const Search *s, uint32_t h, int d, int alpha, int beta) {
        int l = -SCORE_MAX, u = SCORE_MAX;
        if (!has_node_any_level_get_bounds(s, h, d, &l, &u)) return TRANSPOSITION_TABLE_NOT_HAS_NODE;
        if (u <= alpha) return u;
        if (l >= beta) return l;
        return TRANSPOSITION_TABLE_HAS_NODE;
    }
    bool has_node_any_level_get_bounds(const Search *s, uint32_t h, int d, int *l, int *u) {
        auto &b = bucket(h);
        if (!possible_hit(b, s->board)) return false;
        std::lock_guard lock(b.lanes[0].lock);
        if (!find(b, s->board, 0, 0, true)) return false;
        if (auto *data = find(b, s->board, d, s->mpc_level, s->can_use_tt_nonexact_bounds())) data->get_bounds(l, u);
        return true;
    }
    void prefetch(uint32_t h) {
#if USE_SIMD
        _mm_prefetch(reinterpret_cast<const char *>(&bucket(h)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(&bucket(h)) + 64, _MM_HINT_T0);
#endif
    }
};
