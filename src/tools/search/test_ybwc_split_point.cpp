// clang++ -O2 -pthread -std=c++20 test_ybwc_split_point.cpp -o <test executable>
#include <array>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include "../../engine/ybwc_split_point.hpp"

using Bound = Ybwc_split_point::Bound;
void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

void stale_lower_bound() {
    Ybwc_split_point split(10, 50, 10, 0, true);
    split.add_move(1); split.add_move(2);
    auto slow = split.take();
    auto fast = split.take();
    require(split.publish(fast, 30, 2, Bound::exact), "publish improved PV");
    require(!split.publish(slow, 20, 1, Bound::lower), "obsolete lower bound needs a new search");
    require(split.result().completed == (uint64_t{1} << 2), "unproved move must remain incomplete");
    slow.alpha = split.alpha();
    require(split.publish(slow, 42, 1, Bound::exact), "re-search discovers better move");
    require(split.result().value == 42 && split.result().best_move == 1, "retain re-search result");
}

void old_upper_bound_and_cutoff() {
    Ybwc_split_point split(10, 50, 10, 0, true);
    for (int i = 1; i <= 4; ++i) split.add_move(i);
    auto old = split.take();
    auto improving = split.take();
    require(split.publish(improving, 30, 2, Bound::exact), "PV improvement");
    require(split.publish(old, 8, 1, Bound::upper), "old fail-low remains safe at higher alpha");
    require(split.result().best_move == 2, "fail-low cannot replace best move");
    auto cutoff = split.take();
    require(split.publish(cutoff, 52, 3, Bound::lower), "beta cutoff");
    require(!split.searching() && !split.take(), "cutoff stops new work");
    require(split.result().value == 52, "completed cutoff must survive cancellation");
}

void parallel_claims() {
    for (int repetition = 0; repetition < 100; ++repetition) {
        Ybwc_split_point split(-64, 65, -64, -1, true);
        for (int i = 0; i < 64; ++i) split.add_move(i);
        std::array<std::atomic<int>, 64> visits{};
        std::vector<std::thread> workers;
        for (int w = 0; w < 8; ++w) workers.emplace_back([&] {
            while (auto work = split.take()) {
                ++visits[work.move_index];
                std::this_thread::yield();
                if (!split.publish(work, work.move_index, work.move_index, Bound::exact)) std::terminate();
            }
        });
        for (auto &worker : workers) worker.join();
        for (auto &n : visits) require(n.load() == 1, "each sibling must be claimed once");
        const auto result = split.result();
        require(result.value == 63 && result.alpha == 63 && result.best_move == 63, "parallel max");
        require(result.completed == UINT64_MAX, "all moves accounted for");
    }
}

bool try_admit(Ybwc_split_point &split, int max_workers) {
    std::lock_guard lock(split.mutex());
    return split.try_admit_joiner_locked(max_workers);
}

void join_admission() {
    Ybwc_split_point split(0, 1, -64, -1, false);
    for (int i = 0; i < 8; ++i) split.add_move(i);
    int context = 0;
    split.set_join_context(&context, 1);
    auto first = split.take();
    require(!try_admit(split, 4), "no joiner before a handed-out move failed low");
    require(split.publish(first, -3, 0, Bound::upper), "fail low");
    split.register_worker(); // one helper
    require(try_admit(split, 4), "owner + helper + joiner fit in 4");
    require(try_admit(split, 4), "owner + helper + 2 joiners fit in 4");
    require(!try_admit(split, 4), "worker limit");
    while (auto work = split.take()) require(split.publish(work, -5, work.move_index, Bound::upper), "drain");
    require(!try_admit(split, 8), "no joiner once every move is handed out");
    for (int i = 0; i < 3; ++i) split.worker_done(10);
    require(!split.has_running_workers(), "all workers stopped");
    split.wait_retired();
    require(split.worker_nodes() == 30, "worker nodes are accumulated");
    Ybwc_split_point cut(0, 1, -64, -1, false);
    for (int i = 0; i < 3; ++i) cut.add_move(i);
    cut.set_join_context(&context, 1);
    auto a = cut.take();
    require(cut.publish(a, -1, 0, Bound::upper), "fail low");
    auto b = cut.take();
    require(cut.publish(b, 5, 1, Bound::lower), "fail high");
    require(!try_admit(cut, 8), "no joiner after a cutoff");
}

void tree_links_and_wake() {
    Ybwc_split_point root(0, 1, -64, -1, false), a(0, 1, -64, -1, false), b(0, 1, -64, -1, false);
    int context = 0;
    a.set_join_context(&context, 1);
    b.set_join_context(&context, 1);
    for (int i = 0; i < 3; ++i) { a.add_move(i); b.add_move(i); }
    a.link_to(&root);
    b.link_to(&a); // b is a grandchild of root
    {
        std::lock_guard lock(root.mutex());
        require(root.first_child_locked() == &a && a.next_sibling_locked() == nullptr, "child link");
    }
    const uint32_t before = root.wake_sequence();
    root.set_owner_waiting(true);
    auto work = b.take();
    require(b.publish(work, -2, 0, Bound::upper), "fail low");
    require(root.wake_sequence() != before, "a joinable grandchild wakes a waiting ancestor");
    root.set_owner_waiting(false);
    const uint32_t after = root.wake_sequence();
    auto work2 = a.take();
    require(a.publish(work2, -2, 0, Bound::upper), "fail low");
    require(root.wake_sequence() == after, "owners that are not waiting are not woken");
    b.unlink();
    a.unlink();
    std::lock_guard lock(root.mutex());
    require(root.first_child_locked() == nullptr, "unlink");
}

// Owners hand their split's moves to helper threads, and while waiting join
// splits created below them. Every move of every split must be searched once.
void concurrent_helpful_owners() {
    constexpr int n_moves = 12;
    std::atomic<int> searched{0};
    struct Child {
        Ybwc_split_point split{0, 1, -64, -1, false};
    };
    for (int repetition = 0; repetition < 50; ++repetition) {
        searched = 0;
        Ybwc_split_point root(0, 1, -64, -1, false);
        int context = 0;
        root.set_join_context(&context, 1);
        for (int i = 0; i < n_moves; ++i) root.add_move(i);
        // Searching one root move creates a child split with 6 moves that is
        // shared by its searcher and any joiner.
        auto search_root_move = [&](Ybwc_split_point::Work work) {
            Child child;
            child.split.set_join_context(&context, 1);
            for (int i = 0; i < 6; ++i) child.split.add_move(i);
            child.split.link_to(&root);
            while (auto w = child.split.take()) {
                ++searched;
                std::this_thread::yield();
                if (!child.split.publish(w, -1, w.move_index, Bound::upper)) std::terminate();
            }
            while (child.split.has_running_workers()) std::this_thread::yield();
            child.split.unlink();
            child.split.wait_retired();
            ++searched;
            if (!root.publish(work, -1, work.move_index, Bound::upper)) std::terminate();
        };
        std::vector<std::thread> helpers;
        for (int h = 0; h < 3; ++h) {
            root.register_worker();
            helpers.emplace_back([&] {
                while (auto work = root.take()) search_root_move(work);
                root.worker_done(0);
            });
        }
        while (auto work = root.take()) search_root_move(work);
        // The owner joins children until its helpers have finished.
        while (root.has_running_workers()) {
            Ybwc_split_point *target = nullptr;
            {
                std::lock_guard lock(root.mutex());
                for (auto *c = root.first_child_locked(); c && !target; c = c->next_sibling_locked()) {
                    std::lock_guard child_lock(c->mutex());
                    if (c->try_admit_joiner_locked(8)) target = c;
                }
            }
            if (target == nullptr) { std::this_thread::yield(); continue; }
            while (auto w = target->take()) {
                ++searched;
                if (!target->publish(w, -1, w.move_index, Bound::upper)) std::terminate();
            }
            target->worker_done(0);
        }
        for (auto &helper : helpers) helper.join();
        root.wait_retired();
        require(searched == n_moves * 7, "every move searched exactly once");
        std::lock_guard lock(root.mutex());
        require(root.first_child_locked() == nullptr, "children unlinked");
    }
}

int main() {
    try {
        stale_lower_bound(); old_upper_bound_and_cutoff(); parallel_claims();
        join_admission(); tree_links_and_wake(); concurrent_helpful_owners();
        Ybwc_split_point cancelled(0, 1, -64, -1, false);
        cancelled.add_move(0); cancelled.cancel();
        require(!cancelled.take(), "external stop prevents claiming work");
        std::cout << "PASS shared split-point bounds, cancellation, concurrent claims and joins\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
