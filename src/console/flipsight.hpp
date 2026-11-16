/*
    Egaroucid Project
    @file flipsight.hpp
        Native FlipSight bot client (issue #667)
    @license GPL-3.0-or-later
*/
#pragma once
#include <csignal>
#include <iostream>
#include "flipsight_client.hpp"
#include "option.hpp"
#include "info.hpp"

namespace flipsight {
inline volatile std::sig_atomic_t interrupted = 0;
inline void interrupt(int) { interrupted = 1; }

inline MoveResult engine_search(const Leg& leg, bool pondering, int id, bool* searching,
        Millis budget, const Options& options) {
    Board board;
    board.player = leg.turn == 1 ? leg.black : leg.white;
    board.opponent = leg.turn == 1 ? leg.white : leg.black;
    if (board.get_legal() != leg.legal) throw std::runtime_error("Invalid FlipSight legal moves");
    if (pondering) {
        if (!board.is_end()) ai_ponder(board, options.show_log, id, searching);
        return {};
    }
    if (!leg.legal) return {64, 0, 0};
    Search_result result;
    if (options.flipsight_fixed_level) {
        // ai_common's finite-time path searches by time and ignores the requested level.
        // Keep the normal level path and interrupt it only when the game clock requires it.
        std::mutex mutex;
        std::condition_variable wake;
        bool complete = false;
        std::thread deadline([&] {
            std::unique_lock<std::mutex> lock(mutex);
            if (!wake.wait_for(lock, std::chrono::milliseconds(budget), [&] { return complete; })) *searching = false;
        });
        auto finish = [&] {
            { std::lock_guard<std::mutex> lock(mutex); complete = true; }
            wake.notify_all(); deadline.join();
        };
        try {
            result = ai_searching_thread_id(board, options.level, !options.nobook, 0,
                options.n_threads > 1, options.show_log, id, searching);
        } catch (...) { finish(); throw; }
        finish();
    } else {
        result = ai_time_limit(board, !options.nobook, 0, options.n_threads > 1, options.show_log,
            static_cast<uint64_t>(budget), id, searching);
    }
    // An interrupted or extremely short search can finish before depth one.
    if (!is_valid_policy(result.policy) || !(leg.legal & (1ULL << result.policy))) {
        for (int i = 0; i < 64; ++i) if (leg.legal & (1ULL << i)) { result.policy = i; break; }
    }
    return {result.policy, result.value, result.depth};
}

struct NetworkUpdate {
    Json data;
    Millis started = 0, received = 0;
    bool event = false;
};

inline int run(Options* options) {
    Endpoint endpoint(options->flipsight_api);
    // Validate the credential before starting workers or marking the bot online.
    HttpTransport validate(options->flipsight_api, options->flipsight_token);
    interrupted = 0;
    auto old_int = std::signal(SIGINT, interrupt);
    auto old_term = std::signal(SIGTERM, interrupt);
    std::atomic_bool stop(false), denied(false), stream_healthy(false), playing(false), poll_now(true);
    RequestGate gate(options->flipsight_fixed_level && options->level <= 10);
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<NetworkUpdate> updates;
    std::optional<Json> accept;
    std::mutex transport_mutex;
    std::vector<std::unique_ptr<HttpTransport>> transports;
    auto publish = [&](NetworkUpdate update) {
        std::lock_guard<std::mutex> lock(mutex);
        // Stream snapshots are advisory: fall back to a full poll if overloaded.
        if (updates.size() < 128) updates.push_back(std::move(update));
        else poll_now = true;
        wake.notify_all();
    };
    auto call = [&](const Json& body, std::atomic_bool& cancel) {
        std::unique_ptr<HttpTransport> transport;
        {
            std::lock_guard<std::mutex> lock(transport_mutex);
            if (!transports.empty()) { transport = std::move(transports.back()); transports.pop_back(); }
        }
        if (!transport) transport = std::make_unique<HttpTransport>(options->flipsight_api, options->flipsight_token);
        auto reply = request(*transport, gate, body, cancel);
        {
            std::lock_guard<std::mutex> lock(transport_mutex);
            transports.push_back(std::move(transport));
        }
        return reply;
    };
    Controller controller(options->flipsight_threads, options->flipsight_fixed_level ? options->level : -1,
        !options->flipsight_no_ponder,
        [&](const Leg& leg, bool ponder, int id, bool* searching, Millis budget) {
            return engine_search(leg, ponder, id, searching, budget, *options);
        }, call, [](int id, int helpers) {
            if (thread_pool.get_max_thread_size(id) != helpers) thread_pool.set_max_thread_size(id, helpers);
        }, [&](int total_threads) {
            thread_pool.grow(total_threads - 1);
            if (options->show_log) std::cerr << "[FlipSight] allocated search threads = " << total_threads << '\n';
        });
    Json environment = {{"engine", std::string(EGAROUCID_NAME) + " " + EGAROUCID_VERSION},
        {"engine_threads", options->flipsight_threads},
#ifdef _WIN32
        {"os", "Windows"}
#elif __APPLE__
        {"os", "Darwin"}
#else
        {"os", "Linux"}
#endif
    };
    std::thread poller([&] {
        Millis next = 0, last_poll = 0;
        int backoff = 1;
        try {
            while (!stop) {
                std::optional<Json> action;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    wake.wait_for(lock, std::chrono::milliseconds(50), [&] { return stop.load() || accept.has_value(); });
                    if (stop) break;
                    if (accept) { action = *accept; accept.reset(); }
                }
                if (!action && ((now_ms() < next && !poll_now) || now_ms() < last_poll + 1000)) continue;
                if (!action) {
                    action = Json{{"action", "agent_poll"}, {"history", "brief"}, {"environment", environment}};
                    poll_now = false;
                    last_poll = now_ms();
                }
                auto reply = call(*action, stop);
                if (reply.status == 401 || reply.status == 403) { denied = true; stop = true; wake.notify_all(); break; }
                if (reply.ok()) {
                    if ((*action)["action"] == "agent_poll") publish({reply.data, reply.started, reply.received, false});
                    else poll_now = true;
                    backoff = 1;
                    next = now_ms() + (stream_healthy ? (playing ? 10000 : 60000) : (playing ? 1000 : 5000));
                } else {
                    std::cerr << "[FlipSight] API connection/rejection (HTTP " << reply.status << "); retrying\n";
                    next = now_ms() + backoff * 1000;
                    backoff = std::min(30, backoff * 2);
                }
            }
        } catch (...) { stop = true; wake.notify_all(); }
    });
    std::thread stream;
    if (!options->flipsight_no_stream) stream = std::thread([&] {
        int backoff = 1;
        while (!stop) {
            Millis retry_delay = backoff * 1000;
            try {
                HttpTransport transport(options->flipsight_api, options->flipsight_token);
                auto response = transport.stream(stop, [&](const Json& event) {
                    if (event.value("type", std::string()) == "hello") { stream_healthy = true; backoff = 1; }
                    publish({event, 0, now_ms(), true});
                });
                if (response.status == 401 || response.status == 403) { denied = true; stop = true; }
                if (response.status == 429 || response.status == 503) {
                    gate.defer(response.retry_after);
                    retry_delay = std::max(retry_delay, response.retry_after);
                }
            } catch (...) {}
            stream_healthy = false;
            poll_now = true;
            for (Millis i = 0; i < retry_delay && !stop; i += 50) std::this_thread::sleep_for(std::chrono::milliseconds(50));
            backoff = std::min(30, backoff * 2);
        }
    });
    std::cerr << "[FlipSight] Ready; Ctrl+C stops the client.\n";
    int result = 0;
    try {
        while (!stop && !interrupted) {
            std::deque<NetworkUpdate> incoming;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait_for(lock, std::chrono::milliseconds(10));
                incoming.swap(updates);
            }
            for (const auto& update : incoming) {
                try {
                    if (update.event) controller.event(update.data, update.received);
                    else {
                        controller.update(update.data, update.started, update.received);
                        auto challenge = controller.challenge(update.data);
                        if (challenge) {
                            std::lock_guard<std::mutex> lock(mutex);
                            accept = *challenge;
                            wake.notify_all();
                        }
                    }
                } catch (...) {
                    std::cerr << "[FlipSight] Invalid/unsupported server state; refreshing.\n";
                    poll_now = true;
                }
            }
            controller.tick();
            playing = controller.running();
            if (controller.need_poll) { poll_now = true; controller.need_poll = false; wake.notify_all(); }
            if (controller.denied) { denied = true; break; }
        }
    } catch (...) { std::cerr << "[FlipSight] Client failure; stopping.\n"; result = 1; }
    if (stop && !interrupted && !denied) result = 1;
    stop = true;
    wake.notify_all();
    controller.stop();
    poller.join();
    if (stream.joinable()) stream.join();
    std::signal(SIGINT, old_int);
    std::signal(SIGTERM, old_term);
    if (denied) { std::cerr << "[FlipSight] Authentication denied; check the bot token.\n"; result = 1; }
    return result;
}
} // namespace flipsight

inline int flipsight_client(Options* options) {
    try { return flipsight::run(options); }
    catch (...) { std::cerr << "[ERROR] Invalid FlipSight connection settings or HTTPS initialization failed\n"; return 1; }
}
