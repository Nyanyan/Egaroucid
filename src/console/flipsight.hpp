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
    return {result.policy, result.value, result.depth, result.nodes, static_cast<Millis>(result.time), result.probability};
}

struct NetworkUpdate {
    Json data;
    Millis started = 0, received = 0;
    bool event = false;
};

class TableAger {
    std::mutex mutex;
    std::condition_variable wake;
    size_t pending = 0;
    bool stopping = false;
    bool previous_auto_reset;
    std::thread worker;
public:
    explicit TableAger(bool show_log, Event log = {}) : previous_auto_reset(transposition_table_auto_reset_importance) {
        // FlipSight ages once per completed match instead of at the occupancy threshold.
        transposition_table_auto_reset_importance = false;
        try {
            worker = std::thread([this, show_log, log] {
                while (true) {
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        wake.wait(lock, [&] { return stopping || pending; });
                        if (!pending) return;
                        --pending;
                    }
                    auto started = now_ms();
                    transposition_table.reset_importance_concurrent();
                    if (log) log("table_aged", {{"elapsed_ms", now_ms() - started}}, false);
                    else if (show_log) std::cerr << "[FlipSight] transposition table aged after match completion\n";
                }
            });
        } catch (...) { transposition_table_auto_reset_importance = previous_auto_reset; throw; }
    }
    void enqueue() {
        { std::lock_guard<std::mutex> lock(mutex); ++pending; }
        wake.notify_all();
    }
    ~TableAger() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        wake.notify_all();
        worker.join();
        transposition_table_auto_reset_importance = previous_auto_reset;
    }
};

inline int run(Options* options, const Event& log) {
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
        Json fields = {{"action", body.at("action")}};
        for (const auto* key : {"match_id", "leg", "revision", "move"}) if (body.contains(key)) fields[key] = body[key];
        log("api_request", fields, false);
        auto started = now_ms();
        auto reply = request(*transport, gate, body, cancel);
        fields["http_status"] = reply.status; fields["elapsed_ms"] = now_ms() - started;
        fields["gate_wait_ms"] = reply.started ? std::max<Millis>(0, reply.started - started) : 0;
        fields["http_ms"] = reply.received - reply.started;
        fields["code"] = reply.error; fields["cancelled"] = cancel.load(); fields["retry_after_ms"] = reply.retry_after;
        log(reply.ok() ? "api_response" : "api_failed", fields, !reply.ok() && !cancel);
        {
            std::lock_guard<std::mutex> lock(transport_mutex);
            transports.push_back(std::move(transport));
        }
        return reply;
    };
    TableAger ager(options->show_log, log);
    Controller controller(options->flipsight_threads, options->flipsight_fixed_level ? options->level : -1,
        !options->flipsight_no_ponder,
        [&](const Leg& leg, bool ponder, int id, bool* searching, Millis budget) {
            return engine_search(leg, ponder, id, searching, budget, *options);
        }, call, [](int id, int helpers) {
            if (thread_pool.get_max_thread_size(id) != helpers) thread_pool.set_max_thread_size(id, helpers);
        }, [&](int total_threads) {
            thread_pool.grow(total_threads - 1);
            log("threads_allocated", {{"threads", total_threads}}, false);
        }, [&] { ager.enqueue(); }, log);
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
                    log("poll_retry", {{"http_status", reply.status}, {"retry_ms", backoff * 1000}}, true);
                    next = now_ms() + backoff * 1000;
                    backoff = std::min(30, backoff * 2);
                }
            }
        } catch (...) { log("poll_failed", Json::object(), true); stop = true; wake.notify_all(); }
    });
    std::thread stream;
    if (!options->flipsight_no_stream) stream = std::thread([&] {
        int backoff = 1;
        while (!stop) {
            Millis retry_delay = backoff * 1000;
            try {
                HttpTransport transport(options->flipsight_api, options->flipsight_token);
                auto response = transport.stream(stop, [&](const Json& event) {
                    if (event.value("type", std::string()) == "hello") {
                        stream_healthy = true; backoff = 1;
                        log("stream_connected", Json::object(), false);
                    }
                    publish({event, 0, now_ms(), true});
                });
                if (response.status == 401 || response.status == 403) { denied = true; stop = true; }
                if (response.status == 429 || response.status == 503) {
                    gate.defer(response.retry_after);
                    retry_delay = std::max(retry_delay, response.retry_after);
                }
            } catch (...) { if (!stop) log("stream_failed", Json::object(), true); }
            stream_healthy = false;
            poll_now = true;
            if (!stop) log("stream_retry", {{"retry_ms", retry_delay}}, true);
            for (Millis i = 0; i < retry_delay && !stop; i += 50) std::this_thread::sleep_for(std::chrono::milliseconds(50));
            backoff = std::min(30, backoff * 2);
        }
    });
    Json configuration = {{"reported_threads", options->flipsight_threads}, {"search_threads", options->n_threads},
        {"level", options->flipsight_fixed_level ? options->level : -1}, {"ponder", !options->flipsight_no_ponder}};
#if USE_CHANGEABLE_HASH_LEVEL
    configuration["hash_level"] = options->hash_level;
#endif
    log("ready", configuration, false);
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
                } catch (const Json::exception& error) {
                    log("invalid_server_state", {{"source", update.event ? "stream" : "poll"}, {"json_error_id", error.id}}, true);
                    poll_now = true;
                } catch (...) {
                    log("invalid_server_state", {{"source", update.event ? "stream" : "poll"}}, true);
                    poll_now = true;
                }
            }
            controller.tick();
            playing = controller.running();
            if (controller.need_poll) { poll_now = true; controller.need_poll = false; wake.notify_all(); }
            if (controller.denied) { denied = true; break; }
        }
    } catch (...) { log("client_failed", Json::object(), true); result = 1; }
    if (stop && !interrupted && !denied) result = 1;
    stop = true;
    wake.notify_all();
    controller.stop();
    poller.join();
    if (stream.joinable()) stream.join();
    std::signal(SIGINT, old_int);
    std::signal(SIGTERM, old_term);
    if (denied) { log("authentication_denied", Json::object(), true); result = 1; }
    log("stopped", {{"exit_code", result}}, false);
    return result;
}
} // namespace flipsight

inline int flipsight_client(Options* options) {
    try {
        auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        auto path = options->flipsight_log_file.empty() ?
            std::filesystem::path(options->flipsight_log_dir) / ("flipsight_" + std::to_string(stamp) + ".jsonl") :
            std::filesystem::path(options->flipsight_log_file);
        flipsight::SessionLog session(path, options->flipsight_token);
        flipsight::Event log = [&](const std::string& event, flipsight::Json fields, bool error) { session.write(event, std::move(fields), error); };
        log("session_started", {{"log_file", std::filesystem::absolute(path).string()}}, false);
        try { return flipsight::run(options, log); }
        catch (...) { log("startup_failed", {{"reason", "Invalid connection settings or HTTPS initialization failed"}}, true); return 1; }
    } catch (...) { std::cerr << "[FlipSight] Cannot initialize session logging; check the log directory.\n"; return 1; }
}
