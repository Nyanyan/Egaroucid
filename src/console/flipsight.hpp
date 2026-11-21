/*
    Egaroucid Project
    @file flipsight.hpp
        Native FlipSight bot client (issue #667)
    @license GPL-3.0-or-later
*/
#pragma once
#include <csignal>
#include <iostream>
#include "flipsight_session.hpp"
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
    if (options->flipsight_no_stream) {
        log("unsupported_option", {{"reason", "-flipsightnostream is unavailable: API v2 requires bot-stream"}}, true);
        return 1;
    }
    Endpoint endpoint(options->flipsight_api);
    // Validate the credential before starting workers or marking the bot online.
    HttpTransport validate(options->flipsight_api, options->flipsight_token);
    interrupted = 0;
    auto old_int = std::signal(SIGINT, interrupt);
    auto old_term = std::signal(SIGTERM, interrupt);
    RequestGate gate(options->flipsight_fixed_level && options->level <= 10);
    std::mutex transport_mutex;
    std::vector<std::unique_ptr<HttpTransport>> transports;
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
    environment["practice"] = false;
    environment["ponder"] = !options->flipsight_no_ponder;
#if USE_CHANGEABLE_HASH_LEVEL
    auto hash_level = options->hash_level;
#else
    auto hash_level = DEFAULT_HASH_LEVEL;
#endif
    environment["hash_mb"] = ((hash_sizes[hash_level] + TRANSPOSITION_TABLE_N_LOOP - 1) * sizeof(Hash_node) + 1048575) / 1048576;
    auto stream = [&](std::atomic_bool& stop, const std::function<void(const Json&)>& consume, const std::atomic_bool& reconnect) {
        HttpTransport transport(options->flipsight_api, options->flipsight_token);
        return transport.stream(stop, consume, &reconnect);
    };
    Json configuration = {{"reported_threads", options->flipsight_threads}, {"search_threads", options->n_threads},
        {"level", options->flipsight_fixed_level ? options->level : -1}, {"ponder", !options->flipsight_no_ponder}};
#if USE_CHANGEABLE_HASH_LEVEL
    configuration["hash_level"] = options->hash_level;
#endif
    log("ready", configuration, false);
    int result = run_session(controller, call, stream, environment, log, [] { return interrupted != 0; });
    std::signal(SIGINT, old_int);
    std::signal(SIGTERM, old_term);
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
