/* Egaroucid Project: FlipSight API v2 stream session. GPL-3.0-or-later. */
#pragma once
#include "flipsight_client.hpp"

namespace flipsight {
using Stream = std::function<HttpResponse(std::atomic_bool&, const std::function<void(const Json&)>&, const std::atomic_bool&)>;

inline int run_session(Controller& controller, const Call& call, const Stream& stream,
        const Json& environment, const Event& log, const std::function<bool()>& interrupted) {
    struct Update { Json event; Millis started, received; };
    std::atomic_bool stop(false), reconnect(false), denied(false);
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<Update> incoming;
    std::deque<Json> commands{Json{{"action", "agent_environment"}, {"environment", environment}}};
    auto publish = [&](Json event, Millis started = 0) {
        std::lock_guard<std::mutex> lock(mutex);
        if (incoming.size() < 128) incoming.push_back({std::move(event), started, now_ms()});
        else { incoming.clear(); incoming.push_back({{{"type", "resync_required"}}, 0, now_ms()}); reconnect = true; }
        wake.notify_all();
    };
    controller.disconnected();
    std::thread command_worker([&] {
        try {
            while (!stop) {
                Json command;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    wake.wait(lock, [&] { return stop || !commands.empty(); });
                    if (stop) break;
                    command = std::move(commands.front()); commands.pop_front();
                }
                int attempt = 0;
                while (!stop) {
                    Reply reply;
                    try { reply = call(command, stop); } catch (...) {}
                    if (reply.status == 401 || reply.status == 403) { denied = true; stop = true; wake.notify_all(); break; }
                    if (reply.ok()) break;
                    bool transient = reply.status == 0 || reply.status == 408 || reply.status == 429 || reply.status >= 500;
                    if (!transient) {
                        log("command_failed", {{"action", command.at("action")}, {"http_status", reply.status}}, true);
                        if (command.at("action") == "agent_accept") publish({{"type", "command_rejected"}, {"match_id", command.at("match_id")}});
                        break;
                    }
                    Millis delay = std::max<Millis>(reply.retry_after, std::min<Millis>(30000, 250LL << std::min(attempt++, 7)));
                    log("command_retry", {{"action", command.at("action")}, {"http_status", reply.status}, {"retry_ms", delay}}, true);
                    std::unique_lock<std::mutex> lock(mutex);
                    wake.wait_for(lock, std::chrono::milliseconds(delay), [&] { return stop.load(); });
                }
            }
        } catch (...) { log("command_failed", Json::object(), true); stop = true; wake.notify_all(); }
    });
    std::thread stream_worker([&] {
        int backoff = 1;
        while (!stop) {
            reconnect = false;
            auto began = now_ms();
            Millis retry = backoff * 1000;
            Millis server_retry = 0;
            try {
                auto response = stream(stop, [&](const Json& event) { publish(event, began); }, reconnect);
                if (response.status == 401 || response.status == 403) { denied = true; stop = true; }
                server_retry = response.retry_after;
                retry = std::max(retry, response.retry_after);
            } catch (...) { if (!stop) log("stream_failed", Json::object(), true); }
            publish({{"type", "disconnected"}});
            if (stop) break;
            if (now_ms() - began >= 30000) { backoff = 1; retry = std::max<Millis>(1000, server_retry); }
            log("stream_retry", {{"retry_ms", retry}}, true);
            std::unique_lock<std::mutex> lock(mutex);
            wake.wait_for(lock, std::chrono::milliseconds(retry), [&] { return stop.load(); });
            backoff = std::min(30, backoff * 2);
        }
    });
    bool awaiting_hello = true;
    std::set<Millis> accepted;
    int result = 0;
    try {
        while (!stop && !interrupted()) {
            std::deque<Update> updates;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait_for(lock, std::chrono::milliseconds(10));
                incoming.swap(updates);
            }
            for (const auto& update : updates) {
                const auto type = update.event.value("type", std::string());
                if (type == "command_rejected") {
                    accepted.erase(integer(update.event.at("match_id"))); controller.need_resync = true;
                    continue;
                }
                if (type == "disconnected" || type == "resync_required") {
                    controller.disconnected(); awaiting_hello = true;
                    continue;
                }
                if (awaiting_hello && type != "hello") continue;
                try {
                    controller.event(update.event, update.received, update.started);
                    if (type == "hello") { awaiting_hello = false; log("stream_connected", {{"api", 2}}, false); }
                } catch (const Json::exception& error) {
                    log("invalid_server_state", {{"source", "stream"}, {"json_error_id", error.id}}, true);
                    controller.need_resync = true;
                } catch (...) {
                    log("invalid_server_state", {{"source", "stream"}}, true);
                    controller.need_resync = true;
                }
            }
            if (controller.need_resync) {
                controller.need_resync = false;
                controller.disconnected(); awaiting_hello = true; reconnect = true;
            }
            controller.tick();
            if (!awaiting_hello) {
                auto challenge = controller.pending_challenge(accepted);
                if (challenge && accepted.insert(integer(challenge->at("match_id"))).second) {
                    std::lock_guard<std::mutex> lock(mutex);
                    commands.push_back(*challenge); wake.notify_all();
                }
            }
            if (controller.denied) { denied = true; break; }
        }
    } catch (...) { log("client_failed", Json::object(), true); result = 1; }
    if (stop && !interrupted() && !denied) result = 1;
    stop = true; reconnect = true; wake.notify_all();
    controller.stop();
    command_worker.join(); stream_worker.join();
    if (denied) { log("authentication_denied", Json::object(), true); result = 1; }
    log("stopped", {{"exit_code", result}}, false);
    return result;
}
} // namespace flipsight
