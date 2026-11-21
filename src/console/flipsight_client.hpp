/*
    Egaroucid Project
    @file flipsight_client.hpp
        Nonblocking FlipSight match, search, and move lifecycle
    @license GPL-3.0-or-later
*/
#pragma once
#include <condition_variable>
#include <deque>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include "flipsight_http.hpp"
#include "flipsight_log.hpp"

namespace flipsight {
struct Reply {
    int status = 0;
    Json data;
    Millis started = 0, received = 0;
    std::string error;
    Millis retry_after = 0;
    bool ok() const { return status == 200 && data.is_object() && data.value("ok", false); }
};
using Call = std::function<Reply(const Json&, std::atomic_bool&)>;

// All poll and move workers share the same cooldown and request budget.
class RequestGate {
    std::mutex mutex;
    std::deque<Millis> history;
    Millis next = 0, cooldown = 0, interval;
public:
    explicit RequestGate(bool low_level) : interval(low_level ? 300 : 0) {}
    bool wait(std::atomic_bool& stop) {
        while (!stop) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto now = now_ms();
                while (!history.empty() && history.front() <= now - 60000) history.pop_front();
                Millis ready = std::max(next, cooldown);
                if (interval && history.size() >= 200) ready = std::max(ready, history.front() + 60000);
                if (now >= ready) {
                    next = now + interval;
                    history.push_back(now);
                    return true;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    }
    void defer(Millis delay) {
        std::lock_guard<std::mutex> lock(mutex);
        cooldown = std::max(cooldown, now_ms() + std::max<Millis>(1000, delay));
    }
};

inline Reply request(HttpTransport& transport, RequestGate& gate, const Json& body, std::atomic_bool& stop) {
    Reply reply;
    if (!gate.wait(stop)) return reply;
    try {
        auto response = transport.call(body, stop);
        reply.status = response.status;
        reply.started = response.started;
        reply.received = response.received;
        reply.retry_after = response.retry_after;
        if (response.status == 429 || response.status == 503) gate.defer(response.retry_after);
        if (!response.body.empty() && response.status != 401 && response.status != 403) {
            auto data = Json::parse(response.body);
            if (data.is_object()) {
                reply.data = data;
                // Only a known identifier may leave the response parser.
                static const std::set<std::string> codes = {"illegal_move", "stale_revision", "not_your_turn",
                    "leg_finished", "match not running", "match timed out", "stale room revision",
                    "invalid move coordinate", "move outside board", "bot busy", "challenge expired or answered",
                    "engine thread limit exceeded", "both bots must be connected", "unsupported game mode",
                    "bot already has an active match", "bot challenge repeat limit", "site bot variant matches are full"};
                if (data.contains("error") && data["error"].is_string() && codes.count(data["error"].get<std::string>()))
                    reply.error = data["error"].get<std::string>();
            }
        }
    } catch (...) { if (reply.status == 200) reply.status = 0; }
    return reply;
}

struct ClockSync {
    Millis upper_offset = 0, sampled = 0;
    void observe(Millis server, Millis started, Millis received) {
        upper_offset = server - started;
        sampled = received;
    }
    std::optional<Millis> project(Millis server, Millis received) const {
        if (!sampled) return {};
        Millis projected = server - upper_offset;
        if (projected > received + 1000 || received - projected > 120000) return {};
        return std::min(projected, received);
    }
};

struct MoveResult { int policy = 64, value = 0, depth = 0; uint64_t nodes = 0; Millis elapsed = 0; int probability = 0; };
using Search = std::function<MoveResult(const Leg&, bool, int, bool*, Millis)>;
using Quota = std::function<void(int, int)>;
using Grow = std::function<void(int)>;
using Age = std::function<void()>;

class Controller {
public:
    struct Job {
        Leg leg;
        int id;
        bool pondering, searching = true, sent = false, refresh = false, rejected = false;
        int attempts = 0;
        Millis started = 0, failed_at = 0, ready_at = 0;
        std::atomic_bool cancel{false};
        std::future<MoveResult> search;
        std::future<Reply> send;
        std::optional<MoveResult> result;
    };
private:
    struct Match {
        Json snapshot;
        Millis server_now = -1;
        bool divided = false;
        std::vector<Leg> legs;
        int board_count = 0;
        bool seen_running = false, completed = false;
    };
    std::map<Millis, Match> matches;
    Json bot_info = Json::object();
    bool synchronized = true;
    std::map<std::pair<Millis, int>, std::unique_ptr<Job>> jobs;
    std::vector<std::unique_ptr<Job>> retired;
    std::map<Millis, std::array<int, 2>> ids;
    Search search_fn;
    Call call_fn;
    Quota quota_fn;
    Grow grow_fn;
    Age age_fn;
    Event log_fn;
    int threads_per_board, level, allocated_threads;
    bool ponder;
    Millis transport_ms = 200;

    void log(const std::string& event, Json fields = Json::object(), bool error = false) const {
        if (log_fn) log_fn(event, std::move(fields), error);
    }
    static Json job_fields(const Job& job) {
        return {{"match_id", job.leg.match_id}, {"leg", job.leg.index}, {"revision", job.leg.revision},
            {"remaining_ms", std::max<Millis>(0, job.leg.deadline - now_ms())}, {"ponder", job.pondering},
            {"thread_id", job.id}, {"black", job.leg.black}, {"white", job.leg.white},
            {"legal", job.leg.legal}, {"color", job.leg.color}, {"turn", job.leg.turn}};
    }

    static bool ready(const std::future<MoveResult>& future) {
        return future.valid() && future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
    }
    void retire(std::unique_ptr<Job> job) {
        if (!job->pondering && job->leg.deadline <= now_ms()) log("clock_expired", job_fields(*job), true);
        log(job->pondering ? "ponder_stopping" : "job_retired", job_fields(*job));
        job->searching = false;
        job->cancel = true;
        retired.push_back(std::move(job));
    }
    bool id_available(int id) const {
        for (const auto& job : retired) if (job->id == id) return false;
        return true;
    }
    std::array<int, 2> allocate_ids(Millis match_id) {
        if (ids.count(match_id)) return ids.at(match_id);
        std::set<int> used;
        for (const auto& item : ids) { used.insert(item.second[0]); used.insert(item.second[1]); }
        for (int i = 0; i < 16; i += 2) if (!used.count(i) && !used.count(i + 1)) {
            ids[match_id] = {i, i + 1};
            return {i, i + 1};
        }
        throw std::runtime_error("FlipSight supports at most eight concurrent matches");
    }
public:
    Json bot_id;
    bool need_resync = false, denied = false;
    ClockSync clock;

    Controller(int reported_threads, int fixed_level, bool use_ponder, Search search, Call call, Quota quota,
            Grow grow = [](int) {}, Age age = [] {}, Event logger = {})
        : search_fn(std::move(search)), call_fn(std::move(call)), quota_fn(std::move(quota)),
          grow_fn(std::move(grow)), age_fn(std::move(age)), log_fn(std::move(logger)), threads_per_board(reported_threads), level(fixed_level),
          allocated_threads(2 * reported_threads), ponder(use_ponder) {
        grow_fn(allocated_threads);
    }
    ~Controller() { stop(); }
    size_t job_count() const { return jobs.size(); }
    const Job* job(Millis match_id, int leg) const {
        auto found = jobs.find({match_id, leg});
        return found == jobs.end() ? nullptr : found->second.get();
    }
    bool running() const {
        for (const auto& item : matches) if (active(item.second.snapshot) && !item.second.completed) return true;
        return false;
    }
    static bool match_completed(const Json& snapshot) {
        auto ended = [](const Json& item) {
            auto status = item.value("status", std::string());
            return status == "finished" || status == "aborted" || status == "timeout" || status == "stopped";
        };
        if (ended(snapshot)) return true;
        const auto state = snapshot.value("state", Json());
        if (!state.is_object() || !state.contains("legs") || !state["legs"].is_array() || state["legs"].empty()) return false;
        return std::all_of(state["legs"].begin(), state["legs"].end(), ended);
    }
    void update(const Json& payload, Millis started, Millis received, bool full = true) {
        if (payload.contains("bot")) {
            bot_info = payload.at("bot");
            auto id = payload.at("bot").at("id");
            if (bot_id != id) log("connected", {{"bot_id", id}});
            bot_id = id;
        }
        auto server = integer(payload.at("server_now_ms"));
        if (full) clock.observe(server, started, received);
        for (const auto& snapshot : payload.at("matches")) {
            auto match_id = integer(snapshot.at("id"));
            auto found = matches.find(match_id);
            if (found != matches.end() && found->second.server_now > server) {
                log("stale_snapshot", {{"match_id", match_id}, {"server_now_ms", server},
                    {"current_server_now_ms", found->second.server_now}}); continue;
            }
            if (found != matches.end()) {
                const auto& previous = found->second.snapshot;
                if (match_id < 0) {
                    auto room = snapshot.value("room_created_at", Json()), old_room = previous.value("room_created_at", Json());
                    if (room < old_room || (room == old_room && snapshot.value("round_no", 0) < previous.value("round_no", 0))) continue;
                }
                bool same_round = snapshot.value("room_created_at", Json()) == previous.value("room_created_at", Json()) &&
                    snapshot.value("round_no", Json()) == previous.value("round_no", Json());
                auto state = snapshot.value("state", Json()), old_state = previous.value("state", Json());
                if (same_round && state.is_object() && old_state.is_object() && state.contains("revision") &&
                        old_state.contains("revision") && integer(state["revision"]) < integer(old_state["revision"])) continue;
            }
            auto legs = parse_match(snapshot, bot_id, server, started);
            bool same_game = found != matches.end() &&
                found->second.snapshot.value("round_no", Json()) == snapshot.value("round_no", Json()) &&
                found->second.snapshot.value("room_created_at", Json()) == snapshot.value("room_created_at", Json());
            bool ended = match_completed(snapshot);
            if (same_game && found->second.completed && !ended) continue;
            if (same_game && active(snapshot)) {
                bool stale = false;
                for (const auto& leg : legs) for (const auto& previous : found->second.legs)
                    if (leg.index == previous.index && leg.revision < previous.revision) stale = true;
                if (stale) continue;
            }
            bool divided = same_game && found->second.divided;
            divided = divided || !same_line(legs);
            int board_count = static_cast<int>(legs.size());
            if (snapshot.contains("state") && snapshot["state"].is_object() && snapshot["state"].contains("legs"))
                board_count = static_cast<int>(snapshot["state"]["legs"].size());
            if (same_game) board_count = std::max(board_count, found->second.board_count);
            bool previous_running = found != matches.end() && found->second.seen_running;
            bool previous_completed = found != matches.end() && found->second.completed;
            bool age_previous = previous_running && !previous_completed && (!same_game || ended);
            bool seen_running = (same_game && previous_running) || (!ended && active(snapshot) && !legs.empty());
            for (const auto& leg : legs) {
                bool changed = true;
                if (same_game) for (const auto& old : found->second.legs)
                    if (old.index == leg.index && old.same_turn(leg) && old.published == leg.published) changed = false;
                if (changed) log("leg_state", {{"match_id", match_id}, {"leg", leg.index}, {"revision", leg.revision},
                    {"turn", leg.turn}, {"color", leg.color}, {"published", leg.published},
                    {"remaining_ms", std::max<Millis>(0, leg.deadline - received)}, {"black", leg.black},
                    {"white", leg.white}, {"legal", leg.legal}});
            }
            if (seen_running && (!same_game || !previous_running))
                log("match_started", {{"match_id", match_id}, {"boards", board_count}, {"round_no", snapshot.value("round_no", Json())}});
            auto state = snapshot.value("state", Json::object());
            if (seen_running && state.is_object() && state.contains("legs") && state["legs"].is_array()) {
                const auto& sources = state["legs"];
                for (size_t i = 0; i < sources.size(); ++i) {
                    const auto& source = sources[i];
                    if (!source.is_object()) continue;
                    auto scores = source.value("result_discs", source.value("discs", Json()));
                    if (active(source) || !scores.is_array() || scores.size() != 2 ||
                            !scores[0].is_number_integer() || !scores[1].is_number_integer()) continue;
                    auto old_state = same_game ? found->second.snapshot.value("state", Json()) : Json();
                    if (old_state.is_object() && old_state.contains("legs") && old_state["legs"].is_array() &&
                            i < old_state["legs"].size() && !active(old_state["legs"][i]) &&
                            old_state["legs"][i].value("result_discs", old_state["legs"][i].value("discs", Json())) == scores) continue;
                    int own = same_id(snapshot.value("bot_a", Json()), bot_id) == (i == 0) ? 0 : 1;
                    log("leg_finished", {{"match_id", match_id}, {"leg", i}, {"revision", source.value("revision", Json(0))},
                        {"value", integer(scores[own]) - integer(scores[1 - own])}, {"result_discs", scores}});
                }
            }
            if (age_previous) {
                Json fields = {{"match_id", match_id}, {"status", snapshot.value("status", std::string())},
                    {"finish_reason", state.is_object() ? state.value("finish_reason", Json()) : Json()}};
                auto totals = state.is_object() ? state.value("totals", Json()) : Json();
                if (ended && totals.is_array() && totals.size() == 2 && totals[0].is_number_integer() && totals[1].is_number_integer()) {
                    int own = same_id(snapshot.value("bot_a", Json()), bot_id) ? 0 : 1;
                    fields["final_value"] = integer(totals[own]) - integer(totals[1 - own]);
                }
                log("match_finished", fields);
            }
            matches[match_id] = {snapshot, server, divided, legs, board_count, seen_running, ended};
            if (age_previous) age_fn();
            for (const auto& leg : legs) {
                if (!leg.published) log("leg_unpublished", {{"match_id", match_id}, {"leg", leg.index}, {"revision", leg.revision}});
                auto existing = jobs.find({match_id, leg.index});
                if (existing == jobs.end() || !existing->second->leg.same_turn(leg)) continue;
                auto& job = *existing->second;
                job.leg.deadline = leg.deadline;
                if (job.refresh && received >= job.failed_at && (full || started >= job.failed_at)) {
                    log("move_retry_refreshed", job_fields(job));
                    job.refresh = false;
                    job.ready_at = std::max(job.ready_at, received + std::min<Millis>(5000, 250 * (1 << std::min(job.attempts, 4))));
                }
            }
        }
        // Missing matches are retained until explicitly finished or their clocks expire.
    }
    void disconnected() { synchronized = false; }
    std::optional<Json> pending_challenge(const std::set<Millis>& accepted = {}) const {
        if (!synchronized) return {};
        Json snapshots = Json::array();
        int reserved = 0;
        for (const auto& item : matches) {
            if (accepted.count(item.first) && item.second.snapshot.value("status", std::string()) == "pending") ++reserved;
            else snapshots.push_back(item.second.snapshot);
        }
        auto bot = bot_info;
        int capacity = std::clamp(bot.value("max_games", 2), 1, 8) - reserved;
        if (capacity <= 0) return {};
        bot["max_games"] = capacity;
        return challenge({{"bot", bot}, {"matches", snapshots}});
    }
    void event(const Json& event, Millis received, Millis started = 0) {
        auto type = event.value("type", std::string());
        if (type == "hello") {
            if (event.value("api", 0) != 2) throw std::runtime_error("FlipSight API v2 is required");
            if (bot_id != event.at("bot_id")) log("connected", {{"bot_id", event.at("bot_id")}});
            bot_id = event.at("bot_id");
            clock.observe(integer(event.at("server_now_ms")), started ? started : received, received);
            synchronized = false;
            return;
        }
        if (type != "match" && type != "state") return;
        if (event.value("view", std::string()) != "agent") return;
        if (bot_id.is_null()) { need_resync = true; return; }
        auto projected = clock.project(integer(event.at("server_now_ms")), received);
        if (!projected) { log("stream_clock_refresh"); need_resync = true; return; }
        if (type == "state") {
            update(event, *projected, received);
            std::set<Millis> present;
            for (const auto& match : event.at("matches")) present.insert(integer(match.at("id")));
            for (auto& entry : matches) if (!present.count(entry.first) && !entry.second.completed) {
                if (entry.second.seen_running) age_fn();
                entry.second.completed = true;
                log("match_removed", {{"match_id", entry.first}});
            }
            synchronized = true;
            return;
        }
        if (!synchronized) return;
        log("stream_match", {{"match_id", event.at("match").value("id", Json())}, {"server_now_ms", event.at("server_now_ms")},
            {"lag_ms", received - *projected}});
        update({{"matches", Json::array({event.at("match")})}, {"server_now_ms", event.at("server_now_ms")}},
            *projected, received, false);
    }
    std::optional<Json> challenge(const Json& payload) const {
        // A bot's website max_games setting remains the admission limit.
        auto bot = payload.value("bot", Json::object());
        int limit = std::clamp(bot.value("max_games", 2), 1, 8);
        int load = 0;
        for (const auto& match : payload.at("matches")) {
            if (active(match) && (same_id(match.value("bot_a", Json()), bot_id) ||
                    same_id(match.value("bot_b", Json()), bot_id))) ++load;
        }
        if (load >= limit) return {};
        for (const auto& match : payload.at("matches")) {
            if (match.value("status", std::string()) == "pending" && supported(match) &&
                    same_id(match.value("bot_b", Json()), bot_id))
                return Json{{"action", "agent_accept"}, {"match_id", match.at("id")}};
        }
        return {};
    }
    void tick(Millis now = now_ms()) {
        for (auto i = retired.begin(); i != retired.end();) {
            auto& job = **i;
            bool search_done = !job.search.valid() || ready(job.search);
            bool send_done = !job.send.valid() || job.send.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
            if (search_done && send_done) {
                log("job_stopped", job_fields(job));
                if (job.search.valid()) try { job.search.get(); } catch (...) {}
                if (job.send.valid()) try { job.send.get(); } catch (...) {}
                i = retired.erase(i);
            } else ++i;
        }
        // Keep computed moves across reconnects; fresh state decides whether they can be reused.
        if (!synchronized) return;
        std::map<std::pair<Millis, int>, std::pair<Leg, bool>> desired;
        int active_boards = 0;
        for (auto& item : matches) {
            auto& match = item.second;
            if (!active(match.snapshot) || match.completed) continue;
            // Keep the pair's allocation until the entire match finishes, including hidden legs.
            active_boards += match.board_count;
            std::vector<Leg> live;
            for (const auto& leg : match.legs) if (leg.running && leg.published && now < leg.deadline) live.push_back(leg);
            if (live.empty()) continue;
            bool ours = std::any_of(live.begin(), live.end(), [](const Leg& leg) { return leg.ours(); });
            bool selected_ponder = false;
            // When the line is shared, pondering the most advanced position is sufficient.
            std::stable_sort(live.begin(), live.end(), [](const Leg& a, const Leg& b) { return a.history.size() > b.history.size(); });
            for (const auto& leg : live) {
                bool pondering = !leg.ours();
                if (pondering && (!ponder || (!match.divided && (ours || selected_ponder)))) continue;
                desired[{item.first, leg.index}] = {leg, pondering};
                if (pondering) selected_ponder = true;
            }
        }
        int required_threads = std::max(2 * threads_per_board, active_boards * threads_per_board);
        if (required_threads > allocated_threads) {
            grow_fn(required_threads);
            allocated_threads = required_threads;
            log("threads_grown", {{"threads", allocated_threads}, {"active_boards", active_boards}});
        }
        for (auto i = jobs.begin(); i != jobs.end();) {
            auto found = desired.find(i->first);
            if (found == desired.end() || !i->second->leg.same_turn(found->second.first) ||
                    i->second->pondering != found->second.second) {
                retire(std::move(i->second));
                i = jobs.erase(i);
            } else ++i;
        }
        // Release IDs only after every old search and HTTP request has finished.
        for (auto i = ids.begin(); i != ids.end();) {
            bool used = false;
            for (const auto& entry : desired) if (entry.first.first == i->first) used = true;
            for (const auto& old : retired) if (old->leg.match_id == i->first) used = true;
            if (!used) i = ids.erase(i); else ++i;
        }
        for (const auto& entry : desired) {
            if (jobs.count(entry.first)) continue;
            auto allocated = allocate_ids(entry.first.first);
            int id = allocated[entry.first.second];
            if (!id_available(id)) continue;
            auto job = std::make_unique<Job>();
            job->leg = entry.second.first;
            job->pondering = entry.second.second;
            job->id = id;
            job->started = now;
            jobs.emplace(entry.first, std::move(job));
        }
        for (const auto& allocation : ids) {
            std::array<bool, 2> thinking{}, pondering{};
            for (int i = 0; i < 2; ++i) {
                auto entry = jobs.find({allocation.first, i});
                if (entry != jobs.end() && !entry->second->sent && !entry->second->result) {
                    (entry->second->pondering ? pondering : thinking)[i] = true;
                }
            }
            bool shared = !matches.at(allocation.first).divided && !(thinking[0] && thinking[1]);
            int boards = matches.at(allocation.first).board_count;
            auto plan = thread_plan(threads_per_board * std::max(1, boards), shared, thinking, pondering);
            if (!shared && boards == 2) {
                for (auto& count : plan) if (count) count = threads_per_board;
            }
            for (int i = 0; i < 2; ++i) quota_fn(allocation.second[i], std::max(0, plan[i] - 1));
        }
        for (auto& entry : jobs) {
            auto& job = *entry.second;
            if (!job.search.valid() && !job.result && !job.sent) {
                auto budget = search_budget(job.leg.deadline - now);
                auto fields = job_fields(job); fields["budget_ms"] = budget; fields["level"] = level;
                log(job.pondering ? "ponder_started" : "search_started", fields);
                job.search = std::async(std::launch::async, [this, &job, budget, leg = job.leg] {
                    return search_fn(leg, job.pondering, job.id, &job.searching, budget);
                });
            }
            if (ready(job.search)) {
                try {
                    auto result = job.search.get();
                    if (!job.pondering) {
                        if (result.policy != 64 && (result.policy < 0 || result.policy >= 64 ||
                                !(job.leg.legal & (1ULL << result.policy))))
                            throw std::runtime_error("Illegal FlipSight search result");
                        if (result.policy == 64 && job.leg.legal != 0)
                            throw std::runtime_error("Illegal FlipSight pass");
                        job.result = result;
                        auto fields = job_fields(job);
                        fields["move"] = result.policy == 64 ? -1 : 63 - result.policy;
                        fields["value"] = result.value; fields["depth"] = result.depth;
                        fields["nodes"] = result.nodes; fields["probability"] = result.probability;
                        fields["elapsed_ms"] = now_ms() - job.started;
                        fields["engine_elapsed_ms"] = result.elapsed;
                        log("search_finished", fields);
                        job.ready_at = send_at(level, job.started, job.leg.deadline,
                            64 - std::popcount(job.leg.black | job.leg.white), transport_ms);
                        if (job.ready_at > now) {
                            auto waiting = job_fields(job); waiting["wait_ms"] = job.ready_at - now;
                            log("move_pacing", waiting);
                        }
                    } else job.sent = true;
                } catch (...) { log("search_failed", job_fields(job), true); job.sent = true; need_resync = true; }
            }
            if (job.send.valid() && job.send.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                Reply reply;
                try { reply = job.send.get(); } catch (...) {}
                if (reply.status == 401 || reply.status == 403) denied = true;
                if (reply.ok()) {
                    auto fields = job_fields(job); fields["http_status"] = reply.status;
                    fields["http_ms"] = reply.received - reply.started;
                    fields["move"] = job.result->policy == 64 ? -1 : 63 - job.result->policy;
                    fields["remaining_source"] = "deadline";
                    try {
                        const auto& source = reply.data.at("match").at("state").at("legs").at(job.leg.index);
                        auto remaining = clock_ms(source.at("clocks_ms").at(job.leg.color - 1));
                        if (source.value("turn", Json()) == job.leg.color && source.contains("turn_started_ms"))
                            remaining -= std::max<Millis>(0, integer(reply.data.at("server_now_ms")) - integer(source.at("turn_started_ms"))) +
                                std::max<Millis>(0, now_ms() - reply.started);
                        fields["remaining_ms"] = std::max<Millis>(0, remaining);
                        fields["remaining_source"] = "server";
                    } catch (...) {}
                    log("move_acknowledged", fields);
                    job.sent = true;
                    if (reply.received >= reply.started) transport_ms = std::clamp<Millis>(reply.received - reply.started, 50, 5000);

                } else {
                    auto fields = job_fields(job); fields["http_status"] = reply.status;
                    fields["code"] = reply.error; fields["attempt"] = job.attempts;
                    log("move_failed", fields, true);
                    job.refresh = true;
                    job.failed_at = now_ms();
                    job.rejected = reply.error == "illegal_move" || (reply.status >= 400 && reply.status < 500 &&
                        reply.status != 408 && reply.status != 409 && reply.status != 429);
                    if (job.rejected || job.attempts >= 5) job.sent = true;
                    need_resync = true;
                }
            }
            // Pacing delays only this move. Other searches, moves, and polls keep running.
            Millis emergency = std::max<Millis>(100, transport_ms * 2);
            bool can_send = now >= job.ready_at || job.leg.deadline - now < emergency + 1000;
            if (job.result && !job.sent && !job.refresh && !job.send.valid() && can_send && !job.cancel &&
                    now < job.leg.deadline && !(job.leg.match_id < 0 && job.result->policy == 64)) {
                ++job.attempts;
                auto body = job.leg.move_request(job.result->policy);
                auto fields = job_fields(job); fields["move"] = body["move"]; fields["attempt"] = job.attempts;
                log("move_sending", fields);
                job.send = std::async(std::launch::async, [this, &job, body] { return call_fn(body, job.cancel); });
            }
        }
    }
    void stop() {
        for (auto& item : jobs) retire(std::move(item.second));
        jobs.clear();
        for (auto& job : retired) {
            if (job->search.valid()) try { job->search.get(); } catch (...) {}
            if (job->send.valid()) try { job->send.get(); } catch (...) {}
        }
        retired.clear();
    }
};
} // namespace flipsight
