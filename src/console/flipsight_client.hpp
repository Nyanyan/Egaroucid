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

namespace flipsight {
struct Reply {
    int status = 0;
    Json data;
    Millis started = 0, received = 0;
    std::string error;
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
        if (response.status == 429 || response.status == 503) gate.defer(response.retry_after);
        if (!response.body.empty() && response.status != 401 && response.status != 403) {
            auto data = Json::parse(response.body);
            if (data.is_object()) {
                reply.data = data;
                // Only a known identifier may leave the response parser.
                if (data.value("error", std::string()) == "illegal_move") reply.error = "illegal_move";
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
        if (!sampled || received - sampled > 120000) return {};
        Millis projected = server - upper_offset;
        if (projected > received + 1000 || received - projected > 120000) return {};
        return std::min(projected, received);
    }
};

struct MoveResult { int policy = 64, value = 0, depth = 0; };
using Search = std::function<MoveResult(const Leg&, bool, int, bool*, Millis)>;
using Quota = std::function<void(int, int)>;

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
    };
    std::map<Millis, Match> matches;
    std::map<std::pair<Millis, int>, std::unique_ptr<Job>> jobs;
    std::vector<std::unique_ptr<Job>> retired;
    std::map<Millis, std::array<int, 2>> ids;
    Search search_fn;
    Call call_fn;
    Quota quota_fn;
    int threads, level;
    bool ponder;
    Millis transport_ms = 200;

    static bool ready(const std::future<MoveResult>& future) {
        return future.valid() && future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
    }
    void retire(std::unique_ptr<Job> job) {
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
    bool need_poll = false, denied = false;
    ClockSync clock;

    Controller(int total_threads, int fixed_level, bool use_ponder, Search search, Call call, Quota quota)
        : search_fn(std::move(search)), call_fn(std::move(call)), quota_fn(std::move(quota)),
          threads(total_threads), level(fixed_level), ponder(use_ponder) {}
    ~Controller() { stop(); }
    size_t job_count() const { return jobs.size(); }
    const Job* job(Millis match_id, int leg) const {
        auto found = jobs.find({match_id, leg});
        return found == jobs.end() ? nullptr : found->second.get();
    }
    bool running() const {
        for (const auto& item : matches) if (active(item.second.snapshot)) return true;
        return false;
    }
    void update(const Json& payload, Millis started, Millis received, bool full = true) {
        if (payload.contains("bot")) bot_id = payload.at("bot").at("id");
        auto server = integer(payload.at("server_now_ms"));
        if (full) clock.observe(server, started, received);
        for (const auto& snapshot : payload.at("matches")) {
            auto match_id = integer(snapshot.at("id"));
            auto found = matches.find(match_id);
            if (found != matches.end() && found->second.server_now > server) continue;
            auto legs = parse_match(snapshot, bot_id, server, started);
            bool same_game = found != matches.end() &&
                found->second.snapshot.value("round_no", Json()) == snapshot.value("round_no", Json()) &&
                found->second.snapshot.value("room_created_at", Json()) == snapshot.value("room_created_at", Json());
            if (same_game && active(snapshot)) {
                bool stale = false;
                for (const auto& leg : legs) for (const auto& previous : found->second.legs)
                    if (leg.index == previous.index && leg.revision < previous.revision) stale = true;
                if (stale) continue;
            }
            bool divided = same_game && found->second.divided;
            divided = divided || !same_line(legs);
            matches[match_id] = {snapshot, server, divided, legs};
            for (const auto& leg : legs) {
                auto existing = jobs.find({match_id, leg.index});
                if (existing == jobs.end() || !existing->second->leg.same_turn(leg)) continue;
                auto& job = *existing->second;
                job.leg.deadline = leg.deadline;
                if (job.refresh && full && started >= job.failed_at) {
                    job.refresh = false;
                    job.ready_at = std::max(job.ready_at, received + std::min<Millis>(5000, 250 * (1 << std::min(job.attempts, 4))));
                }
            }
        }
        // Missing matches are retained until explicitly finished or their clocks expire.
    }
    void event(const Json& event, Millis received) {
        auto type = event.value("type", std::string());
        if (type == "hello") { need_poll = true; return; }
        if (type != "match") return;
        if (event.value("view", std::string()) != "agent" || !event.contains("match") || bot_id.is_null()) {
            need_poll = true; return;
        }
        auto projected = clock.project(integer(event.at("server_now_ms")), received);
        if (!projected) { need_poll = true; return; }
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
                if (job.search.valid()) try { job.search.get(); } catch (...) {}
                if (job.send.valid()) try { job.send.get(); } catch (...) {}
                i = retired.erase(i);
            } else ++i;
        }
        std::map<std::pair<Millis, int>, std::pair<Leg, bool>> desired;
        int active_pairs = 0;
        for (auto& item : matches) {
            auto& match = item.second;
            if (!active(match.snapshot)) continue;
            std::vector<Leg> live;
            for (const auto& leg : match.legs) if (leg.running && leg.published && now < leg.deadline) live.push_back(leg);
            if (live.empty()) continue;
            ++active_pairs;
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
            auto plan = thread_plan(std::max(1, threads / std::max(1, active_pairs)), shared, thinking, pondering);
            for (int i = 0; i < 2; ++i) quota_fn(allocation.second[i], std::max(0, plan[i] - 1));
        }
        for (auto& entry : jobs) {
            auto& job = *entry.second;
            if (!job.search.valid() && !job.result && !job.sent) {
                auto budget = search_budget(job.leg.deadline - now);
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
                        job.ready_at = send_at(level, job.started, job.leg.deadline,
                            64 - std::popcount(job.leg.black | job.leg.white), transport_ms);
                    } else job.sent = true;
                } catch (...) { job.sent = true; need_poll = true; }
            }
            if (job.send.valid() && job.send.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                Reply reply;
                try { reply = job.send.get(); } catch (...) {}
                if (reply.status == 401 || reply.status == 403) denied = true;
                if (reply.ok()) {
                    job.sent = true;
                    if (reply.received >= reply.started) transport_ms = std::clamp<Millis>(reply.received - reply.started, 50, 5000);
                    if (reply.data.contains("match") && reply.data.contains("server_now_ms")) {
                        // Apply on the next tick: updating jobs inside this iteration would invalidate references.
                        completed.push_back({reply.data, reply.started, reply.received});
                    } else need_poll = true;
                } else {
                    job.refresh = true;
                    job.failed_at = now_ms();
                    job.rejected = reply.error == "illegal_move" || (reply.status >= 400 && reply.status < 500 &&
                        reply.status != 408 && reply.status != 409 && reply.status != 429);
                    if (job.rejected || job.attempts >= 5) job.sent = true;
                    need_poll = true;
                }
            }
            // Pacing delays only this move. Other searches, moves, and polls keep running.
            Millis emergency = std::max<Millis>(100, transport_ms * 2);
            bool can_send = now >= job.ready_at || job.leg.deadline - now < emergency + 1000;
            if (job.result && !job.sent && !job.refresh && !job.send.valid() && can_send && !job.cancel &&
                    now < job.leg.deadline && !(job.leg.match_id < 0 && job.result->policy == 64)) {
                ++job.attempts;
                auto body = job.leg.move_request(job.result->policy);
                job.send = std::async(std::launch::async, [this, &job, body] { return call_fn(body, job.cancel); });
            }
        }
        for (const auto& completion : completed) {
            update({{"matches", Json::array({completion.data.at("match")})},
                    {"server_now_ms", completion.data.at("server_now_ms")}}, completion.started, completion.received);
        }
        completed.clear();
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
private:
    struct Completion { Json data; Millis started, received; };
    std::vector<Completion> completed;
};
} // namespace flipsight
