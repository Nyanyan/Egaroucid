// Issue #667: exercise the native lifecycle with deterministic offline engines/APIs.
#include <iostream>
#include "../../console/flipsight_client.hpp"
using namespace flipsight;

void require(bool condition, const char* text) { if (!condition) throw std::runtime_error(text); }
Json fixture() {
    std::vector<int> cells(64, 0);
    cells[27] = cells[36] = 2; cells[28] = cells[35] = 1;
    Json leg = {{"board", cells}, {"turn", 1}, {"revision", 0}, {"status", "active"},
        {"clocks_ms", {60000, 60000}}, {"turn_started_ms", 100000}, {"history", Json::array()},
        {"legal_moves", {19, 26, 37, 44}}};
    Json match = {{"id", 123}, {"bot_a", 1}, {"bot_b", 2}, {"status", "running"},
        {"state", {{"version", 1}, {"status", "active"}, {"started_ms", 100000}, {"legs", {leg, leg}}}}};
    return {{"ok", true}, {"bot", {{"id", 1}, {"max_games", 2}}}, {"matches", {match}}, {"server_now_ms", 100000}};
}
void play(Json& leg, int square, int participant) {
    int color = leg.at("turn").get<int>();
    auto cells = leg.at("board").get<std::vector<int>>();
    for (int dr = -1; dr <= 1; ++dr) for (int dc = -1; dc <= 1; ++dc) {
        if (!dr && !dc) continue;
        int row = square / 8 + dr, col = square % 8 + dc;
        std::vector<int> captured;
        while (row >= 0 && row < 8 && col >= 0 && col < 8 && cells[row * 8 + col] == 3 - color) {
            captured.push_back(row * 8 + col); row += dr; col += dc;
        }
        if (row >= 0 && row < 8 && col >= 0 && col < 8 && cells[row * 8 + col] == color)
            for (int flip : captured) cells[flip] = color;
    }
    cells[square] = color;
    leg["board"] = cells;
    leg["turn"] = 3 - color;
    leg["revision"] = leg["revision"].get<int>() + 1;
    leg["history"].push_back({{"color", color}, {"move", square}, {"participant", participant}});
    uint64_t black = 0, white = 0;
    for (int i = 0; i < 64; ++i) {
        if (cells[i] == 1) black |= 1ULL << (63 - i);
        if (cells[i] == 2) white |= 1ULL << (63 - i);
    }
    auto legal = legal_moves(black, white, 3 - color);
    leg["legal_moves"] = Json::array();
    for (int i = 0; i < 64; ++i) if (legal & (1ULL << (63 - i))) leg["legal_moves"].push_back(i);
}
void settle(Controller& controller, const std::function<bool()>& done, Millis advance = 0) {
    auto end = now_ms() + 3000;
    while (!done() && now_ms() < end) {
        controller.tick(now_ms() + advance);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    require(done(), "asynchronous operation did not complete");
}
MoveResult first_move(const Leg& leg) {
    for (int i = 0; i < 64; ++i) if (leg.legal & (1ULL << i)) return {i, 2, 3};
    return {};
}
int main() {
    std::map<int, int> quotas;
    auto quota = [&](int id, int count) { quotas[id] = count; };
    auto instant = [](const Leg& leg, bool, int, bool*, Millis) { return first_move(leg); };
    auto success = [](const Json&, std::atomic_bool&) { return Reply{200, {{"ok", true}}, now_ms(), now_ms(), {}}; };
    {
        auto data = fixture();
        Controller controller(4, -1, true, instant, success, quota);
        controller.update(data, now_ms(), now_ms()); controller.tick();
        require(controller.job_count() == 1 && quotas[0] == 7 && quotas[1] == 0, "shared game must use all threads without duplicate ponder");
        controller.stop();
    }
    {
        auto data = fixture();
        play(data["matches"][0]["state"]["legs"][1], 19, 1);
        std::atomic<int> entered{0};
        std::atomic_bool release{false};
        auto blocking = [&](const Leg& leg, bool, int, bool*, Millis) {
            entered.fetch_add(1);
            while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return first_move(leg);
        };
        Controller controller(4, -1, false, blocking, success, quota);
        controller.update(data, now_ms(), now_ms()); controller.tick();
        auto end = now_ms() + 1000;
        while (entered < 2 && now_ms() < end) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        bool both_started = entered == 2;
        release = true;
        require(both_started && quotas[0] == 3 && quotas[1] == 3, "both own turns must start concurrently with half the threads");
        controller.stop();
    }
    {
        auto data = fixture();
        play(data["matches"][0]["state"]["legs"][0], 19, 0);
        play(data["matches"][0]["state"]["legs"][1], 26, 1);
        Controller controller(4, -1, true, instant, success, quota);
        controller.update(data, now_ms(), now_ms()); controller.tick();
        require(controller.job_count() == 2 && controller.job(123, 0)->pondering && !controller.job(123, 1)->pondering,
            "divergence must run search and ponder together");
        require(quotas[0] == 3 && quotas[1] == 3, "divergence quota");
        controller.stop();
    }
    {
        auto data = fixture();
        std::atomic<int> calls{0};
        Json original;
        // Keep the response helper independent of cancellation for this fake API.
        auto api = [&](const Json& body, std::atomic_bool&) {
            if (calls++ == 0) { original = body; return Reply{}; }
            require(original == body, "retry changed the computed move/revision");
            return Reply{200, {{"ok", true}}, now_ms(), now_ms(), {}};
        };
        Controller controller(4, -1, false, instant, api, quota);
        controller.update(data, now_ms(), now_ms());
        settle(controller, [&] { auto j = controller.job(123, 0); return j && j->refresh; });
        for (int i = 0; i < 5; ++i) controller.tick();
        require(calls == 1, "uncertain move replayed without a fresh poll");
        data["server_now_ms"] = 100001;
        controller.update(data, now_ms(), now_ms());
        settle(controller, [&] { auto j = controller.job(123, 0); return j && j->sent; }, 1000);
        require(calls == 2, "fresh unchanged revision should retry exactly once");
        controller.stop();
    }
    {
        auto data = fixture();
        play(data["matches"][0]["state"]["legs"][0], 19, 0);
        play(data["matches"][0]["state"]["legs"][1], 26, 1);
        Controller controller(2, -1, false, instant, success, quota);
        controller.update(data, now_ms(), now_ms()); controller.tick();
        require(controller.job_count() == 1 && quotas[0] == 0 && quotas[1] == 1,
            "a diverged board must keep M threads even when the other board is not pondering");
        controller.stop();
    }
    {
        auto data = fixture();
        std::atomic<int> calls{0};
        auto illegal = [&](const Json&, std::atomic_bool&) {
            ++calls; return Reply{409, {}, now_ms(), now_ms(), "illegal_move"};
        };
        Controller controller(1, -1, false, instant, illegal, quota);
        controller.update(data, now_ms(), now_ms());
        settle(controller, [&] { auto j = controller.job(123, 0); return j && j->rejected; });
        data["server_now_ms"] = 100001;
        controller.update(data, now_ms(), now_ms());
        for (int i = 0; i < 5; ++i) controller.tick(now_ms() + 2000);
        require(calls == 1, "illegal move was replayed");
        controller.stop();
    }
    {
        auto data = fixture();
        play(data["matches"][0]["state"]["legs"][0], 19, 0);
        play(data["matches"][0]["state"]["legs"][0], 18, 1);
        std::atomic<int> calls{0};
        Controller controller(2, 3, false, instant, [&](const Json&, std::atomic_bool&) {
            ++calls; return Reply{200, {{"ok", true}}, now_ms(), now_ms(), {}};
        }, quota);
        controller.update(data, now_ms(), now_ms());
        settle(controller, [&] { auto j = controller.job(123, 0); return j && j->result.has_value(); });
        auto job = controller.job(123, 0);
        require(job && calls == 0, "low level sent before pacing delay");
        settle(controller, [&] { auto j = controller.job(123, 0); return j && j->sent; }, 1100);
        require(calls == 1, "paced move not sent");
        auto older = fixture(); older["server_now_ms"] = 99999;
        controller.update(older, now_ms(), now_ms());
        controller.tick();
        require(controller.job(123, 0)->sent, "stale state restarted a sent search");
        auto missing = data; missing["matches"] = Json::array(); missing["server_now_ms"] = 100002;
        controller.update(missing, now_ms(), now_ms()); controller.tick();
        require(controller.job_count() == 1, "temporary omission discarded an active game");
        data["matches"][0]["status"] = "finished"; data["server_now_ms"] = 100003;
        controller.update(data, now_ms(), now_ms()); controller.tick();
        require(controller.job_count() == 0, "finished game left a search alive");
    }
    {
        auto data = fixture();
        Controller controller(1, -1, false, instant, success, quota);
        controller.update(data, now_ms(), now_ms());
        data["matches"] = Json::array({{{"id", 124}, {"bot_a", 2}, {"bot_b", 1}, {"status", "pending"}, {"state", nullptr}}});
        require(controller.challenge(data).has_value(), "standard pending challenge rejected");
        data["matches"][0]["game_mode"] = "anti";
        require(!controller.challenge(data), "unsupported challenge accepted");
        ClockSync clock; clock.observe(10000, 100, 300);
        require(clock.project(10500, 800) == 600, "stream clock did not retain network delay");
        require(!clock.project(200000, 1000), "future stream clock accepted");
    }
    {
        auto data = fixture();
        Controller controller(2, 10, false, instant, success, quota);
        controller.update(data, 100, 100);
        Json event = {{"type", "match"}, {"view", "agent"}, {"match", data["matches"][0]}, {"server_now_ms", 100100}};
        controller.event(event, 3000);
        require(controller.stream_lagging && controller.need_poll, "delayed notifications must request fast polling even on an open stream");
        controller.need_poll = false; event["server_now_ms"] = 103000;
        controller.event(event, 3110);
        require(!controller.stream_lagging && !controller.need_poll, "fresh notifications must restore normal stream polling");
    }
    {
        auto data = fixture();
        Json acknowledged;
        auto reply_with_clock = [match = data["matches"][0]](const Json& body, std::atomic_bool&) mutable {
            play(match["state"]["legs"][0], body.at("move").get<int>(), 0);
            match["state"]["legs"][0]["clocks_ms"][0] = 43210;
            return Reply{200, {{"ok", true}, {"match", match}, {"server_now_ms", 100001}}, now_ms(), now_ms(), {}};
        };
        Controller controller(2, -1, false, instant, reply_with_clock, quota, [](int) {}, [] {},
            [&](const std::string& event, Json fields, bool) { if (event == "move_acknowledged") acknowledged = std::move(fields); });
        controller.update(data, now_ms(), now_ms());
        settle(controller, [&] { return !acknowledged.is_null(); });
        require(acknowledged.at("remaining_ms") == 43210 && acknowledged.at("remaining_source") == "server" &&
                acknowledged.contains("move"), "played-move output must use the server's remaining clock and actual move");
        controller.stop();
    }
    {
        auto data = fixture();
        Controller controller(1, -1, false, instant, success, quota);
        controller.update(data, now_ms(), now_ms()); controller.tick();
        data["matches"][0]["state"]["legs"][0]["revision"] = 1;
        data["server_now_ms"] = 100001;
        controller.update(data, now_ms(), now_ms()); controller.tick();
        require(!controller.job(123, 0), "unpublished board must not authorize a search or move");
        controller.stop();
    }
    {
        auto data = fixture();
        data["matches"][0]["id"] = -10;
        data["matches"][0]["round_no"] = 1;
        data["matches"][0]["room_created_at"] = "first-room";
        Controller controller(1, -1, false, instant, success, quota);
        controller.update(data, now_ms(), now_ms());
        settle(controller, [&] { auto j = controller.job(-10, 0); return j && j->sent; });
        data["matches"][0]["round_no"] = 2;
        data["matches"][0]["room_created_at"] = "second-room";
        data["server_now_ms"] = 100002;
        controller.update(data, now_ms(), now_ms());
        settle(controller, [&] { auto j = controller.job(-10, 0); return j && j->leg.round == 2 && j->sent; });
        controller.stop();
    }
    {
        auto data = fixture();
        std::vector<int> allocations;
        std::atomic_bool release{false};
        auto blocking = [&](const Leg& leg, bool, int, bool*, Millis) {
            while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return first_move(leg);
        };
        Controller controller(2, -1, false, blocking, success, quota,
            [&](int total) { allocations.push_back(total); });
        controller.update(data, now_ms(), now_ms()); controller.tick();
        auto original = controller.job(123, 0);
        Json pair = data["matches"][0]; pair["id"] = 124;
        data["matches"].push_back(pair); data["server_now_ms"] = 100001;
        controller.update(data, now_ms(), now_ms()); controller.tick();
        bool pair_ok = allocations == std::vector<int>{4, 8} && quotas[0] == 3 && quotas[2] == 3;
        Json normal = pair; normal["id"] = -125;
        normal["state"]["legs"].erase(1);
        data["matches"].push_back(normal); data["server_now_ms"] = 100002;
        controller.update(data, now_ms(), now_ms()); controller.tick();
        bool normal_ok = allocations == std::vector<int>{4, 8, 10} && quotas[0] == 3 && quotas[2] == 3 && quotas[4] == 1;
        bool original_kept = original == controller.job(123, 0);
        data["matches"][1]["status"] = "finished"; data["server_now_ms"] = 100003;
        controller.update(data, now_ms(), now_ms()); controller.tick();
        bool no_shrink = allocations == std::vector<int>{4, 8, 10};
        release = true;
        controller.stop();
        require(pair_ok && normal_ok && original_kept && no_shrink,
            "live match admission must grow by 2M/M and retain existing searches and per-match quotas");
    }
    {
        auto data = fixture();
        int ages = 0;
        Controller controller(2, -1, false, instant, success, quota, [](int) {}, [&] { ++ages; });
        controller.update(data, now_ms(), now_ms()); controller.tick();
        data["matches"][0]["state"]["legs"][0]["status"] = "finished";
        data["server_now_ms"] = 100001;
        controller.update(data, now_ms(), now_ms()); controller.tick();
        require(ages == 0 && controller.running(), "one finished leg must not age a paired match");
        data["matches"][0]["state"]["legs"][1]["status"] = "finished";
        data["server_now_ms"] = 100002;
        controller.update(data, now_ms(), now_ms()); controller.tick();
        require(ages == 1 && !controller.running() && !controller.job_count(), "both finished legs must age exactly once");
        data["matches"][0]["status"] = "finished"; data["server_now_ms"] = 100003;
        controller.update(data, now_ms(), now_ms());
        controller.update(data, now_ms(), now_ms()); controller.tick();
        require(ages == 1, "duplicate finish snapshots aged the same match twice");
        auto late = fixture(); late["server_now_ms"] = 100003;
        controller.update(late, now_ms(), now_ms()); controller.tick();
        require(ages == 1 && !controller.running(), "late running snapshot resurrected a completed match");
        auto history = fixture(); history["matches"][0]["id"] = 999;
        history["matches"][0]["status"] = "finished"; history["server_now_ms"] = 100004;
        controller.update(history, now_ms(), now_ms());
        require(ages == 1, "historical finished matches must not age the table on startup");
        auto normal = fixture(); normal["matches"][0]["id"] = -10;
        normal["matches"][0]["round_no"] = 1;
        normal["matches"][0]["state"]["legs"].erase(1); normal["server_now_ms"] = 100005;
        controller.update(normal, now_ms(), now_ms()); controller.tick();
        normal["matches"][0]["status"] = "finished"; normal["server_now_ms"] = 100006;
        controller.update(normal, now_ms(), now_ms()); controller.tick();
        require(ages == 2, "a normal game must age once on completion");
        normal["matches"][0]["status"] = "running"; normal["matches"][0]["round_no"] = 2;
        normal["server_now_ms"] = 100007;
        controller.update(normal, now_ms(), now_ms()); controller.tick();
        normal["matches"][0]["round_no"] = 3; normal["server_now_ms"] = 100008;
        controller.update(normal, now_ms(), now_ms()); controller.tick();
        require(ages == 3, "room round rollover must complete the preceding round once");
        normal["matches"][0]["status"] = "aborted"; normal["server_now_ms"] = 100009;
        controller.update(normal, now_ms(), now_ms()); controller.tick();
        require(ages == 4, "aborted active game must age once");
        controller.stop();
    }
    {
        auto data = fixture();
        int ages = 0;
        Json other = data["matches"][0]; other["id"] = 124;
        data["matches"].push_back(other);
        Controller controller(2, -1, false, instant, success, quota, [](int) {}, [&] { ++ages; });
        controller.update(data, now_ms(), now_ms()); controller.tick();
        auto running_job = controller.job(124, 0);
        data["matches"][0]["status"] = "finished"; data["server_now_ms"] = 100001;
        controller.update(data, now_ms(), now_ms()); controller.tick();
        require(ages == 1 && running_job == controller.job(124, 0), "aging a completed match must preserve another active match");
        controller.stop();
    }
    {
        auto data = fixture(); data["bot"]["id"] = 2;
        std::vector<std::string> events;
        Controller controller(2, 10, false, instant, success, quota, [](int) {}, [] {},
            [&](const std::string& event, Json, bool) { events.push_back(event); });
        controller.update(data, now_ms(), now_ms());
        settle(controller, [&] { auto j = controller.job(123, 1); return j && j->sent; }, 1100);
        auto& finished = data["matches"][0]["state"]["legs"][0];
        finished = {{"status", "finished"}, {"turn", nullptr}, {"clocks_ms", nullptr}};
        auto& live = data["matches"][0]["state"]["legs"][1];
        play(live, 19, 1); play(live, 18, 0);
        live["history"].push_back({{"auto", true}, {"participant", nullptr}, {"color", 2}, {"move", -1}});
        data["server_now_ms"] = 100001;
        controller.update(data, now_ms(), now_ms());
        settle(controller, [&] { auto j = controller.job(123, 1); return j && j->leg.revision == 2 && j->sent; }, 1100);
        require(std::count(events.begin(), events.end(), "search_finished") == 2 &&
                std::count(events.begin(), events.end(), "move_acknowledged") == 2,
            "a finished leg or an auto-pass must not stall the next move; both moves must be logged");
        controller.stop();
    }
    {
        auto path = std::filesystem::temp_directory_path() / ("egaroucid_flipsight_log_" + std::to_string(now_ms()) + ".jsonl");
        std::ostringstream output, errors;
        {
            SessionLog log(path, "fsbm_PRIVATE_TEST_TOKEN", output, errors);
            std::thread a([&] {
                log.write("search_finished", {{"match_id", 123}, {"leg", 0}, {"revision", 2}, {"value", 2}, {"depth", 10}, {"probability", 100}});
                log.write("search_finished", {{"match_id", 123}, {"leg", 1}, {"revision", 3}, {"value", -1}, {"depth", 10}, {"probability", 100}});
                log.write("leg_state", {{"match_id", 123}, {"black", 1122}});
            });
            std::thread b([&] { log.write("move_failed", {{"http_status", 409}, {"detail", "fsbm_PRIVATE_TEST_TOKEN fsbm_OTHER_SECRET"}}, true); });
            a.join(); b.join();
            log.write("leg_finished", {{"match_id", 123}, {"leg", 0}, {"revision", 50}, {"value", 4}});
            log.write("match_finished", {{"match_id", 123}, {"final_value", 7}});
        }
        std::ifstream input(path);
        std::string line, saved;
        int records = 0;
        while (std::getline(input, line)) {
            auto record = Json::parse(line);
            require(record.contains("event") && record.contains("time") && record.contains("severity"), "log record missing metadata");
            saved += line; ++records;
        }
        input.close(); std::filesystem::remove(path);
        require(records == 6 && output.str().find("Evaluation match 123:") != std::string::npos && errors.str().find("move_failed") != std::string::npos,
            "logger must flush events to the file and route normal/error events to stdout/stderr");
        require(output.str().find("match_eval=+1(latest)") != std::string::npos &&
                output.str().find("match_eval=+7(final)") != std::string::npos && output.str().find("depth=final") != std::string::npos,
            "console must show per-board evaluations and the bot's match total, including final results");
        require(output.str().find("leg_state") == std::string::npos && output.str().find("1122") == std::string::npos &&
                saved.find("leg_state") != std::string::npos, "detailed events must be saved without flooding the console");
        require((saved + output.str() + errors.str()).find("fsbm_") == std::string::npos, "logger exposed a bot credential");
    }
    {
        RequestGate gate(true);
        std::atomic_bool stop(false);
        auto start = now_ms();
        require(gate.wait(stop) && gate.wait(stop) && gate.wait(stop), "request gate refused active requests");
        require(now_ms() - start >= 600, "low-level request spacing was not enforced");
        gate.defer(60000);
        stop = true;
        require(!gate.wait(stop), "cancellation ignored while waiting for HTTP cooldown");
    }
    std::cout << "PASS: shared/divergent games, simultaneous searches, ponder, move recovery, pacing, stale/missing states, admission, clocks\n";
}
