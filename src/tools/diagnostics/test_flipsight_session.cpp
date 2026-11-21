// API v2 lifecycle without a token, network access, or engine resources.
#include "../../console/flipsight_session.hpp"
#include <iostream>
using namespace flipsight;

void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
Json match_fixture() {
    std::vector<int> board(64); board[27] = board[36] = 2; board[28] = board[35] = 1;
    Json leg = {{"board", board}, {"turn", 1}, {"revision", 1}, {"status", "active"}, {"history", Json::array()},
        {"clocks_ms", {60000, 60000}}, {"turn_started_ms", 100000}, {"legal_moves", {19, 26, 37, 44}}};
    return {{"id", 123}, {"bot_a", 2}, {"bot_b", 1}, {"status", "pending"},
        {"state", {{"version", 1}, {"revision", 1}, {"status", "active"}, {"legs", {leg, leg}}}}};
}
int main() try {
    std::atomic<int> environments{0}, accepts{0}, moves{0}, connections{0}, searches{0};
    std::atomic_bool finished(false);
    Json original;
    auto call = [&](const Json& body, std::atomic_bool&) {
        auto action = body.at("action").get<std::string>();
        if (action == "agent_environment") ++environments;
        else if (action == "agent_accept") ++accepts;
        else if (action == "agent_move") {
            if (++moves == 1) { original = body; return Reply{}; }
            require(body == original, "reconnected move changed revision or coordinate");
        } else throw std::runtime_error("v2 sent an obsolete or unexpected action");
        return Reply{200, {{"ok", true}}, now_ms(), now_ms(), {}};
    };
    auto stream = [&](std::atomic_bool& stop, const std::function<void(const Json&)>& consume, const std::atomic_bool& reconnect) {
        int connection = ++connections;
        Millis server = 100000 + connection * 100;
        consume({{"type", "hello"}, {"api", 2}, {"bot_id", 1}, {"server_now_ms", server}});
        auto match = match_fixture();
        if (connection > 1) match["status"] = "running";
        consume({{"type", "state"}, {"view", "agent"}, {"ok", true}, {"bot", {{"id", 1}}},
            {"matches", {match}}, {"server_now_ms", server + 1}});
        if (connection == 1) {
            while (!stop && !accepts) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            match["status"] = "running";
            consume({{"type", "match"}, {"view", "agent"}, {"match", match}, {"server_now_ms", server + 2}});
            while (!stop && !reconnect) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            while (!stop && moves < 2) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            match["status"] = "finished"; match["state"]["revision"] = 2;
            consume({{"type", "match"}, {"view", "agent"}, {"match", match}, {"server_now_ms", server + 2}});
            while (!stop) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return HttpResponse{200};
    };
    auto logger = [&](const std::string& event, Json, bool) { if (event == "match_finished") finished = true; };
    Controller controller(2, -1, false, [&](const Leg& leg, bool, int, bool*, Millis) {
        ++searches;
        for (int i = 0; i < 64; ++i) if (leg.legal & (1ULL << i)) return MoveResult{i, 2, 10};
        return MoveResult{};
    }, call, [](int, int) {}, [](int) {}, [] {}, logger);
    auto began = now_ms();
    int result = run_session(controller, call, stream, {{"engine_threads", 2}}, logger,
        [&] { return finished || now_ms() - began > 8000; });
    require(result == 0 && finished && environments == 1 && accepts == 1 && moves == 2 && connections == 2 && searches == 1,
        "v2 must report environment once, accept stream challenges, reconnect and replay only after fresh state");
    Controller admission(2, -1, false, {}, call, [](int, int) {});
    auto pending = match_fixture(), other = pending; other["id"] = 124;
    admission.update({{"bot", {{"id", 1}, {"max_games", 1}}}, {"matches", {pending, other}},
        {"server_now_ms", 100000}}, now_ms(), now_ms());
    require(!admission.pending_challenge({123}), "in-flight admissions exceeded the website match limit");

    auto match = match_fixture(); match["status"] = "running";
    Controller ordering(2, -1, false, {}, call, [](int, int) {});
    auto update = [&](Json value, Millis server) { ordering.update({{"bot", {{"id", 1}}}, {"matches", {value}}, {"server_now_ms", server}}, now_ms(), now_ms()); };
    match["state"]["revision"] = 10; update(match, 100000);
    auto stale = match; stale["status"] = "finished"; stale["state"]["revision"] = 9; update(stale, 100001);
    require(ordering.running(), "stale match revision overwrote a live match");
    ordering.event({{"type", "state"}, {"view", "agent"}, {"bot", {{"id", 1}}}, {"matches", Json::array()},
        {"server_now_ms", 100002}}, now_ms());
    require(!ordering.running(), "reconnect state left an absent match alive");
    auto human = match_fixture(); human["id"] = -4; human["status"] = "running";
    human["room_created_at"] = "2026-11-21 12:00:00"; human["round_no"] = 4;
    human["state"]["legs"].erase(1);
    update(human, 100003);
    auto old_room = human; old_room["room_created_at"] = "2026-11-20 12:00:00";
    old_room["round_no"] = 100; old_room["status"] = "finished"; update(old_room, 100004);
    require(ordering.running(), "an old human room overwrote its replacement");
    auto old_round = human; old_round["round_no"] = 3; old_round["status"] = "finished"; update(old_round, 100005);
    require(ordering.running(), "an old human round overwrote the current round");
    human["round_no"] = 5; human["state"]["revision"] = 0; human["status"] = "finished"; update(human, 100006);
    require(!ordering.running(), "a new human round with a lower revision was rejected");

    int ages = 0, leg_results = 0, final_value = 0;
    Controller scoring(2, -1, false, {}, call, [](int, int) {}, [](int) {}, [&] { ++ages; },
        [&](const std::string& event, Json fields, bool) {
            if (event == "leg_finished") ++leg_results;
            if (event == "match_finished") final_value = fields.at("final_value").get<int>();
        });
    match["state"]["revision"] = 1;
    scoring.update({{"bot", {{"id", 1}}}, {"matches", {match}}, {"server_now_ms", 100000}}, now_ms(), now_ms());
    match["state"]["revision"] = 2; match["state"]["totals"] = {60, 68};
    match["state"]["legs"][0]["status"] = "timeout"; match["state"]["legs"][0]["result_discs"] = {0, 64};
    match["state"]["legs"][1]["status"] = "stopped"; match["state"]["legs"][1]["discs"] = {4, 60};
    scoring.update({{"matches", {match}}, {"server_now_ms", 100001}}, now_ms(), now_ms());
    require(!scoring.running() && ages == 1 && leg_results == 2 && final_value == 8,
        "timeout/stopped results must use result_discs or discs and age the completed pair once");
    auto no_moves = match_fixture(); no_moves["status"] = "running";
    std::vector<int> pass_board(64); pass_board[0] = 2; pass_board[1] = 1;
    for (auto& leg : no_moves["state"]["legs"]) { leg["board"] = pass_board; leg["legal_moves"] = Json::array(); }
    int pass_calls = 0;
    auto pass_api = [&](const Json& body, std::atomic_bool&) {
        require(body.at("move") == -1, "paired no-move position must send an explicit pass");
        ++pass_calls; return Reply{200, {{"ok", true}}, now_ms(), now_ms(), {}};
    };
    auto pass_search = [](const Leg&, bool, int, bool*, Millis) { return MoveResult{}; };
    Controller passes(2, -1, false, pass_search, pass_api, [](int, int) {});
    passes.update({{"bot", {{"id", 1}}}, {"matches", {no_moves}}, {"server_now_ms", 100000}}, now_ms(), now_ms());
    auto deadline = now_ms() + 500;
    while (now_ms() < deadline) {
        passes.tick();
        auto job = passes.job(123, 1); if (job && job->sent) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    passes.stop(); require(pass_calls == 1, "paired pass was not sent exactly once");
    std::atomic_bool stop(false), reconnect(false);
    {
        StreamWatchdog watchdog(stop, &reconnect, 100);
        for (int i = 0; i < 4; ++i) { std::this_thread::sleep_for(std::chrono::milliseconds(30)); watchdog.received(); }
        require(!watchdog.cancelled, "ping events did not keep the stream alive");
        reconnect = true;
        auto deadline = now_ms() + 500;
        while (!watchdog.cancelled && now_ms() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        require(watchdog.cancelled, "resynchronization did not cancel the old connection");
    }
    reconnect = false;
    {
        StreamWatchdog watchdog(stop, &reconnect, 60);
        auto deadline = now_ms() + 500;
        while (!watchdog.cancelled && now_ms() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        require(watchdog.cancelled, "silent stream did not time out");
    }
    std::cout << "PASS: v2 hello/state/match, stream admission, one environment report, reconnect recovery and revision ordering\n";
} catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
