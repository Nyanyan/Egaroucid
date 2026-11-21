// Offline protocol checks for issue #667. No account or network is used.
#include <iostream>
#include "../../console/flipsight_http.hpp"
using namespace flipsight;

void require(bool condition, const char* text) {
    if (!condition) throw std::runtime_error(text);
}
template<class F> void rejects(F test) {
    bool rejected = false;
    try { test(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid input accepted");
}
Json fixture() {
    std::vector<int> cells(64, 0);
    cells[27] = cells[36] = 2; cells[28] = cells[35] = 1;
    Json leg = {{"board", cells}, {"turn", 1}, {"revision", 0}, {"status", "active"},
        {"clocks_ms", {60000, 60000}}, {"turn_started_ms", 10000}, {"history", Json::array()},
        {"legal_moves", {19, 26, 37, 44}}};
    return {{"id", 123}, {"bot_a", 1}, {"bot_b", 2}, {"status", "running"},
        {"state", {{"version", 1}, {"status", "active"}, {"started_ms", 10000}, {"legs", {leg, leg}}}}};
}
int main() {
    auto match = fixture();
    auto legs = parse_match(match, 1, 12000, 50000);
    require(legs.size() == 2 && legs[0].ours() && !legs[1].ours(), "paired colors");
    require(legs[0].deadline == 78000, "first move deadline");
    require(legs[0].legal == ((1ULL << 44) | (1ULL << 37) | (1ULL << 26) | (1ULL << 19)), "coordinate mask");
    require(legs[0].move_request(44)["move"] == 19 && legs[0].move_request(64)["move"] == -1, "coordinate conversion");
    require(same_line(legs), "initial shared line");
    legs[0].history = {{1, 19}}; legs[1].history = {{1, 26}};
    require(!same_line(legs), "diverged histories");
    legs[1].history.clear();
    require(same_line(legs), "lagging shared leg");
    require(thread_plan(8, true, {true, false}, {false, true}) == std::array<int,2>{8,0}, "full threads shared");
    require(thread_plan(8, false, {true, true}, {false, false}) == std::array<int,2>{4,4}, "simultaneous searches");
    require(thread_plan(3, false, {true, false}, {false, true}) == std::array<int,2>{1,2}, "odd threads and ponder");
    require(thread_plan(1, false, {true, true}, {false, false}) == std::array<int,2>{1,1}, "one-thread simultaneous roots");
    require(send_at(10, 1000, 61000, 60, 200) == 2000, "low-level pacing");
    require(send_at(10, 1000, 2000, 60, 200) == 1000, "urgent move pacing");
    require(send_at(11, 1000, 61000, 60, 200) == 1000, "higher-level pacing");
    require(send_at(-1, 1000, 61000, 60, 200) == 1000, "time allocation pacing");
    match["state"]["legs"][0]["revision"] = 1;
    require(parse_match(match, 1, 12000, 50000)[0].published, "v2 boards are authoritative even when revision exceeds history length");
    match = fixture();
    match["state"]["legs"][0]["history"].push_back({{"auto", true}, {"participant", nullptr}, {"color", 2}, {"move", -1}});
    require(parse_match(match, 1, 12000, 50000).size() == 2, "server auto-pass with null participant must remain playable");
    match["state"]["legs"][0] = {{"status", "finished"}, {"turn", nullptr}, {"clocks_ms", nullptr}};
    auto remaining = parse_match(match, 2, 12000, 50000);
    require(remaining.size() == 1 && remaining[0].index == 1 && remaining[0].ours(), "finished null-valued leg must not stop the live leg");
    match = fixture();
    match["id"] = -7; match["round_no"] = 3; match["room_created_at"] = "room";
    legs = parse_match(match, 1, 12000, 50000);
    require(legs[0].deadline == 108000 && legs[0].move_request(44)["round_no"] == 3, "human clocks and identity");
    match = fixture(); match["state"]["game_mode"] = "anti";
    require(parse_match(match, 1, 12000, 50000).empty(), "unsupported rules");
    match = fixture(); match["state"]["legs"][0]["board"][0] = true;
    rejects([&] { parse_match(match, 1, 12000, 50000); });
    rejects([] { Endpoint("http://example.com/api"); });
    rejects([] { Endpoint("https://user@example.com/api"); });
    rejects([] { Endpoint("https://example.com/api?token=x"); });
    rejects([] { Endpoint("https://example.com:0/api"); });
    Endpoint endpoint("https://example.com:8443/api");
    require(endpoint.port == 8443 && endpoint.path == "/api", "endpoint parsing");
    std::vector<Json> events;
    EventLines lines([&](const Json& event) { events.push_back(event); });
    lines.append("{\"type\":\"hel", 12);
    std::string tail = "lo\"}\r\n\n{\"type\":\"ping\"}\n";
    lines.append(tail.data(), tail.size());
    require(events.size() == 2 && events[0]["type"] == "hello", "chunked NDJSON");
    rejects([&] { std::string huge(MAX_EVENT + 1, 'x'); lines.append(huge.data(), huge.size()); });
    std::cout << "PASS: FlipSight protocol, clocks, paired scheduling, pacing, endpoint validation, NDJSON\n";
}
