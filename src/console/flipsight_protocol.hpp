/*
    Egaroucid Project
    @file flipsight_protocol.hpp
        FlipSight bot API state and scheduling rules (issue #667)
    @license GPL-3.0-or-later
*/
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "lib/json.hpp"

namespace flipsight {
using Json = nlohmann::json;
using Millis = int64_t;

inline Millis now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline bool active(const Json& item) {
    const auto status = item.value("status", std::string());
    return status == "active" || status == "running" || status == "playing";
}

inline Millis integer(const Json& value) {
    if (!value.is_number_integer()) throw std::runtime_error("Invalid FlipSight integer");
    return value.get<Millis>();
}

inline Millis clock_ms(const Json& value) {
    if (value.is_number_integer()) return integer(value);
    if (!value.is_number_float()) throw std::runtime_error("Invalid FlipSight clock");
    double number = value.get<double>();
    if (!std::isfinite(number) || number >= static_cast<double>(std::numeric_limits<Millis>::max()) ||
            number < static_cast<double>(std::numeric_limits<Millis>::min())) throw std::runtime_error("Invalid FlipSight clock");
    return static_cast<Millis>(std::floor(number));
}

inline bool same_id(const Json& a, const Json& b) {
    if (a.is_null() || b.is_null()) return false;
    return (a.is_string() ? a.get<std::string>() : a.dump()) ==
        (b.is_string() ? b.get<std::string>() : b.dump());
}

struct Leg {
    Millis match_id = 0, revision = -1;
    int index = 0, turn = 0, color = 0;
    uint64_t black = 0, white = 0, legal = 0;
    Millis deadline = 0;
    bool running = false, published = true;
    Json round = nullptr, room = nullptr;
    std::vector<std::pair<int, int>> history;

    bool ours() const { return running && published && turn == color; }
    bool same_position(const Leg& other) const {
        return black == other.black && white == other.white && turn == other.turn;
    }
    bool same_turn(const Leg& other) const {
        return match_id == other.match_id && index == other.index && revision == other.revision &&
            round == other.round && room == other.room && same_position(other) && color == other.color;
    }
    Json move_request(int engine_policy) const {
        Json result = {{"action", "agent_move"}, {"match_id", match_id}, {"leg", index},
            {"revision", revision}, {"move", engine_policy == 64 ? -1 : 63 - engine_policy}};
        if (match_id < 0) {
            result["round_no"] = round;
            result["room_created_at"] = room;
        }
        return result;
    }
};

inline bool supported(const Json& match) {
    auto state = match.value("state", Json::object());
    if (!state.is_object()) state = Json::object();
    return state.value("game_mode", match.value("game_mode", std::string("std"))) == "std" &&
        state.value("board_size", 8) == 8 &&
        (!state.contains("blocked_squares") || state["blocked_squares"].empty());
}

inline uint64_t legal_moves(uint64_t black, uint64_t white, int turn) {
    uint64_t own = turn == 1 ? black : white, enemy = turn == 1 ? white : black, legal = 0;
    for (int square = 0; square < 64; ++square) {
        if ((black | white) & (1ULL << (63 - square))) continue;
        for (int dr = -1; dr <= 1; ++dr) for (int dc = -1; dc <= 1; ++dc) {
            if (!dr && !dc) continue;
            int row = square / 8 + dr, col = square % 8 + dc;
            bool captured = false;
            while (row >= 0 && row < 8 && col >= 0 && col < 8 &&
                    (enemy & (1ULL << (63 - row * 8 - col)))) {
                captured = true; row += dr; col += dc;
            }
            if (captured && row >= 0 && row < 8 && col >= 0 && col < 8 &&
                    (own & (1ULL << (63 - row * 8 - col)))) legal |= 1ULL << (63 - square);
        }
    }
    return legal;
}

// observed_at is the HTTP request start, a conservative lower bound on server_now_ms.
inline std::vector<Leg> parse_match(const Json& match, const Json& bot_id,
        Millis server_now, Millis observed_at) {
    std::vector<Leg> result;
    if (!active(match) || !supported(match)) return result;
    int participant = same_id(match.value("bot_a", Json()), bot_id) ? 0 :
        same_id(match.value("bot_b", Json()), bot_id) ? 1 : -1;
    if (participant < 0) return result;
    const auto& state = match.at("state");
    if (state.value("version", 0) != 1) throw std::runtime_error("Unsupported FlipSight state version");
    if (!active(state)) return result;
    const auto& legs = state.at("legs");
    if (!legs.is_array() || legs.empty() || legs.size() > 2)
        throw std::runtime_error("Invalid FlipSight legs");
    bool played = false, history_known = true;
    for (const auto& leg : legs) {
        history_known = history_known && leg.contains("history");
        for (const auto& move : leg.value("history", Json::array())) {
            if (move.contains("auto") && move["auto"] == true) continue;
            if (move.contains("participant") && move["participant"].is_number_integer() &&
                    integer(move["participant"]) == participant && move.contains("move") &&
                    move["move"].is_number_integer() && integer(move["move"]) >= 0) played = true;
        }
    }
    for (size_t i = 0; i < legs.size(); ++i) {
        const auto& source = legs[i];
        // Finished legs may have null turns/clocks. They must not stop the other leg.
        if (!active(source)) continue;
        Leg leg;
        leg.match_id = integer(match.at("id"));
        if (!leg.match_id) throw std::runtime_error("Invalid FlipSight match id");
        leg.index = static_cast<int>(i);
        leg.revision = integer(source.at("revision"));
        if (leg.revision < 0) throw std::runtime_error("Invalid FlipSight revision");
        leg.round = match.value("round_no", Json());
        leg.room = match.value("room_created_at", Json());
        leg.running = active(source);
        // v2 snapshots carry authoritative boards; revisions need not equal history length.
        leg.published = true;
        leg.turn = static_cast<int>(integer(source.at("turn")));
        leg.color = participant == static_cast<int>(i) ? 1 : 2;
        if (leg.turn != 1 && leg.turn != 2) {
            if (leg.running) throw std::runtime_error("Invalid FlipSight turn");
            continue;
        }
        const auto& cells = source.at("board");
        if (!cells.is_array() || cells.size() != 64) throw std::runtime_error("Invalid FlipSight board");
        for (int square = 0; square < 64; ++square) {
            auto cell = integer(cells[square]);
            if (cell < 0 || cell > 2) throw std::runtime_error("Invalid FlipSight cell");
            if (cell == 1) leg.black |= 1ULL << (63 - square);
            if (cell == 2) leg.white |= 1ULL << (63 - square);
        }
        if (source.contains("legal_moves")) {
            if (!source["legal_moves"].is_array()) throw std::runtime_error("Invalid FlipSight legal moves");
            for (const auto& move : source["legal_moves"]) {
                auto square = integer(move);
                if (square < 0 || square >= 64 || cells[static_cast<size_t>(square)] != 0)
                    throw std::runtime_error("Invalid FlipSight legal square");
                leg.legal |= 1ULL << (63 - square);
            }
        } else {
            leg.legal = legal_moves(leg.black, leg.white, leg.turn);
        }
        if (leg.published && leg.running && leg.legal != legal_moves(leg.black, leg.white, leg.turn))
            throw std::runtime_error("FlipSight legal moves disagree with standard Othello");
        const auto& clocks = source.at("clocks_ms");
        if (!clocks.is_array() || clocks.size() != 2) throw std::runtime_error("Invalid FlipSight clocks");
        Millis remaining = clock_ms(clocks[leg.color - 1]);
        if (leg.turn == leg.color && source.contains("turn_started_ms") && !source["turn_started_ms"].is_null())
            remaining -= std::max<Millis>(0, server_now - integer(source["turn_started_ms"]));
        leg.deadline = observed_at + std::max<Millis>(0, remaining);
        if (leg.match_id > 0 && history_known && !played && state.contains("started_ms"))
            leg.deadline = std::min(leg.deadline, observed_at + integer(state["started_ms"]) + 30000 - server_now);
        for (const auto& move : source.value("history", Json::array())) {
            leg.history.emplace_back(move.at("color").get<int>(), move.at("move").get<int>());
        }
        result.push_back(leg);
    }
    return result;
}

inline bool same_line(const std::vector<Leg>& legs) {
    if (legs.size() < 2) return true;
    const auto& a = legs[0];
    const auto& b = legs[1];
    if (!std::equal(a.history.begin(), a.history.begin() + std::min(a.history.size(), b.history.size()), b.history.begin()))
        return false;
    if (a.history.size() == b.history.size()) return a.same_position(b);
    return true;
}

// Counts include each search's calling thread; the pool receives counts minus one.
inline std::array<int, 2> thread_plan(int total, bool shared, const std::array<bool, 2>& thinking,
        const std::array<bool, 2>& pondering) {
    std::array<int, 2> result{0, 0};
    for (int i = 0; i < 2; ++i) if (thinking[i] || pondering[i]) result[i] = 1;
    if (shared) {
        int selected = thinking[0] ? 0 : thinking[1] ? 1 : pondering[0] ? 0 : pondering[1] ? 1 : -1;
        result = {0, 0};
        if (selected >= 0) result[selected] = total;
    } else if (result[0] && result[1]) {
        result[0] = std::max(1, total / 2);
        result[1] = std::max(1, total - result[0]);
    } else {
        for (auto& count : result) if (count) count = total;
    }
    return result;
}

inline Millis search_budget(Millis remaining) {
    return std::max<Millis>(1, remaining - std::min<Millis>(5000, remaining / 2));
}

inline Millis send_at(int level, Millis started, Millis deadline, int empties, Millis transport_ms) {
    Millis turns_left = std::max(1, (empties + 1) / 2);
    Millis required = turns_left * (1000 + transport_ms) + 1000;
    return level >= 0 && level <= 10 && deadline - started > required ? started + 1000 : started;
}
} // namespace flipsight
