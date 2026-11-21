/* Egaroucid Project: FlipSight session logging. GPL-3.0-or-later. */
#pragma once
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <thread>
#include "flipsight_protocol.hpp"

namespace flipsight {
using Event = std::function<void(const std::string&, Json, bool)>;

class SessionLog {
    struct Record { std::string event; Json fields; bool error; std::chrono::system_clock::time_point time; };
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<Record> queue;
    bool stopping = false;
    std::ofstream file;
    std::string token;
    std::thread worker;
    std::ostream& output;
    std::ostream& errors;
    struct Evaluation { int value = 0, depth = 0, probability = 0; Millis revision = 0; bool final = false; };
    struct MatchSummary { int boards = 2; std::map<int, Evaluation> values; };
    std::map<Millis, MatchSummary> matches;

    static std::string signed_value(int value) { return (value >= 0 ? "+" : "") + std::to_string(value); }
    std::string evaluation(Millis id, std::optional<int> final_total = {}) {
        auto& match = matches[id];
        std::string text = "Evaluation match " + std::to_string(id) + ":";
        int total = 0;
        bool complete = true;
        for (int i = 0; i < match.boards; ++i) {
            text += " leg" + std::to_string(i) + "=";
            auto found = match.values.find(i);
            if (found == match.values.end()) { text += '?'; complete = false; continue; }
            const auto& value = found->second;
            total += value.value;
            text += signed_value(value.value) + "@r" + std::to_string(value.revision) + "[depth=" +
                (value.final ? "final" : value.depth < 0 ? "book" : std::to_string(value.depth) + "@" + std::to_string(value.probability) + "%") +
                "](" + (value.final ? "final" : "search") + ")";
        }
        return text + "; match_eval=" + (final_total ? signed_value(*final_total) : complete ? signed_value(total) : "?") +
            (final_total ? "(final)" : "(latest)") + " [bot disc difference].";
    }
    std::string console_message(const Record& record) {
        const auto& f = record.fields;
        const auto& event = record.event;
        auto id = f.value("match_id", Millis(0));
        auto identity = [&] { return "match " + std::to_string(id) + " leg " + std::to_string(f.value("leg", 0)) +
            " rev " + std::to_string(f.value("revision", Millis(0))); };
        if (event == "session_started") return "Logs: " + f.value("log_file", std::string());
        if (event == "ready") return "Engine ready. actual_threads=" + std::to_string(f.value("search_threads", 0)) +
            ", reported_threads_per_board=" + std::to_string(f.value("reported_threads", 0)) +
            ", level=" + std::to_string(f.value("level", -1)) + ", hash=" + std::to_string(f.value("hash_level", 0)) +
            ", ponder=" + (f.value("ponder", false) ? "on" : "off") + ". Connecting to FlipSight.";
        if (event == "connected") return "Connected: bot " + f.at("bot_id").dump() + ".";
        if (event == "match_started") {
            matches[id] = {}; matches[id].boards = f.value("boards", 2);
            return "Match started: match " + std::to_string(id) + ".";
        }
        if (event == "search_finished" || event == "leg_finished") {
            auto& value = matches[id].values[f.value("leg", 0)];
            if (value.final && event != "leg_finished") return {};
            value = {f.value("value", 0), f.value("depth", 0), f.value("probability", 0),
                f.value("revision", Millis(0)), event == "leg_finished"};
            return evaluation(id);
        }
        if (event == "match_finished") return "Match finished: match " + std::to_string(id) + ". " +
            evaluation(id, f.contains("final_value") ? std::optional<int>(f.at("final_value").get<int>()) : std::nullopt);
        if (event == "move_acknowledged") return "Played " + identity() + " move " + std::to_string(f.value("move", -1)) +
            "; " + std::to_string(f.value("remaining_ms", Millis(0))) + " ms left.";
        if (event == "threads_grown") return "Allocated search threads: " + std::to_string(f.value("threads", 0)) + ".";
        if (event == "stream_retry") return "Event stream interrupted; reconnecting in " + std::to_string(f.value("retry_ms", Millis(0))) + " ms to refresh state.";
        if (event == "stream_connected") return "Event stream connected.";
        if (event == "command_retry") return "API request interrupted (HTTP " + std::to_string(f.value("http_status", 0)) +
            "); retrying in " + std::to_string(f.value("retry_ms", Millis(0))) + " ms.";
        if (event == "stopped") return "Stopped (exit " + std::to_string(f.value("exit_code", 0)) + ").";
        if (event == "api_failed" || event == "stream_failed") return {};
        if (record.error) return "ERROR: " + event + (id ? " for " + identity() : "") +
            (f.contains("http_status") ? " (HTTP " + std::to_string(f.at("http_status").get<int>()) + ")" : "") +
            (f.contains("code") && !f.at("code").get<std::string>().empty() ? ": " + f.at("code").get<std::string>() : "") +
            (f.contains("reason") ? ": " + f.at("reason").get<std::string>() : "") + ". See session log.";
        return {};
    }

    std::string redact(std::string text) const {
        if (!token.empty()) {
            size_t at = 0;
            while ((at = text.find(token, at)) != std::string::npos) {
                text.replace(at, token.size(), "[REDACTED]"); at += 10;
            }
        }
        static const std::regex pattern("fsbm_[A-Za-z0-9_.-]+");
        return std::regex_replace(text, pattern, "[REDACTED]");
    }
    static std::string timestamp(std::chrono::system_clock::time_point now) {
        auto seconds = std::chrono::system_clock::to_time_t(now);
        std::tm utc{};
#ifdef _WIN32
        gmtime_s(&utc, &seconds);
#else
        gmtime_r(&seconds, &utc);
#endif
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
        std::ostringstream text;
        text << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3) << std::setfill('0') << milliseconds << 'Z';
        return text.str();
    }
    void consume() {
        while (true) {
            Record record;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stopping || !queue.empty(); });
                if (queue.empty()) return;
                record = std::move(queue.front()); queue.pop_front();
            }
            try {
                auto time = timestamp(record.time);
                Json data = record.fields;
                data["time"] = time; data["event"] = record.event;
                data["severity"] = record.error ? "error" : "info";
                std::string message;
                try { message = console_message(record); }
                catch (...) { errors << "[FlipSight] Console summary formatting failed; detailed event saved." << std::endl; }
                if (!message.empty()) {
                    data["message"] = message;
                    auto& stream = record.error ? errors : output;
                    stream << time + " " + redact(message) + '\n';
                    stream.flush();
                }
                file << redact(data.dump()) << std::endl;
                if (!file) errors << "[FlipSight] Log write failed; check disk space and permissions." << std::endl;
            } catch (...) { errors << "[FlipSight] Log formatting or output failed." << std::endl; }
        }
    }
public:
    SessionLog(const std::filesystem::path& path, const std::string& credential,
            std::ostream& out = std::cout, std::ostream& err = std::cerr)
        : token(credential), output(out), errors(err) {
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
        file.open(path, std::ios::app);
        if (!file) throw std::runtime_error("Cannot open FlipSight session log");
        worker = std::thread([this] { consume(); });
    }
    void write(const std::string& event, Json fields = Json::object(), bool error = false) noexcept {
        try {
            { std::lock_guard<std::mutex> lock(mutex); queue.push_back({event, std::move(fields), error, std::chrono::system_clock::now()}); }
            wake.notify_all();
        } catch (...) { errors << "[FlipSight] Cannot enqueue log record." << std::endl; }
    }
    ~SessionLog() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        wake.notify_all(); worker.join();
    }
};
} // namespace flipsight
