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
#include <mutex>
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
                auto& stream = record.error ? errors : output;
                stream << time + " [FlipSight] " + record.event + ' ' + redact(record.fields.dump()) + '\n';
                stream.flush();
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
