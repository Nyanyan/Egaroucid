/*
    Egaroucid Project
    @file flipsight_http.hpp
        Authenticated HTTPS and NDJSON transport for FlipSight
    @license GPL-3.0-or-later
*/
#pragma once
#include <atomic>
#include <cctype>
#include <ctime>
#include <functional>
#include <memory>
#include <thread>
#include "flipsight_protocol.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winhttp.h>
#ifdef _MSC_VER
#pragma comment(lib, "winhttp.lib")
#endif
#else
#include <curl/curl.h>
#endif

namespace flipsight {
constexpr size_t MAX_RESPONSE = 1024 * 1024;
constexpr size_t MAX_EVENT = 256 * 1024;

struct Endpoint {
    std::string host, path, origin;
    unsigned short port = 443;
    explicit Endpoint(const std::string& url) {
        if (url.rfind("https://", 0) != 0 || url.find_first_of("\r\n\t @?#\\") != std::string::npos)
            throw std::runtime_error("FlipSight API must be a credential-free HTTPS URL without query or fragment");
        auto slash = url.find('/', 8);
        std::string authority = url.substr(8, slash == std::string::npos ? slash : slash - 8);
        path = slash == std::string::npos ? "/" : url.substr(slash);
        origin = "https://" + authority;
        auto colon = authority.find(':');
        host = authority.substr(0, colon);
        if (host.empty() || !std::all_of(host.begin(), host.end(), [](unsigned char c) {
                return std::isalnum(c) || c == '-' || c == '.';
            })) throw std::runtime_error("Invalid FlipSight API host");
        if (colon != std::string::npos) {
            std::string text = authority.substr(colon + 1);
            if (text.empty() || text.size() > 5 || !std::all_of(text.begin(), text.end(), [](unsigned char c) {
                    return std::isdigit(c);
                })) throw std::runtime_error("Invalid FlipSight API port");
            int value = std::stoi(text);
            if (value < 1 || value > 65535) throw std::runtime_error("Invalid FlipSight API port");
            port = static_cast<unsigned short>(value);
        }
    }
};

struct HttpResponse {
    int status = 0;
    std::string body;
    Millis retry_after = 0, started = 0, received = 0;
};

inline Millis retry_seconds(const std::string& text) {
    try {
        size_t used = 0;
        auto seconds = std::stoll(text, &used);
        if (used == text.size() && seconds >= 0) return std::min<Millis>(86400, seconds) * 1000;
    } catch (...) {}
    return 1000;
}

class EventLines {
    std::string pending;
    std::function<void(const Json&)> consume;
public:
    explicit EventLines(std::function<void(const Json&)> callback) : consume(std::move(callback)) {}
    void append(const char* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            if (data[i] == '\n') {
                if (!pending.empty() && pending != "\r") {
                    Json event = Json::parse(pending);
                    if (event.is_object()) consume(event);
                }
                pending.clear();
            } else {
                pending += data[i];
                if (pending.size() > MAX_EVENT) throw std::runtime_error("FlipSight event too large");
            }
        }
    }
};

#ifdef _WIN32
inline DWORD read_winhttp(HINTERNET request, char* buffer, DWORD capacity, bool streaming) {
    DWORD count = capacity;
    if (streaming) {
        // A full-size read can wait for later NDJSON events to fill the buffer.
        if (!WinHttpQueryDataAvailable(request, &count)) throw std::runtime_error("FlipSight HTTPS availability query failed");
        if (!count) return 0;
        count = std::min(count, capacity);
    }
    DWORD received = 0;
    if (!WinHttpReadData(request, buffer, count, &received)) throw std::runtime_error("FlipSight HTTPS read failed");
    return received;
}
#endif

class StreamWatchdog {
    std::atomic_bool finished{false};
    std::atomic<Millis> last_event{now_ms()};
    std::thread worker;
public:
    std::atomic_bool cancelled{false};
    StreamWatchdog(const std::atomic_bool& stop, const std::atomic_bool* reconnect, Millis idle_ms = 15000) {
        worker = std::thread([this, &stop, reconnect, idle_ms] {
            while (!finished) {
                if (stop || (reconnect && reconnect->load()) || now_ms() - last_event >= idle_ms) { cancelled = true; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }
    void received() { last_event = now_ms(); }
    ~StreamWatchdog() { finished = true; worker.join(); }
};

class HttpTransport {
    Endpoint api;
    std::string token;
#ifdef _WIN32
    struct Handle {
        HINTERNET value = nullptr;
        explicit Handle(HINTERNET h) : value(h) {
            if (!h) throw std::runtime_error("FlipSight HTTPS initialization failed");
        }
        ~Handle() { if (value) WinHttpCloseHandle(value); }
    };
    Handle session;
    Handle connection;
    static std::wstring wide(const std::string& text) { return std::wstring(text.begin(), text.end()); }

    HttpResponse perform(const std::string& path, const std::string* body, std::atomic_bool& stop,
            const std::function<void(const char*, size_t)>& chunk) {
        auto method = body ? L"POST" : L"GET";
        HINTERNET raw = WinHttpOpenRequest(connection.value, method, wide(path).c_str(), nullptr,
            WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
        if (!raw) throw std::runtime_error("FlipSight HTTPS request failed");
        std::atomic<HINTERNET> request(raw);
        std::atomic_bool finished(false);
        std::thread cancellation([&] {
            while (!finished && !stop) std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (stop) { auto h = request.exchange(nullptr); if (h) WinHttpCloseHandle(h); }
        });
        auto cleanup = [&] {
            finished = true;
            cancellation.join();
            auto h = request.exchange(nullptr);
            if (h) WinHttpCloseHandle(h);
        };
        HttpResponse response;
        try {
            DWORD policy = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
            if (!WinHttpSetOption(raw, WINHTTP_OPTION_REDIRECT_POLICY, &policy, sizeof(policy)))
                throw std::runtime_error("Cannot disable FlipSight HTTP redirects");
            WinHttpSetTimeouts(raw, 5000, 5000, 8000, body ? 8000 : 15000);
            std::wstring headers = L"Authorization: Bearer " + wide(token) +
                (body ? L"\r\nContent-Type: application/json\r\nAccept: application/json\r\n" :
                    L"\r\nAccept: application/x-ndjson\r\n");
            response.started = now_ms();
            if (!WinHttpSendRequest(raw, headers.c_str(), static_cast<DWORD>(headers.size()),
                    body ? const_cast<char*>(body->data()) : WINHTTP_NO_REQUEST_DATA,
                    body ? static_cast<DWORD>(body->size()) : 0, body ? static_cast<DWORD>(body->size()) : 0, 0) ||
                    !WinHttpReceiveResponse(raw, nullptr)) throw std::runtime_error("FlipSight HTTPS connection failed");
            DWORD status = 0, length = sizeof(status);
            if (!WinHttpQueryHeaders(raw, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                    WINHTTP_HEADER_NAME_BY_INDEX, &status, &length, WINHTTP_NO_HEADER_INDEX))
                throw std::runtime_error("FlipSight HTTPS status missing");
            response.status = static_cast<int>(status);
            wchar_t retry[128]{};
            length = sizeof(retry);
            if (WinHttpQueryHeaders(raw, WINHTTP_QUERY_CUSTOM, L"Retry-After", retry, &length, WINHTTP_NO_HEADER_INDEX)) {
                std::wstring value(retry);
                std::string ascii;
                for (wchar_t c : value) ascii += static_cast<char>(c);
                response.retry_after = retry_seconds(ascii);
                SYSTEMTIME date{};
                if (WinHttpTimeToSystemTime(retry, &date)) {
                    FILETIME target{}, current{};
                    SystemTimeToFileTime(&date, &target);
                    GetSystemTimeAsFileTime(&current);
                    ULARGE_INTEGER a{}, b{};
                    a.LowPart = target.dwLowDateTime; a.HighPart = target.dwHighDateTime;
                    b.LowPart = current.dwLowDateTime; b.HighPart = current.dwHighDateTime;
                    response.retry_after = a.QuadPart > b.QuadPart ? std::min<Millis>(86400000, (a.QuadPart - b.QuadPart) / 10000) : 0;
                }
            }
            if (body || response.status == 200) {
                char buffer[8192];
                while (!stop) {
                    DWORD received = read_winhttp(raw, buffer, sizeof(buffer), static_cast<bool>(chunk));
                    if (!received) break;
                    if (chunk) chunk(buffer, received);
                    else {
                        response.body.append(buffer, received);
                        if (response.body.size() > MAX_RESPONSE) throw std::runtime_error("FlipSight response too large");
                    }
                }
            }
            response.received = now_ms();
        } catch (...) { cleanup(); throw; }
        cleanup();
        return response;
    }
#else
    struct CurlGlobal {
        CurlGlobal() { if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) throw std::runtime_error("HTTPS initialization failed"); }
        ~CurlGlobal() { curl_global_cleanup(); }
    };
    CURL* handle;
    HttpResponse perform(const std::string& path, const std::string* body, std::atomic_bool& stop,
            const std::function<void(const char*, size_t)>& chunk) {
        curl_easy_reset(handle);
        HttpResponse response;
        struct Context {
            HttpResponse* response;
            const std::function<void(const char*, size_t)>* chunk;
            CURL* handle;
            bool failed = false;
        } context{&response, &chunk, handle};
        curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, ("Authorization: Bearer " + token).c_str());
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, body ? "Accept: application/json" : "Accept: application/x-ndjson");
        curl_easy_setopt(handle, CURLOPT_URL, (api.origin + path).c_str());
        curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 0L);
        curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, 5000L);
        curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, body ? 8000L : 0L);
        curl_easy_setopt(handle, CURLOPT_LOW_SPEED_LIMIT, 1L);
        curl_easy_setopt(handle, CURLOPT_LOW_SPEED_TIME, 15L);
        curl_easy_setopt(handle, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(handle, CURLOPT_XFERINFODATA, &stop);
        curl_easy_setopt(handle, CURLOPT_XFERINFOFUNCTION, +[](void* p, curl_off_t, curl_off_t, curl_off_t, curl_off_t) -> int {
            return static_cast<std::atomic_bool*>(p)->load() ? 1 : 0;
        });
        curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, +[](char* data, size_t size, size_t count, void* ptr) -> size_t {
            auto& ctx = *static_cast<Context*>(ptr);
            try {
                long status = 0;
                curl_easy_getinfo(ctx.handle, CURLINFO_RESPONSE_CODE, &status);
                if (*ctx.chunk && status != 200) return size * count;
                if (*ctx.chunk) (*ctx.chunk)(data, size * count);
                else {
                    ctx.response->body.append(data, size * count);
                    if (ctx.response->body.size() > MAX_RESPONSE) throw std::runtime_error("response too large");
                }
                return size * count;
            } catch (...) { ctx.failed = true; return 0; }
        });
        curl_easy_setopt(handle, CURLOPT_WRITEDATA, &context);
        curl_easy_setopt(handle, CURLOPT_HEADERFUNCTION, +[](char* data, size_t size, size_t count, void* ptr) -> size_t {
            std::string header(data, size * count);
            auto colon = header.find(':');
            if (colon != std::string::npos) {
                std::string name = header.substr(0, colon);
                std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (name == "retry-after") {
                    auto text = header.substr(colon + 1);
                    text.erase(0, text.find_first_not_of(" \t"));
                    text.erase(text.find_last_not_of(" \t\r\n") + 1);
                    auto& response = *static_cast<HttpResponse*>(ptr);
                    response.retry_after = retry_seconds(text);
                    auto date = curl_getdate(text.c_str(), nullptr);
                    if (date >= 0) response.retry_after = std::clamp<Millis>((date - std::time(nullptr)) * 1000, 0, 86400000);
                }
            }
            return size * count;
        });
        curl_easy_setopt(handle, CURLOPT_HEADERDATA, &response);
        if (body) {
            curl_easy_setopt(handle, CURLOPT_POSTFIELDS, body->c_str());
            curl_easy_setopt(handle, CURLOPT_POSTFIELDSIZE, static_cast<long>(body->size()));
        }
        response.started = now_ms();
        auto code = curl_easy_perform(handle);
        response.received = now_ms();
        long status = 0;
        curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &status);
        response.status = static_cast<int>(status);
        curl_slist_free_all(headers);
        if (code != CURLE_OK || context.failed) throw std::runtime_error("FlipSight HTTPS transfer failed");
        return response;
    }
#endif
public:
    HttpTransport(const std::string& url, const std::string& credential) : api(url), token(credential)
#ifdef _WIN32
        , session(WinHttpOpen(L"Egaroucid FlipSight", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
            WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0))
        , connection(WinHttpConnect(session.value, wide(api.host).c_str(), api.port, 0))
#endif
    {
        if (token.empty() || !std::all_of(token.begin(), token.end(), [](unsigned char c) { return c >= 33 && c <= 126; }))
            throw std::runtime_error("Missing or invalid FlipSight token");
#ifndef _WIN32
        static CurlGlobal global;
        handle = curl_easy_init();
        if (!handle) throw std::runtime_error("FlipSight HTTPS initialization failed");
#endif
    }
    HttpTransport(const HttpTransport&) = delete;
    HttpTransport& operator=(const HttpTransport&) = delete;
    ~HttpTransport() {
#ifndef _WIN32
        curl_easy_cleanup(handle);
#endif
    }
    HttpResponse call(const Json& request, std::atomic_bool& stop) {
        auto body = request.dump();
        return perform(api.path, &body, stop, {});
    }
    HttpResponse stream(std::atomic_bool& stop, const std::function<void(const Json&)>& consume,
            const std::atomic_bool* reconnect = nullptr) {
        StreamWatchdog watchdog(stop, reconnect);
        EventLines lines([&](const Json& event) { watchdog.received(); consume(event); });
        return perform("/flipsight_api/bot-stream", nullptr, watchdog.cancelled,
            [&](const char* data, size_t size) { lines.append(data, size); });
    }
};
} // namespace flipsight
