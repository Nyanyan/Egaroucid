// Exercise the production WinHTTP reader against a live, slow loopback stream.
#define NOMINMAX
#include <winsock2.h>
#include "../../console/flipsight_http.hpp"
#include <future>
#include <iostream>
#include <sstream>
#pragma comment(lib, "ws2_32.lib")

flipsight::Millis measure(bool streaming) {
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data)) throw std::runtime_error("WSAStartup failed");
    SOCKET listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) || listen(listener, 1))
        throw std::runtime_error("Cannot open loopback test server");
    int size = sizeof(address);
    getsockname(listener, reinterpret_cast<sockaddr*>(&address), &size);
    std::promise<void> release;
    auto released = release.get_future();
    auto server = std::async(std::launch::async, [&] {
        SOCKET peer = accept(listener, nullptr, nullptr);
        if (peer == INVALID_SOCKET) throw std::runtime_error("Loopback accept failed");
        char request[4096]; recv(peer, request, sizeof(request), 0);
        std::string head = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
        std::string event = "{\"type\":\"hello\"}\n";
        std::ostringstream chunk; chunk << std::hex << event.size() << "\r\n" << event << "\r\n";
        std::string first = head + chunk.str();
        send(peer, first.data(), static_cast<int>(first.size()), 0);
        // Keep the response open without enough data to fill an 8 KB read.
        released.wait_for(std::chrono::seconds(2));
        send(peer, "0\r\n\r\n", 5, 0);
        closesocket(peer);
    });
    auto session = WinHttpOpen(L"FlipSight loopback test", WINHTTP_ACCESS_TYPE_NO_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    auto connection = WinHttpConnect(session, L"127.0.0.1", ntohs(address.sin_port), 0);
    auto request = WinHttpOpenRequest(connection, L"GET", L"/", nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
    WinHttpSetTimeouts(request, 1000, 1000, 1000, 3000);
    if (!WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
            !WinHttpReceiveResponse(request, nullptr)) throw std::runtime_error("Loopback HTTP request failed");
    bool hello = false;
    flipsight::EventLines lines([&](const flipsight::Json& event) { hello = event.at("type") == "hello"; });
    auto started = flipsight::now_ms();
    char buffer[8192];
    while (!hello) {
        auto received = flipsight::read_winhttp(request, buffer, sizeof(buffer), streaming);
        if (!received) break;
        lines.append(buffer, received);
    }
    auto elapsed = flipsight::now_ms() - started;
    release.set_value(); server.get();
    WinHttpCloseHandle(request); WinHttpCloseHandle(connection); WinHttpCloseHandle(session);
    closesocket(listener); WSACleanup();
    if (!hello) throw std::runtime_error("Loopback stream did not deliver its event");
    return elapsed;
}

int main() try {
    auto legacy = measure(false);
    auto streaming = measure(true);
    if (legacy < 1500 || streaming >= 500) throw std::runtime_error("Streaming regression did not distinguish buffered and immediate reads");
    std::cout << "PASS: small NDJSON event delivery: legacy=" << legacy << " ms, streaming=" << streaming << " ms\n";
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n'; return 1;
}
