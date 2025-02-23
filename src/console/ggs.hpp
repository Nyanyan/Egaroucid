/*
    Egaroucid Project

    @file ggs.hpp
        Telnet client for Generic Game Server https://skatgame.net/mburo/ggs/
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0 license
*/
#pragma once
#include <winsock2.h>
#include <ws2tcpip.h>
#include "./../engine/engine_all.hpp"
#include "option.hpp"
#pragma comment(lib, "ws2_32.lib")

#define GGS_URL "skatgame.net"
#define GGS_PORT 5000
#define GGS_REPLY_HEADER "GGS> "

int ggs_init(WSADATA &wsaData, struct sockaddr_in &server, SOCKET &sock) {
    // Winsockの初期化
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "Failed to initialize Winsock. Error Code: " << WSAGetLastError() << std::endl;
        return 1;
    }

    // ソケットの作成
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        std::cerr << "Could not create socket. Error Code: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return 1;
    }

    // サーバーのURLをIPアドレスに解決
    const char* hostname = "skatgame.net"; // サーバーのURLを指定
    struct hostent* he = gethostbyname(hostname);
    if (he == nullptr) {
        std::cerr << "Failed to resolve hostname. Error Code: " << WSAGetLastError() << std::endl;
        closesocket(sock);
        WSACleanup();
        return 1;
    }

    // サーバーのアドレスとポートの設定
    server.sin_addr.s_addr = *(u_long*)he->h_addr_list[0];
    server.sin_family = AF_INET;
    server.sin_port = htons(5000); // Telnetのデフォルトポートを指定

    // サーバーに接続
    if (connect(sock, (struct sockaddr*)&server, sizeof(server)) < 0) {
        std::cerr << "Connection failed. Error Code: " << WSAGetLastError() << std::endl;
        closesocket(sock);
        WSACleanup();
        return 1;
    }
    return 0;
}

void ggs_close(SOCKET &sock) {
    closesocket(sock);
    WSACleanup();
}

int ggs_send_message(SOCKET &sock, std::string msg) {
    if (send(sock, msg.c_str(), msg.length(), 0) < 0) {
        return 1;
    }
    return 0;
}

void print_ggs_reply(std::string str) {
    std::stringstream ss(str);
    std::string line;
    while (std::getline(ss, line, '\n')) {
        std::cout << GGS_REPLY_HEADER << line << std::endl;
    }
}

std::string ggs_receive_message(SOCKET &sock) {
    char server_reply[2000];
    int recv_size;
    std::string res;
    if ((recv_size = recv(sock, server_reply, 2000, 0)) == SOCKET_ERROR) {
        std::cerr << "Recv failed. Error Code: " << WSAGetLastError() << std::endl;
        res = "";
    } else {
        server_reply[recv_size] = '\0';
        res = server_reply;
    }
    print_ggs_reply(res);
    return res;
}

void ggs_client(Options *options) {
    WSADATA wsaData;
    SOCKET sock;
    struct sockaddr_in server;
    std::string server_reply;
    
    // connect to GGS server
    if (ggs_init(wsaData, server, sock) != 0) {
        std::cout << "[ERROR] [FATAL] Failed to Connect" << std::endl;
        return;
    }
    std::cout << "Connected to server!" << std::endl;
    server_reply = ggs_receive_message(sock);

    // login
    ggs_send_message(sock, options->ggs_username + "\n");
    server_reply = ggs_receive_message(sock);
    ggs_send_message(sock, options->ggs_password + "\n");
    server_reply = ggs_receive_message(sock);


    ggs_close(sock);
}
