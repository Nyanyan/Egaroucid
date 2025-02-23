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

int ggs_connect(WSADATA &wsaData, struct sockaddr_in &server, SOCKET &sock) {
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

void ggs_print_cyan(std::string str) {
    std::stringstream ss(str);
    std::string line;
    std::cout << "\033[36m";
    while (std::getline(ss, line, '\n')) {
        std::cout << GGS_REPLY_HEADER << line << std::endl;
    }
    std::cout << "\033[0m";
}

void ggs_print_green(std::string str) {
    std::stringstream ss(str);
    std::string line;
    std::cout << "\033[32m";
    while (std::getline(ss, line, '\n')) {
        std::cout << GGS_REPLY_HEADER << line << std::endl;
    }
    std::cout << "\033[0m";
}

int ggs_send_message(SOCKET &sock, std::string msg) {
    if (send(sock, msg.c_str(), msg.length(), 0) < 0) {
        return 1;
    }
    ggs_print_cyan(msg);
    return 0;
}

void set_socket_timeout(SOCKET &sock, int timeout) {
    struct timeval tv;
    tv.tv_sec = timeout / 1000;  // 秒
    tv.tv_usec = (timeout % 1000) * 1000;  // マイクロ秒
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
}

std::string ggs_receive_message_timeout(SOCKET &sock, int timeout) {
    set_socket_timeout(sock, timeout);
    char server_reply[2000];
    int recv_size;
    std::string res;
    if ((recv_size = recv(sock, server_reply, sizeof(server_reply), 0)) == SOCKET_ERROR) {
        if (WSAGetLastError() == WSAETIMEDOUT) {
            //std::cerr << "Recv timed out." << std::endl;
        } else {
            std::cerr << "Recv failed. Error Code: " << WSAGetLastError() << std::endl;
        }
        res = "";
    } else {
        server_reply[recv_size] = '\0';
        res = server_reply;
        ggs_print_green(res);
    }
    return res;
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
        ggs_print_green(res);
    }
    return res;
}

std::string ggs_get_os_info(std::string str) {
    std::stringstream ss(str);
    std::string line;
    while (std::getline(ss, line, '\n')) {
        if (line.substr(0, 4) == "/os:") {
            return line;
        }
    }
    return "";
}

void ggs_client(Options *options) {
    WSADATA wsaData;
    SOCKET sock;
    struct sockaddr_in server;
    std::string server_reply;
    
    // connect to GGS server
    if (ggs_connect(wsaData, server, sock) != 0) {
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
    
    // while (true) {
    //     server_reply = ggs_receive_message(sock);
    //     std::string os_info = ggs_get_os_info(server_reply);
    //     std::cout << os_info << std::endl;
    //     break;
    // }

    ggs_close(sock);
}
