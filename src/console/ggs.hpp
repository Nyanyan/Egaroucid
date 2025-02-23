/*
    Egaroucid Project

    @file ggs.hpp
        Telnet client for Generic Game Server https://skatgame.net/mburo/ggs/
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0 license
*/
#pragma once
#include "./../engine/engine_all.hpp"
#include "option.hpp"
#pragma comment(lib, "ws2_32.lib")

#define GGS_URL "skatgame.net"
#define GGS_PORT 5000
#define GGS_READY "READY"
#define GGS_REPLY_HEADER "GGS> "

struct GGS_Board {
    int last_move;
    std::string player_black;
    uint64_t remaining_seconds_black;
    std::string player_white;
    uint64_t remaining_seconds_white;
    Board board;
    int player_to_move;

    GGS_Board() {
        last_move = -1;
        player_black = "";
        remaining_seconds_black = 0;
        player_white = "";
        remaining_seconds_white = 0;
        board.player = 0;
        board.opponent = 0;
        player_to_move = -1;
    }

    bool is_valid() {
        return (board.player != 0 || board.opponent != 0) && player_to_move != -1;
    }
};

std::vector<std::string> split_by_space(const std::string &str) {
    std::vector<std::string> tokens;
    std::istringstream iss(str);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> split_by_delimiter(const std::string &str, const std::string &delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = str.find(delimiter);
    
    while (end != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }
    
    tokens.push_back(str.substr(start));
    return tokens;
}

std::string remove_spaces(const std::string &str) {
    std::string result = str;
    result.erase(std::remove(result.begin(), result.end(), ' '), result.end());
    return result;
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

int ggs_connect(WSADATA &wsaData, struct sockaddr_in &server, SOCKET &sock) {
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "Failed to initialize Winsock. Error Code: " << WSAGetLastError() << std::endl;
        return 1;
    }

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        std::cerr << "Could not create socket. Error Code: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return 1;
    }

    
    const char* hostname = GGS_URL;
    struct hostent* he = gethostbyname(hostname);
    if (he == nullptr) {
        std::cerr << "Failed to resolve hostname. Error Code: " << WSAGetLastError() << std::endl;
        closesocket(sock);
        WSACleanup();
        return 1;
    }

    server.sin_addr.s_addr = *(u_long*)he->h_addr_list[0];
    server.sin_family = AF_INET;
    server.sin_port = htons(GGS_PORT);
    
    /*
    const char* hostname = GGS_URL;
    struct addrinfo hints, *result;
    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    if (getaddrinfo(hostname, std::to_string(GGS_PORT).c_str(), &hints, &result) != 0) {
        std::cerr << "Failed to resolve hostname. Error Code: " << WSAGetLastError() << std::endl;
        closesocket(sock);
        WSACleanup();
        return 1;
    }

    memcpy(&server, result->ai_addr, result->ai_addrlen);
    freeaddrinfo(result);
    */

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
    ggs_print_cyan(msg);
    return 0;
}

std::vector<std::string> ggs_receive_message(SOCKET *sock) {
    char server_reply[2000];
    int recv_size;
    std::vector<std::string> res;
    if ((recv_size = recv(*sock, server_reply, 2000, 0)) == SOCKET_ERROR) {
        std::cerr << "Recv failed. Error Code: " << WSAGetLastError() << std::endl;
    } else {
        server_reply[recv_size] = '\0';
        res = split_by_delimiter(server_reply, GGS_READY);
        ggs_print_green(server_reply);
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

std::string ggs_get_user_input() {
    std::string res;
    std::getline(std::cin, res);
    return res;
}

bool ggs_is_board_info(std::string line) {
    std::vector<std::string> words = split_by_space(line);
    if (words.size() >= 2) {
        return words[1] == "update" || words[1] == "join";
    }
    return false;
}

std::string ggs_board_get_id(std::string line) {
    std::vector<std::string> words = split_by_space(line);
    if (words.size() >= 3) {
        return words[2];
    }
    return "";
}

GGS_Board ggs_get_board(std::string str) {
    GGS_Board res;
    std::string board_str;
    std::stringstream ss(str);
    std::string line;
    int n_board_identifier_found = 0;
    while (std::getline(ss, line, '\n')) {
        std::vector<std::string> words = split_by_space(line);
        if (line[0] == '|') {
            if (line.substr(0, 10) == "|0 move(s)") {
                continue;
            }
            // board
            if (line.find("A B C D E F G H") != std::string::npos) {
                ++n_board_identifier_found;
                continue;
            }
            if (n_board_identifier_found == 1) { // board info
                std::string board_str_part = line.substr(3, 16);
                board_str += remove_spaces(board_str_part);
                continue;
            }

            // which to move
            if (line == "|* to move") {
                res.player_to_move = BLACK;
                continue;
            } else if (line == "|O to move") {
                res.player_to_move = WHITE;
                continue;
            }

            // last move
            if (line.substr(0, 3) == "|  ") {
                if (words.size() >= 3) {
                    if (words[2].size() >= 2) {
                        res.last_move = get_coord_from_chars(words[2][0], words[2][1]);
                        continue;
                    }
                }
            }

            // users
            if (words.size() >= 4) {
                std::string player_id = words[0].substr(1, words[0].size() - 1);
                std::string remaining_time_minute = words[3].substr(0, 2);
                std::string remaining_time_second = words[3].substr(3, 2);
                uint64_t remaining_seconds = std::stoi(remaining_time_minute) * 60 + std::stoi(remaining_time_second);
                if (words[2][0] == '*') {
                    res.player_black = player_id;
                    res.remaining_seconds_black = remaining_seconds;
                } else if (words[2][0] == 'O') {
                    res.player_white = player_id;
                    res.remaining_seconds_white = remaining_seconds;
                }
            }
        }
    }
    if (res.player_to_move == BLACK) {
        board_str += " *";
    } else if (res.player_to_move == WHITE) {
        board_str += " O";
    }
    res.board.from_str(board_str);

    std::cerr << "black " << res.player_black << " " << res.remaining_seconds_black << std::endl;
    std::cerr << "white " << res.player_white << " " << res.remaining_seconds_white << std::endl;
    std::cerr << res.player_to_move << " to move" << std::endl;
    std::cerr << board_str << std::endl;
    res.board.print();

    return res;
}

void ggs_client(Options *options) {
    WSADATA wsaData;
    SOCKET sock;
    struct sockaddr_in server;
    
    // connect to GGS server
    if (ggs_connect(wsaData, server, sock) != 0) {
        std::cout << "[ERROR] [FATAL] Failed to Connect" << std::endl;
        return;
    }
    std::cout << "Connected to server!" << std::endl;
    ggs_receive_message(&sock);

    // login
    ggs_send_message(sock, options->ggs_username + "\n");
    ggs_receive_message(&sock);
    ggs_send_message(sock, options->ggs_password + "\n");
    ggs_receive_message(&sock);

    // initialize
    ggs_send_message(sock, "ms /os\n");
    ggs_receive_message(&sock);
    ggs_send_message(sock, "ts client -\n");
    ggs_receive_message(&sock);
    
    std::future<std::string> user_input_f;
    std::future<std::vector<std::string>> ggs_message_f;
    while (true) {
        std::vector<std::string> server_replies;
        if (user_input_f.valid()) {
            if (user_input_f.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                std::string user_input = user_input_f.get();
                if (user_input == "exit") {
                    break;
                }
                ggs_send_message(sock, user_input + "\n");
            }
        } else {
            user_input_f = std::async(std::launch::async, ggs_get_user_input);
        }
        if (ggs_message_f.valid()) {
            if (ggs_message_f.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                server_replies = ggs_message_f.get();
                std::cerr << "server_replies.size() " << server_replies.size() << std::endl;
                for (std::string server_reply: server_replies) {
                    std::cerr << server_reply << std::endl;
                }
            }
        } else {
            ggs_message_f = std::async(std::launch::async, ggs_receive_message, &sock);
        }
        if (server_replies.size()) {
            for (std::string server_reply: server_replies) {
                if (server_reply.size()) {
                    std::cout << "see " << server_reply << std::endl;
                    std::string os_info = ggs_get_os_info(server_reply);
                    std::cout << "os_info " << os_info << std::endl;
                    if (ggs_is_board_info(os_info)) {
                        std::cout << "getting board info" << std::endl;
                        GGS_Board ggs_board = ggs_get_board(server_reply);
                        if (ggs_board.is_valid()) {
                            bool need_to_move = 
                                (ggs_board.player_black == options->ggs_username && ggs_board.player_to_move == BLACK) || 
                                (ggs_board.player_white == options->ggs_username && ggs_board.player_to_move == WHITE);
                            if (need_to_move) {
                                
                            } else {
                                
                            }
                        }
                    }
                }
            }
        }
    }

    // close connection
    ggs_close(sock);
}
