/*
    Egaroucid Project

    @file ggs.hpp
        Telnet client for Generic Game Server https://skatgame.net/mburo/ggs/
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/
#pragma once
#include "./../engine/engine_all.hpp"
#include "option.hpp"
#include "util.hpp"
#pragma comment(lib, "ws2_32.lib")

#define GGS_URL "skatgame.net"
#define GGS_PORT 5000
#define GGS_READY "READY"
#define GGS_REPLY_HEADER "GGS RECV> "
#define GGS_SEND_HEADER "GGS SEND> "
#define GGS_INFO_HEADER "GGS INFO> "

#define GGS_NON_SYNCHRO_ID 0

#define GGS_SEND_EMPTY_INTERVAL 180000ULL // 3 minutes

#define GGS_USE_PONDER true
#define GGS_N_PONDER_PARALLEL 1


struct GGS_Board {
    std::string game_id;
    bool is_synchro;
    int synchro_id;
    int last_move;
    std::string player_black;
    uint64_t remaining_seconds_black;
    std::string player_white;
    uint64_t remaining_seconds_white;
    Board board;
    int player_to_move;

    GGS_Board() {
        game_id = "";
        is_synchro = false;
        synchro_id = -1;
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

struct GGS_Match {
    std::string game_id;
    std::string initial_board;
    std::string transcript;
    int result_black;
    std::string player_black;
    std::string player_white;
    
    void init() {
        game_id = "";
    }

    bool is_initialized() {
        return game_id == "";
    }
};

void ggs_print_send(std::string str, Options *options) { // cyan
    std::stringstream ss(str);
    std::string line;
    std::ofstream ofs;
    if (options->ggs_log_to_file) {
        ofs.open(options->ggs_log_file, std::ios::app);
    }
    std::cout << "\033[36m";
    while (std::getline(ss, line, '\n')) {
        std::cout << GGS_SEND_HEADER << line << std::endl;
        if (options->ggs_log_to_file) {
            ofs << GGS_SEND_HEADER << line << std::endl;
        }
    }
    std::cout << "\033[0m";
    if (options->ggs_log_to_file) {
        ofs.close();
    }
}

void ggs_print_receive(std::string str, Options *options) { // green
    std::stringstream ss(str);
    std::string line;
    std::ofstream ofs;
    if (options->ggs_log_to_file) {
        ofs.open(options->ggs_log_file, std::ios::app);
    }
    std::cout << "\033[32m";
    while (std::getline(ss, line, '\n')) {
        std::cout << GGS_REPLY_HEADER << line << std::endl;
        if (options->ggs_log_to_file) {
            ofs << GGS_REPLY_HEADER << line;
        }
    }
    std::cout << "\033[0m";
    if (options->ggs_log_to_file) {
        ofs.close();
    }
}

void ggs_print_info(std::string str, Options *options) { // yellow
    std::stringstream ss(str);
    std::string line;
    std::ofstream ofs;
    if (options->ggs_log_to_file) {
        ofs.open(options->ggs_log_file, std::ios::app);
    }
    std::cout << "\033[33m";
    while (std::getline(ss, line, '\n')) {
        std::cout << GGS_INFO_HEADER << line << std::endl;
        if (options->ggs_log_to_file) {
            ofs << GGS_INFO_HEADER << line << std::endl;
        }
    }
    std::cout << "\033[0m";
    if (options->ggs_log_to_file) {
        ofs.close();
    }
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

int ggs_send_message(SOCKET &sock, std::string msg, Options *options) {
    if (send(sock, msg.c_str(), msg.length(), 0) < 0) {
        return 1;
    }
    ggs_print_send(msg, options);
    return 0;
}

std::vector<std::string> ggs_receive_message(SOCKET *sock, Options *options) {
    char server_reply[20000];
    int recv_size;
    std::vector<std::string> res;
    if ((recv_size = recv(*sock, server_reply, 20000, 0)) == SOCKET_ERROR) {
        std::cerr << "Recv failed. Error Code: " << WSAGetLastError() << std::endl;
    } else {
        server_reply[recv_size] = '\0';
        res = split_by_delimiter(server_reply, GGS_READY);
        ggs_print_receive(server_reply, options);
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

bool ggs_is_match_start(std::string line, std::string username) {
    std::vector<std::string> words = split_by_space(line);
    if (std::find(words.begin(), words.end(), username) != words.end()) {
        if (words.size() >= 3) {
            return words[1] == "+" && words[2] == "match";
        }
    }
    return false;
}

bool ggs_is_match_end(std::string line, std::string username) {
    std::vector<std::string> words = split_by_space(line);
    if (std::find(words.begin(), words.end(), username) != words.end()) {
        if (words.size() >= 3) {
            return words[1] == "-" && words[2] == "match";
        }
    }
    return false;
}

bool ggs_is_board_info(std::string line) {
    std::vector<std::string> words = split_by_space(line);
    if (words.size() >= 2) {
        return words[1] == "update" || words[1] == "join";
    }
    return false;
}

bool ggs_is_match_request(std::string line, std::string username) {
    std::vector<std::string> words = split_by_space(line);
    if (words.size() >= 10) {
        return words[1] == "+" && (words[7] == "R" || words[7] == "S") && (words[4] == username || words[9] == username);
        //return words[1] == "+" && words[7] == "R" && (words[4] == username || words[9] == username);
    }
    return false;
}

std::string ggs_match_request_get_id(std::string line) {
    std::vector<std::string> words = split_by_space(line);
    if (words.size() >= 3) {
        return words[2];
    }
    return "";
}

bool ggs_is_join_tournament_message(std::string line) {
    // something like "nyanyan: tell /td join .1" or "nyanyan: t /td join .1"
    std::vector<std::string> words = split_by_space(line);
    if (words.size() >= 5) {
        return (words[1] == "tell" || words[1] == "t") && words[2] == "/td" && words[3] == "join";
    }
    return false;
}

std::string ggs_join_tournament_get_cmd(std::string line) {
    // something like "nyanyan: tell /td join .1" or "nyanyan: t /td join .1"
    std::vector<std::string> words = split_by_space(line);
    std::string res = "";
    for (int i = 1; i < 5; ++i) {
        res += words[i];
        res += " ";
    }
    return res;
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
    std::string os_info = ggs_get_os_info(str);
    std::vector<std::string> os_info_words = split_by_space(os_info);
    if (os_info_words.size() < 3) {
        std::cerr << "ggs_get_board failed: id invalid" << std::endl;
        return res;
    }
    bool is_join = os_info_words[1] == "join"; // /os: join .4.1 s8r18 K?
    res.game_id = os_info_words[2]; // /os: update .4.1 s8r18 K?
    int game_id_dot_count = 0;
    for (int i = 0; (i = res.game_id.find('.', i)) != std::string::npos; i++) {
        game_id_dot_count++;
    }
    res.is_synchro = game_id_dot_count == 2;
    if (res.is_synchro) {
        std::vector<std::string> ids = split_by_delimiter(res.game_id, ".");
        try {
            res.synchro_id = std::stoi(ids[ids.size() - 1]);
        } catch (const std::invalid_argument& e) {
            std::cerr << "ggs_get_board failed: synchro_id invalid" << std::endl;
            res.synchro_id = -1;
        } catch (const std::out_of_range& e) {
            std::cerr << "ggs_get_board failed: synchro_id out of range" << std::endl;
            res.synchro_id = -1;
        }
    }
    std::string board_str;
    std::stringstream ss(str);
    std::string line;
    int n_board_identifier_found = 0;
    int n_board_identifier_used = 1;
    while (std::getline(ss, line, '\n')) {
        std::vector<std::string> words = split_by_space(line);
        if (line[0] == '|') {
            if (line.find(" move(s)") != std::string::npos) {
                if (line.substr(0, 10) != "|0 move(s)") { // happens in stored game
                    //std::cout << "stored game" << std::endl;
                    std::string line2;
                    while (line2.substr(0, 10) != "|* to move" && line2.substr(0, 10) != "|O to move") {
                        std::getline(ss, line2, '\n'); // skip starting board
                    }
                }
                continue;
            }
            // board
            if (line.find("A B C D E F G H") != std::string::npos) {
                ++n_board_identifier_found;
                continue;
            }
            if (n_board_identifier_found == 1) { // board info
                std::string board_str_part;
                for (char c : line) {
                    if (c == '-' || c == '*' || c == 'O') {
                        board_str_part += c;
                    }
                }
                board_str += remove_spaces(board_str_part);
                continue;
            }

            // which to move
            if (line.substr(0, 10) == "|* to move") {
                res.player_to_move = BLACK;
                continue;
            } else if (line.substr(0, 10) == "|O to move") {
                res.player_to_move = WHITE;
                continue;
            }

            // last move
            if (!is_join) {
                if (line.substr(0, 2) == "| ") {
                    if (words.size() >= 3) {
                        if (words[1][words[1].size() - 1] == ':' && words[2].size() >= 2) {
                            res.last_move = get_coord_from_chars(words[2][0], words[2][1]);
                            continue;
                        }
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

    // std::cerr << "game_id " << res.game_id << std::endl;
    // std::cerr << "is_synchro " << res.is_synchro << std::endl;
    // std::cerr << "synchro_id " << res.synchro_id << std::endl;
    // std::cerr << "black " << res.player_black << " " << res.remaining_seconds_black << std::endl;
    // std::cerr << "white " << res.player_white << " " << res.remaining_seconds_white << std::endl;
    // std::cerr << res.player_to_move << " to move" << std::endl;
    // std::cerr << board_str << std::endl;
    // res.board.print();

    return res;
}

Search_result ggs_search(GGS_Board ggs_board, Options *options, thread_id_t thread_id, bool *searching) {
    Search_result search_result;
    if (ggs_board.board.get_legal()) {

        uint64_t remaining_time_msec = 0;
        if (ggs_board.player_to_move == BLACK) {
            remaining_time_msec = ggs_board.remaining_seconds_black * 1000;
        } else {
            remaining_time_msec = ggs_board.remaining_seconds_white * 1000;
        }
        if (remaining_time_msec > 5000) {
            remaining_time_msec -= 5000;
        } else {
            remaining_time_msec = std::max<uint64_t>(remaining_time_msec * 0.1, 1ULL);
        }

        // // special code for s8r14 5 min
        uint64_t strt = tim();
        // if (ggs_board.board.n_discs() == 14 && remaining_time_msec > 30000) {
        //     std::cerr << "s8r14 first move special selfplay" << std::endl;
        //     bool new_searching = true;
        //     std::future<std::vector<Ponder_elem>> ponder_future = std::async(std::launch::async, ai_ponder, ggs_board.board, true, thread_id, &new_searching);
        //     if (ponder_future.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
        //         new_searching = false;
        //     }
        //     ponder_future.get();
        //     /*
        //     std::vector<Ponder_elem> move_list = ai_get_values(ggs_board.board, true, 4000, thread_id);
        //     double best_value = move_list[0].value;
        //     int n_good_moves = 0;
        //     for (const Ponder_elem &elem: move_list) {
        //         if (elem.value >= best_value - AI_TL_ADDITIONAL_SEARCH_THRESHOLD * 3.0) {
        //             ++n_good_moves;
        //         }
        //     }
        //     if (n_good_moves >= 2) {
        //         ai_additional_selfplay(ggs_board.board, true, move_list, n_good_moves, AI_TL_ADDITIONAL_SEARCH_THRESHOLD * 3.0, 10000, thread_id);
        //     }
        //     */
        //     std::cerr << std::endl;
        // }
        remaining_time_msec -= tim() - strt;

        search_result = ai_time_limit(ggs_board.board, true, 0, true, options->show_log, remaining_time_msec, thread_id, searching);
    } else { // pass
        search_result.policy = MOVE_PASS;
    }
    return search_result;
}

void ggs_send_move(GGS_Board &ggs_board, SOCKET &sock, Search_result search_result, Options *options) {
    std::string ggs_move_cmd;
    if (search_result.policy == MOVE_PASS) {
        ggs_move_cmd = "t /os play " + ggs_board.game_id + " pa";
    } else {
        ggs_move_cmd = "t /os play " + ggs_board.game_id + " " + idx_to_coord(search_result.policy) + "/" + std::to_string(search_result.value);
    }
    ggs_send_message(sock, ggs_move_cmd + "\n", options);
}

void ggs_start_ponder(std::future<std::vector<Ponder_elem>> ponder_futures[][GGS_N_PONDER_PARALLEL], Board board, bool show_log, int synchro_id, bool ponder_searchings[]) {
    ponder_searchings[synchro_id] = true;
    for (int ponder_i = 0; ponder_i < GGS_N_PONDER_PARALLEL; ++ponder_i) {
        ponder_futures[synchro_id][ponder_i] = std::async(std::launch::async, ai_ponder, board, show_log, synchro_id, &ponder_searchings[synchro_id]); // set ponder
    }
}

void ggs_terminate_ponder(std::future<std::vector<Ponder_elem>> ponder_futures[][GGS_N_PONDER_PARALLEL], bool ponder_searchings[], int synchro_id) {
    ponder_searchings[synchro_id] = false; // terminate ponder
    std::vector<std::vector<Ponder_elem>> results;
    for (int ponder_i = 0; ponder_i < GGS_N_PONDER_PARALLEL; ++ponder_i) {
        if (ponder_futures[synchro_id][ponder_i].valid()) {
            results.emplace_back(ponder_futures[synchro_id][ponder_i].get());
        }
    }
    for (std::vector<Ponder_elem> result: results) {
        print_ponder_result(result);
    }
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
    ggs_print_info("Connected to server!", options);
    ggs_receive_message(&sock, options);

    // login
    ggs_send_message(sock, options->ggs_username + "\n", options);
    ggs_receive_message(&sock, options);
    ggs_send_message(sock, options->ggs_password + "\n", options);
    ggs_receive_message(&sock, options);

    // initialize
    ggs_send_message(sock, "ms /os\n", options);
    ggs_receive_message(&sock, options);
    ggs_send_message(sock, "ts client -\n", options);
    ggs_receive_message(&sock, options);
    
    std::future<std::string> user_input_f;
    std::future<std::vector<std::string>> ggs_message_f;
    std::future<Search_result> ai_futures[2];
    bool ai_searchings[2] = {false, false};
    GGS_Board ggs_boards_searching[2];
    std::future<std::vector<Ponder_elem>> ponder_futures[2][GGS_N_PONDER_PARALLEL];
    bool ponder_searchings[2] = {false, false};
    GGS_Board ggs_boards[2][HW2 + 1];
    int ggs_boards_n_discs[2] = {0, 0};
    bool match_playing = false;
    int thread_sizes[2];
    int thread_sizes_before[2];
    for (int i = 0; i < 2; ++i) {
        thread_sizes[i] = 0;
    }
    bool playing_synchro_game = false;
    bool playing_same_board = true;
    GGS_Match matches[2];
    for (int i = 0; i < 2; ++i) {
        matches[i].init();
    }
    uint64_t last_sent_time = tim();
    while (true) {
        if (tim() - last_sent_time > GGS_SEND_EMPTY_INTERVAL) {
            ggs_send_message(sock, "\n", options);
            last_sent_time = tim();
        }
        // check user input
        if (user_input_f.valid()) {
            if (user_input_f.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                std::string user_input = user_input_f.get();
                ggs_send_message(sock, user_input + "\n", options);
                last_sent_time = tim();
                if (user_input == "quit") {
                    global_searching = false;
                    for (int ai_i = 0; ai_i < 2; ++ai_i) {
                        if (ai_futures[ai_i].valid()) {
                            ai_futures[ai_i].get();
                        }
                        for (int ponder_i = 0; ponder_i < GGS_N_PONDER_PARALLEL; ++ponder_i) {
                            if (ponder_futures[ai_i][ponder_i].valid()) {
                                ponder_futures[ai_i][ponder_i].get();
                            }
                        }
                    }
                    break;
                }
            }
        } else {
            user_input_f = std::async(std::launch::async, ggs_get_user_input);
        }
        // check search result
        if (match_playing) {
            // check ai search & send move
            for (int i = 0; i < 2; ++i) {
                if (ai_searchings[i]) {
                    if (ai_futures[i].valid()) {
                        if (ai_futures[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                            Search_result search_result = ai_futures[i].get();
                            ai_searchings[i] = false;
                            ggs_send_move(ggs_boards_searching[i], sock, search_result, options);
                            last_sent_time = tim();
                        }
                    }
                }
            }
            // check ponder ends
            for (int i = 0; i < 2; ++i) {
                if (ponder_searchings[i]) {
                    for (int ponder_i = 0; ponder_i < GGS_N_PONDER_PARALLEL; ++ponder_i) {
                        if (ponder_futures[i][ponder_i].valid()) {
                            if (ponder_futures[i][ponder_i].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                                ponder_futures[i][ponder_i].get();
                                ponder_searchings[i] = false;
                                ggs_print_info("ponder end " + std::to_string(i), options);
                            }
                        }
                    }
                }
            }
        }
        // check server reply
        std::vector<std::string> server_replies;
        if (ggs_message_f.valid()) {
            if (ggs_message_f.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                server_replies = ggs_message_f.get();
            }
        } else {
            ggs_message_f = std::async(std::launch::async, ggs_receive_message, &sock, options); // ask ggs message
        }
        bool new_calculation_start = false;
        if (server_replies.size()) {
            // match start
            for (std::string server_reply: server_replies) {
                if (server_reply.size()) {
                    std::string os_info = ggs_get_os_info(server_reply);
                    // match start
                    if (ggs_is_match_start(os_info, options->ggs_username)) {
                        ggs_print_info("match start!", options);
                        match_playing = true;
                        playing_same_board = true;
                        for (int i = 0; i < 2; ++i) {
                            ai_searchings[i] = false;
                            ponder_searchings[i] = false;
                        }
                        for (int i = 0; i < 2; ++i) {
                            matches[i].init();
                        }
                    }
                }
            }
            // set board info & update match
            for (std::string server_reply: server_replies) {
                if (server_reply.size()) {
                    std::string os_info = ggs_get_os_info(server_reply);
                    if (ggs_is_board_info(os_info)) {
                        GGS_Board ggs_board = ggs_get_board(server_reply);
                        if (ggs_board.is_valid()) {
                            if (ggs_board.player_black == options->ggs_username || ggs_board.player_white == options->ggs_username) { // related to me
                                ggs_print_info("ggs board synchro id " + std::to_string(ggs_board.synchro_id), options);
                                // set board info
                                if (ggs_board.is_synchro) { // synchro game
                                    int n_discs = ggs_board.board.n_discs();
                                    ggs_boards[ggs_board.synchro_id][n_discs] = ggs_board;
                                    ggs_boards_n_discs[ggs_board.synchro_id] = n_discs;
                                    //std::cerr << "synchro game memo " << "n_discs " << ggs_board.board.n_discs() << " " << ggs_board.board.to_str() << std::endl;
                                }
                                // update match
                                int match_idx = GGS_NON_SYNCHRO_ID;
                                if (ggs_board.is_synchro) {
                                    match_idx = ggs_board.synchro_id;
                                }
                                if (matches[match_idx].is_initialized()) {
                                    matches[match_idx].game_id = ggs_board.game_id;
                                    matches[match_idx].player_black = ggs_board.player_black;
                                    matches[match_idx].player_white = ggs_board.player_white;
                                    matches[match_idx].initial_board = ggs_board.board.to_str(ggs_board.player_to_move);
                                    matches[match_idx].result_black = -99;
                                    matches[match_idx].transcript = "";
                                }
                                ggs_print_info("match log received " + std::to_string(match_idx) + " " + std::to_string(ggs_board.last_move) + " " + idx_to_coord(ggs_board.last_move), options);
                                if (is_valid_policy(ggs_board.last_move)) {
                                    matches[match_idx].transcript += idx_to_coord(ggs_board.last_move);
                                    if (
                                        (ggs_board.player_to_move == BLACK && ggs_board.player_black == options->ggs_username) || 
                                        (ggs_board.player_to_move == WHITE && ggs_board.player_white == options->ggs_username)
                                    ) {
                                        std::cerr << "opponent moved " << idx_to_coord(ggs_board.last_move) << std::endl;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // match end
            for (std::string server_reply: server_replies) {
                if (server_reply.size()) {
                    std::string os_info = ggs_get_os_info(server_reply);
                    // match end
                    if (ggs_is_match_end(os_info, options->ggs_username)) {
                        ggs_print_info("match end!", options);
                        match_playing = false;
                        for (int i = 0; i < 2; ++i) {
                            ai_searchings[i] = false;
                            ponder_searchings[i] = false;
                        }
                        if (options->ggs_game_log_to_file) {
                            for (int i = 0; i < 2; ++i) {
                                if (!matches[i].is_initialized()) {
                                    std::string datetime = get_current_datetime_for_file();
                                    std::string filename = options->ggs_game_log_dir + "/" + datetime + "_" + matches[i].game_id + ".txt";
                                    std::ofstream ofs(filename, std::ios::app);
                                    if (!ofs) {
                                        ggs_print_info("Can't open gamelog file " + filename, options);
                                    } else {
                                        int black_score = 0;
                                        Board board(matches[i].initial_board);
                                        int player_sgn = matches[i].initial_board[matches[i].initial_board.size() - 1] == 'X' ? 1 : -1;
                                        Flip flip;
                                        for (int j = 0; j < matches[i].transcript.size(); j += 2) {
                                            if (board.get_legal() == 0) {
                                                board.pass();
                                                player_sgn *= -1;
                                            }
                                            int coord = get_coord_from_chars(matches[i].transcript[j], matches[i].transcript[j + 1]);
                                            calc_flip(&flip, &board, coord);
                                            board.move_board(&flip);
                                            player_sgn *= -1;
                                        }
                                        if (board.is_end()) {
                                            matches[i].result_black = player_sgn * board.score_player();
                                        } else {
                                            matches[i].result_black = -99;
                                        }
                                        ofs << matches[i].game_id << std::endl;
                                        ofs << "black: " << matches[i].player_black << std::endl;
                                        ofs << "white: " << matches[i].player_white << std::endl;
                                        ofs << "initial board: " << matches[i].initial_board << std::endl;
                                        ofs << "transcript: " << matches[i].transcript << std::endl;
                                        ofs << "black's score: " << matches[i].result_black << std::endl;
                                        ofs.close();
                                    }
                                }
                            }
                        }
                        transposition_table.init();
                        ggs_print_info("clearned TT up", options);
                    }
                }
            }
            // receive match request
            if (options->ggs_accept_request) {
                if (!match_playing) {
                    for (std::string server_reply: server_replies) {
                        if (server_reply.size()) {
                            std::string os_info = ggs_get_os_info(server_reply);
                            // match end
                            if (ggs_is_match_request(os_info, options->ggs_username)) {
                                std::string request_id = ggs_match_request_get_id(os_info);
                                std::string accept_cmd = "ts accept " + request_id;
                                ggs_send_message(sock, accept_cmd + "\n", options);
                                last_sent_time = tim();
                                ggs_print_info("match request accepted " + request_id, options);
                            }
                        }
                    }
                }
            }
            // join tournament
            if (options->ggs_route_join_tournament) {
                for (std::string server_reply: server_replies) {
                    if (server_reply.size()) {
                        if (ggs_is_join_tournament_message(server_reply)) {
                            std::string join_tournament_cmd = ggs_join_tournament_get_cmd(server_reply);
                            ggs_send_message(sock, join_tournament_cmd + "\n", options);
                            last_sent_time = tim();
                            ggs_print_info("join tournament " + join_tournament_cmd, options);
                        }
                    }
                }
            }
            // board processing
            for (std::string server_reply: server_replies) {
                if (server_reply.size()) {
                    //std::cout << "see " << server_reply << std::endl;
                    std::string os_info = ggs_get_os_info(server_reply);
                    // processing board
                    if (ggs_is_board_info(os_info)) {
                        //std::cout << "getting board info" << std::endl;
                        GGS_Board ggs_board = ggs_get_board(server_reply);
                        if (ggs_board.is_valid()) {
                            if (ggs_board.player_black == options->ggs_username || ggs_board.player_white == options->ggs_username) { // related to me
                                bool need_to_move = 
                                    (ggs_board.player_black == options->ggs_username && ggs_board.player_to_move == BLACK) || 
                                    (ggs_board.player_white == options->ggs_username && ggs_board.player_to_move == WHITE);
                                if (ggs_board.is_synchro) { // synchro game
                                    playing_synchro_game = true;
                                    int n_discs = ggs_board.board.n_discs();
                                    if (playing_same_board && (ggs_boards[0][n_discs].board == ggs_boards[1][n_discs].board || ggs_boards_n_discs[ggs_board.synchro_id] != ggs_boards_n_discs[ggs_board.synchro_id ^ 1])) {
                                        // std::string msg = "synchro playing same board or opponent has not played " + ggs_board.board.to_str();
                                        // ggs_print_info(msg);
                                        // playing_same_board = true;
                                    } else {
                                        // std::string msg = "synchro game separated " + ggs_board.board.to_str();
                                        // ggs_print_info(msg);
                                        playing_same_board = false;
                                    }
                                    if (need_to_move) { // Egaroucid should move
                                        ggs_terminate_ponder(ponder_futures, ponder_searchings, ggs_board.synchro_id);
                                        if (!ggs_board.board.is_end()) {
                                            ai_searchings[ggs_board.synchro_id] = true;
                                            ggs_boards_searching[ggs_board.synchro_id] = ggs_board;
#if GGS_USE_PONDER
                                            ai_futures[ggs_board.synchro_id] = std::async(std::launch::async, ggs_search, ggs_board, options, ggs_board.synchro_id, &ai_searchings[ggs_board.synchro_id]); // set search
#else
                                            ai_futures[ggs_board.synchro_id] = std::async(std::launch::async, ggs_search, ggs_board, options, THREAD_ID_NONE, &ai_searchings[ggs_board.synchro_id]); // set search
#endif
                                            // ggs_start_ponder(ponder_futures, ggs_board.board, options->show_log, ggs_board.synchro_id, ponder_searchings);
                                            new_calculation_start = true;
                                            std::string msg = "Egaroucid thinking... " + ggs_board.game_id + " " + ggs_board.board.to_str(ggs_board.player_to_move);
                                            ggs_print_info(msg, options);
                                        }
                                    } else { // Opponent's move
#if GGS_USE_PONDER
                                        if (!ggs_board.board.is_end()) {
                                            ggs_start_ponder(ponder_futures, ggs_board.board, options->show_log, ggs_board.synchro_id, ponder_searchings);
                                            // ponder_futures[ggs_board.synchro_id] = std::async(std::launch::async, ai_ponder, ggs_board.board, options->show_log, ggs_board.synchro_id, &ponder_searchings[ggs_board.synchro_id]); // set ponder
                                            new_calculation_start = true;
                                            std::string msg = "Egaroucid pondering... " + ggs_board.game_id + " " + ggs_board.board.to_str(ggs_board.player_to_move);
                                            ggs_print_info(msg, options);
                                        }
#endif
                                    }
                                } else { // non-synchro game
                                    playing_synchro_game = false;
                                    if (need_to_move) { // Egaroucid should move
                                        ggs_terminate_ponder(ponder_futures, ponder_searchings, GGS_NON_SYNCHRO_ID);
                                        if (!ggs_board.board.is_end()) {
                                            ai_searchings[GGS_NON_SYNCHRO_ID] = true;
                                            ggs_boards_searching[GGS_NON_SYNCHRO_ID] = ggs_board;
                                            ai_futures[GGS_NON_SYNCHRO_ID] = std::async(std::launch::async, ggs_search, ggs_board, options, THREAD_ID_NONE, &ai_searchings[GGS_NON_SYNCHRO_ID]); // set search
                                            // ggs_start_ponder(ponder_futures, ggs_board.board, options->show_log, GGS_NON_SYNCHRO_ID, ponder_searchings);
                                        }
                                    } else { // Opponent's move
#if GGS_USE_PONDER
                                        if (!ggs_board.board.is_end()) {
                                            ggs_start_ponder(ponder_futures, ggs_board.board, options->show_log, ggs_board.synchro_id, ponder_searchings);
                                            // ponder_futures[GGS_NON_SYNCHRO_ID] = std::async(std::launch::async, ai_ponder, ggs_board.board, options->show_log, THREAD_ID_NONE, &ponder_searchings[GGS_NON_SYNCHRO_ID]); // set ponder
                                        }
#endif
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
#if GGS_USE_PONDER
        // thread manager
        thread_sizes_before[0] = thread_sizes[0];
        thread_sizes_before[1] = thread_sizes[1];
        if (playing_synchro_game) {
            int full_threads = thread_pool.size();
            int full_threads_enhanced = std::max((int)(full_threads * 1.2), full_threads + 3);
            int reduced_threads = full_threads_enhanced / 2;
            int prioritized_threads = full_threads_enhanced * 0.75;
            int non_prioritized_threads = full_threads_enhanced - prioritized_threads;
            prioritized_threads = std::min(prioritized_threads, full_threads);
            // if (playing_same_board) {
            //     if (ai_searchings[0]) { // 0 is searching
            //         if (ai_searchings[1]) { // 1 is searching
            //             // not occurs
            //         } else if (ponder_searchings[1]) { // 1 is pondering
            //             thread_sizes[0] = full_threads; // full threads for 0
            //             thread_sizes[1] = 0; // off 1's ponder
            //         } else { // 1 is waiting
            //             thread_sizes[0] = full_threads; // full threads for 0
            //             thread_sizes[1] = 0; // off 1
            //         }
            //     } else if (ponder_searchings[0]) { // 0 is pondering
            //         if (ai_searchings[1]) { // 1 is searching
            //             thread_sizes[0] = 0; // off 0's ponder
            //             thread_sizes[1] = full_threads; // full threads for 1
            //         } else if (ponder_searchings[1]) { // 1 is pondering
            //             thread_sizes[0] = reduced_threads; // half & half
            //             thread_sizes[1] = reduced_threads; // half & half
            //         } else { // 1 is waiting
            //             thread_sizes[0] = full_threads; // full threads for 0
            //             thread_sizes[1] = 0; // off 1
            //         }
            //     } else { // 0 is waiting
            //         thread_sizes[0] = 0; // off 0
            //         thread_sizes[1] = full_threads; // full threads for 1
            //     }
            // } else {
                if (ai_searchings[0]) { // 0 is searching
                    if (ai_searchings[1]) { // 1 is searching
                        // not occurs
                    } else if (ponder_searchings[1]) { // 1 is pondering
                        thread_sizes[0] = prioritized_threads;
                        thread_sizes[1] = non_prioritized_threads;
                    } else { // 1 is waiting
                        thread_sizes[0] = full_threads;
                        thread_sizes[1] = 0;
                    }
                } else if (ponder_searchings[0]) { // 0 is pondering
                    if (ai_searchings[1]) { // 1 is searching
                        thread_sizes[0] = non_prioritized_threads;
                        thread_sizes[1] = prioritized_threads;
                    } else if (ponder_searchings[1]) { // 1 is pondering
                        thread_sizes[0] = reduced_threads;
                        thread_sizes[1] = reduced_threads;
                    } else { // 1 is waiting
                        thread_sizes[0] = full_threads;
                        thread_sizes[1] = 0;
                    }
                } else { // 0 is waiting
                    thread_sizes[0] = 0; // off 0
                    thread_sizes[1] = full_threads; // full threads for 1
                }
            // }
        } else {
            thread_sizes[GGS_NON_SYNCHRO_ID] = thread_pool.size(); // full threads for non-synchro game
        }
        // update thread size
        if (thread_sizes[0] != thread_sizes_before[0]) {
            thread_pool.set_max_thread_size(0, thread_sizes[0]);
        }
        if (thread_sizes[1] != thread_sizes_before[1]) {
            thread_pool.set_max_thread_size(1, thread_sizes[1]);
        }
        if (new_calculation_start || thread_sizes[0] != thread_sizes_before[0] || thread_sizes[1] != thread_sizes_before[1]) {
            std::string msg = "thread info synchro " + std::to_string(playing_synchro_game) + " same " + std::to_string(playing_same_board) + " ai " + std::to_string(ai_searchings[0]) + " " + std::to_string(ai_searchings[1]) + " ponder " + std::to_string(ponder_searchings[0]) + " " + std::to_string(ponder_searchings[1]) + " thread size " + std::to_string(thread_sizes[0]) + " " + std::to_string(thread_sizes[1]);
            ggs_print_info(msg, options);
        }
#endif
    }

    // close connection
    ggs_close(sock);
}