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

#define GGS_NON_SYNCHRO_ID 0

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
    char server_reply[20000];
    int recv_size;
    std::vector<std::string> res;
    if ((recv_size = recv(*sock, server_reply, 20000, 0)) == SOCKET_ERROR) {
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

bool ggs_is_match_start(std::string line) {
    std::vector<std::string> words = split_by_space(line);
    if (words.size() >= 3) {
        return words[1] == "+" && words[2] == "match";
    }
    return false;
}

bool ggs_is_match_end(std::string line) {
    std::vector<std::string> words = split_by_space(line);
    if (words.size() >= 3) {
        return words[1] == "-" && words[2] == "match";
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
            if (line.substr(0, 10) == "|* to move") {
                res.player_to_move = BLACK;
                continue;
            } else if (line.substr(0, 10) == "|O to move") {
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
        if (remaining_time_msec > 6000) {
            remaining_time_msec -= 5000;
        }
        std::cerr << "Egaroucid thinking... remaining " << remaining_time_msec << " ms" << std::endl;
        search_result = ai_time_limit(ggs_board.board, true, 0, true, options->show_log, remaining_time_msec, thread_id, searching);
    } else { // pass
        search_result.policy = MOVE_PASS;
    }
    return search_result;
}

void ggs_send_move(GGS_Board &ggs_board, SOCKET &sock, Search_result search_result) {
    std::string ggs_move_cmd;
    if (search_result.policy == MOVE_PASS) {
        ggs_move_cmd = "t /os play " + ggs_board.game_id + " pa";
    } else {
        ggs_move_cmd = "t /os play " + ggs_board.game_id + " " + idx_to_coord(search_result.policy) + "/" + std::to_string(search_result.value);
    }
    ggs_send_message(sock, ggs_move_cmd + "\n");
}

void ggs_terminate_ponder(std::future<std::vector<Ponder_elem>> ponder_futures[], bool ponder_searchings[], int synchro_id) {
    ponder_searchings[synchro_id] = false; // terminate ponder
    if (ponder_futures[synchro_id].valid()) {
        ponder_futures[synchro_id].get();
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
    std::future<Search_result> ai_futures[2];
    bool ai_searchings[2] = {false, false};
    GGS_Board ggs_boards_searching[2];
    std::future<std::vector<Ponder_elem>> ponder_futures[2];
    bool ponder_searchings[2] = {false, false};
    GGS_Board ggs_boards[2][HW2];
    int ggs_boards_n_discs[2] = {0, 0};
    bool match_playing = false;
    int thread_sizes[2];
    int thread_sizes_before[2];
    for (int i = 0; i < 2; ++i) {
        thread_sizes[i] = 0;
    }
    bool playing_synchro_game = false;
    bool playing_same_board = true;
    while (true) {
        std::cerr << "ai " << ai_searchings[0] << " " << ai_searchings[1] << "  ponder " << ponder_searchings[0] << " " << ponder_searchings[1] << "  thread size " << thread_sizes[0] << " " << thread_sizes[1] << "  n_using " << thread_pool.get_n_using_thread(0) << " " << thread_pool.get_n_using_thread(1) << std::endl;
        // check user input
        if (user_input_f.valid()) {
            if (user_input_f.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                std::string user_input = user_input_f.get();
                if (user_input == "exit" || user_input == "quit") {
                    break;
                }
                ggs_send_message(sock, user_input + "\n");
            }
        } else {
            user_input_f = std::async(std::launch::async, ggs_get_user_input);
        }
        // check ai search & send move
        if (match_playing) {
            for (int i = 0; i < 2; ++i) {
                if (ai_searchings[i]) {
                    if (ai_futures[i].valid()) {
                        if (ai_futures[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                            Search_result search_result = ai_futures[i].get();
                            ai_searchings[i] = false;
                            ggs_send_move(ggs_boards_searching[i], sock, search_result);
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
            ggs_message_f = std::async(std::launch::async, ggs_receive_message, &sock); // ask ggs message
        }
        if (server_replies.size()) {
            // set board info
            for (std::string server_reply: server_replies) {
                if (server_reply.size()) {
                    std::string os_info = ggs_get_os_info(server_reply);
                    if (ggs_is_board_info(os_info)) {
                        GGS_Board ggs_board = ggs_get_board(server_reply);
                        if (ggs_board.is_valid()) {
                            if (ggs_board.player_black == options->ggs_username || ggs_board.player_white == options->ggs_username) { // related to me
                                if (ggs_board.is_synchro) { // synchro game
                                    int n_discs = ggs_board.board.n_discs();
                                    ggs_boards[ggs_board.synchro_id][n_discs] = ggs_board;
                                    ggs_boards_n_discs[ggs_board.synchro_id] = n_discs;
                                    std::cerr << "synchro game memo " << "n_discs " << ggs_board.board.n_discs() << " " << ggs_board.board.to_str() << std::endl;
                                }
                            }
                        }
                    }
                }
            }
            // processing
            for (std::string server_reply: server_replies) {
                if (server_reply.size()) {
                    //std::cout << "see " << server_reply << std::endl;
                    std::string os_info = ggs_get_os_info(server_reply);
                    if (os_info.size()) {
                        std::cout << "os_info " << os_info << std::endl;
                    }
                    // match start
                    if (ggs_is_match_start(os_info)) {
                        std::cerr << "match start!" << std::endl;
                        match_playing = true;
                        for (int i = 0; i < 2; ++i) {
                            ai_searchings[i] = false;
                            ponder_searchings[i] = false;
                        }
                    }
                    // match end
                    if (ggs_is_match_end(os_info)) {
                        std::cerr << "match end!" << std::endl;
                        match_playing = false;
                        for (int i = 0; i < 2; ++i) {
                            ai_searchings[i] = false;
                            ponder_searchings[i] = false;
                        }
                        transposition_table.init();
                    }
                    // processing board
                    if (ggs_is_board_info(os_info)) {
                        std::cout << "getting board info" << std::endl;
                        GGS_Board ggs_board = ggs_get_board(server_reply);
                        if (ggs_board.is_valid()) {
                            if (ggs_board.player_black == options->ggs_username || ggs_board.player_white == options->ggs_username) { // related to me
                                bool need_to_move = 
                                    (ggs_board.player_black == options->ggs_username && ggs_board.player_to_move == BLACK) || 
                                    (ggs_board.player_white == options->ggs_username && ggs_board.player_to_move == WHITE);
                                if (ggs_board.is_synchro) { // synchro game
                                    playing_synchro_game = true;
                                    int n_discs = ggs_board.board.n_discs();
                                    if (ggs_boards[0][n_discs].board == ggs_boards[1][n_discs].board || ggs_boards_n_discs[ggs_board.synchro_id] > ggs_boards_n_discs[ggs_board.synchro_id ^ 1]) {
                                        std::cerr << "synchro playing same board or opponent has not played " << ggs_board.board.to_str() << std::endl;
                                        playing_same_board = true;
                                    } else {
                                        std::cerr << "synchro game separated " << ggs_board.board.to_str() << std::endl;
                                        playing_same_board = false;
                                    }
                                    if (need_to_move) { // Egaroucid should move
                                        ggs_terminate_ponder(ponder_futures, ponder_searchings, ggs_board.synchro_id);
                                        ai_searchings[ggs_board.synchro_id] = true;
                                        ggs_boards_searching[ggs_board.synchro_id] = ggs_board;
                                        ai_futures[ggs_board.synchro_id] = std::async(std::launch::async, ggs_search, ggs_board, options, ggs_board.synchro_id, &ai_searchings[ggs_board.synchro_id]); // set search
                                    } else { // Opponent's move
                                        ponder_searchings[ggs_board.synchro_id] = true;
                                        ponder_futures[ggs_board.synchro_id] = std::async(std::launch::async, ai_ponder, ggs_board.board, options->show_log, ggs_board.synchro_id, &ponder_searchings[ggs_board.synchro_id]); // set ponder
                                    }
                                } else { // non-synchro game
                                    playing_synchro_game = false;
                                    if (need_to_move) { // Egaroucid should move
                                        ggs_terminate_ponder(ponder_futures, ponder_searchings, GGS_NON_SYNCHRO_ID);
                                        ai_searchings[GGS_NON_SYNCHRO_ID] = true;
                                        ggs_boards_searching[GGS_NON_SYNCHRO_ID] = ggs_board;
                                        ai_futures[GGS_NON_SYNCHRO_ID] = std::async(std::launch::async, ggs_search, ggs_board, options, THREAD_ID_NONE, &ai_searchings[GGS_NON_SYNCHRO_ID]); // set search
                                    } else { // Opponent's move
                                        ponder_searchings[GGS_NON_SYNCHRO_ID] = true;
                                        ponder_futures[GGS_NON_SYNCHRO_ID] = std::async(std::launch::async, ai_ponder, ggs_board.board, options->show_log, THREAD_ID_NONE, &ponder_searchings[GGS_NON_SYNCHRO_ID]); // set ponder
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // thread manager
        thread_sizes_before[0] = thread_sizes[0];
        thread_sizes_before[1] = thread_sizes[1];
        if (playing_synchro_game) {
            int full_threads = thread_pool.size();
            int reduced_threads = thread_pool.size() / 2;
            if (playing_same_board) {
                if (ai_searchings[0]) { // 0 is searching
                    if (ai_searchings[1]) { // 1 is searching
                        // not occurs
                    } else if (ponder_searchings[1]) { // 1 is pondering
                        thread_sizes[0] = full_threads; // full threads for 0
                        thread_sizes[1] = 0; // off 1's ponder
                    } else { // 1 is waiting
                        thread_sizes[0] = full_threads; // full threads for 0
                        thread_sizes[1] = 0; // off 1
                    }
                } else if (ponder_searchings[0]) { // 0 is pondering
                    if (ai_searchings[1]) { // 1 is searching
                        thread_sizes[0] = 0; // off 0's ponder
                        thread_sizes[1] = full_threads; // full threads for 1
                    } else if (ponder_searchings[1]) { // 1 is pondering
                        thread_sizes[0] = reduced_threads; // half & half
                        thread_sizes[1] = reduced_threads; // half & half
                    } else { // 1 is waiting
                        thread_sizes[0] = full_threads; // full threads for 0
                        thread_sizes[1] = 0; // off 1
                    }
                } else { // 0 is waiting
                    thread_sizes[0] = 0; // off 0
                    thread_sizes[1] = full_threads; // full threads for 1
                }
            } else {
                thread_sizes[0] = reduced_threads; // half & half
                thread_sizes[1] = reduced_threads; // half & half
                for (int i = 0; i < 2; ++i) {
                    if (!ai_searchings[i] && !ponder_searchings[i]) { // i is waiting
                        thread_sizes[i ^ 1] = full_threads; // full threads for i ^ 1
                    }
                }
            }
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
    }

    // close connection
    ggs_close(sock);
}
