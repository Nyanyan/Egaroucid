// Offline paired-game replay with the real Lv.10 engine, bot API revisions and auto-passes.
#define main egaroucid_console_main
#include "../../Egaroucid_for_Console.cpp"
#undef main

namespace fs = flipsight;
class TestServer {
    struct Game {
        Board board;
        int turn = 1, color;
        int revision = 0;
        fs::Millis remaining = 60000, started = fs::now_ms();
        bool finished = false;
        fs::Json history = fs::Json::array();
    };
    std::mutex mutex;
    std::array<Game, 2> games;
    int moves = 0, passes = 0;

    void play(Game& game, int policy, bool automatic = false) {
        game.history.push_back({{"color", game.turn}, {"move", policy == 64 ? -1 : 63 - policy},
            {"participant", automatic ? fs::Json() : fs::Json(game.turn == game.color ? 0 : 1)}, {"auto", automatic}});
        if (!automatic) ++game.revision;
        if (policy == 64) { game.board.pass(); ++passes; }
        else { Flip flip; calc_flip(&flip, &game.board, policy); game.board.move_board(&flip); ++moves; }
        game.turn = 3 - game.turn;
        game.started = fs::now_ms();
    }
    void advance(Game& game, bool opponent_ready) {
        while (!game.finished) {
            uint64_t legal = game.board.get_legal();
            if (!legal) {
                Board passed = game.board; passed.pass();
                if (!passed.get_legal()) { game.finished = true; break; }
                play(game, 64, true); continue;
            }
            if (game.turn == game.color) break;
            if (!opponent_ready || fs::now_ms() - game.started < 120) break;
            int policy = 0; while (!(legal & (1ULL << policy))) ++policy;
            play(game, policy);
        }
    }
    fs::Json snapshot_unlocked() {
        fs::Json legs = fs::Json::array();
        bool all_finished = true;
        for (const auto& game : games) {
            all_finished = all_finished && game.finished;
            uint64_t black = game.turn == 1 ? game.board.player : game.board.opponent;
            uint64_t white = game.turn == 1 ? game.board.opponent : game.board.player;
            std::vector<int> board(64, 0), legal;
            for (int i = 0; i < 64; ++i) {
                auto bit = 1ULL << (63 - i);
                if (black & bit) board[i] = 1;
                if (white & bit) board[i] = 2;
                if (game.board.get_legal() & bit) legal.push_back(i);
            }
            std::array<fs::Millis, 2> clocks{60000, 60000}; clocks[game.color - 1] = game.remaining;
            legs.push_back({{"board", board}, {"legal_moves", legal}, {"revision", game.revision},
                {"turn", game.finished ? fs::Json() : fs::Json(game.turn)}, {"status", game.finished ? "finished" : "active"},
                {"history", game.history}, {"clocks_ms", game.finished ? fs::Json() : fs::Json(clocks)},
                {"turn_started_ms", game.started}, {"result_discs", {std::popcount(black), std::popcount(white)}}});
        }
        return {{"id", 123}, {"bot_a", 1}, {"bot_b", 2}, {"status", all_finished ? "finished" : "running"},
            {"state", {{"status", all_finished ? "finished" : "active"}, {"version", 1}, {"revision", moves + passes}, {"legs", legs}, {"game_mode", "std"}}}};
    }
public:
    TestServer() {
        Board opening; opening.reset();
        for (int i = 0; i < 8; ++i) {
            auto legal = opening.get_legal(); int policy = 0;
            while (!(legal & (1ULL << policy))) ++policy;
            Flip flip; calc_flip(&flip, &opening, policy); opening.move_board(&flip);
        }
        for (int i = 0; i < 2; ++i) { games[i].board = opening; games[i].color = i == 0 ? 1 : 2; advance(games[i], false); }
    }
    fs::Json payload() {
        std::lock_guard<std::mutex> lock(mutex);
        return {{"bot", {{"id", 1}}}, {"server_now_ms", fs::now_ms()}, {"matches", fs::Json::array({snapshot_unlocked()})}};
    }
    bool advance_opponents() {
        std::lock_guard<std::mutex> lock(mutex);
        int before = moves + passes;
        for (auto& game : games) advance(game, true);
        return before != moves + passes;
    }
    fs::Reply call(const fs::Json& body, std::atomic_bool&) {
        std::lock_guard<std::mutex> lock(mutex);
        auto started = fs::now_ms();
        auto& game = games.at(body.at("leg").get<int>());
        int policy = body.at("move").get<int>(); policy = policy < 0 ? 64 : 63 - policy;
        if (game.finished || game.turn != game.color || body.at("revision") != game.revision)
            throw std::runtime_error("Replay received a stale move");
        if (policy == 64 || !(game.board.get_legal() & (1ULL << policy))) throw std::runtime_error("Replay received an illegal move");
        game.remaining -= started - game.started;
        if (game.remaining <= 0) throw std::runtime_error("Replay exceeded the actual game clock");
        play(game, policy); advance(game, false);
        return {200, {{"ok", true}, {"server_now_ms", fs::now_ms()}, {"match", snapshot_unlocked()}}, started, fs::now_ms(), {}};
    }
    bool finished() { std::lock_guard<std::mutex> lock(mutex); return games[0].finished && games[1].finished; }
    void report() { std::cout << "Replay explicit moves=" << moves << ", server auto-passes=" << passes << '\n'; }
};

int main(int argc, char** argv) try {
    if (argc != 3) return 1;
    std::string binary_path = std::filesystem::absolute(argv[1]).string() + "/";
    if (!std::filesystem::exists(binary_path + "resources/eval.egev2")) throw std::runtime_error("Replay needs console resources");
    std::vector<std::string> args = {"test", "-flipsight", "offline-test-token", "-fsthread", "2", "-hash", "20", "-level", "10", "-nobook"};
    std::vector<char*> raw; for (auto& arg : args) raw.push_back(arg.data());
    auto options = get_options(get_commandline_options(static_cast<int>(raw.size()), raw.data()), binary_path);
    init_console(options, binary_path);
    TestServer server;
    std::ostringstream output, errors;
    fs::SessionLog session(argv[2], "offline-test-token", output, errors);
    bool logged_finish = false;
    int ponders = 0, stops = 0;
    fs::Controller controller(2, 10, true,
        [&](const fs::Leg& leg, bool ponder, int id, bool* searching, fs::Millis budget) {
            return fs::engine_search(leg, ponder, id, searching, budget, options);
        }, [&](const fs::Json& body, std::atomic_bool& cancel) { return server.call(body, cancel); },
        [](int id, int helpers) { thread_pool.set_max_thread_size(id, helpers); },
        [](int total) { thread_pool.grow(total - 1); }, [] {},
        [&](const std::string& event, fs::Json fields, bool error) {
            if (event == "match_finished") logged_finish = true;
            if (event == "ponder_started") ++ponders;
            if (event == "ponder_stopping") ++stops;
            session.write(event, std::move(fields), error);
        });
    auto began = fs::now_ms();
    controller.event({{"type", "hello"}, {"api", 2}, {"bot_id", 1}, {"server_now_ms", began}}, began, began);
    auto initial = server.payload(); initial["type"] = "state"; initial["view"] = "agent";
    controller.event(initial, fs::now_ms());
    int revision = initial["matches"][0]["state"]["revision"].get<int>();
    while (!logged_finish && fs::now_ms() - began < 45000) {
        server.advance_opponents();
        auto state = server.payload();
        auto next_revision = state["matches"][0]["state"]["revision"].get<int>();
        if (next_revision != revision) {
            revision = next_revision;
            controller.event({{"type", "match"}, {"view", "agent"}, {"match", state["matches"][0]},
                {"server_now_ms", state["server_now_ms"]}}, fs::now_ms());
        }
        controller.tick();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    bool finished = server.finished() && logged_finish;
    controller.stop();
    if (!finished) throw std::runtime_error("Paired replay stalled; inspect its session log");
    if (!ponders || ponders != stops) throw std::runtime_error("Replay must start and stop real pondering");
    server.report();
    std::cout << "Replay ponder starts/stops=" << ponders << '/' << stops << '\n';
    std::cout << "PASS: Lv.10 paired replay completes with auto-passes, finished/null legs, move acknowledgements and session logging\n";
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
