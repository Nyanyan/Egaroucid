// Issue #667: verify the actual console engine adapter without connecting to FlipSight.
#define main egaroucid_console_main
#include "../../Egaroucid_for_Console.cpp"
#undef main
#include <cassert>

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "Usage: test_flipsight_engine <directory-containing-resources>\n"; return 1; }
    std::string binary_path = std::filesystem::absolute(argv[1]).string() + "/";
    std::vector<std::string> arguments = {"test", "-flipsight", "offline-test-token", "-level", "3",
        "-thread", "4", "-hash", "18", "-nobook"};
    std::vector<char*> raw;
    for (auto& text : arguments) raw.push_back(text.data());
    auto options = get_options(get_commandline_options(static_cast<int>(raw.size()), raw.data()), binary_path);
    init_console(options, binary_path);
    Board board; board.reset();
    flipsight::Leg leg;
    leg.match_id = 123; leg.color = leg.turn = 1; leg.running = true;
    leg.black = board.player; leg.white = board.opponent; leg.legal = board.get_legal();
    leg.deadline = flipsight::now_ms() + 60000;
    for (int id = 0; id < 2; ++id) thread_pool.set_max_thread_size(id, 1);
    bool searching[2] = {true, true};
    auto a = std::async(std::launch::async, [&] { return flipsight::engine_search(leg, false, 0, &searching[0], 1000, options); });
    auto b = std::async(std::launch::async, [&] { return flipsight::engine_search(leg, false, 1, &searching[1], 1000, options); });
    auto first = a.get(), second = b.get();
    if (first.depth != 3 || second.depth != 3 || !(leg.legal & (1ULL << first.policy)) ||
            !(leg.legal & (1ULL << second.policy))) throw std::runtime_error("fixed-level concurrent engine search failed");
    options.level = 60;
    auto started = flipsight::now_ms();
    auto timed = flipsight::engine_search(leg, false, 0, &searching[0], 10, options);
    if (!(leg.legal & (1ULL << timed.policy)) || flipsight::now_ms() - started > 3000)
        throw std::runtime_error("fixed level did not honor the available time");
    options.flipsight_fixed_level = false;
    searching[0] = true;
    auto automatic = flipsight::engine_search(leg, false, 0, &searching[0], 30, options);
    if (!(leg.legal & (1ULL << automatic.policy))) throw std::runtime_error("game-clock search failed");
    options.flipsight_fixed_level = true; options.level = 0;
    searching[0] = true;
    auto random = flipsight::engine_search(leg, false, 0, &searching[0], 100, options);
    if (!(leg.legal & (1ULL << random.policy))) throw std::runtime_error("level zero failed");
    searching[0] = true;
    auto ponder = std::async(std::launch::async, [&] { return flipsight::engine_search(leg, true, 0, &searching[0], 1000, options); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    searching[0] = false;
    if (ponder.wait_for(std::chrono::seconds(3)) != std::future_status::ready) throw std::runtime_error("ponder did not stop");
    ponder.get();
    std::cout << "PASS: native fixed-level, simultaneous roots, time cap, game-clock search, level zero, ponder cancellation\n";
}
