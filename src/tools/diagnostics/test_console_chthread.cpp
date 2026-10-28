/*
    Issue #644: runtime thread-count changes through the console dispatcher.

    Build and run from the repository root:
    g++ -O1 -std=c++20 -march=native -pthread src/tools/diagnostics/test_console_chthread.cpp -o test_console_chthread.exe
    ./test_console_chthread.exe
*/
#include <sstream>
#include <stdexcept>
#include "../../engine/engine_all.hpp"
#include "../../console/console_all.hpp"

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Redirect_stream {
    std::ios& stream;
    std::ios::iostate previous_state;
    std::streambuf* previous_buffer;

    Redirect_stream(std::ios& stream_, std::streambuf* buffer)
        : stream(stream_), previous_state(stream.rdstate()), previous_buffer(stream.rdbuf(buffer)) {
        stream.clear();
    }
    ~Redirect_stream() {
        stream.rdbuf(previous_buffer);
        stream.clear(previous_state);
    }
};

std::string command(const std::string& text, Board_info* board, State* state, Options* options) {
    std::istringstream input(text + '\n');
    std::ostringstream output;
    Redirect_stream in(std::cin, input.rdbuf());
    Redirect_stream out(std::cout, output.rdbuf());
    Redirect_stream err(std::cerr, output.rdbuf());
    check_command(board, state, options);
    return output.str();
}

void check_worker() {
    bool pushed = false;
    std::future<int> result;
    const uint64_t deadline = tim() + 2000;
    while (!pushed && tim() < deadline) {
        result = thread_pool.push(&pushed, [] { return 644; });
        if (!pushed) std::this_thread::yield();
    }
    require(pushed, "No usable worker after changing the thread count");
    require(result.get() == 644, "Worker did not execute the submitted task");
}

int main() {
    try {
        Options options{};
        options.n_threads = 1;
        options.level = 3;
        Board_info board;
        board.reset();
        State state;
        state.remaining_time_msec_black = 123456;
        state.remaining_time_msec_white = 654321;
        const Board original = board.board;
        const size_t original_history = board.boards.size();

        for (int count : {4, 1, 3, 3, 2, 1}) {
            const auto output = command("chthread " + std::to_string(count), &board, &state, &options);
            require(output.find("[ERROR]") == std::string::npos, "Valid thread count rejected");
            require(options.n_threads == count, "Option not updated");
            require(thread_pool.size() == count - 1, "Worker count must exclude the main thread");
            if (count > 1) check_worker();
        }
        for (const std::string& arg : {"", "0", "-1", "abc", "2x", "2 3", "1.5", "999999999999999999999999"}) {
            const auto output = command("chthread " + arg, &board, &state, &options);
            require(output.find("[ERROR]") != std::string::npos, "Invalid thread count accepted");
            require(options.n_threads == 1 && thread_pool.size() == 0, "Invalid input changed the count");
        }
        command("chthread   2  ", &board, &state, &options);
        require(options.n_threads == 2 && thread_pool.size() == 1, "Surrounding whitespace rejected");
        require(board.board.player == original.player && board.board.opponent == original.opponent,
            "Changing threads modified the position");
        require(board.boards.size() == original_history, "Changing threads modified the history");
        require(state.remaining_time_msec_black == 123456 && state.remaining_time_msec_white == 654321,
            "Changing threads reset the clocks");
        require(options.level == 3, "Changing threads modified the search level");
        require(command("help", &board, &state, &options).find("chthread <n_threads>") != std::string::npos,
            "chthread missing from interactive help");

        // The dispatcher must join pondering before resizing the worker pool.
        options.ponder = true;
        state.ponder_searching = true;
        std::atomic<bool> ponder_finished{false};
        state.ponder_future = std::async(std::launch::async, [&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            ponder_finished = true;
            return std::vector<Ponder_elem>{};
        });
        command("chthread 3", &board, &state, &options);
        require(ponder_finished && !state.ponder_future.valid() && !state.ponder_searching,
            "Pondering was not joined before returning from chthread");
        require(options.n_threads == 3 && thread_pool.size() == 2, "Ponder-enabled resize failed");
        check_worker();
        thread_pool.resize(0);

        std::cout << "PASS: resizing, worker execution, invalid input, state preservation, help, and ponder join\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
