#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int BLACK = 0, WHITE = 1, SCORE_UNDEFINED = 100000000, MOVE_UNDEFINED = 125;
bool global_searching = true, searching = true;
int light_calls = 0, range_calls = 0, flips = 0;
enum Mode { NORMAL, GLOBAL_LIGHT, LOCAL_LIGHT, INVALID_LIGHT, ILLEGAL_LIGHT,
    GLOBAL_RANGE, LOCAL_RANGE, INVALID_RANGE, ILLEGAL_RANGE,
    GLOBAL_FALLBACK, LOCAL_FALLBACK, INVALID_FALLBACK, ILLEGAL_FALLBACK } mode;
bool is_valid_policy(int policy) { return 0 <= policy && policy < 64; }
uint64_t tim() { return 0; }
struct Flip {};
struct Board {
    void reset() {}
    bool is_end() { return false; }
    uint64_t get_legal() { return 1ULL << 19; }
    void pass() {}
    void move_board(Flip*) {}
};
void calc_flip(Flip*, Board* board, int policy) {
    if (!is_valid_policy(policy) || !(board->get_legal() & (1ULL << policy)))
        throw std::runtime_error("generator passed an undefined or illegal move to calc_flip");
    ++flips;
}
struct Search_result { int policy = 19, value = 0; };
Search_result ai_accept_loss(Board, int, int) {
    ++light_calls;
    if (mode == GLOBAL_LIGHT || (mode == GLOBAL_FALLBACK && range_calls)) global_searching = false;
    if (mode == LOCAL_LIGHT || (mode == LOCAL_FALLBACK && range_calls)) searching = false;
    if (mode == INVALID_LIGHT || (mode == INVALID_FALLBACK && range_calls)) return {MOVE_UNDEFINED, SCORE_UNDEFINED};
    if (mode == ILLEGAL_LIGHT || (mode == ILLEGAL_FALLBACK && range_calls)) return {20, 0};
    return {};
}
Search_result ai_range(Board, int, int, int, bool*) {
    ++range_calls;
    if (mode == GLOBAL_RANGE) global_searching = false;
    if (mode == LOCAL_RANGE) searching = false;
    if (mode == INVALID_RANGE) return {MOVE_UNDEFINED, 0};
    if (mode == ILLEGAL_RANGE) return {20, 0};
    if (mode >= GLOBAL_FALLBACK && range_calls == 1) return {MOVE_UNDEFINED, SCORE_UNDEFINED};
    return {};
}
#include "random_board_generator_methods.hpp"

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void reset(Mode next) {
    mode = next;
    global_searching = searching = true;
    light_calls = range_calls = flips = 0;
}
int main() {
    try {
        for (bool global_stop: {false, true}) {
            reset(NORMAL);
            if (global_stop) global_searching = false;
            else searching = false;
            require(random_board_generator(-10, 10, 6, 1, 15, 20, &searching).empty()
                    && light_calls == 0 && range_calls == 0 && flips == 0,
                    "an already-cancelled generator still searched or returned moves");
        }
        std::cout << "PASS: either cancellation flag prevents all search work\n";
        for (Mode next: {GLOBAL_LIGHT, LOCAL_LIGHT, INVALID_LIGHT, ILLEGAL_LIGHT,
                        GLOBAL_RANGE, LOCAL_RANGE, INVALID_RANGE, ILLEGAL_RANGE,
                        GLOBAL_FALLBACK, LOCAL_FALLBACK, INVALID_FALLBACK, ILLEGAL_FALLBACK}) {
            reset(next);
            require(random_board_generator(-10, 10, 6, 1, 15, 20, &searching).empty(),
                    "cancelled or invalid search returned a partial transcript");
            const bool in_light = next <= ILLEGAL_LIGHT;
            require(flips == (in_light ? 0 : 4) && range_calls == (in_light ? 0 : 1)
                    && light_calls == (in_light ? 1 : next >= GLOBAL_FALLBACK ? 5 : 4),
                    "cancelled/invalid search continued moving or retried");
        }
        std::cout << "PASS: cancellation, undefined policy, and illegal policy stop light/range/fallback phases\n";
        for (int count: {0, 1, 2, 6, 60}) {
            reset(NORMAL);
            auto moves = random_board_generator(-10, 10, count, 1, 15, 20, &searching);
            require(moves.size() == count && flips == count
                    && std::all_of(moves.begin(), moves.end(), [](int policy) { return policy == 19; }),
                    "normal generation no longer returns the complete requested transcript");
        }
        std::cout << "PASS: successful generation preserves requested lengths and legal policies\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
