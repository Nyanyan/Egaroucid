#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int BLACK = 0, WHITE = 1, GRAPH_MODE_NORMAL = 0, HW2 = 64, HW2_M1 = 63, MOVE_UNDEFINED = 125;
bool global_searching = true;
struct Board {
    int discs = 4;
    uint64_t get_legal() const { return 1ULL << 19; }
    bool is_end() const { return false; }
    int n_discs() const { return discs; }
};
struct Search_result { int policy = 19, value = 0; };
Search_result ai(Board, int, bool, int, bool, bool) { return {}; }
Search_result ai_loss(Board, int, bool, int, bool, bool, int) { return {}; }
bool is_valid_policy(int policy) { return 0 <= policy && policy < 64; }
int calc_ai_type(Search_result) { return 1; }
int get_ai_loss_graph_value(const std::vector<int>&, int) { return 0; }
double myrandom() { return 0.5; }
int idx_to_coord(int policy) { return policy; }
struct History { Board board; int player = BLACK, v = 0, level = 0; };
struct Data {
    History history_elem;
    struct {
        int branch = GRAPH_MODE_NORMAL;
        std::vector<History> nodes[2]{{History{}}, {History{}}};
    } graph_resources;
    struct {
        bool ai_put_black = false, ai_put_white = false, force_specified_openings = false;
        bool pause_when_pass = false, accept_ai_loss = false, use_book = false;
        int level = 1;
        std::vector<int> max_loss_by_move, loss_percentage_by_move;
    } menu_elements;
    struct { int get_one(Board) { return 19; } } forced_openings;
};
struct Scene {
    Data data;
    struct { bool ai_thinking = false; std::future<Search_result> ai_future; } ai_status;
    bool changing_scene = false, need_start_game_button = false, pausing_in_pass = false;
    bool putting_1_move_by_ai = false, ai_calculation_interrupted = false;
    int moves_applied = 0;
    Data& getData() { return data; }
    void stop_calculating() { global_searching = false; reset_ai(); ai_calculation_interrupted = false; }
    void resume_calculating() { global_searching = true; }
    void update_auto_ai_turn_timer() {}
    void move_processing(int) {
        stop_calculating();
        ++moves_applied;
        ++data.history_elem.board.discs;
        data.graph_resources.nodes[0].push_back(data.history_elem);
        resume_calculating();
    }
#include "calculation_lifecycle_methods.hpp"
};
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void finish_move(Scene& scene) {
    scene.ai_status.ai_future.wait();
    scene.ai_move();
}
void check_cancel_single_move() {
    Scene scene;
    scene.putting_1_move_by_ai = true;
    scene.ai_move();
    require(scene.ai_status.ai_thinking && scene.putting_1_move_by_ai,
            "starting AI search lost the single-move request");
    scene.stop_calculating();  // History change, new game, profile change, etc.
    scene.resume_calculating();
    require(!scene.putting_1_move_by_ai && !scene.should_ai_move(),
            "cancelled single-move request leaked to a different position");
    require(!scene.ai_status.ai_future.valid() && !scene.ai_status.ai_thinking,
            "cancel left an AI future or its input lock");
    std::cout << "PASS: cancelling a single move clears its request, future, and lock\n";
}
void check_normal_and_forced_single_move() {
    for (bool forced: {false, true}) {
        Scene scene;
        global_searching = true;
        scene.data.menu_elements.force_specified_openings = forced;
        scene.putting_1_move_by_ai = true;
        scene.ai_move();
        if (!forced) {
            require(scene.putting_1_move_by_ai && scene.should_ai_move(),
                    "search setup cancelled the requested move");
            finish_move(scene);
        }
        require(scene.moves_applied == 1 && !scene.putting_1_move_by_ai && !scene.should_ai_move(),
                "single move continued after playing a forced opening");
    }
    std::cout << "PASS: normal search and forced opening each play exactly one requested move\n";
}
void check_pause_single_move() {
    Scene scene;
    global_searching = true;
    scene.putting_1_move_by_ai = true;
    scene.ai_move();
    global_searching = false;
    scene.ai_calculation_interrupted = true;
    scene.ai_move();
    require(scene.moves_applied == 0 && scene.putting_1_move_by_ai,
            "pause applied or discarded the requested move");
    global_searching = true;
    finish_move(scene);  // Discard the interrupted result.
    require(scene.moves_applied == 0 && scene.putting_1_move_by_ai,
            "resuming applied an interrupted result or lost its request");
    scene.ai_move();
    finish_move(scene);
    require(scene.moves_applied == 1 && !scene.putting_1_move_by_ai,
            "resumed single move did not finish exactly once");
    std::cout << "PASS: pause retains a same-position request and resume discards the interrupted result\n";
}
int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "all";
        if (mode == "all" || mode == "cancel") check_cancel_single_move();
        if (mode == "all" || mode == "single") check_normal_and_forced_single_move();
        if (mode == "all" || mode == "resume") check_pause_single_move();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
