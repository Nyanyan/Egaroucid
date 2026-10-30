#include <algorithm>
#include <atomic>
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
std::atomic<int> random_starts{0};
std::vector<int> random_result;
std::vector<int> random_board_generator(int, int, int, int, int, int, bool*) {
    ++random_starts;
    return random_result;
}
struct Search_result { int policy = 19, value = 0; };
Search_result ai(Board, int, bool, int, bool, bool) { return {}; }
Search_result ai_loss(Board, int, bool, int, bool, bool, int) { return {}; }
bool is_valid_policy(int policy) { return 0 <= policy && policy < 64; }
int calc_ai_type(Search_result) { return 1; }
int get_ai_loss_graph_value(const std::vector<int>&, int) { return 0; }
double myrandom() { return 0.5; }
int idx_to_coord(int policy) { return policy; }
struct History {
    Board board;
    int player = BLACK, v = 0, level = 0;
    void reset() { *this = History{}; }
};
struct Data {
    History history_elem;
    struct {
        int branch = GRAPH_MODE_NORMAL;
        std::vector<History> nodes[2]{{History{}}, {History{}}};
        void init() { branch = 0; nodes[0].clear(); nodes[1].clear(); }
    } graph_resources;
    struct {
        bool ai_put_black = false, ai_put_white = false, force_specified_openings = false;
        bool pause_when_pass = false, accept_ai_loss = false, use_book = false;
        int level = 1;
        int generate_random_board_score_range_min = -64, generate_random_board_score_range_max = 64;
        int generate_random_board_moves = 2;
        std::vector<int> max_loss_by_move, loss_percentage_by_move;
    } menu_elements;
    struct { int get_one(Board) { return 19; } } forced_openings;
    struct { int resets = 0; void init() { ++resets; } } game_information;
};
struct Scene {
    Data data;
    struct {
        bool ai_thinking = false, analyzing = false;
        bool random_board_generator_calculating = false;
        bool hint_calculating = false, hint_calculated = false, pv_calculating = false, pv_calculated = false;
        bool local_strategy_calculating = false, local_strategy_calculated = false;
        bool local_strategy_policy_calculating = false, local_strategy_policy_calculated = false;
        std::future<Search_result> ai_future;
        std::future<std::vector<int>> random_board_generator_future;
        std::future<void> hint_future, pv_future, local_strategy_future, local_strategy_policy_future;
    } ai_status;
    struct { bool umigame_calculating = false, umigame_calculated = false; } umigame_status;
    struct { bool book_accuracy_calculating = false, book_accuracy_calculated = false; } book_accuracy_status;
    bool changing_scene = false, need_start_game_button = false, pausing_in_pass = false;
    bool putting_1_move_by_ai = false, ai_calculation_interrupted = false;
    bool hint_calculation_interrupted = false, analyze_calculation_interrupted = false;
    bool pv_calculation_interrupted = false, local_strategy_calculation_interrupted = false;
    bool local_strategy_policy_calculation_interrupted = false, umigame_calculation_interrupted = false;
    bool book_accuracy_calculation_interrupted = false, random_board_generation_interrupted = false;
    bool random_position_marked = false;
    int moves_applied = 0;
    Data& getData() { return data; }
    void stop_calculating() {
        global_searching = false;
        clear_interrupted_calculation_flags();
        reset_ai();
        reset_random_board_generator();
    }
    void update_auto_ai_turn_timer() {}
    void reset_turn_timer_state() {}
    void mark_current_position_as_random_generated() { random_position_marked = true; }
    void need_start_game_button_calculation() {}
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
std::future<std::vector<int>> ready_moves(std::vector<int> moves) {
    std::promise<std::vector<int>> promise;
    promise.set_value(std::move(moves));
    return promise.get_future();
}
void existing_history(Scene& scene) {
    scene.data.history_elem.board.discs = 7;
    scene.data.graph_resources.nodes[0].resize(4);
}
void require_history_preserved(Scene& scene) {
    require(scene.data.history_elem.board.discs == 7 && scene.data.graph_resources.nodes[0].size() == 4
            && scene.data.game_information.resets == 0 && scene.moves_applied == 0,
            "random generation discarded or modified the existing game before success");
}
void check_random_lifecycle() {
    Scene scene;
    global_searching = true;
    existing_history(scene);
    random_result.clear();
    int starts_before = random_starts;
    scene.request_random_board();
    scene.ai_status.random_board_generator_future.wait();
    require_history_preserved(scene);
    scene.pause_calculating();
    require(!global_searching && !scene.ai_status.random_board_generator_calculating
            && scene.random_board_generation_interrupted,
            "pause did not cancel random generation and release its input lock");
    scene.check_random_board_generater();
    require(!global_searching, "paused random result silently resumed calculation");
    require_history_preserved(scene);
    scene.resume_calculating();
    scene.ai_status.random_board_generator_future.wait();
    require(random_starts == starts_before + 2 && scene.ai_status.random_board_generator_calculating
            && !scene.random_board_generation_interrupted, "resume did not restart the cancelled generator");
    require_history_preserved(scene);
    scene.pause_calculating();
    scene.stop_calculating();  // Moving through history permanently cancels the request.
    starts_before = random_starts;
    scene.resume_calculating();
    require(random_starts == starts_before && !scene.ai_status.random_board_generator_future.valid()
            && !scene.ai_status.random_board_generator_calculating,
            "a cancelled generator restarted on a different position");
    require_history_preserved(scene);
    std::cout << "PASS: generation keeps the game; pause unlocks it; resume restarts; navigation cancels\n";
}
void check_random_completion() {
    for (bool empty: {true, false}) {
        Scene scene;
        global_searching = true;
        existing_history(scene);
        scene.ai_status.random_board_generator_calculating = true;
        scene.ai_status.random_board_generator_future = ready_moves(empty ? std::vector<int>{} : std::vector<int>{19, 19});
        scene.check_random_board_generater();
        if (empty) {
            require_history_preserved(scene);
            require(!scene.random_position_marked, "failed generation marked an unrelated position");
        } else {
            require(scene.moves_applied == 2 && scene.data.history_elem.board.discs == 6
                    && scene.data.game_information.resets == 1 && scene.random_position_marked,
                    "successful generation was not applied exactly once");
        }
        require(global_searching && !scene.ai_status.random_board_generator_calculating,
                "completed generator left its input lock");
    }
    Scene paused;
    global_searching = false;
    existing_history(paused);
    paused.ai_status.random_board_generator_calculating = true;
    paused.ai_status.random_board_generator_future = ready_moves({19, 19});
    paused.check_random_board_generater();
    require(!global_searching && paused.ai_status.random_board_generator_future.valid(),
            "completion consumed a result while calculation was paused");
    require_history_preserved(paused);
    paused.stop_calculating();
    std::cout << "PASS: paused/failed results preserve history; successful results apply exactly once\n";
}
void check_random_join_before_resume() {
    Scene scene;
    global_searching = false;
    scene.random_board_generation_interrupted = true;
    scene.ai_status.random_board_generator_future = std::async(std::launch::deferred, [&scene]() {
        require(!global_searching && !scene.ai_status.random_board_generator_calculating,
                "resume re-enabled cancellation flags before joining the old generator");
        return std::vector<int>{19};
    });
    random_result.clear();
    scene.resume_calculating();
    scene.stop_calculating();
    std::cout << "PASS: old generator is joined before either cancellation flag is re-enabled\n";
}
void check_background_restart() {
    for (int task = 0; task < 4; ++task) for (bool get_while_paused: {false, true}) {
        Scene scene;
        bool* calculating[] = {&scene.ai_status.hint_calculating, &scene.ai_status.pv_calculating,
            &scene.ai_status.local_strategy_calculating, &scene.ai_status.local_strategy_policy_calculating};
        bool* calculated[] = {&scene.ai_status.hint_calculated, &scene.ai_status.pv_calculated,
            &scene.ai_status.local_strategy_calculated, &scene.ai_status.local_strategy_policy_calculated};
        bool* interrupted[] = {&scene.hint_calculation_interrupted, &scene.pv_calculation_interrupted,
            &scene.local_strategy_calculation_interrupted, &scene.local_strategy_policy_calculation_interrupted};
        std::future<void>* futures[] = {&scene.ai_status.hint_future, &scene.ai_status.pv_future,
            &scene.ai_status.local_strategy_future, &scene.ai_status.local_strategy_policy_future};
        void (Scene::*getters[])() = {&Scene::try_hint_get, &Scene::try_pv_get,
            &Scene::try_local_strategy_get, &Scene::try_local_strategy_policy_get};
        std::promise<void> promise;
        *futures[task] = promise.get_future();
        *calculating[task] = true;
        global_searching = true;
        scene.pause_calculating();
        promise.set_value();
        if (get_while_paused) {
            (scene.*getters[task])();
            require(*calculated[task] && *interrupted[task], "paused preview lost its restart marker");
        }
        scene.resume_calculating();
        if (!get_while_paused) (scene.*getters[task])();
        require(!*calculating[task] && !*calculated[task] && !*interrupted[task],
                "interrupted background result prevented fresh calculation on resume");
    }
    std::cout << "PASS: hint, PV, local strategy, and policy restart for both future completion orders\n";
}
int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "all";
        if (mode == "all" || mode == "cancel") check_cancel_single_move();
        if (mode == "all" || mode == "single") check_normal_and_forced_single_move();
        if (mode == "all" || mode == "resume") check_pause_single_move();
        if (mode == "all" || mode == "random") {
            check_random_lifecycle();
            check_random_completion();
            check_random_join_before_resume();
        }
        if (mode == "all" || mode == "background") check_background_restart();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
