#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int BLACK = 0, WHITE = 1, GRAPH_MODE_NORMAL = 0, GRAPH_MODE_INSPECT = 1;
constexpr int BOOK_CHANGE_NO_CELL = -1;
bool global_searching = true;

struct Board {
    int discs = 10;
    bool ended = false;
    int n_discs() const { return discs; }
    bool is_end() const { return ended; }
};
struct History { Board board; int player = BLACK; };
struct Data {
    History history_elem;
    struct {
        int branch = GRAPH_MODE_NORMAL;
        std::vector<History> nodes[2]{{History{}}, {History{}}};
    } graph_resources;
    struct { bool ai_put_black = true, ai_put_white = true; } menu_elements;
    struct { int changing = BOOK_CHANGE_NO_CELL; } book_information;
    struct { bool opened = false; bool active() const { return opened; } } menu;
};
struct Scene {
    Data data;
    struct { bool analyzing = false, random_board_generator_calculating = false; } ai_status;
    bool need_start_game_button = false, pausing_in_pass = false, putting_1_move_by_ai = false;
    int graph_inputs = 0;
    Data &getData() { return data; }
    void interact_graph() { ++graph_inputs; }
#include "selfplay_pause_methods.hpp"
};

void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

void check_pause_and_resume() {
    Scene scene;
    global_searching = true;
    require(scene.should_ai_move(), "active selfplay lost its AI turn");
    scene.route_graph_input();
    require(scene.graph_inputs == 0, "active selfplay must retain its input lock");

    global_searching = false;
    require(!scene.should_ai_move(), "paused selfplay still claims the AI turn");
    scene.route_graph_input();
    require(scene.graph_inputs == 1, "paused selfplay blocked history/graph input");
    require(scene.data.menu_elements.ai_put_black && scene.data.menu_elements.ai_put_white,
            "pausing must preserve the selected AI players");

    global_searching = true;
    require(scene.should_ai_move(), "resume did not restore the AI turn");
    scene.route_graph_input();
    require(scene.graph_inputs == 1, "resumed selfplay lost its input lock");
    std::cout << "PASS: pause releases navigation; resume retains selfplay settings\n";
}

void check_turn_conditions() {
    for (int ai = 0; ai < 4; ++ai) for (int player: {BLACK, WHITE})
    for (bool running: {false, true}) for (bool one_move: {false, true}) {
        Scene scene;
        scene.data.menu_elements.ai_put_black = ai & 1;
        scene.data.menu_elements.ai_put_white = ai & 2;
        scene.data.history_elem.player = player;
        scene.putting_1_move_by_ai = one_move;
        global_searching = running;
        bool controlled = ai & (player == BLACK ? 1 : 2);
        require(scene.should_ai_move() == (running && (controlled || one_move)),
                "human/AI or one-move turn condition changed");
        scene.route_graph_input();
        require(scene.graph_inputs == int(!running || (!controlled && !one_move)),
                "navigation gate differs from the current AI turn");
    }
    global_searching = true;
    Scene scene;
    scene.need_start_game_button = true;
    require(!scene.should_ai_move(), "AI moved before game start");
    scene.need_start_game_button = false;
    scene.pausing_in_pass = true;
    require(!scene.should_ai_move(), "AI ignored pass pause");
    scene.pausing_in_pass = false;
    scene.data.graph_resources.branch = GRAPH_MODE_INSPECT;
    require(!scene.should_ai_move(), "inspection history triggered selfplay");
    scene.data.graph_resources.branch = GRAPH_MODE_NORMAL;
    scene.data.history_elem.board.discs = 8;
    require(!scene.should_ai_move(), "earlier history triggered selfplay");
    scene.data.history_elem.board.discs = 10;
    scene.data.history_elem.board.ended = true;
    scene.putting_1_move_by_ai = true;
    require(!scene.should_ai_move(), "game over triggered an AI move");
    std::cout << "PASS: human/AI turns, single move, start, pass, history, and game over\n";
}

void check_other_input_locks() {
    global_searching = false;
    for (int lock = 0; lock < 3; ++lock) {
        Scene scene;
        if (lock == 0) scene.ai_status.random_board_generator_calculating = true;
        if (lock == 1) scene.data.book_information.changing = 12;
        if (lock == 2) scene.data.menu.opened = true;
        scene.route_graph_input();
        require(scene.graph_inputs == 0, "pause bypassed an unrelated input lock");
    }
    std::cout << "PASS: random generation, book edit, and menu locks remain effective\n";
}

void check_analysis_pause_and_resume() {
    Scene scene;
    scene.data.menu_elements.ai_put_black = false;
    scene.data.menu_elements.ai_put_white = false;
    scene.ai_status.analyzing = true;
    global_searching = true;
    require(scene.is_analysis_running(), "active analysis was not recognized");
    scene.route_graph_input();
    require(scene.graph_inputs == 0 && !scene.operation_menu_is_unlocked(),
            "active analysis lost its navigation/menu lock");

    global_searching = false;
    scene.route_graph_input();
    require(scene.graph_inputs == 1 && scene.operation_menu_is_unlocked(),
            "paused analysis blocked history, graph, or menu navigation");
    require(!scene.is_analysis_running() && scene.ai_status.analyzing,
            "pause destroyed the pending analysis state instead of releasing its input lock");

    global_searching = true;
    scene.route_graph_input();
    require(scene.graph_inputs == 1 && !scene.operation_menu_is_unlocked() && scene.is_analysis_running(),
            "resumed analysis did not reacquire its navigation/menu lock");

    scene.ai_status.analyzing = false;
    scene.route_graph_input();
    require(scene.graph_inputs == 2 && scene.operation_menu_is_unlocked(),
            "finished or cancelled analysis left an input lock");
    std::cout << "PASS: analysis pause releases history/graph/menu locks, retains pending state, and resumes\n";
}

int main(int argc, char **argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "all";
        if (mode != "analysis") {
            check_pause_and_resume();
            check_turn_conditions();
        }
        if (mode != "selfplay") check_analysis_pause_and_resume();
        check_other_input_locks();
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
