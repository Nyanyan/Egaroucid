#pragma once
#include <iostream>
#include <future>
#include "./ai.hpp"
#include "function/language.hpp"
#include "function/menu.hpp"
#include "function/graph.hpp"
#include "function/opening.hpp"
#include "function/button.hpp"
#include "function/radio_button.hpp"
#include "gui_common.hpp"

#define HINT_SINGLE_TASK_N_THREAD 4

bool compare_value_cell(pair<int, int>& a, pair<int, int>& b) {
	return a.first > b.first;
}

bool compare_hint_info(Hint_info& a, Hint_info& b) {
	return a.value > b.value;
}

void draw_board(Fonts fonts, Colors colors, History_elem history_elem) {
	String coord_x = U"abcdefgh";
	for (int i = 0; i < HW; ++i) {
		fonts.font15_bold(i + 1).draw(Arg::center(BOARD_SX - BOARD_COORD_SIZE, BOARD_SY + BOARD_CELL_SIZE * i + BOARD_CELL_SIZE / 2), colors.dark_gray);
		fonts.font15_bold(coord_x[i]).draw(Arg::center(BOARD_SX + BOARD_CELL_SIZE * i + BOARD_CELL_SIZE / 2, BOARD_SY - BOARD_COORD_SIZE - 2), colors.dark_gray);
	}
	for (int i = 0; i < HW_M1; ++i) {
		Line(BOARD_SX + BOARD_CELL_SIZE * (i + 1), BOARD_SY, BOARD_SX + BOARD_CELL_SIZE * (i + 1), BOARD_SY + BOARD_CELL_SIZE * HW).draw(BOARD_CELL_FRAME_WIDTH, colors.dark_gray);
		Line(BOARD_SX, BOARD_SY + BOARD_CELL_SIZE * (i + 1), BOARD_SX + BOARD_CELL_SIZE * HW, BOARD_SY + BOARD_CELL_SIZE * (i + 1)).draw(BOARD_CELL_FRAME_WIDTH, colors.dark_gray);
	}
	Circle(BOARD_SX + 2 * BOARD_CELL_SIZE, BOARD_SY + 2 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(colors.dark_gray);
	Circle(BOARD_SX + 2 * BOARD_CELL_SIZE, BOARD_SY + 6 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(colors.dark_gray);
	Circle(BOARD_SX + 6 * BOARD_CELL_SIZE, BOARD_SY + 2 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(colors.dark_gray);
	Circle(BOARD_SX + 6 * BOARD_CELL_SIZE, BOARD_SY + 6 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(colors.dark_gray);
	RoundRect(BOARD_SX, BOARD_SY, BOARD_CELL_SIZE * HW, BOARD_CELL_SIZE * HW, BOARD_ROUND_DIAMETER).drawFrame(0, BOARD_ROUND_FRAME_WIDTH, colors.white);
	Flip flip;
	int board_arr[HW2];
	history_elem.board.translate_to_arr(board_arr, history_elem.player);
	for (int cell = 0; cell < HW2; ++cell) {
		int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
		int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
		if (board_arr[cell] == BLACK) {
			Circle(x, y, DISC_SIZE).draw(colors.black);
		}
		else if (board_arr[cell] == WHITE) {
			Circle(x, y, DISC_SIZE).draw(colors.white);
		}
	}
	if (history_elem.policy != -1) {
		int x = BOARD_SX + (HW_M1 - history_elem.policy % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
		int y = BOARD_SY + (HW_M1 - history_elem.policy / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
		Circle(x, y, LEGAL_SIZE).draw(colors.red);
	}
}

class Main_scene : public App::Scene {
private:
	Graph graph;
	Move_board_button_status move_board_button_status;
	AI_status ai_status;

public:
	Main_scene(const InitData& init) : IScene{ init } {
		cerr << "main scene loading" << endl;
		getData().menu = create_menu(&getData().menu_elements);
		graph.sx = GRAPH_SX;
		graph.sy = GRAPH_SY;
		graph.size_x = GRAPH_WIDTH;
		graph.size_y = GRAPH_HEIGHT;
		graph.resolution = GRAPH_RESOLUTION;
		graph.font = getData().fonts.font15;
		graph.font_size = 15;
		if (getData().graph_resources.need_init) {
			getData().graph_resources.init();
			getData().graph_resources.nodes[getData().graph_resources.put_mode].emplace_back(getData().history_elem);
		}
		cerr << "main scene loaded" << endl;
	}

	void update() override {
		Scene::SetBackground(getData().colors.green);

		// multi threading
		if (getData().menu_elements.n_threads != thread_pool.size()) {
			stop_calculating();
			thread_pool.resize(getData().menu_elements.n_threads);
			cerr << "thread pool resized to " << thread_pool.size() << endl;
			resume_calculating();
		}

		// init
		getData().graph_resources.delta = 0;

		// opening
		update_opening();

		// menu
		menu_game();
		menu_manipulate();
		menu_in_out();
		menu_book();

		// analyze
		if (ai_status.analyzing) {
			analyze_get_task();
		}

		bool graph_interact_ignore = ai_status.analyzing;
		// transcript move
		if (!graph_interact_ignore && !getData().menu.active()) {
			interact_graph();
		}
		update_n_discs();

		bool move_ignore = ai_status.analyzing;
		// move
		bool ai_should_move =
			getData().graph_resources.put_mode == GRAPH_MODE_NORMAL &&
			((getData().history_elem.player == BLACK && getData().menu_elements.ai_put_black) || (getData().history_elem.player == WHITE && getData().menu_elements.ai_put_white)) &&
			getData().history_elem.board.n_discs() == getData().graph_resources.nodes[GRAPH_MODE_NORMAL][getData().graph_resources.nodes[GRAPH_MODE_NORMAL].size() - 1].board.n_discs();
		if (!move_ignore) {
			if (ai_should_move) {
				ai_move();
			}
			else if (!getData().menu.active()) {
				interact_move();
			}
		}

		// board drawing
		draw_board(getData().fonts, getData().colors, getData().history_elem);

		bool hint_ignore = ai_should_move || ai_status.analyzing;

		// hint / legalcalculating & drawing
		if (!hint_ignore) {
			if (getData().menu_elements.use_disc_hint) {
				if (!ai_status.hint_calculating && ai_status.hint_level < getData().menu_elements.level) {
					hint_init_calculating();
				}
				hint_do_task();
				uint64_t legal_ignore = draw_hint();
				if (getData().menu_elements.show_legal) {
					draw_legal(legal_ignore);
				}
			}
			else if (getData().menu_elements.show_legal) {
				draw_legal(0);
			}
		}

		// graph drawing
		if (getData().menu_elements.show_graph) {
			graph.draw(getData().graph_resources.nodes[0], getData().graph_resources.nodes[1], getData().graph_resources.n_discs);
		}

		// info drawing
		draw_info();

		// opening on cell drawing
		if (getData().menu_elements.show_opening_on_cell) {
			draw_opening_on_cell();
		}

		// menu drawing
		getData().menu.draw();
	}

	void draw() const override {

	}

private:
	void reset_hint() {
		ai_status.hint_level = HINT_NOT_CALCULATING;
		ai_status.hint_available = false;
		ai_status.hint_calculating = false;
		ai_status.hint_task_stack.clear();
	}

	void reset_ai() {
		ai_status.ai_thinking = false;
	}

	void reset_analyze() {
		ai_status.analyzing = false;
		ai_status.analyze_task_stack.clear();
	}

	void stop_calculating() {
		cerr << "terminating calculation" << endl;
		global_searching = false;
		if (ai_status.ai_future.valid()) {
			ai_status.ai_future.get();
		}
		for (int i = 0; i < HW2; ++i) {
			if (ai_status.hint_future[i].valid()) {
				ai_status.hint_future[i].get();
			}
		}
		for (int i = 0; i < ANALYZE_SIZE; ++i) {
			if (ai_status.analyze_future[i].valid()) {
				ai_status.analyze_future[i].get();
			}
		}
		cerr << "calculation terminated" << endl;
		reset_ai();
		reset_hint();
		reset_analyze();
		cerr << "reset all calculations" << endl;
	}

	void resume_calculating() {
		global_searching = true;
	}

	void menu_game() {
		if (getData().menu_elements.start_game) {
			stop_calculating();
			getData().history_elem.reset();
			getData().graph_resources.init();
			getData().graph_resources.nodes[getData().graph_resources.put_mode].emplace_back(getData().history_elem);
			resume_calculating();
		}
		if (getData().menu_elements.analyze && !ai_status.ai_thinking && !ai_status.analyzing) {
			stop_calculating();
			init_analyze();
			resume_calculating();
		}
	}

	void menu_in_out() {
		if (getData().menu_elements.input_transcript) {
			changeScene(U"Import_transcript", SCENE_FADE_TIME);
		}
		if (getData().menu_elements.input_board) {
			changeScene(U"Import_board", SCENE_FADE_TIME);
		}
		if (getData().menu_elements.edit_board) {
			changeScene(U"Edit_board", SCENE_FADE_TIME);
		}
		if (getData().menu_elements.input_game) {
			changeScene(U"Import_game", SCENE_FADE_TIME);
		}
		if (getData().menu_elements.copy_transcript) {
			copy_transcript();
		}
		if (getData().menu_elements.save_game) {
			changeScene(U"Export_game", SCENE_FADE_TIME);
		}
	}

	void menu_manipulate() {
		if (getData().menu_elements.stop_calculating) {
			stop_calculating();
			ai_status.hint_level = HINT_INF_LEVEL;
			resume_calculating();
		}
		if (!ai_status.analyzing) {
			if (getData().menu_elements.backward) {
				--getData().graph_resources.n_discs;
				getData().graph_resources.delta = -1;
			}
			if (getData().menu_elements.forward) {
				++getData().graph_resources.n_discs;
				getData().graph_resources.delta = 1;
			}
		}
		if (getData().menu_elements.convert_180) {
			stop_calculating();
			getData().history_elem.board.board_rotate_180();
			if (0 <= getData().history_elem.policy && getData().history_elem.policy < HW2) {
				getData().history_elem.policy = HW2_M1 - getData().history_elem.policy;
			}
			if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy < HW2) {
				getData().history_elem.next_policy = HW2_M1 - getData().history_elem.next_policy;
			}
			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
					getData().graph_resources.nodes[i][j].board.board_rotate_180();
					if (0 <= getData().graph_resources.nodes[i][j].policy && getData().graph_resources.nodes[i][j].policy < HW2) {
						getData().graph_resources.nodes[i][j].policy = HW2_M1 - getData().graph_resources.nodes[i][j].policy;
					}
					if (0 <= getData().graph_resources.nodes[i][j].next_policy && getData().graph_resources.nodes[i][j].next_policy < HW2) {
						getData().graph_resources.nodes[i][j].next_policy = HW2_M1 - getData().graph_resources.nodes[i][j].next_policy;
					}
				}
			}
			resume_calculating();
		}
		if (getData().menu_elements.convert_blackline) {
			stop_calculating();
			getData().history_elem.board.board_black_line_mirror();
			if (0 <= getData().history_elem.policy && getData().history_elem.policy < HW2) {
				int x = getData().history_elem.policy % HW;
				int y = getData().history_elem.policy / HW;
				getData().history_elem.policy = (HW_M1 - x) * HW + (HW_M1 - y);
			}
			if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy < HW2) {
				int x = getData().history_elem.next_policy % HW;
				int y = getData().history_elem.next_policy / HW;
				getData().history_elem.next_policy = (HW_M1 - x) * HW + (HW_M1 - y);
			}
			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
					getData().graph_resources.nodes[i][j].board.board_black_line_mirror();
					if (0 <= getData().graph_resources.nodes[i][j].policy && getData().graph_resources.nodes[i][j].policy < HW2) {
						int x = getData().graph_resources.nodes[i][j].policy % HW;
						int y = getData().graph_resources.nodes[i][j].policy / HW;
						getData().graph_resources.nodes[i][j].policy = (HW_M1 - x) * HW + (HW_M1 - y);
					}
					if (0 <= getData().graph_resources.nodes[i][j].next_policy && getData().graph_resources.nodes[i][j].next_policy < HW2) {
						int x = getData().graph_resources.nodes[i][j].next_policy % HW;
						int y = getData().graph_resources.nodes[i][j].next_policy / HW;
						getData().graph_resources.nodes[i][j].next_policy = (HW_M1 - x) * HW + (HW_M1 - y);
					}
				}
			}
			resume_calculating();
		}
		if (getData().menu_elements.convert_whiteline) {
			stop_calculating();
			getData().history_elem.board.board_white_line_mirror();
			if (0 <= getData().history_elem.policy && getData().history_elem.policy < HW2) {
				int x = getData().history_elem.policy % HW;
				int y = getData().history_elem.policy / HW;
				getData().history_elem.policy = x * HW + y;
			}
			if (0 <= getData().history_elem.next_policy && getData().history_elem.next_policy < HW2) {
				int x = getData().history_elem.next_policy % HW;
				int y = getData().history_elem.next_policy / HW;
				getData().history_elem.next_policy = x * HW + y;
			}
			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < (int)getData().graph_resources.nodes[i].size(); ++j) {
					getData().graph_resources.nodes[i][j].board.board_white_line_mirror();
					if (0 <= getData().graph_resources.nodes[i][j].policy && getData().graph_resources.nodes[i][j].policy < HW2) {
						int x = getData().graph_resources.nodes[i][j].policy % HW;
						int y = getData().graph_resources.nodes[i][j].policy / HW;
						getData().graph_resources.nodes[i][j].policy = x * HW + y;
					}
					if (0 <= getData().graph_resources.nodes[i][j].next_policy && getData().graph_resources.nodes[i][j].next_policy < HW2) {
						int x = getData().graph_resources.nodes[i][j].next_policy % HW;
						int y = getData().graph_resources.nodes[i][j].next_policy / HW;
						getData().graph_resources.nodes[i][j].next_policy = x * HW + y;
					}
				}
			}
			resume_calculating();
		}
	}

	void menu_book() {
		if (getData().menu_elements.book_import) {
			changeScene(U"Import_book", SCENE_FADE_TIME);
		}
		if (getData().menu_elements.book_reference) {
			changeScene(U"Refer_book", SCENE_FADE_TIME);
		}
	}

	void interact_graph() {
		getData().graph_resources.n_discs = graph.update_n_discs(getData().graph_resources.nodes[0], getData().graph_resources.nodes[1], getData().graph_resources.n_discs);
		if (!KeyLeft.pressed() && !KeyA.pressed()) {
			move_board_button_status.left_pushed = BUTTON_NOT_PUSHED;
		}
		if (!KeyRight.pressed() && !KeyD.pressed()) {
			move_board_button_status.right_pushed = BUTTON_NOT_PUSHED;
		}

		if (MouseX1.down() || KeyLeft.down() || KeyA.down() || (move_board_button_status.left_pushed != BUTTON_NOT_PUSHED && tim() - move_board_button_status.left_pushed >= BUTTON_LONG_PRESS_THRESHOLD)) {
			--getData().graph_resources.n_discs;
			getData().graph_resources.delta = -1;
			if (KeyLeft.down() || KeyA.down()) {
				move_board_button_status.left_pushed = tim();
			}
		}
		else if (MouseX2.down() || KeyRight.down() || KeyD.down() || (move_board_button_status.right_pushed != BUTTON_NOT_PUSHED && tim() - move_board_button_status.right_pushed >= BUTTON_LONG_PRESS_THRESHOLD)) {
			++getData().graph_resources.n_discs;
			getData().graph_resources.delta = 1;
			if (KeyRight.down() || KeyD.down()) {
				move_board_button_status.right_pushed = tim();
			}
		}
	}

	void update_n_discs() {
		int max_n_discs = getData().graph_resources.nodes[getData().graph_resources.put_mode].back().board.n_discs();
		getData().graph_resources.n_discs = min(getData().graph_resources.n_discs, max_n_discs);
		int min_n_discs = getData().graph_resources.nodes[GRAPH_MODE_NORMAL][0].board.n_discs();
		if (getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
			min_n_discs = min(min_n_discs, getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs());
		}
		getData().graph_resources.n_discs = max(getData().graph_resources.n_discs, min_n_discs);
		if (getData().graph_resources.put_mode == GRAPH_MODE_INSPECT && getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
			if (getData().graph_resources.n_discs < getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs()) {
				getData().graph_resources.put_mode = GRAPH_MODE_NORMAL;
				getData().graph_resources.nodes[1].clear();
			}
		}
		int node_idx = getData().graph_resources.node_find(getData().graph_resources.put_mode, getData().graph_resources.n_discs);
		if (node_idx == -1 && getData().graph_resources.put_mode == GRAPH_MODE_INSPECT) {
			getData().graph_resources.nodes[GRAPH_MODE_INSPECT].clear();
			int node_idx_0 = getData().graph_resources.node_find(GRAPH_MODE_NORMAL, getData().graph_resources.n_discs);
			if (node_idx_0 == -1) {
				cerr << "history vector element not found 0" << endl;
				return;
			}
			getData().graph_resources.nodes[GRAPH_MODE_INSPECT].emplace_back(getData().graph_resources.nodes[GRAPH_MODE_NORMAL][node_idx_0]);
			node_idx = getData().graph_resources.node_find(getData().graph_resources.put_mode, getData().graph_resources.n_discs);
		}
		while (node_idx == -1) {
			//cerr << "history vector element not found 1" << endl;
			getData().graph_resources.n_discs += getData().graph_resources.delta;
			node_idx = getData().graph_resources.node_find(GRAPH_MODE_NORMAL, getData().graph_resources.n_discs);
		}
		if (getData().history_elem.board != getData().graph_resources.nodes[getData().graph_resources.put_mode][node_idx].board) {
			stop_calculating();
			resume_calculating();
		}
		getData().history_elem = getData().graph_resources.nodes[getData().graph_resources.put_mode][node_idx];
	}

	void move_processing(int_fast8_t cell) {
		int parent_idx = getData().graph_resources.node_find(getData().graph_resources.put_mode, getData().history_elem.board.n_discs());
		if (parent_idx != -1) {
			if (getData().graph_resources.nodes[getData().graph_resources.put_mode][parent_idx].next_policy == HW2_M1 - cell && parent_idx + 1 < (int)getData().graph_resources.nodes[getData().graph_resources.put_mode].size()) {
				++getData().graph_resources.n_discs;
				return;
			}
			getData().graph_resources.nodes[getData().graph_resources.put_mode][parent_idx].next_policy = HW2_M1 - cell;
			while (getData().graph_resources.nodes[getData().graph_resources.put_mode].size() > parent_idx + 1) {
				getData().graph_resources.nodes[getData().graph_resources.put_mode].pop_back();
			}
		}
		Flip flip;
		calc_flip(&flip, &getData().history_elem.board, HW2_M1 - cell);
		getData().history_elem.board.move_board(&flip);
		getData().history_elem.policy = HW2_M1 - cell;
		getData().history_elem.next_policy = -1;
		getData().history_elem.v = GRAPH_IGNORE_VALUE;
		getData().history_elem.level = -1;
		getData().history_elem.player ^= 1;
		if (getData().history_elem.board.get_legal() == 0ULL) {
			getData().history_elem.board.pass();
			getData().history_elem.player ^= 1;
		}
		getData().graph_resources.nodes[getData().graph_resources.put_mode].emplace_back(getData().history_elem);
		getData().graph_resources.n_discs++;
		reset_hint();
	}

	void interact_move() {
		uint64_t legal = getData().history_elem.board.get_legal();
		for (int_fast8_t cell = 0; cell < HW2; ++cell) {
			if (1 & (legal >> (HW2_M1 - cell))) {
				int x = cell % HW;
				int y = cell / HW;
				Rect cell_rect(BOARD_SX + x * BOARD_CELL_SIZE, BOARD_SY + y * BOARD_CELL_SIZE, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
				if (cell_rect.leftClicked()) {
					if (getData().graph_resources.put_mode == GRAPH_MODE_NORMAL) {
						int parent_idx = getData().graph_resources.node_find(GRAPH_MODE_NORMAL, getData().history_elem.board.n_discs());
						if (parent_idx != -1) {
							bool go_to_inspection_mode =
								getData().history_elem.board.n_discs() != getData().graph_resources.nodes[GRAPH_MODE_NORMAL][getData().graph_resources.nodes[GRAPH_MODE_NORMAL].size() - 1].board.n_discs() &&
								HW2_M1 - cell != getData().graph_resources.nodes[GRAPH_MODE_NORMAL][parent_idx].next_policy;
							if (go_to_inspection_mode) {
								getData().graph_resources.put_mode = GRAPH_MODE_INSPECT;
							}
						}
					}
					stop_calculating();
					move_processing(cell);
					resume_calculating();
				}
			}
		}
	}

	void ai_move() {
		uint64_t legal = getData().history_elem.board.get_legal();
		if (!ai_status.ai_thinking) {
			if (legal) {
				ai_status.ai_future = async(launch::async, ai, getData().history_elem.board, getData().menu_elements.level, getData().menu_elements.use_book, true, true);
				ai_status.ai_thinking = true;
			}
		}
		else if (ai_status.ai_future.valid()) {
			if (ai_status.ai_future.wait_for(chrono::seconds(0)) == future_status::ready) {
				Search_result search_result = ai_status.ai_future.get();
				if (1 & (legal >> search_result.policy)) {
					int sgn = getData().history_elem.player == 0 ? 1 : -1;
					move_processing(HW2_M1 - search_result.policy);
					getData().graph_resources.nodes[getData().graph_resources.put_mode].back().v = sgn * search_result.value;
					getData().graph_resources.nodes[getData().graph_resources.put_mode].back().level = getData().menu_elements.level;
				}
				ai_status.ai_thinking = false;
			}
		}
	}

	void update_opening() {
		string new_opening = opening.get(getData().history_elem.board, getData().history_elem.player ^ 1);
		if (new_opening.size() && getData().history_elem.opening_name != new_opening) {
			getData().history_elem.opening_name = new_opening;
			int node_idx = getData().graph_resources.node_find(getData().graph_resources.put_mode, getData().graph_resources.n_discs);
			if (node_idx == -1) {
				cerr << "history vector element not found 2" << endl;
				return;
			}
			getData().graph_resources.nodes[getData().graph_resources.put_mode][node_idx].opening_name = new_opening;
		}
	}

	Menu create_menu(Menu_elements* menu_elements) {
		Menu menu;
		menu_title title;
		menu_elem menu_e, side_menu, side_side_menu;
		Font menu_font = getData().fonts.font12;



		title.init(language.get("play", "game"));

		menu_e.init_button(language.get("play", "new_game"), &menu_elements->start_game);
		title.push(menu_e);
		menu_e.init_button(language.get("play", "analyze"), &menu_elements->analyze);
		title.push(menu_e);

		menu.push(title);




		title.init(language.get("settings", "settings"));

		menu_e.init_check(language.get("ai_settings", "use_book"), &menu_elements->use_book, menu_elements->use_book);
		title.push(menu_e);
		menu_e.init_bar(language.get("ai_settings", "level"), &menu_elements->level, menu_elements->level, 0, 60);
		title.push(menu_e);
		menu_e.init_bar(language.get("settings", "thread", "thread"), &menu_elements->n_threads, menu_elements->n_threads, 1, 32);
		title.push(menu_e);

		menu_e.init_check(language.get("settings", "play", "ai_put_black"), &menu_elements->ai_put_black, menu_elements->ai_put_black);
		title.push(menu_e);
		menu_e.init_check(language.get("settings", "play", "ai_put_white"), &menu_elements->ai_put_white, menu_elements->ai_put_white);
		title.push(menu_e);

		menu.push(title);




		title.init(language.get("display", "display"));

		menu_e.init_button(language.get("display", "hint", "hint"), &menu_elements->dummy);
		side_menu.init_check(language.get("display", "hint", "disc_value"), &menu_elements->use_disc_hint, menu_elements->use_disc_hint);
		menu_e.push(side_menu);
		side_menu.init_bar(language.get("display", "hint", "disc_value_number"), &menu_elements->n_disc_hint, menu_elements->n_disc_hint, 1, SHOW_ALL_HINT);
		menu_e.push(side_menu);
		side_menu.init_check(language.get("display", "hint", "umigame_value"), &menu_elements->use_umigame_value, menu_elements->use_umigame_value);
		menu_e.push(side_menu);
		title.push(menu_e);

		menu_e.init_check(language.get("display", "legal"), &menu_elements->show_legal, menu_elements->show_legal);
		title.push(menu_e);
		menu_e.init_check(language.get("display", "graph"), &menu_elements->show_graph, menu_elements->show_graph);
		title.push(menu_e);
		menu_e.init_check(language.get("display", "opening_on_cell"), &menu_elements->show_opening_on_cell, menu_elements->show_opening_on_cell);
		title.push(menu_e);
		menu_e.init_check(language.get("display", "log"), &menu_elements->show_log, menu_elements->show_log);
		title.push(menu_e);

		menu.push(title);




		title.init(language.get("operation", "operation"));

		menu_e.init_button(language.get("operation", "stop_calculating"), &menu_elements->stop_calculating);
		title.push(menu_e);

		menu_e.init_button(language.get("operation", "forward"), &menu_elements->forward);
		title.push(menu_e);
		menu_e.init_button(language.get("operation", "backward"), &menu_elements->backward);
		title.push(menu_e);

		menu_e.init_button(language.get("operation", "convert", "convert"), &menu_elements->dummy);
		side_menu.init_button(language.get("operation", "convert", "vertical"), &menu_elements->convert_180);
		menu_e.push(side_menu);
		side_menu.init_button(language.get("operation", "convert", "black_line"), &menu_elements->convert_blackline);
		menu_e.push(side_menu);
		side_menu.init_button(language.get("operation", "convert", "white_line"), &menu_elements->convert_whiteline);
		menu_e.push(side_menu);
		title.push(menu_e);

		menu.push(title);



		title.init(language.get("in_out", "in_out"));

		menu_e.init_button(language.get("in_out", "in"), &menu_elements->dummy);
		side_menu.init_button(language.get("in_out", "input_transcript"), &menu_elements->input_transcript);
		menu_e.push(side_menu);
		side_menu.init_button(language.get("in_out", "input_board"), &menu_elements->input_board);
		menu_e.push(side_menu);
		side_menu.init_button(language.get("in_out", "edit_board"), &menu_elements->edit_board);
		menu_e.push(side_menu);
		side_menu.init_button(language.get("in_out", "input_game"), &menu_elements->input_game);
		menu_e.push(side_menu);
		title.push(menu_e);

		menu_e.init_button(language.get("in_out", "out"), &menu_elements->dummy);
		side_menu.init_button(language.get("in_out", "output_transcript"), &menu_elements->copy_transcript);
		menu_e.push(side_menu);
		side_menu.init_button(language.get("in_out", "output_game"), &menu_elements->save_game);
		menu_e.push(side_menu);
		title.push(menu_e);

		menu.push(title);




		title.init(language.get("book", "book"));

		menu_e.init_button(language.get("book", "import"), &menu_elements->book_import);
		title.push(menu_e);
		menu_e.init_button(language.get("book", "book_reference"), &menu_elements->book_reference);
		title.push(menu_e);
		menu_e.init_button(language.get("book", "settings"), &menu_elements->dummy);
		side_menu.init_bar(language.get("book", "depth"), &menu_elements->book_learn_depth, menu_elements->book_learn_depth, 0, 60);
		menu_e.push(side_menu);
		side_menu.init_bar(language.get("book", "accept"), &menu_elements->book_learn_error, menu_elements->book_learn_error, 0, 64);
		menu_e.push(side_menu);
		title.push(menu_e);
		menu_e.init_button(language.get("book", "start_learn"), &menu_elements->book_start_learn);
		title.push(menu_e);

		menu.push(title);




		title.init(language.get("help", "help"));
		menu_e.init_button(language.get("help", "how_to_use"), &menu_elements->usage);
		title.push(menu_e);
		menu_e.init_button(language.get("help", "bug_report"), &menu_elements->bug_report);
		title.push(menu_e);
		menu_e.init_check(language.get("help", "auto_update_check"), &menu_elements->auto_update_check, menu_elements->auto_update_check);
		title.push(menu_e);
		menu_e.init_button(language.get("help", "license"), &menu_elements->license);
		title.push(menu_e);
		menu.push(title);





		title.init(U"Language");
		for (int i = 0; i < (int)getData().resources.language_names.size(); ++i) {
			menu_e.init_radio(language_name.get(getData().resources.language_names[i]), &menu_elements->languages[i], menu_elements->languages[i]);
			title.push(menu_e);
		}
		menu.push(title);




		menu.init(0, 0, menu_font, getData().resources.checkbox);
		return menu;
	}

	void draw_legal(uint64_t ignore) {
		Flip flip;
		uint64_t legal = getData().history_elem.board.get_legal();
		for (int cell = 0; cell < HW2; ++cell) {
			int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
			int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
			if (1 & (legal >> (HW2_M1 - cell))) {
				if (HW2_M1 - cell == getData().history_elem.next_policy) {
					if (getData().history_elem.player == WHITE) {
						Circle(x, y, DISC_SIZE).draw(ColorF(getData().colors.white, 0.2));
					}
					else {
						Circle(x, y, DISC_SIZE).draw(ColorF(getData().colors.black, 0.2));
					}
				}
				if ((1 & (ignore >> (HW2_M1 - cell))) == 0)
					Circle(x, y, LEGAL_SIZE).draw(getData().colors.cyan);
			}
		}
	}

	void draw_info() {
		if (getData().history_elem.board.get_legal()) {
			getData().fonts.font20(Format(getData().history_elem.board.n_discs() - 3) + language.get("info", "moves")).draw(INFO_SX, INFO_SY);
			if (getData().history_elem.player == BLACK) {
				getData().fonts.font20(language.get("info", "black")).draw(INFO_SX + 100, INFO_SY);
			}
			else {
				getData().fonts.font20(language.get("info", "white")).draw(INFO_SX + 100, INFO_SY);
			}
		}
		else {
			getData().fonts.font20(language.get("info", "game_end")).draw(INFO_SX, INFO_SY);
		}
		getData().fonts.font15(language.get("info", "opening_name") + U": " + Unicode::FromUTF8(getData().history_elem.opening_name)).draw(INFO_SX, INFO_SY + 30);
		Circle(INFO_SX + INFO_DISC_RADIUS, INFO_SY + 75, INFO_DISC_RADIUS).draw(getData().colors.black);
		Circle(INFO_SX + INFO_DISC_RADIUS, INFO_SY + 110, INFO_DISC_RADIUS).draw(getData().colors.white);
		if (getData().history_elem.player == BLACK) {
			getData().fonts.font20(getData().history_elem.board.count_player()).draw(Arg::leftCenter(INFO_SX + 40, INFO_SY + 75));
			getData().fonts.font20(getData().history_elem.board.count_opponent()).draw(Arg::leftCenter(INFO_SX + 40, INFO_SY + 110));
		}
		else {
			getData().fonts.font20(getData().history_elem.board.count_opponent()).draw(Arg::leftCenter(INFO_SX + 40, INFO_SY + 75));
			getData().fonts.font20(getData().history_elem.board.count_player()).draw(Arg::leftCenter(INFO_SX + 40, INFO_SY + 110));
		}
		getData().fonts.font15(language.get("common", "level") + Format(getData().menu_elements.level)).draw(INFO_SX, INFO_SY + 135);
		int mid_depth, end_depth;
		get_level_depth(getData().menu_elements.level, &mid_depth, &end_depth);
		getData().fonts.font15(language.get("info", "lookahead_0") + Format(mid_depth) + language.get("info", "lookahead_1")).draw(INFO_SX, INFO_SY + 160);
		getData().fonts.font15(language.get("info", "complete_0") + Format(end_depth) + language.get("info", "complete_1")).draw(INFO_SX, INFO_SY + 185);
	}

	uint64_t draw_hint() {
		uint64_t res = 0ULL;
		if (ai_status.hint_available) {
			vector<Hint_info> hint_infos;
			for (int cell = 0; cell < HW2; ++cell) {
				if (ai_status.hint_use_stable[cell]) {
					Hint_info hint_info;
					hint_info.value = ai_status.hint_values_stable[cell];
					hint_info.cell = cell;
					hint_info.type = ai_status.hint_types_stable[cell];
					hint_infos.emplace_back(hint_info);
				}
			}
			sort(hint_infos.begin(), hint_infos.end(), compare_hint_info);
			if (hint_infos.size()) {
				int sgn = getData().history_elem.player == 0 ? 1 : -1;
				int node_idx = getData().graph_resources.node_find(getData().graph_resources.put_mode, getData().graph_resources.n_discs);
				if (node_idx != -1) {
					if (getData().graph_resources.nodes[getData().graph_resources.put_mode][node_idx].level < hint_infos[0].type) {
						getData().graph_resources.nodes[getData().graph_resources.put_mode][node_idx].v = sgn * (int)round(hint_infos[0].value);
						getData().graph_resources.nodes[getData().graph_resources.put_mode][node_idx].level = hint_infos[0].type;
					}
				}
			}
			int n_disc_hint = min((int)hint_infos.size(), getData().menu_elements.n_disc_hint);
			for (int i = 0; i < n_disc_hint; ++i) {
				int sx = BOARD_SX + (hint_infos[i].cell % HW) * BOARD_CELL_SIZE;
				int sy = BOARD_SY + (hint_infos[i].cell / HW) * BOARD_CELL_SIZE;
				Color color = getData().colors.white;
				if (hint_infos[i].value == hint_infos[0].value) {
					color = getData().colors.cyan;
				}
				getData().fonts.font15_bold((int)round(hint_infos[i].value)).draw(sx + 2, sy, color);
				if (hint_infos[i].type == HINT_TYPE_BOOK) {
					getData().fonts.font10(U"book").draw(sx + 2, sy + 16, color);
				}
				else if (hint_infos[i].type > HINT_MAX_LEVEL) {
					getData().fonts.font10(Format(hint_infos[i].type) + U"%").draw(sx + 2, sy + 16, color);
				}
				else {
					getData().fonts.font10(U"Lv." + Format(hint_infos[i].type)).draw(sx + 2, sy + 16, color);
				}
				res |= 1ULL << (HW2_M1 - hint_infos[i].cell);
			}
		}
		return res;
	}

	void draw_opening_on_cell() {
		uint64_t legal = getData().history_elem.board.get_legal();
		for (int cell = 0; cell < HW2; ++cell) {
			int x = HW_M1 - cell % HW;
			int y = HW_M1 - cell / HW;
			Rect cell_rect(BOARD_SX + x * BOARD_CELL_SIZE, BOARD_SY + y * BOARD_CELL_SIZE, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
			if ((1 & (legal >> cell)) && cell_rect.mouseOver()) {
				Flip flip;
				calc_flip(&flip, &getData().history_elem.board, cell);
				string openings = opening_many.get(getData().history_elem.board.move_copy(&flip), getData().history_elem.player);
				if (openings.size()) {
					String opening_name = U" " + Unicode::FromUTF8(openings).replace(U" ", U" \n ");
					Vec2 pos = Cursor::Pos();
					pos.x += 20;
					RectF background_rect = getData().fonts.font15_bold(opening_name).region(pos);
					background_rect.draw(getData().colors.white);
					getData().fonts.font15_bold(opening_name).draw(pos, getData().colors.black);
				}
			}
		}
	}

	void hint_init_calculating() {
		uint64_t legal = getData().history_elem.board.get_legal();
		if (ai_status.hint_level == HINT_NOT_CALCULATING) {
			for (int cell = 0; cell < HW2; ++cell) {
				ai_status.hint_values[cell] = HINT_INIT_VALUE;
				ai_status.hint_use[cell] = (bool)(1 & (legal >> (HW2_M1 - cell)));
				ai_status.hint_types[cell] = HINT_NOT_CALCULATING;
			}
		}
		else {
			ai_status.hint_available = true;
		}
		++ai_status.hint_level;
		vector<pair<int, int>> value_cells;
		for (int cell = 0; cell < HW2; ++cell) {
			if (ai_status.hint_use[cell]) {
				value_cells.emplace_back(make_pair(ai_status.hint_values[cell], cell));
			}
		}
		sort(value_cells.begin(), value_cells.end(), compare_value_cell);
		int n_legal = pop_count_ull(legal);
		int hint_adoption_threshold = getData().menu_elements.n_disc_hint + max(1, n_legal * (getData().menu_elements.level - ai_status.hint_level) / getData().menu_elements.level);
		hint_adoption_threshold = min(hint_adoption_threshold, (int)value_cells.size());
		ai_status.hint_task_stack.clear();
		Board board;
		Flip flip;
		int next_task_size = 0;
		int idx = 0;
		for (pair<int, int>& value_cell : value_cells) {
			if (idx++ >= hint_adoption_threshold) {
				break;
			}
			if (ai_status.hint_types[value_cell.second] != HINT_TYPE_BOOK) {
				++next_task_size;
			}
		}
		ai_status.hint_use_multi_thread = next_task_size < getData().menu_elements.n_threads;
		if (ai_status.hint_level <= 10) {
			ai_status.hint_use_multi_thread = false;
		}
		idx = 0;
		for (pair<int, int>& value_cell : value_cells) {
			if (idx++ >= hint_adoption_threshold) {
				break;
			}
			if (ai_status.hint_types[value_cell.second] != HINT_TYPE_BOOK) {
				board = getData().history_elem.board;
				calc_flip(&flip, &board, (uint_fast8_t)(HW2_M1 - value_cell.second));
				board.move_board(&flip);
				ai_status.hint_task_stack.emplace_back(make_pair(value_cell.second, bind(ai_hint, board, ai_status.hint_level, getData().menu_elements.use_book, ai_status.hint_use_multi_thread, false)));
			}
		}
		ai_status.hint_n_doing_tasks = 0;
		ai_status.hint_calculating = true;
		cerr << "hint search level " << ai_status.hint_level << " n_tasks " << ai_status.hint_task_stack.size() << " multi_threading " << ai_status.hint_use_multi_thread << endl;
	}

	void hint_do_task() {
		if (ai_status.hint_n_doing_tasks > 0) {
			for (int cell = 0; cell < HW2; ++cell) {
				if (ai_status.hint_future[cell].valid()) {
					if (ai_status.hint_future[cell].wait_for(chrono::seconds(0)) == future_status::ready) {
						Search_result search_result = ai_status.hint_future[cell].get();
						if (ai_status.hint_values[cell] == HINT_INIT_VALUE || search_result.is_end_search || search_result.depth == SEARCH_BOOK) {
							ai_status.hint_values[cell] = -search_result.value;
						}
						else {
							ai_status.hint_values[cell] -= 1.2 * search_result.value;
							ai_status.hint_values[cell] /= 2.2;
						}
						if (search_result.depth == SEARCH_BOOK) {
							ai_status.hint_types[cell] = HINT_TYPE_BOOK;
						}
						else if (search_result.is_end_search) {
							ai_status.hint_types[cell] = search_result.probability;
						}
						else {
							ai_status.hint_types[cell] = ai_status.hint_level;
						}
						--ai_status.hint_n_doing_tasks;
					}
				}
			}
		}
		if (ai_status.hint_task_stack.size() == 0 && ai_status.hint_n_doing_tasks == 0) {
			for (int cell = 0; cell < HW2; ++cell) {
				ai_status.hint_use_stable[cell] = ai_status.hint_use[cell];
				ai_status.hint_values_stable[cell] = ai_status.hint_values[cell];
				ai_status.hint_types_stable[cell] = ai_status.hint_types[cell];
			}
			ai_status.hint_calculating = false;
		}
		else if (ai_status.hint_task_stack.size()) {
			int loop_time = 0;
			if (ai_status.hint_use_multi_thread) {
				if (max(1, getData().menu_elements.n_threads / HINT_SINGLE_TASK_N_THREAD) - ai_status.hint_n_doing_tasks > 0) {
					loop_time = min((int)ai_status.hint_task_stack.size(), max(1, getData().menu_elements.n_threads / HINT_SINGLE_TASK_N_THREAD) - ai_status.hint_n_doing_tasks);
				}
				else {
					loop_time = 0;
				}
			}
			else {
				loop_time = min((int)ai_status.hint_task_stack.size(), getData().menu_elements.n_threads - ai_status.hint_n_doing_tasks);
			}
			if (loop_time > 0) {
				loop_time = 1;
				for (int i = 0; i < loop_time; ++i) {
					pair<int, function<Search_result()>> task = ai_status.hint_task_stack.back();
					ai_status.hint_task_stack.pop_back();
					ai_status.hint_future[task.first] = async(launch::async, task.second);
				}
				ai_status.hint_n_doing_tasks += loop_time;
			}
		}
	}

	void init_analyze() {
		ai_status.analyze_task_stack.clear();
		int idx = 0;
		for (History_elem& node : getData().graph_resources.nodes[getData().graph_resources.put_mode]) {
			Analyze_info analyze_info;
			analyze_info.idx = idx++;
			analyze_info.sgn = node.player ? -1 : 1;
			analyze_info.board = node.board;
			ai_status.analyze_task_stack.emplace_back(make_pair(analyze_info, bind(ai, node.board, getData().menu_elements.level, getData().menu_elements.use_book, true, true)));
		}
		cerr << "analyze " << ai_status.analyze_task_stack.size() << " tasks" << endl;
		ai_status.analyzing = true;
		analyze_do_task();
	}

	void analyze_do_task() {
		pair<Analyze_info, function<Search_result()>> task = ai_status.analyze_task_stack.back();
		ai_status.analyze_task_stack.pop_back();
		ai_status.analyze_future[task.first.idx] = async(launch::async, task.second);
		ai_status.analyze_sgn[task.first.idx] = task.first.sgn;
		getData().history_elem.board = task.first.board;
		getData().history_elem.policy = -1;
		getData().history_elem.next_policy = -1;
		getData().history_elem.player = task.first.sgn == 1 ? 0 : 1;
		getData().graph_resources.n_discs = getData().history_elem.board.n_discs();
	}

	void analyze_get_task() {
		if (ai_status.analyze_task_stack.size() == 0) {
			ai_status.analyzing = false;
			getData().history_elem = getData().graph_resources.nodes[getData().graph_resources.put_mode].back();
			getData().graph_resources.n_discs = getData().graph_resources.nodes[getData().graph_resources.put_mode].back().board.n_discs();
			return;
		}
		bool task_finished = false;
		for (int i = 0; i < ANALYZE_SIZE; ++i) {
			if (ai_status.analyze_future[i].valid()) {
				if (ai_status.analyze_future[i].wait_for(chrono::seconds(0)) == future_status::ready) {
					Search_result search_result = ai_status.analyze_future[i].get();
					int value = ai_status.analyze_sgn[i] * search_result.value;
					cerr << i << " " << value << endl;
					getData().graph_resources.nodes[getData().graph_resources.put_mode][i].v = value;
					getData().graph_resources.nodes[getData().graph_resources.put_mode][i].level = getData().menu_elements.level;
					task_finished = true;
				}
			}
		}
		if (task_finished) {
			analyze_do_task();
		}
	}

	void copy_transcript() {
		string transcript;
		int inspect_switch_n_discs = INF;
		if (getData().graph_resources.put_mode == 1) {
			if (inspect_switch_n_discs = getData().graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
				inspect_switch_n_discs = getData().graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs();
			}
			else {
				cerr << "no node found in inspect mode" << endl;
			}
		}
		for (History_elem& history_elem : getData().graph_resources.nodes[GRAPH_MODE_NORMAL]) {
			if (history_elem.board.n_discs() >= inspect_switch_n_discs || history_elem.board.n_discs() >= getData().history_elem.board.n_discs()) {
				break;
			}
			if (history_elem.policy != -1) {
				transcript += idx_to_coord(history_elem.policy);
			}
		}
		if (inspect_switch_n_discs != INF) {
			for (History_elem& history_elem : getData().graph_resources.nodes[GRAPH_MODE_INSPECT]) {
				if (history_elem.board.n_discs() >= getData().history_elem.board.n_discs()) {
					break;
				}
				if (history_elem.policy != -1) {
					transcript += idx_to_coord(history_elem.policy);
				}
			}
		}
		cerr << transcript << endl;
		Clipboard::SetText(Unicode::Widen(transcript));
	}
};
