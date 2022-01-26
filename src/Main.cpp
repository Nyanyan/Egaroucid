#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include <fstream>
#include <sstream>
#include <time.h>
#include <queue>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "book.hpp"
#include "evaluate.hpp"
#include "transpose_table.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "book.hpp"
#include "human_value.hpp"
#include "joseki.hpp"
#include "umigame.hpp"
#if USE_MULTI_THREAD
	#include "thread_pool.hpp"
#endif
#include "gui/pulldown.hpp"
#include "gui/graph.hpp"
#include "gui/menu.hpp"
#include "gui/language.hpp"

using namespace std;

#define search_final_define 100
#define search_book_define -1
#define hint_not_calculated_define 0

#define left_left 20
#define left_center 235
#define left_right 490
#define right_left 510
#define right_center 745
#define right_right 980
#define y_center 360

constexpr Color font_color = Palette::White;
constexpr int board_size = 480;
constexpr int board_sx = left_left, board_sy = y_center - board_size / 2, board_cell_size = board_size / hw, board_cell_frame_width = 2, board_frame_width = 7;
constexpr int stone_size = 25, legal_size = 5;
constexpr int graph_sx = 575, graph_sy = 245, graph_width = 415, graph_height = 345, graph_resolution = 10, graph_font_size = 15;
constexpr Color green = Color(36, 153, 114, 100);

struct cell_value {
	int value;
	int depth;
};

bool ai_init() {
	board_init();
	search_init();
	transpose_table_init();
	if (!evaluate_init())
		return false;
	if (!book_init())
		return false;
	if (!joseki_init())
		return false;
	if (!human_value_init())
		return false;
	thread_pool.resize(16);
	return true;
}

bool lang_initialize() {
	return language.init();
}

Menu create_menu(bool *start_game_flag,
	bool *use_ai_flag, bool *human_first, bool *human_second, bool *both_ai,
	bool *use_hint_flag, bool *normal_hint, bool *human_hint, bool *umigame_hint,
	bool *use_value_flag,
	bool *start_book_learn_flag) {
	Menu menu;
	menu_title title;
	menu_elem menu_e, side_menu;
	Font menu_font(15);
	Texture checkbox(U"resources/img/checked.png", TextureDesc::Mipped);

	title.init(language.get("play", "game"));

	menu_e.init_button(language.get("play", "new_game"), start_game_flag);
	title.push(menu_e);

	menu.push(title);

	title.init(language.get("settings", "settings"));

	menu_e.init_check(language.get("settings", "ai_play", "ai_play"), use_ai_flag, *use_ai_flag);
	side_menu.init_radio(language.get("settings", "ai_play", "human_first"), human_first, *human_first);
	menu_e.push(side_menu);
	side_menu.init_radio(language.get("settings", "ai_play", "human_second"), human_second, *human_second);
	menu_e.push(side_menu);
	side_menu.init_radio(language.get("settings", "ai_play", "both_ai"), both_ai, *both_ai);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu_e.init_check(language.get("settings", "hint", "hint"), use_hint_flag, *use_hint_flag);
	side_menu.init_check(language.get("settings", "hint", "stone_value"), normal_hint, *normal_hint);
	menu_e.push(side_menu);
	side_menu.init_check(language.get("settings", "hint", "human_value"), human_hint, *human_hint);
	menu_e.push(side_menu);
	side_menu.init_check(language.get("settings", "hint", "umigame_value"), umigame_hint, *umigame_hint);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu_e.init_check(language.get("settings", "value"), use_value_flag, *use_value_flag);
	title.push(menu_e);

	menu.push(title);



	title.init(language.get("book", "book"));

	menu_e.init_button(language.get("book", "learn"), start_book_learn_flag);
	title.push(menu_e);

	menu.push(title);


	menu.init(0, 0, menu_font, checkbox);
	return menu;
}

pair<bool, board> move_board(board b, bool board_clicked[]) {
	board res;
	for (int cell = 0; cell < hw2; ++cell) {
		if (board_clicked[cell]) {
			res = b.move(cell);
			res.check_player();
			return make_pair(true, res);
		}
	}
	return make_pair(false, b);
}

search_result ai_return_result(search_result result) {
	return result;
}

search_result ai_book_return(board bd, book_value book_result) {
	search_result res;
	res.policy = book_result.policy;
	res.value = book_result.value;
	res.depth = -1;
	res.nps = 0;
	cerr << "book policy " << res.policy << " value " << res.value << endl;
	return res;
}

inline future<search_result> ai(board bd, int depth, int end_depth, int book_accept) {
	constexpr int first_moves[4] = { 19, 26, 37, 44 };
	int policy;
	search_result result;
	if (bd.n == 4) {
		policy = first_moves[myrandrange(0, 4)];
		result.policy = policy;
		result.value = 0;
		result.depth = 0;
		result.nps = 0;
		return async(launch::async, ai_return_result, result);
	}
	book_value book_result = book.get_random(&bd, book_accept);
	if (book_result.policy != -1) {
		return async(launch::async, ai_book_return, bd, book_result);
	}
	else if (bd.n >= hw2 - end_depth) {
		return async(launch::async, endsearch, bd, tim(), false);
	}
	return async(launch::async, midsearch, bd, tim(), depth);
}

cell_value hint_search(board bd, int depth, int end_depth) {
	cell_value res;
	res.value = book.get(&bd);
	if (res.value != -inf) {
		res.depth = search_book_define;
	}
	else if (hw2 - bd.n <= end_depth) {
		res.value = endsearch_value(bd, tim()).value;
		res.depth = end_depth >= 21 ? hw2 - bd.n : search_final_define;
	}
	else {
		res.value = midsearch_value(bd, tim(), depth).value;
		res.depth = depth;
	}
	return res;
}

cell_value analyze_search(board bd, int depth, int end_depth) {
	cell_value res;
	int value = book.get(&bd);
	if (value != -inf) {
		res.value = value;
		res.depth = search_book_define;
	}
	else {
		bool has_legal = false;
		for (int cell = 0; cell < hw2; ++cell) {
			if (bd.legal(cell))
				has_legal = true;
		}
		bool passed = !has_legal;
		if (!has_legal) {
			bd.p = 1 - bd.p;
			for (int cell = 0; cell < hw2; ++cell) {
				if (bd.legal(cell))
					has_legal = true;
			}
		}
		if (!has_legal) {
			res.value = -end_evaluate(&bd);
			res.depth = 0;
		}
		else if (hw2 - bd.n <= end_depth) {
			res.value = endsearch(bd, tim(), false).value;
			if (passed)
				res.value = -res.value;
			res.depth = depth >= 21 ? hw2 - bd.n + 1 : search_final_define;
		}
		else {
			res.value = midsearch(bd, tim(), depth).value;
			if (passed)
				res.value = -res.value;
			res.depth = depth + 1;
		}
	}
	return res;
}

inline void create_vacant_lst(board bd) {
	int bd_arr[hw2];
	bd.translate_to_arr(bd_arr);
	vacant_lst.clear();
	for (int i = 0; i < hw2; ++i) {
		if (bd_arr[i] == vacant)
			vacant_lst.push_back(i);
	}
	if (bd.n < hw2_m1)
		sort(vacant_lst.begin(), vacant_lst.end(), cmp_vacant);
}

int find_history_idx(vector<board> history, int history_place) {
	for (int i = 0; i < (int)history.size(); ++i){
		if (history[i].n - 4 == history_place)
			return i;
	}
	return 0;
}

void initialize_draw(future<bool> *f, bool *initializing, bool *initialize_failed, Font font, Font small_font, Texture icon, Texture logo) {
	icon.scaled((double)(left_right - left_left) / icon.width()).draw(left_left, y_center - (left_right - left_left) / 2);
	logo.scaled((double)(left_right - left_left) / logo.width() * 0.75).draw(right_left, y_center - 30);
	if (!(*initialize_failed)) {
		font(U"起動中...").draw(right_left, y_center + font.fontSize(), font_color);
		if (f->wait_for(chrono::seconds(0)) == future_status::ready) {
			if (f->get()) {
				*initializing = false;
			}
			else {
				*initialize_failed = true;
			}
		}
	}
	else {
		small_font(U"起動失敗\nresourcesフォルダを確認してください").draw(right_left, y_center + font.fontSize(), font_color);
	}
}

void lang_initialize_failed_draw(Font font, Font small_font, Texture icon, Texture logo) {
	icon.scaled((double)(left_right - left_left) / icon.width()).draw(left_left, y_center - (left_right - left_left) / 2);
	logo.scaled((double)(left_right - left_left) / logo.width() * 0.75).draw(right_left, y_center - 30);
	small_font(U"言語パックを読み込めませんでした\nresourcesフォルダを確認してください").draw(right_left, y_center + font.fontSize(), font_color);
}

void board_draw(Rect board_cells[], board b, bool use_hint_flag, bool normal_hint, bool human_hint, bool umigame_hint,
	const int hint_state[], const int hint_value[], const int hint_depth[], Font normal_font, Font normal_depth_font) {
	for (int i = 0; i < hw_m1; ++i) {
		Line(board_sx + board_cell_size * (i + 1), board_sy, board_sx + board_cell_size * (i + 1), board_sy + board_cell_size * hw).draw(board_cell_frame_width, Palette::Black);
		Line(board_sx, board_sy + board_cell_size * (i + 1), board_sx + board_cell_size * hw, board_sy + board_cell_size * (i + 1)).draw(board_cell_frame_width, Palette::Black);
	}
	Circle(board_sx + 2 * board_cell_size, board_sy + 2 * board_cell_size, 5).draw(Palette::Black);
	Circle(board_sx + 2 * board_cell_size, board_sy + 6 * board_cell_size, 5).draw(Palette::Black);
	Circle(board_sx + 6 * board_cell_size, board_sy + 2 * board_cell_size, 5).draw(Palette::Black);
	Circle(board_sx + 6 * board_cell_size, board_sy + 6 * board_cell_size, 5).draw(Palette::Black);
	RoundRect(board_sx, board_sy, board_cell_size * hw, board_cell_size * hw, 20).draw(green).drawFrame(0, board_frame_width, Palette::White);
	int board_arr[hw2], max_cell_value = -inf;
	b.translate_to_arr(board_arr);
	for (int cell = 0; cell < hw2; ++cell) {
		int x = board_sx + (cell % hw) * board_cell_size + board_cell_size / 2;
		int y = board_sy + (cell / hw) * board_cell_size + board_cell_size / 2;
		if (board_arr[cell] == black) {
			Circle(x, y, stone_size).draw(Palette::Black);
		}
		else if (board_arr[cell] == white) {
			Circle(x, y, stone_size).draw(Palette::White);
		}
		else if (b.p != vacant) {
			if (b.legal(cell)) {
				if (use_hint_flag && normal_hint && hint_state[cell] >= 2) {
					max_cell_value = max(max_cell_value, hint_value[cell]);
				}
				if (!use_hint_flag || (!normal_hint && !human_hint && !umigame_hint)) {
					Circle(x, y, legal_size).draw(Palette::Blue);
				}
			}
		}
		if (b.policy == cell) {
			Circle(x, y, legal_size).draw(Palette::Red);
		}

	}
	if (use_hint_flag && b.p != vacant) {
		bool hint_shown[hw2];
		for (int i = 0; i < hw2; ++i) {
			hint_shown[i] = false;
		}
		if (normal_hint) {
			for (int cell = 0; cell < hw2; ++cell) {
				if (b.legal(cell)) {
					if (hint_state[cell] >= 2) {
						Color color = Palette::White;
						if (hint_value[cell] == max_cell_value)
							color = Palette::Cyan;
						int x = board_sx + (cell % hw) * board_cell_size + 3;
						int y = board_sy + (cell / hw) * board_cell_size + 3;
						normal_font(hint_value[cell]).draw(x, y, color);
						y += 19;
						if (hint_depth[cell] == search_book_define) {
							normal_depth_font(U"book").draw(x, y, color);
						}
						else if (hint_depth[cell] == search_final_define) {
							normal_depth_font(U"100%").draw(x, y, color);
						}
						else {
							normal_depth_font(hint_depth[cell], U"手").draw(x, y, color);
						}
						hint_shown[cell] = true;
					}
					else {
						int x = board_sx + (cell % hw) * board_cell_size + board_cell_size / 2;
						int y = board_sy + (cell / hw) * board_cell_size + board_cell_size / 2;
						Circle(x, y, legal_size).draw(Palette::Blue);
					}
				}
			}
		}
	}
}

void reset_hint(int hint_state[], future<cell_value> hint_future[]) {
	global_searching = false;
	for (int i = 0; i < hw2; ++i) {
		if (hint_state[i] % 2 == 1) {
			hint_future[i].get();
		}
		hint_state[i] = hint_not_calculated_define;
	}
	global_searching = true;
}

void reset_ai(bool *ai_thinking, future<search_result> *ai_future) {
	if (*ai_thinking) {
		global_searching = false;
		ai_future->get();
		global_searching = true;
		*ai_thinking = false;
	}
}

bool not_finished(board bd) {
	return bd.p == 0 || bd.p == 1;
}

void Main() {
	Size window_size = Size(1000, 720);
	Window::Resize(window_size);
	Window::SetStyle(WindowStyle::Sizable);
	Scene::SetResizeMode(ResizeMode::Keep);
	Window::SetTitle(U"Egaroucid5.3.0");
	System::SetTerminationTriggers(UserAction::NoAction);
	Scene::SetBackground(green);
	Console.open();

	bool start_game_flag;
	bool use_ai_flag = true, human_first = true, human_second = false, both_ai = false;
	bool use_hint_flag = true, normal_hint = true, human_hint = true, umigame_hint = true;
	bool use_value_flag = true;
	bool start_book_learn_flag;
	Texture icon(U"resources/img/icon.png", TextureDesc::Mipped);
	Texture logo(U"resources/img/logo.png", TextureDesc::Mipped);
	Rect board_cells[hw2];
	for (int cell = 0; cell < hw2; ++cell) {
		board_cells[cell] = Rect(board_sx + (cell % hw) * board_cell_size, board_sy + (cell / hw) * board_cell_size, board_cell_size, board_cell_size);
	}
	Font graph_font(graph_font_size);
	Graph graph;
	graph.sx = graph_sx;
	graph.sy = graph_sy;
	graph.size_x = graph_width;
	graph.size_y = graph_height;
	graph.resolution = graph_resolution;
	graph.font = graph_font;
	graph.font_size = graph_font_size;
	Font font50(50);
	Font font20(20);
	Font normal_hint_font(18);
	Font normal_hint_depth_font(10);

	board bd;
	bool board_clicked[hw2];
	vector<board> history, fork_history;
	int history_place = 0;
	bool fork_mode = false;
	int hint_value[hw2], hint_state[hw2], hint_depth[hw2];
	future<cell_value> hint_future[hw2];

	future<bool> initialize_future = async(launch::async, ai_init);
	bool initializing = true, initialize_failed = false;

	int lang_initialized = 0;
	future<bool> lang_initialize_future = async(launch::async, lang_initialize);

	Menu menu;

	future<search_result> ai_future;
	bool ai_thinking = false;
	int ai_value = 0;
	int ai_depth1 = 13, ai_depth2 = 20, ai_book_accept = 0;
	int hint_depth1 = 11, hint_depth2 = 18;

	while (System::Update()) {
		if (System::GetUserActions() & UserAction::CloseButtonClicked) {
			reset_hint(hint_state, hint_future);
			reset_ai(&ai_thinking, &ai_future);
			System::Exit();
		}
		if (initializing) {
			if (lang_initialized == 0) {
				if (lang_initialize_future.wait_for(chrono::seconds(0)) == future_status::ready) {
					if (lang_initialize_future.get()) {
						lang_initialized = 1;
					}
					else {
						lang_initialized = 3;
					}
				}
			}
			else if (lang_initialized == 1) {
				menu = create_menu(&start_game_flag,
						&use_ai_flag, &human_first, &human_second, &both_ai,
						&use_hint_flag, &normal_hint, &human_hint, &umigame_hint,
						&use_value_flag,
						&start_book_learn_flag);
				lang_initialized = 2;
			}
			else if (lang_initialized == 2) {
				initialize_draw(&initialize_future, &initializing, &initialize_failed, font50, font20, icon, logo);
				if (!initializing) {
					bd.reset();
					bd.v = -inf;
					history.emplace_back(bd);
					history_place = 0;
					fork_mode = false;
					for (int i = 0; i < hw2; ++i) {
						hint_state[i] = hint_not_calculated_define;
					}
				}
			}
			else {
				lang_initialize_failed_draw(font50, font20, icon, logo);
			}
		}
		else {
			for (int cell = 0; cell < hw2; ++cell) {
				board_clicked[cell] = board_cells[cell].leftClicked() && !menu.active() && bd.legal(cell);
				if (board_clicked[cell])
					global_searching = false;
			}
			if (start_game_flag) {
				cerr << "reset" << endl;
				bd.reset();
				bd.v = -inf;
				history.clear();
				history.emplace_back(bd);
				fork_history.clear();
				history_place = 0;
				fork_mode = false;
				reset_hint(hint_state, hint_future);
			}
			if (not_finished(bd) && (!use_ai_flag || (human_first && bd.p == black) || (human_second && bd.p == white) || fork_mode)) {
				pair<bool, board> moved_board = move_board(bd, board_clicked);
				if (moved_board.first) {
					bd = moved_board.second;
					bd.check_player();
					bd.v = -inf;
					if (fork_mode) {
						while (fork_history.size()) {
							if (fork_history[fork_history.size() - 1].n >= bd.n) {
								fork_history.pop_back();
							}
							else {
								break;
							}
						}
						fork_history.emplace_back(bd);
					}
					else {
						history.emplace_back(bd);
					}
					history_place = bd.n - 4;
					reset_hint(hint_state, hint_future);
				}
				if (not_finished(bd) && use_hint_flag && normal_hint) {
					for (int cell = 0; cell < hw2; ++cell) {
						if (bd.legal(cell) && hint_state[cell] < hint_depth1 * 2) {
							if (hint_state[cell] % 2 == 0) {
								if (hint_state[cell] == hint_depth1 * 2 - 2) {
									hint_future[cell] = async(launch::async, hint_search, bd.move(cell), hint_depth1, hint_depth2);
								}
								else {
									hint_future[cell] = async(launch::async, hint_search, bd.move(cell), hint_state[cell] / 2 + 1, hint_state[cell] / 2 + 1);
								}
								++hint_state[cell];
							}
							else if (hint_future[cell].wait_for(chrono::seconds(0)) == future_status::ready) {
								cell_value hint_result = hint_future[cell].get();
								if (global_searching) {
									if (hint_state[cell] == 1 || hint_result.depth == search_final_define) {
										hint_value[cell] = -hint_result.value;
									}
									else {
										hint_value[cell] -= hint_result.value;
										hint_value[cell] /= 2;
									}
									hint_depth[cell] = hint_result.depth;
								}
								if (hint_result.depth == search_final_define || hint_result.depth == search_book_define) {
									hint_state[cell] = hint_depth1 * 2;
								}
								else {
									++hint_state[cell];
								}
							}
						}
					}
				}
			}
			else if (not_finished(bd) && history[history.size() - 1].n - 4 == history_place) {
				if (ai_thinking) {
					if (ai_future.wait_for(chrono::seconds(0)) == future_status::ready) {
						search_result ai_result = ai_future.get();
						int sgn = (bd.p ? -1 : 1);
						bd = bd.move(ai_result.policy);
						bd.check_player();
						bd.v = sgn * ai_result.value;
						history.emplace_back(bd);
						history_place = bd.n - 4;
						ai_value = ai_result.value;
						ai_thinking = false;
						reset_hint(hint_state, hint_future);
					}
				}
				else {
					global_searching = true;
					create_vacant_lst(bd);
					ai_future = ai(bd, ai_depth1, ai_depth2, ai_book_accept);
					ai_thinking = true;
				}
			}

			int former_history_place = history_place;
			history_place = graph.update_place(history, fork_history, history_place);
			if (history_place != former_history_place) {
				if (ai_thinking) {
					reset_ai(&ai_thinking, &ai_future);
				}
				if (fork_mode && history_place > fork_history[fork_history.size() - 1].n - 4) {
					fork_history.clear();
					bd = history[find_history_idx(history, history_place)];
					create_vacant_lst(bd);
					reset_hint(hint_state, hint_future);
					if (history_place == history[history.size() - 1].n - 4) {
						fork_mode = false;
					}
					else {
						fork_history.emplace_back(bd);
					}
				}
				else if (!fork_mode || (fork_mode && history_place <= fork_history[0].n - 4)) {
					fork_mode = true;
					fork_history.clear();
					bd = history[find_history_idx(history, history_place)];
					fork_history.emplace_back(bd);
					create_vacant_lst(bd);
					reset_hint(hint_state, hint_future);
				}
				else if (fork_mode) {
					bd = fork_history[find_history_idx(fork_history, history_place)];
					create_vacant_lst(bd);
					reset_hint(hint_state, hint_future);
				}
			}

			board_draw(board_cells, bd, use_hint_flag, normal_hint, human_hint, umigame_hint,
				hint_state, hint_value, hint_depth, normal_hint_font, normal_hint_depth_font);
			graph.draw(history, fork_history, history_place);
		}

		menu.draw();
	}

}
