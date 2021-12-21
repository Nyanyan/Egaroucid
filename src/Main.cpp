#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "book.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "book.hpp"
#if USE_MULTI_THREAD
	#include "thread_pool.hpp"
#endif
#include "pulldown.hpp"

using namespace std;

#define final_define_value 100
#define book_define_value -1
#define end_game_define_player 2
#define both_ai_define 100

struct cell_value {
	int value;
	int depth;
};

inline void init() {
	board_init();
	search_init();
	transpose_table_init();
	evaluate_init();
	#if !MPC_MODE && !EVAL_MODE && USE_BOOK
		book_init();
	#endif
	#if USE_MULTI_THREAD
		thread_pool_init();
	#endif
}

inline void create_vacant_lst(board bd, int bd_arr[]) {
	vacant_lst.clear();
	for (int i = 0; i < hw2; ++i) {
		if (bd_arr[i] == vacant)
			vacant_lst.push_back(i);
	}
	if (bd.n < hw2_m1)
		sort(vacant_lst.begin(), vacant_lst.end(), cmp_vacant);
}

cell_value cell_value_search(board bd, int depth, int end_depth) {
	cell_value res;
	int value = book.get(&bd);
	if (value != -inf) {
		res.value = value;
		res.depth = book_define_value;
	}else if (hw2 - bd.n <= end_depth) {
		bool use_mpc = hw2 - bd.n >= 16 ? true : false;
		res.value = nega_scout_final(&bd, false, hw2 - bd.n, -sc_w, sc_w, use_mpc, 1.7);
		res.depth = use_mpc ? hw2 - bd.n : final_define_value;
	} else {
		bool use_mpc = hw2 - bd.n >= 10 ? true : false;
		res.value = nega_scout(&bd, false, depth, -sc_w, sc_w, use_mpc, 1.3);
		res.depth = depth;
	}
	return res;
}

inline future<cell_value> calc_value(board bd, int policy, int depth, int end_depth) {
	board nb = bd.move(policy);
	return async(launch::async, cell_value_search, nb, depth, end_depth);
}

inline int proc_coord(int y, int x) {
	return y * hw + x;
}

search_result return_result(search_result result) {
	return result;
}

search_result book_return(board bd, book_value book_result) {
	search_result res;
	res.policy = book_result.policy;
	res.value = book_result.value;
	return res;
}

inline future<search_result> ai(board bd, int depth, int end_depth, int bd_arr[]) {
	constexpr int first_moves[4] = { 19, 26, 37, 44 };
	int policy;
	search_result result;
	if (bd.n == 4) {
		policy = first_moves[myrandrange(0, 4)];
		result.policy = policy;
		result.value = 0;
		result.depth = 0;
		result.nps = 0;
		return async(launch::async, return_result, result);
	}
	if (bd.n < book_stones) {
		book_value book_result = book.get_half_random(&bd);
		if (book_result.policy != -1) {
			return async(launch::async, book_return, bd, book_result);
		}
	}
	if (bd.n >= hw2 - end_depth)
		return async(launch::async, endsearch, bd, tim());
	return async(launch::async, midsearch, bd, tim(), depth);
}

inline void check_pass(board *bd) {
	bool not_passed = false;
	for (int i = 0; i < hw2; ++i)
		not_passed |= bd->legal(i);
	if (!not_passed) {
		bd->p = 1 - bd->p;
		not_passed = false;
		for (int i = 0; i < hw2; ++i)
			not_passed |= bd->legal(i);
		if (!not_passed)
			bd->p = end_game_define_player;
	}
}

void Main() {

	constexpr int offset_y = 150;
	constexpr int offset_x = 60;
	constexpr int cell_hw = 50;
	constexpr Size cell_size{cell_hw, cell_hw};
	constexpr int stone_size = 20;
	constexpr int legal_size = 5;
	constexpr int first_board[hw2] = {
		vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
		vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
		vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
		vacant,vacant,vacant,white,black,vacant,vacant,vacant,
		vacant,vacant,vacant,black,white,vacant,vacant,vacant,
		vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
		vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
		vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant
	};
	constexpr chrono::duration<int> seconds0 = chrono::seconds(0);

	init();

	Array<Rect> cells;
	Array<Circle> stones, legals;
	vector<int> cell_center_y, cell_center_x;
	board bd;
	int bd_arr[hw2];
	future<search_result> future_result;
	search_result result;
	future<cell_value> future_cell_values[hw2];
	int cell_value_state[hw2];
	int cell_values[hw2];
	int cell_depth[hw2];
	Font cell_value_font(18);
	Font cell_depth_font(12);
	Font coord_ui(40);
	bool playing = false, thinking = false;
	int last_played = -1;
	int depth, end_depth, ai_player, cell_value_depth, cell_value_end_depth;
	bool show_cell_value;
	depth = 12;
	end_depth = 22;
	cell_value_depth = 12;
	cell_value_end_depth = 20;
	ai_player = 0;
	show_cell_value = true;

	const Font pulldown_font{15};
	const Array<String> player_items = { U"先手", U"後手", U"人間同士", U"AI同士"};
	const Array<String> hint_items = {U"ヒントあり", U"ヒントなし"};
	Pulldown pulldown_player{player_items, pulldown_font, Point{125, 0}};
	Pulldown pulldown_hint{ hint_items, pulldown_font, Point{215, 0}};

	for (int i = 0; i < hw; ++i) {
		cell_center_y.push_back(offset_y + i * cell_size.y + cell_hw / 2);
		cell_center_x.push_back(offset_x + i * cell_size.y + cell_hw / 2);
	}

	for (int i = 0; i < hw2; ++i) {
		stones << Circle{ cell_center_x[i % hw], cell_center_y[i / hw], stone_size };
		legals << Circle{ cell_center_x[i % hw], cell_center_y[i / hw], legal_size };
		cells << Rect{ (offset_x + (i % hw) * cell_size.x), (offset_y + (i / hw) * cell_size.y), cell_size};
	}

	while (System::Update()) {

		if (SimpleGUI::Button(U"対局開始", Vec2(0, 0))) {
			int player_idx = pulldown_player.getIndex();
			if (player_idx == 0)
				ai_player = 1;
			else if (player_idx == 1)
				ai_player = 0;
			else if (player_idx == 2)
				ai_player = -1;
			else
				ai_player = both_ai_define;
			playing = true;
			for (int i = 0; i < hw2; ++i)
				bd_arr[i] = first_board[i];
			bd.translate_from_arr(bd_arr, black);
			create_vacant_lst(bd, bd_arr);
			for (int i = 0; i < hw2; ++i)
				cell_value_state[i] = 0;
			last_played = -1;
		}

		for (const auto& cell : cells)
			cell.stretched(-1).draw(Palette::Green);
		for (int i = 0; i < hw; ++i)
			coord_ui((char)('A' + i)).draw(offset_x + i * cell_hw + 10, offset_y - cell_hw);
		for (int i = 0; i < hw; ++i)
			coord_ui(i + 1).draw(offset_x - cell_hw, offset_y + i * cell_hw);
		Circle(offset_x + 2 * cell_hw, offset_y + 2 * cell_hw, 5).draw(Palette::Black);
		Circle(offset_x + 2 * cell_hw, offset_y + 6 * cell_hw, 5).draw(Palette::Black);
		Circle(offset_x + 6 * cell_hw, offset_y + 2 * cell_hw, 5).draw(Palette::Black);
		Circle(offset_x + 6 * cell_hw, offset_y + 6 * cell_hw, 5).draw(Palette::Black);

		if (playing) {
			for (int y = 0; y < hw; ++y) {
				for (int x = 0; x < hw; ++x) {
					int coord = proc_coord(y, x);
					if (bd_arr[coord] == black)
						stones[coord].draw(Palette::Black);
					else if (bd_arr[coord] == white)
						stones[coord].draw(Palette::White);
					if (last_played == coord)
						Circle(cell_center_x[coord % hw], cell_center_y[coord / hw], 5).draw(Palette::Red);
					else if (bd.legal(coord)) {
						if (bd.p != ai_player && bd.p != end_game_define_player && ai_player != both_ai_define) {
							if (cell_value_state[coord] == 0) {
								legals[coord].draw(Palette::Blue);
								if (show_cell_value) {
									future_cell_values[coord] = calc_value(bd, coord, cell_value_depth, cell_value_end_depth);
									cell_value_state[coord] = 1;
								}
							}
							else if (cell_value_state[coord] == 1) {
								legals[coord].draw(Palette::Blue);
								if (future_cell_values[coord].wait_for(seconds0) == future_status::ready) {
									cell_value cell_value_result = future_cell_values[coord].get();
									cell_values[coord] = round(-(double)cell_value_result.value / step);
									cell_depth[coord] = cell_value_result.depth;
									cell_value_state[coord] = 2;
								}
							}
							else if (cell_value_state[coord] == 2) {
								if (show_cell_value) {
									cell_value_font(cell_values[coord]).draw(offset_x + (coord % hw) * cell_hw + 2, offset_y + (coord / hw) * cell_hw);
									if (cell_depth[coord] == final_define_value)
										cell_depth_font(cell_depth[coord], U"%").draw(offset_x + (coord % hw) * cell_hw + 2, offset_y + (coord / hw) * cell_hw + 21);
									else if (cell_depth[coord] == book_define_value)
										cell_depth_font(U"book").draw(offset_x + (coord % hw) * cell_hw + 2, offset_y + (coord / hw) * cell_hw + 21);
									else
										cell_depth_font(cell_depth[coord], U"手").draw(offset_x + (coord % hw) * cell_hw + 2, offset_y + (coord / hw) * cell_hw + 21);
								}
								else
									legals[coord].draw(Palette::Blue);
							}
							if (cells[coord].leftClicked()) {
								bd = bd.move(coord);
								last_played = coord;
								bd.translate_to_arr(bd_arr);
								create_vacant_lst(bd, bd_arr);
								check_pass(&bd);
								for (int i = 0; i < hw2; ++i)
									cell_value_state[i] = 0;
							}
						}
					}
				}
			}
			if ((bd.p == ai_player || ai_player == both_ai_define) && bd.p != end_game_define_player) {
				if (thinking) {
					if (future_result.wait_for(seconds0) == future_status::ready) {
						thinking = false;
						result = future_result.get();
						//Print << (double)result.value / step;
						bd = bd.move(result.policy);
						last_played = result.policy;
						bd.translate_to_arr(bd_arr);
						create_vacant_lst(bd, bd_arr);
						check_pass(&bd);
						if (bd.p == ai_player)
							future_result = ai(bd, depth, end_depth, bd_arr);
					}
				} else {
					thinking = true;
					future_result = ai(bd, depth, end_depth, bd_arr);
				}
			}
		}

		pulldown_hint.update();
		pulldown_hint.draw();
		if (pulldown_hint.getIndex() == 0)
			show_cell_value = true;
		else
			show_cell_value = false;

		pulldown_player.update();
		pulldown_player.draw();

	}
}
