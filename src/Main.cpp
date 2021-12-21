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

using namespace std;

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

search_result calc_value_end(board nb) {
	search_result res;
	res.value = end_evaluate(&nb);
	return res;
}

search_result cell_value_search(board bd, int depth, int end_depth) {
	if (bd.n >= hw2 - end_depth)
		return endsearch(bd, tim());
	return midsearch(bd, tim(), depth);
}

search_result calc_value_p(board bd, int depth, int end_depth) {
	search_result res;
	bool not_passed = false;
	for (int i = 0; i < hw2; ++i)
		not_passed |= bd.legal(i);
	if (!not_passed) {
		bd.p = 1 - bd.p;
		not_passed = false;
		for (int i = 0; i < hw2; ++i)
			not_passed |= bd.legal(i);
		if (!not_passed) {
			res.value = -end_evaluate(&bd);
			return res;
		} else {
			res = cell_value_search(bd, depth, end_depth);
			res.value = -res.value;
			return res;
		}
	}
	res = cell_value_search(bd, depth, end_depth);
	return res;
}

inline future<search_result> calc_value(board bd, int policy) {
	constexpr int depth = 12;
	constexpr int end_depth = 18;
	board nb = bd.move(policy);
	return async(launch::async, calc_value_p, nb, depth, end_depth);
}

inline int proc_coord(int y, int x) {
	return y * hw + x;
}

search_result return_result(search_result result) {
	return result;
}

search_result book_return(board bd, int policy) {
	search_result res = midsearch(bd, tim(), 7);
	res.policy = policy;
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
		policy = book.get(&bd);
		if (policy != -1) {
			return async(launch::async, book_return, bd, policy);
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
			bd->p = 2;
	}
}

void Main() {
	constexpr int offset_y = 60;
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
	future<search_result> future_cell_values[hw2];
	int calc_cell_values[hw2];
	int cell_values[hw2];
	Font cell_value(15);
	int depth, end_depth, ai_player;
	depth = 10;
	end_depth = 20;
	ai_player = 1;

	for (int i = 0; i < hw; ++i) {
		cell_center_y.push_back(offset_y + i * cell_size.y + cell_hw / 2);
		cell_center_x.push_back(offset_x + i * cell_size.y + cell_hw / 2);
	}

	for (int i = 0; i < hw2; ++i) {
		stones << Circle{ cell_center_x[i % hw], cell_center_y[i / hw], stone_size };
		legals << Circle{ cell_center_x[i % hw], cell_center_y[i / hw], legal_size };
		cells << Rect{ (offset_x + (i % hw) * cell_size.x), (offset_y + (i / hw) * cell_size.y), cell_size};
	}

	for (int i = 0; i < hw2; ++i)
		bd_arr[i] = first_board[i];
	bd.translate_from_arr(bd_arr, black);

	for (int i = 0; i < hw2; ++i)
		calc_cell_values[i] = 0;
	create_vacant_lst(bd, bd_arr);

	if (bd.p == ai_player)
		future_result = ai(bd, depth, end_depth, bd_arr);

	while (System::Update()) {

		for (const auto& cell : cells)
			cell.stretched(-1).draw(Palette::Green);

		for (int y = 0; y < hw; ++y) {
			for (int x = 0; x < hw; ++x) {
				int coord = proc_coord(y, x);
				if (bd_arr[coord] == black)
					stones[coord].draw(Palette::Black);
				else if (bd_arr[coord] == white)
					stones[coord].draw(Palette::White);
				else if (bd.legal(coord)) {
					legals[coord].draw(Palette::Blue);
					if (bd.p != ai_player) {
						if (calc_cell_values[coord] == 0) {
							future_cell_values[coord] = calc_value(bd, coord);
							calc_cell_values[coord] = 1;
						} else if (calc_cell_values[coord] == 1) {
							if (future_cell_values[coord].wait_for(seconds0) == future_status::ready){
								cell_values[coord] = round(-(double)future_cell_values[coord].get().value / step);
								calc_cell_values[coord] = 2;
							}
						} else if (calc_cell_values[coord] == 2) {
							cell_value(cell_values[coord]).draw(offset_x + (coord % hw) * cell_hw, offset_y + (coord / hw) * cell_hw);
						}
						if (cells[coord].leftClicked()) {
							bd = bd.move(coord);
							bd.translate_to_arr(bd_arr);
							create_vacant_lst(bd, bd_arr);
							check_pass(&bd);
							for (int i = 0; i < hw2; ++i)
								calc_cell_values[i] = 0;
							if (bd.p == ai_player)
								future_result = ai(bd, depth, end_depth, bd_arr);
						}
					}
				}
			}
		}

		if (bd.p == ai_player) {
			if (future_result.wait_for(seconds0) == future_status::ready) {
				result = future_result.get();
				Print << (double)result.value / step;
				bd = bd.move(result.policy);
				bd.translate_to_arr(bd_arr);
				create_vacant_lst(bd, bd_arr);
				check_pass(&bd);
				if (bd.p == ai_player)
					future_result = ai(bd, depth, end_depth, bd_arr);
			}
		}

	}
}
