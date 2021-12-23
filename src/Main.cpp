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
#include "graph.hpp"

using namespace std;

#define final_define_value 100
#define book_define_value -1
#define both_ai_define 100
#define n_accept_define 0
#define exact_define 1
#define graph_font_size 10

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
		bool use_mpc = hw2 - bd.n >= 18 ? true : false;
		if (hw2 - bd.n <= 5) {
			int cells[5];
			pick_vacant(&bd, cells);
			if (bd.n == hw2 - 5)
				res.value = last5(&bd, false, -sc_w, sc_w, cells[0], cells[1], cells[2], cells[3], cells[4]);
			else if (bd.n == hw2 - 4)
				res.value = last4(&bd, false, -sc_w, sc_w, cells[0], cells[1], cells[2], cells[3]);
			else if (bd.n == hw2 - 3)
				res.value = last3(&bd, false, -sc_w, sc_w, cells[0], cells[1], cells[2]);
			else if (bd.n == hw2 - 2)
				res.value = last2(&bd, false, -sc_w, sc_w, cells[0], cells[1]);
			else if (bd.n == hw2 - 1)
				res.value = last1(&bd, false, cells[0]);
			else
				res.value = end_evaluate(&bd);
		} else {
			transpose_table.init_now();
			res.value = nega_scout_final(&bd, false, hw2 - bd.n, -sc_w, sc_w, use_mpc, 1.7);
		}
		res.depth = use_mpc ? hw2 - bd.n : final_define_value;
	} else {
		bool use_mpc = hw2 - bd.n >= 10 ? true : false;
		transpose_table.init_now();
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

inline future<search_result> ai(board bd, int depth, int end_depth, int bd_arr[], int book_accept) {
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
		book_value book_result = book.get_random(&bd, (double)book_accept);
		if (book_result.policy != -1)
			return async(launch::async, book_return, bd, book_result);
	}
	if (bd.n >= hw2 - end_depth)
		return async(launch::async, endsearch, bd, tim());
	return async(launch::async, midsearch, bd, tim(), depth);
}

inline bool check_pass(board *bd) {
	bool not_passed = false;
	for (int i = 0; i < hw2; ++i)
		not_passed |= bd->legal(i);
	if (!not_passed) {
		bd->p = 1 - bd->p;
		not_passed = false;
		for (int i = 0; i < hw2; ++i)
			not_passed |= bd->legal(i);
		if (!not_passed)
			return true;
	}
	return false;
}

inline String coord_translate(int coord) {
	String res;
	res << (char)((int)'a' + coord % hw);
	res << char('1' + coord / hw);
	return res;
}

void Main() {
	Size window_size = Size(1000, 700);
	Window::Resize(window_size);
	Window::SetTitle(U"Egaroucid5");
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
	Font score_ui(40);
	Font record_ui(20);
	Font value_ui(30);
	Font change_book_ui(20);
	Font input_board_ui(20);
	Font graph_font(graph_font_size);
	Font move_font(30);
	bool playing = false, thinking = false, cell_value_thinking = false, changing_book = false;
	int depth, end_depth, ai_player, cell_value_depth, cell_value_end_depth, book_accept, show_cell_value, show_value, n_moves = 0;
	double value;
	String change_book_value_str = U"";
	String change_book_value_info_str = U"修正した評価値";
	String change_book_value_coord_str = U"";
	int change_book_coord = -1;
	ai_player = 0;
	show_cell_value = 0;
	show_value = 0;
	book_accept = 0;
	String record = U"";
	vector<board> board_history;
	vector<int> last_played;
	bool finished = false;
	int input_board_state = 0;
	double depth_double = 12, end_depth_double = 20, cell_value_depth_double = 10, cell_value_end_depth_double = 18, book_accept_double = 2;

	const Font pulldown_font{15};
	const Array<String> player_items = { U"先手", U"後手", U"人間同士", U"AI同士"};
	const Array<String> hint_items = {U"ヒントあり", U"ヒントなし"};
	const Array<String> value_items = { U"評価値表示", U"評価値非表示"};
	Pulldown pulldown_player{ player_items, pulldown_font, Point{145, 0}, ai_player};
	Pulldown pulldown_hint{hint_items, pulldown_font, Point{235, 0}, show_cell_value};
	Pulldown pulldown_value{value_items, pulldown_font, Point{340, 0}, show_value};

	Graph graph;
	graph.sx = 550;
	graph.sy = 220;
	graph.size_x = 420;
	graph.size_y = 300;
	graph.resolution = 10;
	graph.font_size = graph_font_size;
	graph.font = graph_font;

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
		cell_value_state[i] = 0;

	while (System::Update()) {

		cell_value_thinking = false;
		for (int i = 0; i < hw2; ++i)
			cell_value_thinking = cell_value_thinking || (cell_value_state[i] == 1);
		SimpleGUI::Slider(U"中盤{:.0f}手読み"_fmt(depth_double), depth_double, 1, 60, Vec2(600, 5), 150, 250, !thinking);
		depth = round(depth_double);
		SimpleGUI::Slider(U"終盤{:.0f}空読み"_fmt(end_depth_double), end_depth_double, 1, 60, Vec2(600, 40), 150, 250, !thinking);
		end_depth = round(end_depth_double);
		SimpleGUI::Slider(U"ヒント中盤{:.0f}手読み"_fmt(cell_value_depth_double), cell_value_depth_double, 1, 60, Vec2(550, 75), 200, 250, !cell_value_thinking);
		cell_value_depth = round(cell_value_depth_double);
		SimpleGUI::Slider(U"ヒント終盤{:.0f}空読み"_fmt(cell_value_end_depth_double), cell_value_end_depth_double, 1, 60, Vec2(550, 110), 200, 250, !cell_value_thinking);
		cell_value_end_depth = round(cell_value_end_depth_double);
		SimpleGUI::Slider(U"book誤差{:.0f}石"_fmt(book_accept_double), book_accept_double, 0, 60, Vec2(550, 145), 200, 250);
		book_accept = round(book_accept_double);

		for (const auto& cell : cells)
			cell.stretched(-1).draw(Palette::Green);
		for (int i = 0; i < hw; ++i)
			coord_ui((char)('a' + i)).draw(offset_x + i * cell_hw + 10, offset_y - cell_hw);
		for (int i = 0; i < hw; ++i)
			coord_ui(i + 1).draw(offset_x - cell_hw, offset_y + i * cell_hw);
		Circle(offset_x + 2 * cell_hw, offset_y + 2 * cell_hw, 5).draw(Palette::Black);
		Circle(offset_x + 2 * cell_hw, offset_y + 6 * cell_hw, 5).draw(Palette::Black);
		Circle(offset_x + 6 * cell_hw, offset_y + 2 * cell_hw, 5).draw(Palette::Black);
		Circle(offset_x + 6 * cell_hw, offset_y + 6 * cell_hw, 5).draw(Palette::Black);

		if (SimpleGUI::Button(U"局面入力", Vec2(500, 600))) {
			String board_str;
			if (!Clipboard::GetText(board_str)) {
				input_board_state = 1;
			}
			else {
				bool flag = true;
				int player = -1;
				if (board_str.size() != hw2 + 1) {
					flag = false;
				}
				else {
					for (int i = 0; i < hw2; ++i) {
						if (board_str[i] == '0')
							bd_arr[i] = black;
						else if (board_str[i] == '1')
							bd_arr[i] = white;
						else if (board_str[i] == '.')
							bd_arr[i] = vacant;
						else {
							flag = false;
							break;
						}
					}
					if (board_str[hw2] == '0')
						player = 0;
					else if (board_str[hw2] == '1')
						player = 1;
					else
						flag = false;
				}
				if (flag) {
					bd.translate_from_arr(bd_arr, player);
					input_board_state = 2;
					playing = false;
					record.clear();
				}
				else {
					input_board_state = 1;
				}
			}
		}
		if (input_board_state == 1)
			input_board_ui(U"取得失敗").draw(625, 605);
		else if (input_board_state == 2) {
			input_board_ui(U"取得成功").draw(625, 605);
			for (int y = 0; y < hw; ++y) {
				for (int x = 0; x < hw; ++x) {
					int coord = proc_coord(y, x);
					if (bd_arr[coord] == black)
						stones[coord].draw(Palette::Black);
					else if (bd_arr[coord] == white)
						stones[coord].draw(Palette::White);
				}
			}
		}
		

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
			thinking = false;
			value = 0.0;
			record.clear();
			if (input_board_state != 2) {
				for (int i = 0; i < hw2; ++i)
					bd_arr[i] = first_board[i];
				bd.translate_from_arr(bd_arr, black);
			}
			create_vacant_lst(bd, bd_arr);
			for (int i = 0; i < hw2; ++i)
				cell_value_state[i] = 0;
			change_book_value_str.clear();
			changing_book = false;
			n_moves = 0;
			graph.clear();
			board_history.clear();
			last_played.clear();
			board_history.push_back(bd);
			last_played.push_back(-1);
			finished = false;
			input_board_state = 0;
		}

		record_ui(record).draw(0, 550);

		if (SimpleGUI::Button(U"棋譜コピー", Vec2(0, 50))) {
			String record_copy = record;
			record_copy.replace(U"\n", U"");
			Clipboard::SetText(record_copy);
		}

		if (playing) {
			if (n_moves != 60)
				move_font(n_moves + 1, U"手目").draw(420, 650);
			else
				move_font(U"終局").draw(420, 650);
			bool flag = false;
			for (int i = 0; i < hw2; ++i)
				flag |= (cell_value_state[i] == 1);
			if (flag) {
				SimpleGUI::Button(U"<", Vec2(550, 650), 50, false);
				SimpleGUI::Button(U">", Vec2(600, 650), 50, false);
			} else {
				if (SimpleGUI::Button(U"<", Vec2(550, 650), 50)) {
					if (n_moves >= 1)
						--n_moves;
					bd = board_history[n_moves];
					bd.translate_to_arr(bd_arr);
					create_vacant_lst(bd, bd_arr);
					for (int i = 0; i < hw2; ++i)
						cell_value_state[i] = 0;
				}
				if (SimpleGUI::Button(U">", Vec2(600, 650), 50)) {
					if (n_moves < board_history.size() - 1)
						++n_moves;
					bd = board_history[n_moves];
					bd.translate_to_arr(bd_arr);
					create_vacant_lst(bd, bd_arr);
					for (int i = 0; i < hw2; ++i)
						cell_value_state[i] = 0;
				}
			}
		}

		if (playing) {
			for (int y = 0; y < hw; ++y) {
				for (int x = 0; x < hw; ++x) {
					int coord = proc_coord(y, x);
					if (bd_arr[coord] == black)
						stones[coord].draw(Palette::Black);
					else if (bd_arr[coord] == white)
						stones[coord].draw(Palette::White);
					if (last_played[n_moves] == coord)
						Circle(cell_center_x[coord % hw], cell_center_y[coord / hw], 5).draw(Palette::Red);
					else if (bd.legal(coord)) {
						if ((bd.p != ai_player && ai_player != both_ai_define) || n_moves != board_history.size() - 1){
							if (cell_value_state[coord] == 0) {
								legals[coord].draw(Palette::Blue);
								if (show_cell_value == 0) {
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
								if (show_cell_value == 0) {
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
							if (cells[coord].leftClicked() && !changing_book && !finished && n_moves == board_history.size() - 1) {
								bd = bd.move(coord);
								++n_moves;
								record += coord_translate(coord);
								String record_copy = record;
								record_copy.replace(U"\n", U"");
								if (record_copy.size() % 40 == 0)
									record += U"\n";
								last_played.push_back(coord);
								bd.translate_to_arr(bd_arr);
								create_vacant_lst(bd, bd_arr);
								finished = check_pass(&bd);
								board_history.push_back(bd);
								for (int i = 0; i < hw2; ++i)
									cell_value_state[i] = 0;
							} else if (cells[coord].rightClicked()) {
								if (changing_book && coord == change_book_coord) {
									if (change_book_value_str.size() == 0) {
										changing_book = false;
									} else {
										int change_book_value = ParseOr<int>(change_book_value_str, -1000);
										if (change_book_value == -1000)
											change_book_value_info_str = U"形式エラー";
										else {
											book.change(bd.move(coord), -change_book_value);
											cell_value_state[coord] = 0;
											change_book_value_str.clear();
											changing_book = false;
										}
									}
								} else {
									change_book_value_str.clear();
									changing_book = true;
									change_book_value_coord_str = coord_translate(coord);
									change_book_coord = coord;
								}
							}
						}
					}
				}
			}
			if ((bd.p == ai_player || ai_player == both_ai_define) && !finished && n_moves == board_history.size() - 1) {
				if (thinking) {
					if (future_result.wait_for(seconds0) == future_status::ready) {
						thinking = false;
						result = future_result.get();
						value = (double)result.value / step;
						record += coord_translate(result.policy);
						if (ai_player == both_ai_define && bd.p == white)
							graph.push(bd.n - 4, -(double)result.value / step);
						else
							graph.push(bd.n - 4, (double)result.value / step);
						bd = bd.move(result.policy);
						++n_moves;
						String record_copy = record;
						record_copy.replace(U"\n", U"");
						if (record_copy.size() % 40 == 0)
							record += U"\n";
						last_played.push_back(result.policy);
						bd.translate_to_arr(bd_arr);
						create_vacant_lst(bd, bd_arr);
						finished = check_pass(&bd);
						board_history.push_back(bd);
					}
				} else {
					thinking = true;
					future_result = ai(bd, depth, end_depth, bd_arr, book_accept);
				}
			}
			score_ui(U"黒 ", bd.count(black), U" ", bd.count(white), U" 白").draw(10, 640);
			if (show_value == 0)
				value_ui(U"評価値: ", round(value)).draw(250, 650);
		}

		if (changing_book) {
			if (Key0.down() || KeyNum0.down()) {
				change_book_value_str += U"0";
				change_book_value_info_str = U"修正した評価値";
			} else if (Key1.down() || KeyNum1.down()) {
				change_book_value_str += U"1";
				change_book_value_info_str = U"修正した評価値";
			} else if (Key2.down() || KeyNum2.down()) {
				change_book_value_str += U"2";
				change_book_value_info_str = U"修正した評価値";
			} else if (Key3.down() || KeyNum3.down()) {
				change_book_value_str += U"3";
				change_book_value_info_str = U"修正した評価値";
			} else if (Key4.down() || KeyNum4.down()) {
				change_book_value_str += U"4";
				change_book_value_info_str = U"修正した評価値";
			} else if (Key5.down() || KeyNum5.down()) {
				change_book_value_str += U"5";
				change_book_value_info_str = U"修正した評価値";
			} else if (Key6.down() || KeyNum6.down()) {
				change_book_value_str += U"6";
				change_book_value_info_str = U"修正した評価値";
			} else if (Key7.down() || KeyNum7.down()) {
				change_book_value_str += U"7";
				change_book_value_info_str = U"修正した評価値";
			} else if (Key8.down() || KeyNum8.down()) {
				change_book_value_str += U"8";
				change_book_value_info_str = U"修正した評価値";
			} else if (Key9.down() || KeyNum9.down()) {
				change_book_value_str += U"9";
				change_book_value_info_str = U"修正した評価値";
			} else if (KeyMinus.down()) {
				change_book_value_str += U"-";
				change_book_value_info_str = U"修正した評価値";
			} else if (KeyBackspace.down()) {
				if (change_book_value_str.size())
					change_book_value_str.pop_back();
				change_book_value_info_str = U"修正した評価値";
			}
			change_book_ui(change_book_value_info_str, U"(", change_book_value_coord_str, U"): ", change_book_value_str).draw(670, 660);
		}

		pulldown_hint.update();
		pulldown_hint.draw();
		show_cell_value = pulldown_hint.getIndex();

		pulldown_player.update();
		pulldown_player.draw();

		pulldown_value.update();
		pulldown_value.draw();
		show_value = pulldown_value.getIndex();

		if (show_value == 0)
			graph.draw();
		
	}
}
