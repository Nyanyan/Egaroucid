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
#include "pulldown.hpp"
#include "graph.hpp"

using namespace std;

#define final_define_value 100
#define book_define_value -1
#define both_ai_define 3
#define n_accept_define 0
#define exact_define 1
#define graph_font_size 15

bool book_learning = false;
int book_depth = 30, book_learn_accept = 10;

struct cell_value {
	int value;
	int depth;
};

inline bool init(String *message) {
	board_init();
	search_init();
	transpose_table_init();
	*message = U"基本関数初期化完了　評価関数初期化中";
	if (!evaluate_init())
		return false;
	*message = U"評価関数初期化完了　book初期化中";
	if (!book_init())
		return false;
	*message = U"book初期化完了　定石初期化中";
	if (!joseki_init())
		return false;
	*message = U"定石初期化完了　人間的評価関数初期化中";
	if (!human_value_init())
		return false;
	*message = U"人間的評価関数初期化完了";
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
	--depth;
	--end_depth;
	cell_value res;
	int value = book.get(&bd);
	if (value != -inf) {
		res.value = value;
		res.depth = book_define_value;
	} else {
		if (hw2 - bd.n <= end_depth) {
			//int g = midsearch_value_nomemo(bd, tim(), min(10, hw2 - bd.n)).value;
			int g = 0;
			res.value = endsearch_value(bd, tim(), g).value;
			res.depth = depth >= 21 ? hw2 - bd.n + 1 : final_define_value;
		} else {
			res.value = midsearch_value(bd, tim(), depth).value;
			res.depth = depth + 1;
		}
	}
	return res;
}

inline future<cell_value> calc_value(board bd, int policy, int depth, int end_depth) {
	board nb = bd.move(policy);
	return async(launch::async, cell_value_search, nb, depth, end_depth);
}

inline future<cell_value> calc_value_nopolicy(board bd, int depth, int end_depth) {
	return async(launch::async, cell_value_search, bd, depth, end_depth);
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
	res.depth = -1;
	res.nps = 0;
	cerr << "book policy " << res.policy << " value " << res.value << endl;
	return res;
}

inline future<search_result> ai(board bd, int depth, int end_depth, int bd_arr[], int book_accept, bool *pre_searched) {
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
	book_value book_result = book.get_random(&bd, book_accept);
	if (book_result.policy != -1)
		return async(launch::async, book_return, bd, book_result);
	if (bd.n >= hw2 - end_depth) {
		/*
		if (!*pre_searched) {
			*pre_searched = true;
			return async(launch::async, endsearch, bd, tim(), false);
		}
		*/
		return async(launch::async, endsearch, bd, tim(), false);
	}
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

inline void import_book(string file) {
	cerr << "book import" << endl;
	bool result = book.import_file(file);
	if (result)
		cerr << "book imported" << endl;
	else
		cerr << "book NOT fully imported" << endl;
}

bool operator< (const pair<int, board>& a, const pair<int, board>& b) {
	if (a.first == b.first)
		return a.second.n > b.second.n;
	return a.first < b.first;
};

inline int get_value(board bd, int depth, int end_depth) {
	int value = book.get(&bd);
	if (value == -inf) {
		transpose_table.init_now();
		if (hw2 - bd.n <= end_depth) {
			int g = midsearch_value_nomemo(bd, tim(), min(10, hw2 - bd.n)).value;
			value = endsearch_value(bd, tim(), g).value;
		} else {
			value = midsearch_value_book(bd, tim(), depth).value;
		}
	}
	return value;
}

void learn_book(board bd, int depth, int end_depth, board *bd_ptr, double *value_ptr) {
	cerr << "start learning book" << endl;
	priority_queue<pair<int, board>> que;
	que.push(make_pair(-abs(get_value(bd, depth, end_depth)), bd));
	pair<int, board> popped;
	int weight, i, j, value;
	board b, nb;
	value = get_value(bd, depth, end_depth);
	bd.copy(bd_ptr);
	*value_ptr = (bd.p ? -1 : 1) * value;
	book.reg(bd, value);
	while (!que.empty()) {
		popped = que.top();
		que.pop();
		weight = -popped.first;
		b = popped.second;
		if (b.n - 4 <= book_depth) {
			for (i = 0; i < hw2; ++i) {
				if (b.legal(i)) {
					nb = b.move(i);
					if (!book_learning)
						return;
					value = get_value(nb, depth, end_depth);
					if (abs(value) <= book_learn_accept) {
						nb.copy(bd_ptr);
						*value_ptr = value;
						book.reg(nb, value);
						if (nb.n - 4 <= book_depth)
							que.push(make_pair(-(weight + abs(value)), nb));
					}
				}
			}
		}
	}
	//book_learning = false;
}

umigame_result get_umigame_p(board b) {
	return umigame.get(&b);
}

future<umigame_result> get_umigame(board b) {
	return async(launch::async, get_umigame_p, b);
}

void Main() {
	Size window_size = Size(1000, 720);
	Window::Resize(window_size);
	Window::SetStyle(WindowStyle::Sizable);
	Scene::SetResizeMode(ResizeMode::Keep);
	Window::SetTitle(U"Egaroucid5.1.0");
	System::SetTerminationTriggers(UserAction::NoAction);
	Scene::SetBackground(Palette::White);
	//Console.open();
	stringstream logger_stream;
	cerr.rdbuf(logger_stream.rdbuf());
	string logger;
	String logger_String;
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
	constexpr int human_hint_sub_depth = 2;
	constexpr int human_hint_depth = 6;

	String initialize_message;
	future<bool> future_initialize = async(launch::async, init, &initialize_message);
	bool initialized = false, initialize_failed = false;

	Array<Rect> cells;
	Array<Circle> stones, legals;
	vector<int> cell_center_y, cell_center_x;
	board bd;
	int bd_arr[hw2];
	future<search_result> future_result;
	search_result result;
	future<cell_value> future_cell_values[hw2];
	future<umigame_result> future_umigame[hw2];
	future<cell_value> future_val;
	bool analysys_start = false;
	int analysys_n_moves = 1000;
	int cell_value_state[hw2];
	int umigame_state[hw2];
	umigame_result umigame_values[hw2];
	int cell_values[hw2];
	int cell_depth[hw2];
	Font cell_value_font(18, Typeface::Bold);
	Font human_value_font(15, Typeface::Bold);
	Font cell_depth_font(10);
	Font umigame_font(12, Typeface::Heavy);
	Font coord_ui(40);
	Font score_ui(40);
	Font record_ui(20);
	Font value_ui(30);
	Font change_book_ui(20);
	Font input_board_record_ui(20);
	Font graph_font(graph_font_size);
	Font move_font(30);
	Font saved_ui(20);
	Font copy_ui(20);
	Font joseki_ui(17);
	Font font40(40);
	Font font30(30);
	Font font20(20);
	Font font15(15);
	Color font_color = Palette::Black;
	bool playing = false, thinking = false, cell_value_thinking = false, changing_book = false;
	int depth, end_depth, ai_player, cell_value_depth, cell_value_end_depth, book_accept, n_moves = 0;
	double value = 0.0;
	String change_book_value_str = U"";
	String change_book_value_info_str = U"修正した評価値";
	String change_book_value_coord_str = U"";
	int change_book_coord = -1;
	ai_player = 0;
	int player_default = 0;
	bool hint_default = true, human_hint_default = true, value_default = true, umigame_default = true;
	book_accept = 0;
	String record = U"";
	vector<board> board_history;
	bool finished = false, copied = false;
	int saved = 0;
	int input_board_state = 0, input_record_state = 0;
	double depth_double = 12, end_depth_double = 20, cell_value_depth_double = 10, cell_value_end_depth_double = 18, book_accept_double = 0, book_depth_double = 40, book_learn_accept_double = 10.0;
	int board_start_moves, finish_moves, max_cell_value = -inf, start_moves = 0;
	bool book_changed = false, book_changing = false, closing = false, pre_searched = false, book_learning_button = false;
	future<void> book_import_future, book_learn_future;
	future<vector<search_result_pv>> human_value_future;
	int human_value_state = 0;
	vector<search_result_pv> human_values;
	bool show_log = true;
	TextEditState black_player, white_player, play_memo;

	const Texture icon(U"resources/icon.png", TextureDesc::Mipped);

	ifstream ifs("resources/settings.txt");
	if (!ifs.fail()) {
		string line;
		getline(ifs, line);
		player_default = stoi(line);
		getline(ifs, line);
		hint_default = stoi(line);
		getline(ifs, line);
		human_hint_default = stoi(line);
		getline(ifs, line);
		value_default = stoi(line);
		getline(ifs, line);
		umigame_default = stoi(line);
		getline(ifs, line);
		depth_double = stof(line);
		getline(ifs, line);
		end_depth_double = stof(line);
		getline(ifs, line);
		cell_value_depth_double = stof(line);
		getline(ifs, line);
		cell_value_end_depth_double = stof(line);
		getline(ifs, line);
		book_accept_double = stof(line);
		getline(ifs, line);
		book_depth_double = stof(line);
		getline(ifs, line);
		book_learn_accept_double = stof(line);
		getline(ifs, line);
		show_log = stoi(line);
	}
	ifs.close();
	depth = round(depth_double);
	end_depth = round(end_depth_double);
	cell_value_depth = round(cell_value_depth_double);
	cell_value_end_depth = round(cell_value_end_depth_double);
	book_accept = round(book_accept_double);

	const Font pulldown_font(20);
	const Array<String> player_items = { U"人間先手", U"人間後手", U"人間同士", U"AI同士"};
	Pulldown pulldown_player{ player_items, pulldown_font, Point{480, 0}, player_default};

	Graph graph;
	graph.sx = 550;
	graph.sy = 200;
	graph.size_x = 420;
	graph.size_y = 260;
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

	for (int i = 0; i < hw2; ++i) {
		cell_value_state[i] = 0;
		umigame_state[i] = 0;
	}

	while (System::Update()) {

		SimpleGUI::CheckBox(show_log, U"ログ表示", Point(0, 687));
		if (getline(logger_stream, logger))
			logger_String = Unicode::Widen(logger);
		else
			logger_stream.clear();
		if (show_log)
			font15(logger_String).draw(150, 695, font_color);


		if (System::GetUserActions() & UserAction::CloseButtonClicked) {
			book_learning = false;
			closing = true;
		}
		if (closing){
			if (book_changed) {
				font40(U"bookが変更されました。保存しますか？").draw(0, 0, font_color);
				if (SimpleGUI::Button(U"保存する", Vec2(0, 150))) {
					book.save();
					ofstream ofs("resources/settings.txt");
					if (!ofs.fail()) {
						ofs << pulldown_player.getIndex() << endl;
						ofs << hint_default << endl;
						ofs << human_hint_default << endl;
						ofs << value_default << endl;
						ofs << umigame_default << endl;
						ofs << depth << endl;
						ofs << end_depth << endl;
						ofs << cell_value_depth << endl;
						ofs << cell_value_end_depth << endl;
						ofs << book_accept << endl;
						ofs << book_depth << endl;
						ofs << book_learn_accept << endl;
						ofs << show_log << endl;
					}
					ofs.close();
					System::Exit();
				}
				if (SimpleGUI::Button(U"保存しない", Vec2(0, 300))) {
					ofstream ofs("resources/settings.txt");
					if (!ofs.fail()) {
						ofs << pulldown_player.getIndex() << endl;
						ofs << hint_default << endl;
						ofs << human_hint_default << endl;
						ofs << value_default << endl;
						ofs << umigame_default << endl;
						ofs << depth << endl;
						ofs << end_depth << endl;
						ofs << cell_value_depth << endl;
						ofs << cell_value_end_depth << endl;
						ofs << book_accept << endl;
						ofs << book_depth << endl;
						ofs << book_learn_accept << endl;
						ofs << show_log << endl;
					}
					ofs.close();
					System::Exit();
				}
			} else {
				ofstream ofs("resources/settings.txt");
				if (!ofs.fail()) {
					ofs << pulldown_player.getIndex() << endl;
					ofs << hint_default << endl;
					ofs << human_hint_default << endl;
					ofs << value_default << endl;
					ofs << umigame_default << endl;
					ofs << depth << endl;
					ofs << end_depth << endl;
					ofs << cell_value_depth << endl;
					ofs << cell_value_end_depth << endl;
					ofs << book_accept << endl;
					ofs << book_depth << endl;
					ofs << book_learn_accept << endl;
					ofs << show_log << endl;
				}
				ofs.close();
				System::Exit();
			}
			continue;
		}

		if (!initialized) {
			if (!initialize_failed) {
				if (future_initialize.wait_for(seconds0) == future_status::ready) {
					if (future_initialize.get())
						initialized = true;
					else
						initialize_failed = true;
				}
			}
			double scale = 500.0 / icon.width();
			icon.scaled(scale).draw(500 - 250, 350 - 250);
			if (!initialize_failed)
				font40(U"AI初期化中…").draw(0, 0, font_color);
			else
				font40(U"AI初期化失敗\n繰り返す場合はresourcesフォルダを確認してください。").draw(0, 0, font_color);
			font30(initialize_message).draw(0, 600, font_color);
			continue;
		}

		if (book_changing) {
			font40(U"book追加中…").draw(0, 0, font_color);
			if (book_import_future.wait_for(seconds0) == future_status::ready) {
				book_changing = false;
				book_import_future.get();
			}
			continue;
		} else if (DragDrop::HasNewFilePaths()) {
			for (const auto& dropped : DragDrop::GetDroppedFilePaths()) {
				book_changed = true;
				book_changing = true;
				book_import_future = async(launch::async, import_book, dropped.path.narrow());
				break;
			}
			continue;
		} else if (const auto status = DragDrop::DragOver()) {
			if (status->itemType == DragItemType::FilePaths) {
				font40(U"ドラッグ&ドロップでbookを追加").draw(0, 0, font_color);
				continue;
			}
		}
		SimpleGUI::CheckBox(value_default, U"評価値", Point(250, 0), 120, !book_learning);
		SimpleGUI::CheckBox(hint_default, U"ヒント", Point(355, 0), 120, !book_learning);
		SimpleGUI::CheckBox(human_hint_default, U"人間的ヒント", Point(305, 35), 170, hint_default && !book_learning);
		SimpleGUI::CheckBox(umigame_default, U"うみがめ数", Point(160, 35), 150, hint_default && !book_learning);

		cell_value_thinking = false;
		for (int i = 0; i < hw2; ++i)
			cell_value_thinking = cell_value_thinking || (cell_value_state[i] == 1);
		end_depth_double = max(end_depth_double, depth_double);
		cell_value_end_depth_double = max(cell_value_end_depth_double, cell_value_depth_double);
		SimpleGUI::Slider(U"中盤{:.0f}手読み"_fmt(depth_double), depth_double, 1, 60, Vec2(600, 0), 150, 250, !thinking && !book_learning);
		SimpleGUI::Slider(U"終盤{:.0f}手読み"_fmt(end_depth_double), end_depth_double, 1, 60, Vec2(600, 35), 150, 250, !thinking && !book_learning);
		SimpleGUI::Slider(U"ヒント中盤{:.0f}手読み"_fmt(cell_value_depth_double), cell_value_depth_double, 1, 60, Vec2(550, 70), 200, 250, !cell_value_thinking && hint_default && !book_learning);
		SimpleGUI::Slider(U"ヒント終盤{:.0f}手読み"_fmt(cell_value_end_depth_double), cell_value_end_depth_double, 1, 60, Vec2(550, 105), 200, 250, !cell_value_thinking && hint_default && !book_learning);
		SimpleGUI::Slider(U"book誤差{:.0f}石"_fmt(book_accept_double), book_accept_double, 0, 64, Vec2(550, 140), 200, 250, !book_learning);
		depth = round(depth_double);
		end_depth = round(end_depth_double);
		cell_value_depth = round(cell_value_depth_double);
		cell_value_end_depth = round(cell_value_end_depth_double);
		book_accept = round(book_accept_double);

		Rect{offset_x, offset_y, cell_size.x * hw, cell_size.y * hw}.draw(Palette::Black);
		for (const auto& cell : cells)
			cell.stretched(-1).draw(Palette::Green);
		for (int i = 0; i < hw; ++i)
			coord_ui((char)('a' + i)).draw(offset_x + i * cell_hw + 10, offset_y - cell_hw - 5, font_color);
		for (int i = 0; i < hw; ++i)
			coord_ui(i + 1).draw(offset_x - cell_hw + 2, offset_y + i * cell_hw - 5, font_color);
		Circle(offset_x + 2 * cell_hw, offset_y + 2 * cell_hw, 5).draw(Palette::Black);
		Circle(offset_x + 2 * cell_hw, offset_y + 6 * cell_hw, 5).draw(Palette::Black);
		Circle(offset_x + 6 * cell_hw, offset_y + 2 * cell_hw, 5).draw(Palette::Black);
		Circle(offset_x + 6 * cell_hw, offset_y + 6 * cell_hw, 5).draw(Palette::Black);

		if (SimpleGUI::Button(U"棋譜入力", Vec2(500, 550), 120, !book_learning)) {
			String record_str;
			if (!Clipboard::GetText(record_str)) {
				input_record_state = 1;
			}
			else {
				bool flag = true;
				String record_tmp = U"";
				board_history.clear();
				if (record_str.size() % 2 != 0) {
					flag = false;
				}
				else {
					int y, x;
					for (int i = 0; i < hw2; ++i)
						bd_arr[i] = first_board[i];
					bd.translate_from_arr(bd_arr, black);
					board_history.push_back(bd);
					for (int i = 0; i < record_str.size(); i += 2) {
						x = (int)record_str[i] - (int)'a';
						if (x < 0 || hw <= x) {
							x = (int)record_str[i] - (int)'A';
							if (x < 0 || hw <= x) {
								flag = false;
								break;
							}
						}
						y = (int)record_str[i + 1] - (int)'1';
						if (y < 0 || hw <= y) {
							flag = false;
							break;
						}
						if (bd.legal(y * hw + x)) {
							bd = bd.move(y * hw + x);
						}
						else {
							bd.p = 1 - bd.p;
							if (bd.legal(y * hw + x)) {
								bd = bd.move(y * hw + x);
							}
							else {
								flag = false;
								break;
							}
						}
						board_history.push_back(bd);
						record_tmp += coord_translate(y * hw + x);
						String record_copy = record_tmp;
						record_copy.replace(U"\n", U"");
						if (record_copy.size() % 40 == 0)
							record_tmp += U"\n";
					}
				}
				if (flag) {
					input_record_state = 2;
					bd.translate_to_arr(bd_arr);
					record = record_tmp;
					playing = false;
				}
				else {
					input_record_state = 1;
					board_history.clear();
					record.clear();
				}
			}
		}
		if (input_record_state == 1){
			input_board_record_ui(U"取得失敗").draw(625, 555, font_color);
			playing = false;
		} else if (input_record_state == 2) {
			input_board_record_ui(U"取得成功").draw(625, 555, font_color);
			for (int y = 0; y < hw; ++y) {
				for (int x = 0; x < hw; ++x) {
					int coord = proc_coord(y, x);
					if (bd_arr[coord] == black)
						stones[coord].draw(Palette::Black);
					else if (bd_arr[coord] == white)
						stones[coord].draw(Palette::White);
				}
			}
			playing = false;
		}

		if (SimpleGUI::Button(U"局面入力", Vec2(500, 600), 120, !book_learning)) {
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
		if (input_board_state == 1){
			input_board_record_ui(U"取得失敗").draw(625, 605, font_color);
			playing = false;
		} else if (input_board_state == 2) {
			input_board_record_ui(U"取得成功").draw(625, 605, font_color);
			for (int i = 0; i < hw2; ++i) {
				cell_value_state[i] = 0;
				umigame_state[i] = 0;
			}
			for (int y = 0; y < hw; ++y) {
				for (int x = 0; x < hw; ++x) {
					int coord = proc_coord(y, x);
					if (bd_arr[coord] == black)
						stones[coord].draw(Palette::Black);
					else if (bd_arr[coord] == white)
						stones[coord].draw(Palette::White);
				}
			}
			playing = false;
		}

		if (analysys_n_moves < (int)board_history.size()){
			SimpleGUI::Button(U"棋譜解析", Vec2(0, 50), 120, false);
			if (!analysys_start) {
				board_history[analysys_n_moves].translate_to_arr(bd_arr);
				create_vacant_lst(board_history[analysys_n_moves], bd_arr);
				future_val = calc_value_nopolicy(board_history[analysys_n_moves], depth, end_depth);
				analysys_start = true;
			} else if (future_val.wait_for(seconds0) == future_status::ready) {
				int val = (board_history[analysys_n_moves].p ? -1 : 1) * round((double)future_val.get().value);
				graph.push(analysys_n_moves, val);
				++analysys_n_moves;
				analysys_start = false;
			}
		} else if (SimpleGUI::Button(U"棋譜解析", Vec2(0, 50), 120, !thinking && !book_learning)) {
			graph.clear();
			n_moves = board_history[board_history.size() - 1].n - 4;
			bd = board_history[board_history.size() - 1];
			analysys_n_moves = 0;
			analysys_start = false;
		} else {
			analysys_n_moves = 1000;
		}

		if (!book_learning && !book_learning_button) {
			if (SimpleGUI::Button(U"book学習", Vec2(125, 0), 120, !thinking && playing)) {
				book_learning = true;
				book_learning_button = true;
				book_changed = true;
				book_learn_future = async(launch::async, learn_book, bd, depth, end_depth, &bd, &value);
			}
		} else {
			SimpleGUI::Slider(U"book深さ{:.0f}手"_fmt(book_depth_double), book_depth_double, 0, 64, Vec2(550, 175), 200, 250);
			SimpleGUI::Slider(U"book許容{:.0f}石"_fmt(book_learn_accept_double), book_learn_accept_double, 0, 64, Vec2(550, 210), 200, 250);
			book_depth = (int)book_depth_double;
			book_learn_accept = (int)book_learn_accept_double;
			if (SimpleGUI::Button(U"学習停止", Vec2(125, 0), 120, !thinking)) {
				book_learning = false;
				book_learning_button = false;
				book_learn_future.get();
				for (int i = 0; i < hw2; ++i) {
					cell_value_state[i] = 0;
					umigame_state[i] = 0;
				}
				human_value_state = 0;
			}
		}

		if (book_learning) {
			value_ui(U"評価値: ", round(value)).draw(250, 650, font_color);
		}
		font20(U"メモ:").draw(470, 480, font_color);
		font20(U"先手:").draw(470, 515, font_color);
		font20(U"後手:").draw(730, 515, font_color);
		SimpleGUI::TextBox(play_memo, Vec2(520, 475), 460, !book_learning);
		SimpleGUI::TextBox(black_player, Vec2(520, 510), 200, !book_learning);
		SimpleGUI::TextBox(white_player, Vec2(780, 510), 200, !book_learning);
		if (SimpleGUI::Button(U"対局保存", Vec2(750, 550), 120, !book_learning)) {
			if (playing || finished) {
				String record_copy = record;
				record_copy.replace(U"\n", U"");
				string record_stdstr = record_copy.narrow();
				__time64_t now;
				tm newtime;
				_time64(&now);
				errno_t err = localtime_s(&newtime, &now);
				ostringstream sout;
				string year = to_string(newtime.tm_year + 1900);
				sout << setfill('0') << setw(2) << newtime.tm_mon + 1;
				string month = sout.str();
				sout.str("");
				sout.clear(stringstream::goodbit);
				sout << setfill('0') << setw(2) << newtime.tm_mday;
				string day = sout.str();
				sout.str("");
				sout.clear(stringstream::goodbit);
				sout << setfill('0') << setw(2) << newtime.tm_hour;
				string hour = sout.str();
				sout.str("");
				sout.clear(stringstream::goodbit);
				sout << setfill('0') << setw(2) << newtime.tm_min;
				string minute = sout.str();
				sout.str("");
				sout.clear(stringstream::goodbit);
				sout << setfill('0') << setw(2) << newtime.tm_sec;
				string second = sout.str();
				int player_idx = -1;
				if (ai_player == 0)
					player_idx = 1;
				else if (ai_player == 1)
					player_idx = 0;
				else if (ai_player == 2)
					player_idx = 2;
				else
					player_idx = 3;
				string info = year + month + day + "_" + hour + minute + second + "_" + to_string(depth) + "_" + to_string(end_depth) + "_" + to_string(player_idx);
				ofstream ofs("record/" + info + ".txt");
				if (ofs.fail()) {
					saved = 2;
				}
				else {
					string result = "?";
					if (finished) {
						int int_result = bd.count(0) - bd.count(1);
						int sum_stones = bd.count(0) + bd.count(1);
						if (int_result > 0)
							int_result += hw2 - sum_stones;
						else if (int_result < 0)
							int_result -= hw2 - sum_stones;
						result = to_string(int_result);
					}
					String black_player_text = black_player.text, white_player_text = white_player.text;
					if (black_player_text == U"")
						black_player_text = U"?";
					if (white_player_text == U"")
						white_player_text = U"?";
					ofs << record_stdstr << " " << bd.count(0) << " " << bd.count(1) << " " << result << " " << black_player_text << " " << white_player_text << " " << play_memo.text << endl;
					ofs.close();
					saved = 1;
				}
			}
			else
				saved = 2;
		}
		if (saved == 1) {
			saved_ui(U"成功").draw(900, 550, font_color);
		} else if (saved == 2) {
			saved_ui(U"失敗").draw(900, 550, font_color);
		}

		if (SimpleGUI::Button(U"棋譜コピー", Vec2(750, 600), 120, !book_learning)) {
			String record_copy = record;
			record_copy.replace(U"\n", U"");
			Clipboard::SetText(record_copy);
			copied = true;
		}
		if (copied) {
			copy_ui(U"完了").draw(900, 600, font_color);
		}
		

		if (SimpleGUI::Button(U"対局開始", Vec2(0, 0), 120, !book_learning)) {
			int player_idx = pulldown_player.getIndex();
			if (player_idx == 0)
				ai_player = 1;
			else if (player_idx == 1)
				ai_player = 0;
			else if (player_idx == 2)
				ai_player = 2;
			else
				ai_player = both_ai_define;
			if (ai_player == 0 || ai_player == both_ai_define) {
				black_player.text = U"Egaroucid";
				if (white_player.text == U"Egaroucid" && ai_player != both_ai_define)
					white_player.text = U"";
			}
			if (ai_player == 1 || ai_player == both_ai_define) {
				white_player.text = U"Egaroucid";
				if (black_player.text == U"Egaroucid" && ai_player != both_ai_define)
					black_player.text = U"";
			}
			playing = true;
			thinking = false;
			value = 0.0;
			if (input_board_state != 2 && input_record_state != 2) {
				for (int i = 0; i < hw2; ++i)
					bd_arr[i] = first_board[i];
				bd.translate_from_arr(bd_arr, black);
			}
			create_vacant_lst(bd, bd_arr);
			for (int i = 0; i < hw2; ++i) {
				cell_value_state[i] = 0;
				umigame_state[i] = 0;
			}
			human_value_state = 0;
			change_book_value_str.clear();
			changing_book = false;
			n_moves = bd.n - 4;
			graph.clear();
			if (input_record_state != 2) {
				record.clear();
				board_history.clear();
				board_history.push_back(bd);
			}
			board_start_moves = board_history[0].n - 4;
			bool has_legal = false;
			for (int i = 0; i < hw2; ++i) {
				if (bd.legal(i))
					has_legal = true;
			}
			finished = !has_legal;
			saved = 0;
			copied = false;
			input_board_state = 0;
			input_record_state = 0;
			start_moves = bd.n - 4;
			pre_searched = false;
		}

		record_ui(record).draw(3, 550, font_color);

		if (playing) {
			bool passed2 = true;
			for (int i = 0; i < hw2; ++i) {
				if (bd.legal(i))
					passed2 = false;
			}
			if (passed2) {
				bd.p = 1 - bd.p;
				for (int i = 0; i < hw2; ++i) {
					if (bd.legal(i))
						passed2 = false;
				}
				bd.p = 1 - bd.p;
			}
			if ((finished && n_moves == finish_moves) || passed2)
				move_font(U"終局").draw(420, 650, font_color);
			else
				move_font(bd.n - 3, U"手目").draw(420, 650, font_color);
			bool flag = false;
			for (int i = 0; i < hw2; ++i)
				flag |= (cell_value_state[i] == 1);
			if (flag) {
				SimpleGUI::Button(U"<", Vec2(550, 650), 50, false);
				SimpleGUI::Button(U">", Vec2(600, 650), 50, false);
			} else {
				if (SimpleGUI::Button(U"<", Vec2(550, 650), 50, !book_learning)) {
					if (n_moves - board_start_moves >= 1)
						--n_moves;
					bd = board_history[n_moves - board_start_moves];
					bd.translate_to_arr(bd_arr);
					create_vacant_lst(bd, bd_arr);
					for (int i = 0; i < hw2; ++i) {
						cell_value_state[i] = 0;
						umigame_state[i] = 0;
					}
					human_value_state = 0;
				}
				if (SimpleGUI::Button(U">", Vec2(600, 650), 50, !book_learning)) {
					if (n_moves - board_start_moves < board_history.size() - 1)
						++n_moves;
					bd = board_history[n_moves - board_start_moves];
					bd.translate_to_arr(bd_arr);
					create_vacant_lst(bd, bd_arr);
					for (int i = 0; i < hw2; ++i) {
						cell_value_state[i] = 0;
						umigame_state[i] = 0;
					}
					human_value_state = 0;
				}
			}
		}

		if (book_learning) {
			bd.translate_to_arr(bd_arr);
			for (int y = 0; y < hw; ++y) {
				for (int x = 0; x < hw; ++x) {
					int coord = proc_coord(y, x);
					if (bd_arr[coord] == black)
						stones[coord].draw(Palette::Black);
					else if (bd_arr[coord] == white)
						stones[coord].draw(Palette::White);
				}
			}
		} else if (playing && !book_learning) {
			if (SimpleGUI::Button(U"読み停止", Vec2(880, 680), 120, global_searching))
				global_searching = false;
			/*
			bool all_hint_done_flag = true, has_legal = false;
			for (int cell = 0; cell < hw2; ++cell) {
				if (bd.legal(cell)) {
					has_legal = true;
					if (cell_value_state[cell] % 2 == 1)
						all_hint_done_flag = false;
				}
			}
			*/
			bool has_legal = false;
			max_cell_value = -inf;
			for (int cell = 0; cell < hw2; ++cell) {
				if (bd.legal(cell)) {
					has_legal = true;
					if (cell_value_state[cell] > 0)
						max_cell_value = max(max_cell_value, cell_values[cell]);
				}
			}
			if (!has_legal)
				bd.p = 1 - bd.p;
			for (int y = 0; y < hw; ++y) {
				for (int x = 0; x < hw; ++x) {
					int coord = proc_coord(y, x);
					if (bd_arr[coord] == black)
						stones[coord].draw(Palette::Black);
					else if (bd_arr[coord] == white)
						stones[coord].draw(Palette::White);
					if (bd.policy == coord)
						Circle(cell_center_x[coord % hw], cell_center_y[coord / hw], 5).draw(Palette::Red);
					if (bd.legal(coord)) {
						if ((bd.p != ai_player && ai_player != both_ai_define) || n_moves != board_history.size() - 1){
							if (hint_default) {
								if (cell_value_state[coord] % 2 == 1 && global_searching) {
									if (future_cell_values[coord].wait_for(seconds0) == future_status::ready) {
										cell_value cell_value_result = future_cell_values[coord].get();
										cell_values[coord] = -cell_value_result.value;
										cell_depth[coord] = cell_value_result.depth;
										++cell_value_state[coord];
									}
								}
								else if (cell_value_state[coord] % 2 == 0 && cell_value_state[coord] / 2 + 1 <= cell_value_depth) {
									if (cell_value_state[coord] / 2 + 1 < cell_value_depth)
										future_cell_values[coord] = calc_value(bd, coord, cell_value_state[coord] / 2 + 1, cell_value_state[coord] / 2 + 1);
									else
										future_cell_values[coord] = calc_value(bd, coord, cell_value_state[coord] / 2 + 1, cell_value_end_depth);
									++cell_value_state[coord];
								}
							}
							if (hint_default && cell_value_state[coord] >= 2) {
								Color color = Palette::White;
								if (cell_values[coord] == max_cell_value)
									color = Palette::Cyan;
								cell_value_font(cell_values[coord]).draw(offset_x + (coord % hw) * cell_hw + 2, offset_y + (coord / hw) * cell_hw, color);
								if (cell_depth[coord] == final_define_value)
									cell_depth_font(cell_depth[coord], U"%").draw(offset_x + (coord % hw) * cell_hw + 2, offset_y + (coord / hw) * cell_hw + 19, color);
								else if (cell_depth[coord] == book_define_value)
									cell_depth_font(U"book").draw(offset_x + (coord % hw) * cell_hw + 2, offset_y + (coord / hw) * cell_hw + 19, color);
								else
									cell_depth_font(cell_depth[coord], U"手").draw(offset_x + (coord % hw) * cell_hw + 2, offset_y + (coord / hw) * cell_hw + 19, color);
							}
							else
								legals[coord].draw(Palette::Blue);
							if (umigame_state[coord] == 0) {
								if (hint_default && umigame_default) {
									board moved_bd = bd.move(coord);
									if (book.get(&moved_bd) != -inf) {
										future_umigame[coord] = get_umigame(moved_bd);
										umigame_state[coord] = 1;
									}
								}
							}
							else if (umigame_state[coord] == 1) {
								if (future_umigame[coord].wait_for(seconds0) == future_status::ready) {
									umigame_values[coord] = future_umigame[coord].get();
									umigame_state[coord] = 2;
								}
							}
							else if (umigame_state[coord] == 2) {
								if (hint_default && umigame_default) {
									int umigame_sx = offset_x + (coord % hw) * cell_hw + 2;
									int umigame_sy = offset_y + (coord / hw) * cell_hw + 32;
									RectF black_rect = umigame_font(umigame_values[coord].b).region(umigame_sx, umigame_sy);
									black_rect.draw(Palette::Black);
									umigame_font(umigame_values[coord].b).draw(umigame_sx, umigame_sy, Palette::Green);
									umigame_sx += black_rect.size.x;
									RectF white_rect = umigame_font(umigame_values[coord].w).region(umigame_sx, umigame_sy);
									white_rect.draw(Palette::White);
									umigame_font(umigame_values[coord].w).draw(umigame_sx, umigame_sy, Palette::Green);
								}
							}
							if (cells[coord].leftClicked() && !changing_book){ // && (ai_player == 2 || (!finished && n_moves == board_history.size() - 1 + board_start_moves))) {
								bd = bd.move(coord);
								++n_moves;
								human_value_state = 0;
								global_searching = true;
								record += coord_translate(coord);
								String record_copy = record;
								record_copy.replace(U"\n", U"");
								if (record_copy.size() % 40 == 0)
									record += U"\n";
								bd.translate_to_arr(bd_arr);
								create_vacant_lst(bd, bd_arr);
								finished = check_pass(&bd);
								if (finished)
									finish_moves = n_moves;
								while (board_history.size()) {
									if (board_history[board_history.size() - 1].n >= bd.n)
										board_history.pop_back();
									else
										break;
								}
								while (graph.last_x() >= bd.n - 4)
									graph.pop();
								board_history.push_back(bd);
								for (int i = 0; i < hw2; ++i) {
									cell_value_state[i] = 0;
									umigame_state[i] = 0;
								}
								saved = 0;
								copied = false;
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
											book_changed = true;
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
						value = result.value;
						record += coord_translate(result.policy);
						if (ai_player == both_ai_define && bd.p == white)
							graph.push(bd.n - 4, -result.value);
						else
							graph.push(bd.n - 4, (bd.p ? -1.0 : 1.0) * (double)result.value);
						bd = bd.move(result.policy);
						++n_moves;
						human_value_state = 0;
						global_searching = true;
						String record_copy = record;
						record_copy.replace(U"\n", U"");
						if (record_copy.size() % 40 == 0)
							record += U"\n";
						bd.translate_to_arr(bd_arr);
						create_vacant_lst(bd, bd_arr);
						finished = check_pass(&bd);
						if (finished)
							finish_moves = n_moves;
						board_history.push_back(bd);
						saved = 0;
					}
				} else {
					thinking = true;
					future_result = ai(bd, depth, end_depth, bd_arr, book_accept, &pre_searched);
				}
			}
			score_ui(U"黒 ", bd.count(black), U" ", bd.count(white), U" 白").draw(10, 640, font_color);
			if (value_default)
				value_ui(U"評価値: ", round(value)).draw(250, 650, font_color);

			if (((bd.p != ai_player && ai_player != both_ai_define) || n_moves != board_history.size() - 1) && hint_default && human_hint_default) {
				if (human_value_state == 0) {
					human_value_future = async(launch::async, search_human, bd, tim(), human_hint_depth, human_hint_sub_depth);
					human_value_state = 1;
				} else if (human_value_state == 1 && human_value_future.wait_for(seconds0) == future_status::ready) {
					human_values = human_value_future.get();
					human_value_state = 2;
				} else if (human_value_state == 2) {
					Color color = Palette::Cyan;
					for (const search_result_pv elem : human_values) {
						human_value_font(round(elem.concat_value)).draw(offset_x + (elem.policy % hw) * cell_hw + 32, offset_y + (elem.policy / hw) * cell_hw + 28, color);
						color = Palette::White;
					}
				}
			}
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
			change_book_ui(change_book_value_info_str, U"(", change_book_value_coord_str, U"): ", change_book_value_str).draw(670, 650, font_color);
		}

		joseki_ui(Unicode::FromUTF8(joseki.get(bd))).draw(145, 80, font_color);

		pulldown_player.update();
		pulldown_player.draw();

		if (value_default && !book_learning)
			graph.draw();
		
	}
}
