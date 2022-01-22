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

using namespace std;

Menu create_menu(bool *start_game_flag,
	bool *use_ai_flag, bool *human_first, bool *human_second, bool *both_ai,
	bool *use_hint_flag, bool *normal_hint, bool *human_hint, bool *umigame_hint,
	bool *use_value_flag) {
	Menu menu;
	menu_title title;
	menu_elem menu_e, side_menu;
	Font menu_font(15);
	Texture checkbox(U"resources/img/checked.png", TextureDesc::Mipped);

	title.init(U"対局");

	menu_e.init_button(U"対局開始", start_game_flag);
	title.push(menu_e);

	menu.push(title);



	title.init(U"設定");

	menu_e.init_check(U"AIが着手", use_ai_flag, *use_ai_flag);
	side_menu.init_radio(U"人間先手", human_first, *human_first);
	menu_e.push(side_menu);
	side_menu.init_radio(U"人間後手", human_second, *human_second);
	menu_e.push(side_menu);
	side_menu.init_radio(U"AI同士", both_ai, *both_ai);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu_e.init_check(U"ヒント表示", use_hint_flag, *use_hint_flag);
	side_menu.init_check(U"石差評価", normal_hint, *normal_hint);
	menu_e.push(side_menu);
	side_menu.init_check(U"人間的評価", human_hint, *human_hint);
	menu_e.push(side_menu);
	side_menu.init_check(U"うみがめ数", umigame_hint, *umigame_hint);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu_e.init_check(U"評価値表示", use_value_flag, *use_value_flag);
	title.push(menu_e);

	menu.push(title);


	menu.init(0, 0, menu_font, checkbox);
	return menu;
}

void Main() {
	Size window_size = Size(1000, 720);
	Window::Resize(window_size);
	Window::SetStyle(WindowStyle::Sizable);
	Scene::SetResizeMode(ResizeMode::Keep);
	Window::SetTitle(U"Egaroucid5.2.0");
	//System::SetTerminationTriggers(UserAction::NoAction);
	Scene::SetBackground(Palette::White);
	Console.open();

	/*
	Menu menu;
	menu_title title;
	menu_elem menu_e, side_menu;
	Font menu_font(15);
	Texture checkbox(U"resources/icon.png", TextureDesc::Mipped);

	bool start_game_flag;

	title.init(U"対局");
	menu_e.init_button(U"対局開始", &start_game_flag);
	title.push(menu_e);
	menu.push(title);

	title.init(U"対局");
	menu_e.init_button(U"対局開始", &start_game_flag);
	title.push(menu_e);
	menu.push(title);


	bool clicked1, clicked2, clicked3, clicked4, side_menu_clicked;
	bool radio1, radio2, radio3, radio4, radio5, radio6;
	title.init(U"ふぁいる");
	menu_e.init_button(U"テスト", &clicked1);
	title.push(menu_e);
	menu_e.init_check(U"猫になりたい", &clicked2, false);
	title.push(menu_e);
	menu_e.init_check(U"さいどめにゅー", &side_menu_clicked, false);
	side_menu.init_radio(U"にゃ", &radio1, true);
	menu_e.push(side_menu);
	side_menu.init_radio(U"にぃ", &radio2, false);
	menu_e.push(side_menu);
	side_menu.init_radio(U"にゅ", &radio3, false);
	menu_e.push(side_menu);
	title.push(menu_e);
	menu.push(title);

	title.init(U"ねこねこ");
	menu_e.init_button(U"わっしょい", &clicked3);
	title.push(menu_e);
	menu_e.init_button(U"そいや！！", &clicked4);
	title.push(menu_e);
	menu.push(title);

	title.init(U"直ラジオだよ");
	menu_e.init_radio(U"わ", &radio4, true);
	title.push(menu_e);
	menu_e.init_radio(U"を", &radio5, false);
	title.push(menu_e);
	menu_e.init_radio(U"ん", &radio6, false);
	title.push(menu_e);
	menu.push(title);

	menu.init(0, 0, menu_font, checkbox);
	*/

	bool start_game_flag;
	bool use_ai_flag = true, human_first = true, human_second = false, both_ai = false;
	bool use_hint_flag = true, normal_hint = true, human_hint = true, umigame_hint = true;
	bool use_value_flag = true;
	Menu menu = create_menu(&start_game_flag,
		&use_ai_flag, &human_first, &human_second, &both_ai,
		&use_hint_flag, &normal_hint, &human_hint, &umigame_hint,
		&use_value_flag);

	while (System::Update()) {
		menu.draw();
		//Print << menu.active();
		/*
		if (clicked1)
			Print << U"clicked1";
		if (clicked2)
			Print << U"clicked2";
		if (clicked3)
			Print << U"clicked3";
		if (clicked4)
			Print << U"clicked4";
		*/
	}

}
