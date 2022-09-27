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

class Export_game : public App::Scene {
private:
	String black_player_name;
	String white_player_name;
	String memo;
	Button back_button;
	Button export_main_button;
	Button iexport_this_board_button;
	bool active_cells[3];


public:
	Export_game(const InitData& init) : IScene{ init } {
		back_button.init(BUTTON3_1_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		export_main_button.init(BUTTON3_2_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "export_main"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		export_this_board_button.init(BUTTON3_3_SX, BUTTON3_SY, BUTTON3_WIDTH, BUTTON3_HEIGHT, BUTTON3_RADIUS, language.get("in_out", "export_until_this_board"), getData().fonts.font15, getData().colors.white, getData().colors.black);
		for (int i = 0; i < 3; ++i) {
			active_cells[i] = false;
		}
		active_cells[0] = true;
	}

	void update() override {
		getData().fonts.font25(language.get("in_out", "output_game")).draw(Arg::topCenter(X_CENTER, 10), getData().colors.white);
		getData().fonts.font20(language.get("in_out", "player_name")).draw(Arg::topCenter(X_CENTER, 50), getData().colors.white);
		Rect black_area{ X_CENTER - EXPORT_GAME_PLAYER_WIDTH, 80, EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT };
		Rect white_area{ X_CENTER, 80, EXPORT_GAME_PLAYER_WIDTH, EXPORT_GAME_PLAYER_HEIGHT };
		Circle(X_CENTER - EXPORT_GAME_PLAYER_WIDTH - EXPORT_GAME_RADIUS - 20, 80 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.black);
		Circle(X_CENTER + EXPORT_GAME_PLAYER_WIDTH + EXPORT_GAME_RADIUS + 20, 80 + EXPORT_GAME_RADIUS, EXPORT_GAME_RADIUS).draw(getData().colors.white);
		getData().fonts.font20(language.get("in_out", "memo")).draw(Arg::topCenter(X_CENTER, 110), getData().colors.white);
		Rect memo_area{ X_CENTER - EXPORT_GAME_MEMO_WIDTH / 2, 140, EXPORT_GAME_MEMO_WIDTH, EXPORT_GAME_MEMO_HEIGHT };
		const String editingText = TextInput::GetEditingText();
		bool tab_inputted = false;
		if (active_cells[0]) {
			tab_inputted = black_player_name.narrow().find('\t') != string::npos;
		}
		else if (active_cells[1]) {
			tab_inputted = white_player_name.narrow().find('\t') != string::npos;
		}
		else if (active_cells[2]) {
			tab_inputted = memo.narrow().find('\t') != string::npos;
		}
		if (tab_inputted) {
			for (int i = 0; i < 3; ++i) {
				if (active_cells[i]) {
					active_cells[i] = false;
					active_cells[(i + 1) % 3] = true;
					break;
				}
			}
			black_player_name = black_player_name.replaced(U"\t", U"");
			white_player_name = white_player_name.replaced(U"\t", U"");
			memo = memo.replaced(U"\t", U"");
		}
		if (black_area.leftClicked() || active_cells[0]) {
			black_area.draw(getData().colors.light_cyan).drawFrame(2, getData().colors.black);
			TextInput::UpdateText(black_player_name);
			if (KeyControl.pressed() && KeyV.down()) {
				String clip_text;
				Clipboard::GetText(clip_text);
				black_player_name += clip_text;
			}
			getData().fonts.font15(black_player_name + U'|' + editingText).draw(black_area.stretched(-4), getData().colors.black);
			active_cells[0] = true;
			active_cells[1] = false;
			active_cells[2] = false;
		}
		else {
			black_area.draw(getData().colors.white).drawFrame(2, getData().colors.black);
			getData().fonts.font15(black_player_name).draw(black_area.stretched(-4), getData().colors.black);
		}
		if (white_area.leftClicked() || active_cells[1]) {
			white_area.draw(getData().colors.light_cyan).drawFrame(2, getData().colors.black);
			TextInput::UpdateText(white_player_name);
			if (KeyControl.pressed() && KeyV.down()) {
				String clip_text;
				Clipboard::GetText(clip_text);
				white_player_name += clip_text;
			}
			getData().fonts.font15(white_player_name + U'|' + editingText).draw(white_area.stretched(-4), getData().colors.black);
			active_cells[0] = false;
			active_cells[1] = true;
			active_cells[2] = false;
		}
		else {
			white_area.draw(getData().colors.white).drawFrame(2, getData().colors.black);
			getData().fonts.font15(white_player_name).draw(white_area.stretched(-4), getData().colors.black);
		}
		if (memo_area.leftClicked() || active_cells[2]) {
			memo_area.draw(getData().colors.light_cyan).drawFrame(2, getData().colors.black);
			TextInput::UpdateText(memo);
			if (KeyControl.pressed() && KeyV.down()) {
				String clip_text;
				Clipboard::GetText(clip_text);
				memo += clip_text;
			}
			getData().fonts.font15(memo + U'|' + editingText).draw(memo_area.stretched(-4), getData().colors.black);
			active_cells[0] = false;
			active_cells[1] = false;
			active_cells[2] = true;
		}
		else {
			memo_area.draw(getData().colors.white).drawFrame(2, getData().colors.black);
			getData().fonts.font15(memo).draw(memo_area.stretched(-4), getData().colors.black);
		}
		back_button.draw();
		export_main_button.draw();
		export_this_board_button.draw();
		if (back_button.clicked() || KeyEscape.pressed()) {
			changeScene(U"Main_scene", SCENE_FADE_TIME);
		}
		if (export_main_button.clicked()) {

		}
		if (export_this_board_button.clicked()) {

		}
	}

	void draw() const override {

	}
};
