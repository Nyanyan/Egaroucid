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

class Import_transcript : public App::Scene {
private:
	Button single_back_button;
	Button back_button;
	Button import_button;
	bool done;
	bool failed;
	string transcript;
	vector<History_elem> n_history;

public:
	Import_transcript(const InitData& init) : IScene{ init } {
		single_back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		import_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "import"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		done = false;
		failed = false;
		transcript.clear();
	}

	void update() override {
		Scene::SetBackground(getData().colors.green);
		const int icon_width = (LEFT_RIGHT - LEFT_LEFT) / 2;
		getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
		getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
		int sy = 20 + icon_width + 50;
		if (!done) {
			getData().fonts.font25(language.get("in_out", "input_transcript")).draw(Arg::topCenter(X_CENTER, sy), getData().colors.white);
			Rect text_area{ X_CENTER - 300, sy + 40, 600, 70 };
			text_area.draw(getData().colors.light_cyan).drawFrame(2, getData().colors.black);
			String str = Unicode::Widen(transcript);
			TextInput::UpdateText(str);
			const String editingText = TextInput::GetEditingText();
			bool return_pressed = false;
			if (KeyControl.pressed() && KeyV.down()) {
				String clip_text;
				Clipboard::GetText(clip_text);
				str += clip_text;
			}
			if (str.size()) {
				if (str[str.size() - 1] == '\n') {
					str.replace(U"\n", U"");
					return_pressed = true;
				}
			}
			transcript = str.narrow();
			getData().fonts.font15(str + U'|' + editingText).draw(text_area.stretched(-4), getData().colors.black);
			back_button.draw();
			import_button.draw();
			if (back_button.clicked() || KeyEscape.pressed()) {
				changeScene(U"Main_scene", SCENE_FADE_TIME);
			}
			if (import_button.clicked() || KeyEnter.pressed()) {
				n_history = import_transcript_processing(transcript, &failed);
				done = true;
			}
		}
		else {
			if (!failed) {
				getData().graph_resources.init();
				getData().graph_resources.nodes[0] = n_history;
				getData().graph_resources.n_discs = getData().graph_resources.nodes[0].back().board.n_discs();
				getData().graph_resources.need_init = false;
				changeScene(U"Main_scene", SCENE_FADE_TIME);
			}
			else {
				getData().fonts.font25(language.get("in_out", "import_failed")).draw(Arg::topCenter(X_CENTER, sy), getData().colors.white);
				single_back_button.draw();
				if (single_back_button.clicked() || KeyEscape.pressed()) {
					changeScene(U"Main_scene", SCENE_FADE_TIME);
				}
			}
		}
	}

	void draw() const override {

	}
};
