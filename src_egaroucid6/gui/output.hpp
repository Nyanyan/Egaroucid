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
	Button single_back_button;
	Button back_button;
	Button import_button;
	bool done;
	bool failed;
	string transcript;
	vector<History_elem> n_history;

public:
	Export_game(const InitData& init) : IScene{ init } {
		single_back_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		back_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "back"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		import_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("in_out", "import"), getData().fonts.font25, getData().colors.white, getData().colors.black);
		done = false;
		failed = false;
		transcript.clear();
	}

	void update() override {

	}

	void draw() const override {

	}
};
