#pragma once
#include <Siv3D.hpp>
#include <iostream>
#include <vector>

using namespace std;

#define radio_button_r 3
#define radio_button_margin 20

class Radio_Button_Element {
public:
	Circle circle;
	String str;
	Font font;
	int x;
	int y;
	bool checked;
	RectF region;

public:
	void init(int xx, int yy, Font f, int font_size, String s, bool c) {
		x = xx;
		y = yy;
		font = f;
		circle.x = x + radio_button_margin / 2;
		circle.y = y;
		circle.r = radio_button_r;
		str = s;
		checked = c;
		region = font(str).region(Arg::leftCenter = Vec2{ x + radio_button_margin, y });
		region.x = x;
		region.w += radio_button_margin;
	}

	bool clicked() {
		return region.leftClicked();
	}

	void draw() {
		if (checked) {
			circle.draw(Palette::Cyan);
		}
		font(str).draw(Arg::leftCenter = Vec2{ x + radio_button_margin, y }, Palette::White);
	}
};

class Radio_Button {
public:
	vector<Radio_Button_Element> elems;
	int checked;

public:
	void init() {
		elems.clear();
		checked = 0;
	}

	void push(Radio_Button_Element elem) {
		elems.emplace_back(elem);
	}

	void draw() {
		for (Radio_Button_Element& elem : elems) {
			elem.draw();
		}
		for (int i = 0; i < (int)elems.size(); ++i) {
			if (elems[i].clicked()) {
				checked = i;
			}
		}
		for (int i = 0; i < (int)elems.size(); ++i) {
			elems[i].checked = (checked == i);
		}
	}
};
