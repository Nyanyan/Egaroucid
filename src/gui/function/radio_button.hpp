/*
    Egaroucid Project

	@file radio_button.hpp
		Radio button for GUI
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <Siv3D.hpp>
#include <iostream>
#include <vector>

#define radio_button_r 3
#define radio_button_margin 20

class Radio_button_element {
public:
	Circle circle;
	String str;
	Font font;
	int font_size;
	int x;
	int y;
	bool checked;
	RectF region;

public:
	void init(int xx, int yy, Font f, int fs, String s, bool c) {
		x = xx;
		y = yy;
		font = f;
		font_size = fs;
		circle.x = x + radio_button_margin / 2;
		circle.y = y;
		circle.r = radio_button_r;
		str = s;
		checked = c;
		region = font(str).region(font_size, Arg::leftCenter = Vec2{ x + radio_button_margin, y });
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
		font(str).draw(font_size, Arg::leftCenter = Vec2{ x + radio_button_margin, y }, Palette::White);
	}
};

class Radio_button {
public:
	std::vector<Radio_button_element> elems;
	int checked;

public:
	void init() {
		elems.clear();
		checked = 0;
	}

	void push(Radio_button_element elem) {
		elems.emplace_back(elem);
	}

	void draw() {
		for (Radio_button_element& elem : elems) {
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
