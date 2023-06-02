/*
    Egaroucid Project

	@file button.hpp
		Button for Egaroucid's GUI
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <Siv3D.hpp>
#include <iostream>

class Button {
public:
	RoundRect rect;
	String str;
	Font font;
	int font_size;
	Color button_color;
	Color font_color;
public:
	void init(int x, int y, int w, int h, int r, String s, int fs, Font f, Color c1, Color c2) {
		rect.x = x;
		rect.y = y;
		rect.w = w;
		rect.h = h;
		rect.r = r;
		str = s;
		font_size = fs;
		font = f;
		button_color = c1;
		font_color = c2;
	}

	void draw() {
		rect.draw(button_color);
		font(str).drawAt(font_size, rect.x + rect.w / 2, rect.y + rect.h / 2, font_color);
	}

	void draw(double transparency) {
		rect.draw(ColorF(button_color, transparency));
		font(str).drawAt(font_size, rect.x + rect.w / 2, rect.y + rect.h / 2, ColorF(font_color, transparency));
	}

	bool clicked() {
		return rect.leftClicked();
	}
};

class FrameButton {
public:
	RoundRect rect;
	String str;
	Font font;
	int font_size;
	Color button_color;
	Color font_color;
	Color frame_color;
	int frame_width;
public:
	void init(int x, int y, int w, int h, int r, int fw, String s, int fs, Font f, Color c1, Color c2, Color c3) {
		rect.x = x;
		rect.y = y;
		rect.w = w;
		rect.h = h;
		rect.r = r;
		frame_width = fw;
		str = s;
		font_size = fs;
		font = f;
		button_color = c1;
		font_color = c2;
		frame_color = c3;
	}

	void draw() {
		rect.draw(button_color).drawFrame(frame_width, frame_color);
		font(str).drawAt(font_size, rect.x + rect.w / 2, rect.y + rect.h / 2, font_color);
	}

	void draw(double transparency) {
		rect.draw(ColorF(button_color, transparency)).drawFrame(frame_width, ColorF(frame_color, transparency));
		font(str).drawAt(font_size, rect.x + rect.w / 2, rect.y + rect.h / 2, ColorF(font_color, transparency));
	}

	bool clicked() {
		return rect.leftClicked();
	}
};