#pragma once
#include <Siv3D.hpp>
#include <iostream>
#include <vector>

using namespace std;

constexpr Color menu_color = Palette::Gray;
constexpr Color menu_active_color = Palette::Skyblue;
constexpr Color menu_font_color = Palette::Black;
constexpr int menu_offset = 2;

class menu_elem {
private:
	String str;
	Rect rect;
	Font font;
	bool is_slidebar;
	int *bar_elem;
	bool *is_clicked;
	bool is_active;

public:
	void init(String s, bool *c) {
		str = s;
		is_active = false;
		is_clicked = c;
		is_slidebar = false;
		*is_clicked = false;
	}

	void init(String s, bool sbar, int *belem) {
		str = s;
		is_active = false;
		is_slidebar = sbar;
		bar_elem = belem;
		*is_clicked = false;
	}

	void init_font(Font f) {
		font = f;
	}

	void init_inside(int x, int y, int w, int h) {
		rect.x = x;
		rect.y = y;
		rect.w = w;
		rect.h = h;
	}

	void draw() {
		is_active = rect.mouseOver();
		*is_clicked = rect.leftClicked();
		if (is_active)
			rect.draw(menu_active_color);
		else
			rect.draw(menu_color);
		font(str).draw(rect.x + menu_offset, rect.y + menu_offset, menu_font_color);
	}

	bool clicked() {
		return *is_clicked;
	}

	bool active() {
		return is_active;
	}

	RectF size() {
		return font(str).region(Point{ 0, 0 });
	}

	void not_clicked() {
		*is_clicked = false;
	}
};

class menu_title {
private:
	String str;
	Rect rect;
	Font font;
	bool is_open;
	vector<menu_elem> elems;

public:
	void init(String s) {
		str = s;
		is_open = false;
	}

	void init_font(Font f) {
		font = f;
	}

	void init_inside(int x, int y, int w, int h) {
		rect.x = x;
		rect.y = y;
		rect.w = w;
		rect.h = h;
		int height = h - menu_offset * 2, width = w - menu_offset * 2;
		for (menu_elem &elem : elems) {
			elem.init_font(font);
			RectF r = elem.size();
			height = max(height, (int)r.h);
			width = max(width, (int)r.w);
			cerr << r.h << " " << r.w << endl;
		}
		height += menu_offset * 2;
		width += menu_offset * 2;
		cerr << "title " << height << " " << width << endl;
		int xx = rect.x;
		int yy = rect.y + rect.h;
		for (menu_elem &elem : elems) {
			elem.init_inside(xx, yy, width, height);
			yy += height;
		}
	}

	void push(menu_elem elem) {
		elems.emplace_back(elem);
		cerr << "title " << elems.size() << endl;
	}

	void draw() {
		rect.draw(menu_color);
		font(str).draw(rect.x + menu_offset, rect.y + menu_offset, menu_font_color);
		bool n_is_open = false, clicked = false;
		if (rect.mouseOver()) {
			is_open = true;
			n_is_open = true;
		}
		if (is_open) {
			for (menu_elem &elem : elems) {
				elem.draw();
				n_is_open = n_is_open || elem.active();
				clicked = clicked || elem.clicked();
			}
		}
		else {
			for (menu_elem& elem : elems) {
				elem.not_clicked();
			}
		}
		is_open = n_is_open;
		if (clicked)
			is_open = false;
	}

	RectF size() {
		return font(str).region(Point{ 0, 0 });
	}

	bool open() {
		return is_open;
	}

	void clear() {
		elems.clear();
	}
};

class Menu{
private:
	bool is_open;
	vector<menu_title> menu;

public:
	void push(menu_title elem) {
		menu.emplace_back(elem);
		cerr << menu.size() << endl;
	}

	void init(int x, int y, Font f) {
		int height = 0, width = 0;
		for (menu_title &elem : menu) {
			elem.init_font(f);
			RectF r = elem.size();
			height = max(height, (int)r.h);
			width = max(width, (int)r.w);
		}
		height += menu_offset * 2;
		width += menu_offset * 2;
		cerr << height << " " << width << endl;
		int xx = x;
		int yy = y;
		for (menu_title &elem : menu) {
			elem.init_inside(xx, yy, width, height);
			xx += width;
		}
	}

	void draw() {
		is_open = false;
		for (menu_title &elem : menu) {
			elem.draw();
			is_open = is_open || elem.open();
		}
	}

	bool open() {
		return is_open;
	}
};
