#pragma once
#include <Siv3D.hpp>
#include <iostream>
#include <vector>

using namespace std;

constexpr Color menu_color = Palette::Gainsboro;
constexpr Color menu_active_color = Palette::Lightblue;
constexpr Color menu_font_color = Palette::Black;
constexpr int menu_offset = 2;

#define button_mode 0
#define bar_mode 1
#define check_mode 2

class menu_elem {
private:
	String str;
	Rect rect;
	Font font;
	int mode;
	bool is_active;
	int *bar_elem;
	bool *is_clicked;
	bool *is_checked;
	Texture checkbox;

public:
	void init_button(String s, bool *c) {
		mode = button_mode;
		str = s;
		is_clicked = c;
		*is_clicked = false;
	}

	void init_bar(String s, int *c, int d) {
		mode = bar_mode;
		is_active = false;
		str = s;
		bar_elem = c;
		*bar_elem = d;
	}

	void init_check(String s, bool *c, bool d) {
		mode = check_mode;
		is_active = false;
		str = s;
		is_checked = c;
		*is_checked = d;
	}

	void pre_init(Font f, Texture c) {
		font = f;
		checkbox = c;
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
		font(str).draw(rect.x + rect.h - menu_offset, rect.y + menu_offset, menu_font_color);
		if (mode == check_mode) {
			if (*is_checked) {
				checkbox.scaled((double)(rect.h - 2 * menu_offset) / checkbox.width()).draw(rect.x + menu_offset, rect.y + menu_offset);
			}
			if (*is_clicked) {
				*is_checked = !(*is_checked);
			}
		}
	}

	bool clicked() {
		return *is_clicked;
	}

	bool active() {
		return is_active;
	}

	RectF size() {
		RectF res = font(str).region(Point{ 0, 0 });
		res.w += res.h;
		return res;
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
	Texture checkbox;

public:
	void init(String s) {
		str = s;
		is_open = false;
	}

	void pre_init(Font f, Texture c) {
		font = f;
		checkbox = c;
	}

	void init_inside(int x, int y, int w, int h) {
		rect.x = x;
		rect.y = y;
		rect.w = w;
		rect.h = h;
		int height = h - menu_offset * 2, width = w - menu_offset * 2;
		for (menu_elem &elem : elems) {
			elem.pre_init(font, checkbox);
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

	void init(int x, int y, Font f, Texture c) {
		int height = 0, width = 0;
		for (menu_title &elem : menu) {
			elem.pre_init(f, c);
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
