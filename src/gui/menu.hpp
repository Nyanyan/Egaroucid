#pragma once
#include <Siv3D.hpp>
#include <iostream>
#include <vector>

using namespace std;

constexpr Color menu_color = Palette::Gainsboro;
constexpr Color menu_active_color = Palette::Lightblue;
constexpr Color menu_select_color = Palette::Lightcyan;
constexpr Color menu_font_color = Palette::Black;
constexpr Color radio_color = Palette::Deepskyblue;
constexpr int menu_offset_x = 10;
constexpr int menu_offset_y = 1;
constexpr double radio_ratio = 0.2;

#define button_mode 0
#define bar_mode 1
#define check_mode 2
#define radio_mode 3

class menu_elem {
private:
	String str;
	Rect rect;
	Font font;
	int mode;
	bool has_child;
	vector<menu_elem> children;
	bool is_active;
	bool was_active;
	int *bar_elem;
	bool *is_clicked_p;
	bool is_clicked;
	bool *is_checked;
	bool dammy_clicked;
	Texture checkbox;

public:
	void init_button(String s, bool *c) {
		clear();
		mode = button_mode;
		has_child = false;
		is_active = false;
		was_active = false;
		str = s;
		is_clicked_p = c;
		is_clicked = false;
		*is_clicked_p = is_clicked;
	}

	void init_bar(String s, int *c, int d) {
		clear();
		mode = bar_mode;
		has_child = false;
		is_active = false;
		was_active = false;
		str = s;
		bar_elem = c;
		*bar_elem = d;
		is_clicked = false;
	}

	void init_check(String s, bool *c, bool d) {
		clear();
		mode = check_mode;
		has_child = false;
		is_active = false;
		was_active = false;
		str = s;
		is_checked = c;
		*is_checked = d;
		is_clicked = false;
	}

	void init_radio(String s, bool* c, bool d) {
		clear();
		mode = radio_mode;
		has_child = false;
		is_active = false;
		str = s;
		is_checked = c;
		*is_checked = d;
		is_clicked = false;
	}

	void push(menu_elem ch) {
		has_child = true;
		children.emplace_back(ch);
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
		if (has_child) {
			int height = h - menu_offset_y * 2, width = 0;
			for (menu_elem& elem : children) {
				elem.pre_init(font, checkbox);
				RectF r = elem.size();
				height = max(height, (int)r.h);
				width = max(width, (int)r.w);
			}
			height += menu_offset_y * 2;
			width += menu_offset_x * 2;
			int xx = rect.x + rect.w;
			int yy = rect.y;
			for (menu_elem& elem : children) {
				elem.init_inside(xx, yy, width, height);
				yy += height;
			}
		}
	}

	void update() {
		was_active = is_active;
		is_active = rect.mouseOver();
		is_clicked = rect.leftClicked();
	}

	void draw_noupdate() {
		if (mode == button_mode)
			*is_clicked_p = is_clicked;
		if (is_clicked && mode == check_mode)
			*is_checked = !(*is_checked);
		if (is_active)
			rect.draw(menu_active_color);
		else
			rect.draw(menu_select_color);
		font(str).draw(rect.x + rect.h - menu_offset_y, rect.y + menu_offset_y, menu_font_color);
		if (mode == check_mode) {
			if (*is_checked) {
				checkbox.scaled((double)(rect.h - 2 * menu_offset_y) / checkbox.width()).draw(rect.x + menu_offset_y, rect.y + menu_offset_y);
			}
		}
		else if (mode == radio_mode) {
			if (*is_checked) {
				Circle(rect.x + rect.h / 2, rect.y + rect.h / 2, (int)(rect.h * radio_ratio)).draw(radio_color);
			}
		}
		if (has_child) {
			for (menu_elem& elem : children) {
				elem.update();
				is_active = is_active || (elem.active() && last_active());
			}
			if (is_active) {
				int radio_checked = -1;
				int idx = 0;
				for (menu_elem& elem : children) {
					if (elem.clicked()) {
						if (elem.menu_mode() == radio_mode && !elem.checked()) {
							radio_checked = idx;
						}
						is_clicked = true;
					}
					++idx;
				}
				idx = 0;
				for (menu_elem& elem : children) {
					if (elem.menu_mode() == radio_mode && radio_checked != -1) {
						elem.set_checked(idx == radio_checked);
					}
					elem.draw_noupdate();
					++idx;
				}
			}
		}
	}

	void draw() {
		update();
		draw_noupdate();
	}

	bool clicked() {
		return is_clicked;
	}

	bool active() {
		return is_active;
	}

	bool last_active() {
		return was_active;
	}

	RectF size() {
		RectF res = font(str).region(Point{ 0, 0 });
		res.w += res.h;
		return res;
	}

	void not_clicked() {
		is_clicked = false;
		if (has_child) {
			for (menu_elem& elem : children) {
				elem.not_clicked();
			}
		}
	}

	void clear() {
		has_child = false;
		children.clear();
	}

	int menu_mode() {
		return mode;
	}

	void set_checked(bool v) {
		*is_checked = v;
	}

	bool checked() {
		return *is_checked;
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
		clear();
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
		int height = h - menu_offset_y * 2, width = w - menu_offset_x * 2;
		for (menu_elem &elem : elems) {
			elem.pre_init(font, checkbox);
			RectF r = elem.size();
			height = max(height, (int)r.h);
			width = max(width, (int)r.w);
		}
		height += menu_offset_y * 2;
		width += menu_offset_x * 2;
		int xx = rect.x;
		int yy = rect.y + rect.h;
		for (menu_elem &elem : elems) {
			elem.init_inside(xx, yy, width, height);
			yy += height;
		}
	}

	void push(menu_elem elem) {
		elems.emplace_back(elem);
	}

	void draw() {
		bool n_is_open = false, clicked = false;
		if (rect.mouseOver()) {
			is_open = true;
			n_is_open = true;
		}
		if (is_open) {
			int radio_checked = -1;
			int idx = 0;
			for (menu_elem& elem : elems) {
				elem.update();
				if (elem.clicked() && elem.menu_mode() == radio_mode && !elem.checked()) {
					radio_checked = idx;
				}
				++idx;
			}
			idx = 0;
			for (menu_elem& elem : elems) {
				if (elem.menu_mode() == radio_mode && radio_checked != -1) {
					elem.set_checked(idx == radio_checked);
				}
				elem.draw_noupdate();
				n_is_open = n_is_open || elem.active();
				clicked = clicked || elem.clicked();
				++idx;
			}
		}
		else {
			for (menu_elem& elem : elems) {
				elem.not_clicked();
			}
		}
		if (is_open) {
			rect.draw(menu_select_color);
		}
		else {
			rect.draw(menu_color);
		}
		font(str).draw(Arg::topCenter(rect.x + rect.w / 2, rect.y + menu_offset_y), menu_font_color);
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
	}

	void init(int x, int y, Font f, Texture c) {
		int height = 0, width = 0;
		for (menu_title &elem : menu) {
			elem.pre_init(f, c);
			RectF r = elem.size();
			height = max(height, (int)r.h);
			width = max(width, (int)r.w);
		}
		height += menu_offset_y * 2;
		width += menu_offset_x * 2;
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

	bool active() {
		return is_open;
	}
};
