/*
    Egaroucid Project

	@file menu.hpp
		Menu class
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <Siv3D.hpp>
#include <iostream>
#include <vector>

constexpr Color menu_color = Palette::Gainsboro;
constexpr Color menu_active_color = Palette::Lightblue;
constexpr Color menu_select_color = Palette::Lightcyan;
constexpr Color menu_font_color = Palette::Black;
constexpr Color radio_color = Palette::Deepskyblue;
constexpr Color bar_color = Palette::Lightskyblue;
constexpr Color bar_circle_color = Palette::Deepskyblue;
constexpr int menu_offset_x = 10;
constexpr int menu_offset_y = 1;
constexpr int menu_image_offset_y = 3;
constexpr int menu_child_offset = 2;
constexpr int bar_additional_offset = 20;
constexpr double radio_ratio = 0.2;

#define button_mode 0
#define bar_mode 1
#define check_mode 2
#define radio_mode 3
#define bar_check_mode 4
#define bar_size 140
#define bar_height 14
#define bar_radius 6

#define MENU_TRIANGLE_ACTIVE_TIME 100

class menu_elem {
private:
	String str;
	Rect rect;
	Font font;
	int font_size;
	int mode;
	bool has_child;
	std::vector<menu_elem> children;
	bool is_active;
	uint64_t last_active_on_cell;
	bool was_active;
	int *bar_elem;
	bool *is_clicked_p;
	bool is_clicked;
	bool *is_checked;
	bool dammy_clicked;
	Texture checkbox;
	Texture unchecked;
	int min_elem;
	int max_elem;
	int bar_value_offset;
	Circle bar_circle;
	Rect bar_rect;
	int bar_sx;
	int bar_center_y;
	Texture image;
	bool use_image;

public:
	menu_elem(){
		use_image = false;
	}

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

	void init_bar(String s, int *c, int d, int mn, int mx) {
		clear();
		mode = bar_mode;
		has_child = false;
		is_active = false;
		was_active = false;
		str = s;
		bar_elem = c;
		*bar_elem = d;
		min_elem = mn;
		max_elem = mx;
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

	void init_radio(Texture t, bool* c, bool d) {
		clear();
		mode = radio_mode;
		has_child = false;
		is_active = false;
		is_checked = c;
		*is_checked = d;
		is_clicked = false;
		use_image = true;
		image = t;
	}

	void init_bar_check(String s, int *c, int d, int mn, int mx, bool *e, bool f){
		clear();
		mode = bar_check_mode;
		has_child = false;
		is_active = false;
		was_active = false;
		str = s;
		bar_elem = c;
		*bar_elem = d;
		min_elem = mn;
		max_elem = mx;
		is_checked = e;
		*is_checked = f;
		is_clicked = false;
	}

	void push(menu_elem ch) {
		has_child = true;
		children.emplace_back(ch);
	}

	void pre_init(int fs, Font f, Texture c, Texture u, int h) {
		font_size = fs;
		font = f;
		checkbox = c;
		unchecked = u;
		bar_value_offset = font(U"88").region(font_size, Point{ 0, 0 }).w;
		rect.h = h;
	}

	void init_inside(int x, int y, int w, int h) {
		rect.x = x;
		rect.y = y;
		rect.w = w;
		rect.h = h;
		if (has_child) {
			int height = h - menu_offset_y * 2, width = 0;
			for (menu_elem& elem : children) {
				elem.pre_init(font_size, font, checkbox, unchecked, height);
				RectF r = elem.size();
				height = std::max(height, (int)r.h);
				width = std::max(width, (int)r.w);
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
		if (mode == bar_mode || mode == bar_check_mode) {
			bar_sx = rect.x + rect.w - bar_size - bar_additional_offset;
			bar_center_y = rect.y + rect.h / 2;
			bar_rect.x = bar_sx;
			bar_rect.y = bar_center_y - bar_height / 2;
			bar_rect.w = bar_size;
			bar_rect.h = bar_height;
			bar_circle.x = bar_sx + bar_size * (*bar_elem - min_elem + 5) / (max_elem - min_elem + 10);
			bar_circle.y = bar_center_y;
			bar_circle.r = bar_radius;
		}
	}

	void update() {
		was_active = is_active;
		is_active = rect.mouseOver();
		for (menu_elem& elem : children) {
			elem.update();
			is_active = is_active || (elem.active() && last_active());
		}
		if (is_active){
			last_active_on_cell = tim();
		} else if (tim() - last_active_on_cell < MENU_TRIANGLE_ACTIVE_TIME){
			int x0 = rect.x + rect.w / 2;
			int x1 = rect.x + rect.w;
			int y0 = rect.y + rect.h;
			int y1 = rect.y + rect.h + rect.h * children.size() / 2;
			Point mouse_pos = Cursor::Pos();
			is_active = x0 <= mouse_pos.x && mouse_pos.x <= x1 && y0 <= mouse_pos.y && mouse_pos.y <= y1;
		}
		if (mode == bar_check_mode){
			is_clicked = Rect(rect.x, rect.y, bar_sx - bar_value_offset - rect.x, rect.h).leftClicked();
		} else
			is_clicked = rect.leftClicked();
		if ((mode == bar_mode || (mode == bar_check_mode && (*is_checked))) && bar_rect.leftPressed()) {
			int min_error = INF;
			int cursor_x = Cursor::Pos().x;
			for (int i = min_elem; i <= max_elem; ++i) {
				int x = round((double)bar_sx + 10.0 + (double)(bar_size - 20) * (double)(i - min_elem) / (double)(max_elem - min_elem));
				if (abs(cursor_x - x) < min_error) {
					min_error = abs(cursor_x - x);
					*bar_elem = i;
				}
			}
		}
	}

	void update_button() {
		if (mode == button_mode) {
			*is_clicked_p = is_clicked;
		}
	}

	void draw_noupdate() {
		if (mode == button_mode) {
			*is_clicked_p = is_clicked;
		}
		if (is_clicked && (mode == check_mode || mode == bar_check_mode)) {
			*is_checked = !(*is_checked);
		}
		if (is_active) {
			rect.draw(menu_active_color);
		}
		else {
			rect.draw(menu_select_color);
		}
		if (use_image){
			image.scaled((double)(rect.h - 2 * menu_image_offset_y) / image.height()).draw(rect.x + rect.h - menu_offset_y, rect.y + menu_image_offset_y);
		} else{
			font(str).draw(font_size, rect.x + rect.h - menu_offset_y, rect.y + menu_offset_y, menu_font_color);
		}
		if (mode == bar_mode || mode == bar_check_mode) {
			font(*bar_elem).draw(font_size, bar_sx - menu_offset_x - menu_child_offset - bar_value_offset, rect.y + menu_offset_y, menu_font_color);
			if (mode == bar_check_mode && !(*is_checked))
				bar_rect.draw(ColorF(bar_color, 0.5));
			else
				bar_rect.draw(bar_color);
			bar_circle.x = round((double)bar_sx + 10.0 + (double)(bar_size - 20) * (double)(*bar_elem - min_elem) / (double)(max_elem - min_elem));
			if (mode == bar_check_mode && !(*is_checked))
				Shape2D::Cross(1.5 * bar_circle.r, 6, Vec2{bar_circle.x, bar_circle.y}).draw(ColorF(bar_circle_color, 0.5));
			else
				bar_circle.draw(bar_circle_color);
		}
		if (has_child) {
			font(U">").draw(font_size, rect.x + rect.w - menu_offset_x - menu_child_offset, rect.y + menu_offset_y, menu_font_color);
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
			/*
			else {
				for (menu_elem& elem : children) {
					elem.update_button();
				}
			}
			*/
		}
		if (mode == check_mode || mode == bar_check_mode) {
			if (*is_checked) {
				checkbox.scaled((double)(rect.h - 2 * menu_offset_y) / checkbox.width()).draw(rect.x + menu_offset_y, rect.y + menu_offset_y);
			}
			else {
				unchecked.scaled((double)(rect.h - 2 * menu_offset_y) / unchecked.width()).draw(rect.x + menu_offset_y, rect.y + menu_offset_y);
			}
		}
		else if (mode == radio_mode) {
			if (*is_checked) {
				Circle(rect.x + rect.h / 2, rect.y + rect.h / 2, (int)(rect.h * radio_ratio)).draw(radio_color);
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
		RectF res;
		if (use_image){
			res.x = 0;
			res.y = 0;
			res.h = rect.h - 2 * menu_image_offset_y;
			res.w = (double)res.h * image.width() / image.height();
		} else{
			res = font(str).region(font_size, Point{ 0, 0 });
		}
		res.w += res.h;
		if (mode == bar_mode || mode == bar_check_mode) {
			res.w += bar_size + bar_value_offset + bar_additional_offset;
		}
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
	int font_size;
	bool is_open;
	std::vector<menu_elem> elems;
	Texture checkbox;
	Texture unchecked;

public:
	void init(String s) {
		clear();
		str = s;
		is_open = false;
	}

	void pre_init(int fs, Font f, Texture c, Texture u) {
		font_size = fs;
		font = f;
		checkbox = c;
		unchecked = u;
	}

	void init_inside(int x, int y, int w, int h) {
		rect.x = x;
		rect.y = y;
		rect.w = w;
		rect.h = h;
		int height = h - menu_offset_y * 2, width = w - menu_offset_x * 2;
		for (menu_elem &elem : elems) {
			elem.pre_init(font_size, font, checkbox, unchecked, height);
			RectF r = elem.size();
			height = std::max(height, (int)r.h);
			width = std::max(width, (int)r.w);
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
			bool active_found = false;
			for (menu_elem& elem : elems) {
				if (active_found){
					elem.draw_noupdate();
				} else{
					elem.update();
					active_found |= elem.active();
				}
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
				elem.update_button();
			}
		}
		if (is_open) {
			rect.draw(menu_select_color);
		}
		else {
			rect.draw(menu_color);
		}
		font(str).draw(font_size, Arg::topCenter(rect.x + rect.w / 2, rect.y + menu_offset_y), menu_font_color);
		is_open = n_is_open;
		//if (clicked)
		//	is_open = false;
	}

	void draw_title() {
		rect.draw(menu_color);
		font(str).draw(font_size, Arg::topCenter(rect.x + rect.w / 2, rect.y + menu_offset_y), menu_font_color);
	}

	RectF size() {
		return font(str).region(font_size, Point{ 0, 0 });
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
	std::vector<menu_title> menu;

public:
	void push(menu_title elem) {
		menu.emplace_back(elem);
	}

	void init(int x, int y, int fs, Font f, Texture c, Texture u) {
		int height = 0, width = 0;
		for (menu_title &elem : menu) {
			elem.pre_init(fs, f, c, u);
			RectF r = elem.size();
			height = std::max(height, (int)r.h);
			width = std::max(width, (int)r.w);
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

	void draw_title() {
		is_open = false;
		for (menu_title& elem : menu) {
			elem.draw_title();
		}
	}

	bool active() {
		return is_open;
	}
};