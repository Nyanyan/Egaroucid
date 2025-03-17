/*
    Egaroucid Project

    @file menu.hpp
        Menu class
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <Siv3D.hpp>
#include <iostream>
#include <vector>
#include "click.hpp"

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

constexpr int MENU_MODE_BUTTON = 0;
constexpr int MENU_MODE_BAR = 1;
constexpr int MENU_MODE_CHECK = 2;
constexpr int MENU_MODE_RADIO = 3;
constexpr int MENU_MODE_BAR_CHECK = 4;

constexpr int MENU_BAR_SIZE = 140;
constexpr int MENU_BAR_HEIGHT = 14;
constexpr int MENU_BAR_RADIUS = 6;

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
    bool was_active;
    Click_supporter click_supporter;
    bool is_clicked;

    // bar mode
    int *bar_elem;
    int min_elem;
    int max_elem;
    int bar_value_offset;
    Circle bar_circle;
    Rect bar_rect;
    int bar_sx;
    int bar_center_y;
    bool bar_changeable;

    // button mode
    bool *is_clicked_p;

    // check mode
    bool *is_checked;
    String unchecked_str;
    Texture checkbox;
    Texture unchecked;

    // display image on the menu (for language selection)
    bool use_image;
    Texture image;

public:
    void init_button(String s, bool *c) {
        clear();
        click_supporter.init();
        mode = MENU_MODE_BUTTON;
        has_child = false;
        is_active = false;
        was_active = false;
        str = s;
        bar_changeable = false;
        is_clicked_p = c;
        is_clicked = false;
        *is_clicked_p = is_clicked;
        use_image = false;
    }

    void init_bar(String s, int *c, int d, int mn, int mx) {
        clear();
        click_supporter.init();
        mode = MENU_MODE_BAR;
        has_child = false;
        is_active = false;
        was_active = false;
        str = s;
        bar_elem = c;
        *bar_elem = d;
        bar_changeable = false;
        min_elem = mn;
        max_elem = mx;
        is_clicked = false;
        use_image = false;
    }

    void init_check(String s, bool *c, bool d) {
        clear();
        click_supporter.init();
        mode = MENU_MODE_CHECK;
        has_child = false;
        is_active = false;
        was_active = false;
        str = s;
        bar_changeable = false;
        is_checked = c;
        *is_checked = d;
        is_clicked = false;
        use_image = false;
    }

    void init_radio(String s, bool* c, bool d) {
        clear();
        click_supporter.init();
        mode = MENU_MODE_RADIO;
        has_child = false;
        is_active = false;
        str = s;
        bar_changeable = false;
        is_checked = c;
        *is_checked = d;
        is_clicked = false;
        use_image = false;
    }

    void init_radio(Texture t, bool* c, bool d) {
        clear();
        click_supporter.init();
        mode = MENU_MODE_RADIO;
        has_child = false;
        is_active = false;
        bar_changeable = false;
        is_checked = c;
        *is_checked = d;
        is_clicked = false;
        use_image = true;
        image = t;
    }

    void init_bar_check(String s, int *c, int d, int mn, int mx, bool *e, bool f, String u) {
        clear();
        click_supporter.init();
        mode = MENU_MODE_BAR_CHECK;
        has_child = false;
        is_active = false;
        was_active = false;
        str = s;
        bar_changeable = false;
        bar_elem = c;
        *bar_elem = d;
        min_elem = mn;
        max_elem = mx;
        is_checked = e;
        *is_checked = f;
        unchecked_str = u;
        is_clicked = false;
        use_image = false;
    }

    void push(menu_elem ch) {
        has_child = true;
        children.emplace_back(ch);
    }

    void pre_init(int fs, Font f, Texture c, Texture u, int h, int bvo) {
        font_size = fs;
        font = f;
        checkbox = c;
        unchecked = u;
        bar_value_offset = bvo; //font(U"88").region(font_size, Point{ 0, 0 }).w;
        rect.h = h;
    }

    void init_inside(int x, int y, int w, int h) {
        rect.x = x;
        rect.y = y;
        rect.w = w;
        rect.h = h;
        if (has_child) {
            int height = h - menu_offset_y * 2, width = 0;
            for (menu_elem& child : children) {
                child.pre_init(font_size, font, checkbox, unchecked, height, bar_value_offset);
                std::pair<int, int> child_size = child.size();
                height = std::max(height, child_size.first);
                width = std::max(width, child_size.second);
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
        if (mode == MENU_MODE_BAR || mode == MENU_MODE_BAR_CHECK) {
            bar_sx = rect.x + rect.w - MENU_BAR_SIZE - bar_additional_offset;
            bar_center_y = rect.y + rect.h / 2;
            bar_rect.x = bar_sx;
            bar_rect.y = bar_center_y - MENU_BAR_HEIGHT / 2;
            bar_rect.w = MENU_BAR_SIZE;
            bar_rect.h = MENU_BAR_HEIGHT;
            bar_circle.x = bar_sx + MENU_BAR_SIZE * (*bar_elem - min_elem + 5) / (max_elem - min_elem + 10);
            bar_circle.y = bar_center_y;
            bar_circle.r = MENU_BAR_RADIUS;
        }
    }

    void update() {
        was_active = is_active;
        is_active = rect.mouseOver();
        if (mode == MENU_MODE_BAR || (mode == MENU_MODE_BAR_CHECK && (*is_checked))) {
            // bar active?
            if (bar_rect.leftClicked()) {
                bar_changeable = true;
            } else if (!MouseL.pressed()) {
                bar_changeable = false;
            }
            // bar is active -> this element is active
            is_active |= bar_changeable;
        }
        // if a child bar is active, other children must be inactive
        bool active_child_bar_found = false;
        for (menu_elem& child: children) {
            child.update();
            active_child_bar_found |= child.bar_active();
        }
        if (active_child_bar_found) {
            for (menu_elem& child: children) {
                if (!child.bar_active()) {
                    child.set_inactive();
                }
            }
        }
        // if a child is active, this element is active
        for (menu_elem& child: children) {
            is_active |= (child.active() && last_active());
        }
        // check clicked
        if (mode == MENU_MODE_BAR_CHECK) {
            Rect bar_check_rect = Rect(rect.x, rect.y, bar_sx - bar_value_offset - rect.x, rect.h);
            click_supporter.update(bar_check_rect);
            is_clicked = click_supporter.clicked();
        } else {
            click_supporter.update(rect);
            is_clicked = click_supporter.clicked();
        }
        // set bar position
        if ((mode == MENU_MODE_BAR || (mode == MENU_MODE_BAR_CHECK && (*is_checked))) && bar_changeable) {
            Cursor::RequestStyle(CursorStyle::ResizeLeftRight);
            int min_error = INF;
            int cursor_x = Cursor::Pos().x;
            for (int i = min_elem; i <= max_elem; ++i) {
                int x = round((double)bar_sx + 10.0 + (double)(MENU_BAR_SIZE - 20) * (double)(i - min_elem) / (double)(max_elem - min_elem));
                if (abs(cursor_x - x) < min_error) {
                    min_error = abs(cursor_x - x);
                    *bar_elem = i;
                }
            }
        }
    }

    void update_button() {
        if (mode == MENU_MODE_BUTTON) {
            *is_clicked_p = is_clicked;
        }
    }

    void draw_noupdate() {
        if (mode == MENU_MODE_BUTTON) {
            *is_clicked_p = is_clicked;
        }
        if (is_clicked && (mode == MENU_MODE_CHECK || mode == MENU_MODE_BAR_CHECK)) {
            *is_checked = !(*is_checked);
        }
        if (is_active) {
            rect.draw(menu_active_color);
        } else {
            rect.draw(menu_select_color);
        }
        if (use_image) {
            image.scaled((double)(rect.h - 2 * menu_image_offset_y) / image.height()).draw(rect.x + rect.h - menu_offset_y, rect.y + menu_image_offset_y);
        } else {
            font(str).draw(font_size, rect.x + rect.h - menu_offset_y, rect.y + menu_offset_y, menu_font_color);
        }
        if (mode == MENU_MODE_BAR || mode == MENU_MODE_BAR_CHECK) {
            if (mode == MENU_MODE_BAR_CHECK && !(*is_checked)) {
                font(unchecked_str).draw(font_size, bar_sx - menu_offset_x - menu_child_offset - font(unchecked_str).region(font_size, Point{ 0, 0 }).w, rect.y + menu_offset_y, menu_font_color);
            } else {
                font(*bar_elem).draw(font_size, bar_sx - menu_offset_x - menu_child_offset - bar_value_offset, rect.y + menu_offset_y, menu_font_color);
            }
            if (mode == MENU_MODE_BAR_CHECK && !(*is_checked)) {
                bar_rect.draw(ColorF(bar_color, 0.5));
            } else {
                bar_rect.draw(bar_color);
            }
            bar_circle.x = round((double)bar_sx + 10.0 + (double)(MENU_BAR_SIZE - 20) * (double)(*bar_elem - min_elem) / (double)(max_elem - min_elem));
            if (mode == MENU_MODE_BAR_CHECK && !(*is_checked)) {
                Shape2D::Cross(1.5 * bar_circle.r, 6, Vec2{bar_circle.x, bar_circle.y}).draw(ColorF(bar_circle_color, 0.5));
            } else {
                bar_circle.draw(bar_circle_color);
            }
        }
        if (has_child) {
            font(U">").draw(font_size, rect.x + rect.w - menu_offset_x - menu_child_offset, rect.y + menu_offset_y, menu_font_color);
            if (is_active) {
                int radio_checked = -1;
                int idx = 0;
                for (menu_elem& child: children) {
                    if (child.clicked()) {
                        if (child.menu_mode() == MENU_MODE_RADIO && !child.checked()) {
                            radio_checked = idx;
                        }
                        is_clicked = true;
                    }
                    ++idx;
                }
                idx = 0;
                for (menu_elem& child: children) {
                    if (child.menu_mode() == MENU_MODE_RADIO && radio_checked != -1) {
                        child.set_checked(idx == radio_checked);
                    }
                    child.draw_noupdate();
                    ++idx;
                }
            }
        }
        if (mode == MENU_MODE_CHECK || mode == MENU_MODE_BAR_CHECK) {
            if (*is_checked) {
                checkbox.scaled((double)(rect.h - 2 * menu_offset_y) / checkbox.width()).draw(rect.x + menu_offset_y, rect.y + menu_offset_y);
            } else {
                unchecked.scaled((double)(rect.h - 2 * menu_offset_y) / unchecked.width()).draw(rect.x + menu_offset_y, rect.y + menu_offset_y);
            }
        } else if (mode == MENU_MODE_RADIO) {
            if (*is_checked) {
                Circle(rect.x + rect.h / 2, rect.y + rect.h / 2, (int)(rect.h * radio_ratio)).draw(radio_color);
            }
        }
    }

    void set_inactive() {
        is_active = false;
    }

    void draw() {
        update();
        draw_noupdate();
    }

    bool clicked() {
        return is_clicked;
    }
    
    bool bar_active() {
        return bar_changeable;
    }

    bool active() {
        return is_active;
    }

    bool last_active() {
        return was_active;
    }

    std::pair<int, int> size() {
        int h, w;
        if (use_image) {
            h = rect.h - 2 * menu_image_offset_y;
            w = (double)h * image.width() / image.height();
        } else {
            RectF rect = font(str).region(font_size, Point{ 0, 0 });
            h = rect.h;
            w = rect.w;
        }
        w += h;
        if (mode == MENU_MODE_BAR || mode == MENU_MODE_BAR_CHECK) {
            w += MENU_BAR_SIZE + bar_value_offset + bar_additional_offset;
        }
        return std::make_pair(h, w);
    }

    void set_not_clicked() {
        is_clicked = false;
        if (has_child) {
            for (menu_elem& child: children) {
                child.set_not_clicked();
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
    std::vector<menu_elem> children;
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
        uint64_t strt = tim();
        std::cerr << "a";
        rect.x = x;
        rect.y = y;
        rect.w = w;
        rect.h = h;
        int height = h - menu_offset_y * 2, width = w - menu_offset_x * 2;
        int bar_value_offset = font(U"88").region(font_size, Point{ 0, 0 }).w;
        for (menu_elem &child: children) {
            child.pre_init(font_size, font, checkbox, unchecked, height, bar_value_offset);
            std::pair<int, int> child_size = child.size();
            height = std::max(height, child_size.first);
            width = std::max(width, child_size.second);
        }
        height += menu_offset_y * 2;
        width += menu_offset_x * 2;
        int xx = rect.x;
        int yy = rect.y + rect.h;
        std::cerr << "b " << tim() - strt << " ";
        for (menu_elem &child: children) {
            child.init_inside(xx, yy, width, height);
            yy += height;
        }
        std::cerr << "c " << tim() - strt << std::endl;
    }

    void push(menu_elem elem) {
        children.emplace_back(elem);
    }

    void draw() {
        bool n_is_open = false, clicked = false;
        if (rect.mouseOver()) {
            is_open = true;
            n_is_open = true;
        }
        if (is_open) {
            int radio_checked = -1;
            bool active_child_bar_found = false;
            int idx = 0;
            for (menu_elem& child: children) {
                child.update();
                // radio button check
                if (child.clicked() && child.menu_mode() == MENU_MODE_RADIO && !child.checked()) {
                    radio_checked = idx;
                }
                // active bar check
                active_child_bar_found |= child.bar_active();
                ++idx;
            }
            // radio button update
            idx = 0;
            for (menu_elem& child: children) {
                if (child.menu_mode() == MENU_MODE_RADIO && radio_checked != -1) {
                    child.set_checked(idx == radio_checked);
                }
                ++idx;
            }
            // bar update
            if (active_child_bar_found) {
                for (menu_elem& child: children) {
                    if (!child.bar_active()) {
                        child.set_inactive();
                    }
                }
            }
            // draw children
            for (menu_elem& child: children) {
                child.draw_noupdate();
                n_is_open = n_is_open || child.active();
                clicked = clicked || child.clicked();
                ++idx;
            }
        } else {
            for (menu_elem& child: children) {
                child.set_not_clicked();
                child.update_button();
            }
        }
        if (is_open) {
            rect.draw(menu_select_color);
        } else {
            rect.draw(menu_color);
        }
        font(str).draw(font_size, Arg::topCenter(rect.x + rect.w / 2, rect.y + menu_offset_y), menu_font_color);
        is_open = n_is_open;
        //if (clicked)
        //    is_open = false;
    }

    void draw_title() {
        rect.draw(menu_color);
        font(str).draw(font_size, Arg::topCenter(rect.x + rect.w / 2, rect.y + menu_offset_y), menu_font_color);
    }

    // std::pair<int, int> size() {
    //     RectF rect = font(str).region(font_size, Point{ 0, 0 });
    //     return std::make_pair(rect.h, rect.w);
    // }

    RectF size() {
        return font(str).region(font_size, Point{ 0, 0 });
    }

    bool open() {
        return is_open;
    }

    void clear() {
        children.clear();
    }
};

class Menu {
private:
    bool is_open;
    std::vector<menu_title> menu;

public:
    void push(menu_title elem) {
        menu.emplace_back(elem);
    }

    void init(int x, int y, int fs, Font f, Texture c, Texture u) {
        int height = 0, width = 0;
        for (menu_title &title : menu) {
            title.pre_init(fs, f, c, u);
            RectF r = title.size();
            height = std::max(height, (int)r.h);
            width = std::max(width, (int)r.w);
            // std::pair<int, int> title_size = title.size();
            // height = std::max(height, title_size.first);
            // width = std::max(width, title_size.second);
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