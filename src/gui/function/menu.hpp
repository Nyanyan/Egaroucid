/*
    Egaroucid Project

    @file menu.hpp
        Menu class
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <Siv3D.hpp>
#include <algorithm>
#include <iostream>
#include <vector>
#include "click.hpp"
#include "./../../engine/engine_all.hpp"

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
constexpr int bar_additional_offset = 6;
constexpr double radio_ratio = 0.2;

constexpr int MENU_MODE_BUTTON = 0;
constexpr int MENU_MODE_BAR = 1;
constexpr int MENU_MODE_2BARS = 2;
constexpr int MENU_MODE_CHECK = 3;
constexpr int MENU_MODE_RADIO = 4;
constexpr int MENU_MODE_BAR_CHECK = 5;

constexpr int MENU_BAR_SIZE = 150;
constexpr int MENU_BAR_HEIGHT = 14;
constexpr int MENU_BAR_RADIUS = 6;
constexpr int MENU_BAR_MARGIN_H = 7;
constexpr int MENU_2BARS_MARGIN_H = 14;


// text width size ratio
// text_width_size_ratio[lang_name] = {<ascii No.0's width / font_size>, <ascii No.1's width / font_size>, ...}
std::unordered_map<std::string, std::vector<double>> text_width_size_ratio = {
    {"japanese",                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 2.32031, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.290039, 0.362956, 0.400065, 0.67806, 0.605143, 0.847005, 0.711914, 0.236979, 0.385091, 0.385091, 0.527018, 0.733073, 0.304036, 0.457031, 0.282878, 0.507161, 0.620117, 0.620117, 0.620117, 0.620117, 0.620117, 0.620117, 0.620117, 0.620117, 0.620117, 0.620117, 0.362956, 0.384115, 0.666992, 0.75293, 0.666992, 0.611003, 0.822917, 0.693034, 0.597982, 0.694987, 0.681966, 0.586914, 0.57194, 0.745117, 0.709961, 0.317057, 0.527018, 0.633138, 0.584961, 0.857096, 0.712891, 0.792969, 0.599935, 0.785156, 0.620117, 0.572917, 0.652995, 0.682943, 0.693034, 0.953125, 0.650065, 0.669922, 0.637044, 0.470052, 0.507161, 0.470052, 0.652995, 0.583008, 0.277995, 0.547852, 0.578125, 0.522135, 0.578125, 0.552083, 0.522135, 0.572917, 0.572917, 0.306966, 0.347005, 0.561849, 0.306966, 0.805013, 0.568034, 0.569987, 0.572917, 0.572917, 0.453125, 0.49707, 0.512044, 0.557943, 0.556966, 0.809896, 0.55306, 0.556966, 0.546875, 0.513021, 0.343099, 0.513021, 0.690104, 0}},
    {"english",                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 2.32031, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.290039, 0.362956, 0.400065, 0.67806, 0.605143, 0.847005, 0.711914, 0.236979, 0.385091, 0.385091, 0.527018, 0.733073, 0.304036, 0.457031, 0.282878, 0.507161, 0.620117, 0.620117, 0.620117, 0.620117, 0.620117, 0.620117, 0.620117, 0.620117, 0.620117, 0.620117, 0.362956, 0.384115, 0.666992, 0.75293, 0.666992, 0.611003, 0.822917, 0.693034, 0.597982, 0.694987, 0.681966, 0.586914, 0.57194, 0.745117, 0.709961, 0.317057, 0.527018, 0.633138, 0.584961, 0.857096, 0.712891, 0.792969, 0.599935, 0.785156, 0.620117, 0.572917, 0.652995, 0.682943, 0.693034, 0.953125, 0.650065, 0.669922, 0.637044, 0.470052, 0.507161, 0.470052, 0.652995, 0.583008, 0.277995, 0.547852, 0.578125, 0.522135, 0.578125, 0.552083, 0.522135, 0.572917, 0.572917, 0.306966, 0.347005, 0.561849, 0.306966, 0.805013, 0.568034, 0.569987, 0.572917, 0.572917, 0.453125, 0.49707, 0.512044, 0.557943, 0.556966, 0.809896, 0.55306, 0.556966, 0.546875, 0.513021, 0.343099, 0.513021, 0.690104, 0}},
    {"simplified_chinese",          {0, 0, 0, 0, 0, 0, 0, 0, 0, 1.79167, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.223958, 0.322917, 0.474935, 0.555013, 0.555013, 0.920898, 0.680013, 0.277995, 0.337891, 0.337891, 0.467122, 0.555013, 0.277995, 0.347005, 0.277995, 0.391927, 0.555013, 0.555013, 0.555013, 0.555013, 0.555013, 0.555013, 0.555013, 0.555013, 0.555013, 0.555013, 0.277995, 0.277995, 0.555013, 0.555013, 0.555013, 0.473958, 0.945964, 0.607096, 0.656901, 0.638021, 0.686849, 0.587891, 0.552083, 0.689128, 0.727865, 0.292969, 0.536133, 0.646159, 0.542969, 0.811849, 0.722005, 0.741862, 0.632161, 0.741862, 0.636068, 0.597005, 0.598958, 0.721029, 0.57487, 0.87793, 0.572917, 0.530924, 0.602865, 0.337891, 0.391927, 0.337891, 0.555013, 0.558919, 0.60612, 0.564128, 0.617839, 0.509115, 0.620117, 0.554036, 0.32487, 0.563151, 0.607096, 0.275065, 0.276042, 0.552083, 0.285156, 0.926107, 0.611003, 0.60612, 0.620117, 0.620117, 0.388021, 0.468099, 0.376953, 0.607096, 0.521159, 0.802083, 0.498047, 0.521159, 0.473958, 0.337891, 0.26888, 0.337891, 0.555013, 0}},
    {"traditional_chinese_taiwan",  {0, 0, 0, 0, 0, 0, 0, 0, 0, 1.79167, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.223958, 0.322917, 0.474935, 0.555013, 0.555013, 0.920898, 0.680013, 0.277995, 0.337891, 0.337891, 0.467122, 0.555013, 0.277995, 0.347005, 0.277995, 0.391927, 0.555013, 0.555013, 0.555013, 0.555013, 0.555013, 0.555013, 0.555013, 0.555013, 0.555013, 0.555013, 0.277995, 0.277995, 0.555013, 0.555013, 0.555013, 0.473958, 0.945964, 0.607096, 0.656901, 0.638021, 0.686849, 0.587891, 0.552083, 0.689128, 0.727865, 0.292969, 0.536133, 0.646159, 0.542969, 0.811849, 0.722005, 0.741862, 0.632161, 0.741862, 0.636068, 0.597005, 0.598958, 0.721029, 0.57487, 0.87793, 0.572917, 0.530924, 0.602865, 0.337891, 0.391927, 0.337891, 0.555013, 0.558919, 0.60612, 0.564128, 0.617839, 0.509115, 0.620117, 0.554036, 0.32487, 0.563151, 0.607096, 0.275065, 0.276042, 0.552083, 0.285156, 0.926107, 0.611003, 0.60612, 0.620117, 0.620117, 0.388021, 0.468099, 0.376953, 0.607096, 0.521159, 0.802083, 0.498047, 0.521159, 0.473958, 0.337891, 0.26888, 0.337891, 0.555013, 0}}
};

void generate_text_width_size_ratio(Font font) {
    for (int i = 0; i < 128; ++i) {
        char c = i;
        std::string s = std::string(1, c);
        // std::cerr << i << " " << c << " " << s << " ";
        // std::cerr << font(Unicode::Widen(s)).region(1, {0, 0}).h << " ";
        std::cerr << font(Unicode::Widen(s)).region(1, {0, 0}).w << ", ";
        // std::cerr << std::endl;
    }
    std::cerr << std::endl;
}


int count_ascii(const String& str) {
    int count = 0;
    for (const auto& ch : str) {
        if (InRange<int32>(ch, 0, 127)) {
            ++count;
        }
    }
    return count;
}

double region_ascii(const String& str, int font_size, std::string lang_name, Font font) {
    double width = 0.0;
    std::vector<double> ratio;
    bool ratio_found = false;
    if (text_width_size_ratio.find(lang_name) != text_width_size_ratio.end()) {
        ratio = text_width_size_ratio[lang_name];
        ratio_found = true;
    }
    for (const auto& ch : str) {
        if (InRange<int32>(ch, 0, 127)) {
            if (ratio_found) {
                width += ratio[static_cast<int>(ch)] * font_size;
            } else {
                width += font(ch).region(font_size, Point{0, 0}).w;
            }
        }
    }
    return width;
}

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

    // 2 bars mode
    int *bar_elem1;
    int *bar_elem2;
    Texture arrow_left;
    int bar_active_circle = 0;
    int min_bar_gap = 2;

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

    int cursor_to_bar_value(int cursor_x, bool is_right_elem = false) const {
        if (max_elem == min_elem) {
            return min_elem;
        }
        // For MENU_MODE_2BARS:
        // - Left element (bar_elem1, is_right_elem=false): use right edge (cursor_x + MENU_BAR_RADIUS)
        // - Right element (bar_elem2, is_right_elem=true): use left edge (cursor_x - MENU_BAR_RADIUS)
        int effective_x = cursor_x;
        if (mode == MENU_MODE_2BARS) {
            effective_x = is_right_elem ? (cursor_x - MENU_BAR_RADIUS) : (cursor_x + MENU_BAR_RADIUS);
        }
        double margin = MENU_BAR_MARGIN_H;
        if (mode == MENU_MODE_2BARS) {
            margin = MENU_2BARS_MARGIN_H;
        }
        double ratio = (double)(effective_x - (bar_sx + margin)) / (double)(MENU_BAR_SIZE - margin * 2);
        ratio = std::clamp(ratio, 0.0, 1.0);
        int value = (int)round(ratio * (max_elem - min_elem)) + min_elem;
        return std::clamp(value, min_elem, max_elem);
    }

    int value_to_bar_x(int value, bool is_right_elem = false) const {
        if (max_elem == min_elem) {
            return bar_sx + (MENU_BAR_SIZE / 2);
        }
        double ratio = (double)(value - min_elem) / (double)(max_elem - min_elem);
        ratio = std::clamp(ratio, 0.0, 1.0);
        double margin = MENU_BAR_MARGIN_H;
        if (mode == MENU_MODE_2BARS) {
            margin = MENU_2BARS_MARGIN_H;
        }
        int x = (int)round(bar_sx + margin + (double)(MENU_BAR_SIZE - margin * 2) * ratio);
        // For MENU_MODE_2BARS:
        // - Left element (bar_elem1, is_right_elem=false): value represents right edge, so subtract radius
        // - Right element (bar_elem2, is_right_elem=true): value represents left edge, so add radius
        if (mode == MENU_MODE_2BARS) {
            x = is_right_elem ? (x + MENU_BAR_RADIUS) : (x - MENU_BAR_RADIUS);
        }
        return x;
    }

    void refresh_bar_circles() {
        if (mode == MENU_MODE_BAR || mode == MENU_MODE_BAR_CHECK) {
            bar_circle.x = value_to_bar_x(*bar_elem);
            bar_circle.y = bar_center_y;
            bar_circle.r = MENU_BAR_RADIUS;
        }
    }

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

    void init_button(Texture t, String s, bool *c) {
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
        use_image = true;
        image = t;
    }

    void init_2bars(String s, int *c1, int *c2, int d1, int d2, int mn, int mx, int mgap, Texture al) {
        clear();
        click_supporter.init();
        mode = MENU_MODE_2BARS;
        has_child = false;
        is_active = false;
        was_active = false;
        str = s;
        bar_elem1 = c1;
        bar_elem2 = c2;
        *bar_elem1 = d1;
        *bar_elem2 = d2;
        bar_changeable = false;
        bar_active_circle = 0;
        min_elem = mn;
        max_elem = mx;
        min_bar_gap = mgap;
        is_clicked = false;
        use_image = false;
        arrow_left = al;
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
        bar_active_circle = 0;
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

    void init_inside(int x, int y, int w, int h, std::string lang_name) {
        rect.x = x;
        rect.y = y;
        rect.w = w;
        rect.h = h;
        if (has_child) {
            int height = h - menu_offset_y * 2, width = 0;
            for (menu_elem& child : children) {
                child.pre_init(font_size, font, checkbox, unchecked, height, bar_value_offset);
                std::pair<int, int> child_size = child.size(lang_name);
                height = std::max(height, child_size.first);
                width = std::max(width, child_size.second);
            }
            height += menu_offset_y * 2;
            width += menu_offset_x * 2;
            int xx = rect.x + rect.w;
            int yy = rect.y;
            for (menu_elem& elem : children) {
                elem.init_inside(xx, yy, width, height, lang_name);
                yy += height;
            }
        }
        if (mode == MENU_MODE_BAR || mode == MENU_MODE_2BARS || mode == MENU_MODE_BAR_CHECK) {
            bar_sx = rect.x + rect.w - MENU_BAR_SIZE - bar_additional_offset;
            bar_center_y = rect.y + rect.h / 2;
            bar_rect.x = bar_sx;
            bar_rect.y = bar_center_y - MENU_BAR_HEIGHT / 2;
            bar_rect.w = MENU_BAR_SIZE;
            bar_rect.h = MENU_BAR_HEIGHT;
            refresh_bar_circles();
        }
    }

    void update() {
        was_active = is_active;
        is_active = rect.mouseOver();
        if (mode == MENU_MODE_BAR || mode == MENU_MODE_2BARS || mode == MENU_MODE_BAR_CHECK) {
            refresh_bar_circles();
        }
        if (mode == MENU_MODE_2BARS) {
            const int handle1_x = value_to_bar_x(*bar_elem1, false);
            const int handle2_x = value_to_bar_x(*bar_elem2, true);
            const Circle handle_circle1(handle1_x, bar_center_y, MENU_BAR_RADIUS);
            const Circle handle_circle2(handle2_x, bar_center_y, MENU_BAR_RADIUS);
            const bool circle1_clicked = handle_circle1.leftClicked();
            const bool circle2_clicked = handle_circle2.leftClicked();
            const bool bar_clicked = bar_rect.leftClicked();
            if (circle1_clicked && circle2_clicked) {
                bar_changeable = true;
                const int center_x = (bar_rect.x + bar_rect.x + bar_rect.w) / 2;
                const int dist1 = std::abs(handle1_x - center_x);
                const int dist2 = std::abs(handle2_x - center_x);
                bar_active_circle = (dist1 <= dist2) ? 1 : 2;
            } else if (circle1_clicked) {
                bar_changeable = true;
                bar_active_circle = 1;
            } else if (circle2_clicked) {
                bar_changeable = true;
                bar_active_circle = 2;
            } else if (bar_clicked) {
                bar_changeable = true;
            }
            if (!MouseL.pressed()) {
                bar_changeable = false;
                bar_active_circle = 0;
            }
            if (bar_changeable) {
                Cursor::RequestStyle(CursorStyle::ResizeLeftRight);
                const int cursor_x = Cursor::Pos().x;
                const int circle1_x = handle1_x;
                const int circle2_x = handle2_x;

                if (bar_active_circle == 1) {
                    // Move only the first handle, never modify bar_elem2 here
                    const int cursor_value = cursor_to_bar_value(cursor_x, false);
                    int newv = std::clamp(cursor_value, min_elem, *bar_elem2 - min_bar_gap);
                    *bar_elem1 = newv;
                } else if (bar_active_circle == 2) {
                    // Move only the second handle, never modify bar_elem1 here
                    const int cursor_value = cursor_to_bar_value(cursor_x, true);
                    int newv = std::clamp(cursor_value, *bar_elem1 + min_bar_gap, max_elem);
                    *bar_elem2 = newv;
                } else {
                    // No specific handle active: pick the nearer handle and move only it
                    int dist1 = cursor_x - circle1_x; if (dist1 < 0) dist1 = -dist1;
                    int dist2 = cursor_x - circle2_x; if (dist2 < 0) dist2 = -dist2;
                    if (dist1 <= dist2) {
                        const int cursor_value = cursor_to_bar_value(cursor_x, false);
                        int newv = std::clamp(cursor_value, min_elem, *bar_elem2 - min_bar_gap);
                        *bar_elem1 = newv;
                        bar_active_circle = 1;
                    } else {
                        const int cursor_value = cursor_to_bar_value(cursor_x, true);
                        int newv = std::clamp(cursor_value, *bar_elem1 + min_bar_gap, max_elem);
                        *bar_elem2 = newv;
                        bar_active_circle = 2;
                    }
                }

                // final clamp to ensure within [min_elem, max_elem]
                *bar_elem1 = std::clamp(*bar_elem1, min_elem, max_elem);
                *bar_elem2 = std::clamp(*bar_elem2, min_elem, max_elem);
            }
            is_active |= bar_changeable;
        } else if (mode == MENU_MODE_BAR || (mode == MENU_MODE_BAR_CHECK && (*is_checked))) {
            bar_active_circle = 0;
            if (bar_rect.leftClicked()) {
                bar_changeable = true;
            } else if (!MouseL.pressed()) {
                bar_changeable = false;
            }
            if (bar_changeable) {
                Cursor::RequestStyle(CursorStyle::ResizeLeftRight);
                *bar_elem = cursor_to_bar_value(Cursor::Pos().x);
            }
            is_active |= bar_changeable;
        } else if (!MouseL.pressed()) {
            bar_changeable = false;
            bar_active_circle = 0;
        }
        // if a child bar is active, other children must be inactive
        // Only update children when this element is currently active or was active
        // (to allow smooth transitions). When the parent is inactive, avoid
        // calling child.update() so children don't react while parent is closed.
        bool active_child_bar_found = false;
        for (menu_elem& child: children) {
            if (is_active || was_active) {
                child.update();
                active_child_bar_found |= child.bar_active();
            } else {
                // Make sure inactive parent's children remain inactive and don't
                // keep stale states (like bar_changeable) from previous frames.
                child.set_inactive();
            }
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
            image.scaled((double)(rect.h - 2 * menu_image_offset_y) / (double)image.height()).draw(rect.x + rect.h - menu_offset_y, rect.y + menu_image_offset_y);
            double x_offset = (double)(rect.h - 2 * menu_image_offset_y) / (double)image.height() * (double)image.width();
            font(str).draw(font_size, rect.x + rect.h - menu_offset_y + x_offset, rect.y + menu_offset_y, menu_font_color);
        } else {
            font(str).draw(font_size, rect.x + rect.h - menu_offset_y, rect.y + menu_offset_y, menu_font_color);
        }
        if (mode == MENU_MODE_BAR || mode == MENU_MODE_BAR_CHECK) {
            if (mode == MENU_MODE_BAR_CHECK && !(*is_checked)) {
                font(unchecked_str).draw(font_size, Arg::topRight(bar_sx - menu_child_offset - 4, rect.y + menu_offset_y), menu_font_color);
            } else {
                font(*bar_elem).draw(font_size, Arg::topRight(bar_sx - menu_child_offset - 4, rect.y + menu_offset_y), menu_font_color);
            }
            if (mode == MENU_MODE_BAR_CHECK && !(*is_checked)) {
                bar_rect.draw(ColorF(bar_color, 0.5));
            } else {
                bar_rect.draw(bar_color);
            }
            bar_circle.x = value_to_bar_x(*bar_elem);
            if (mode == MENU_MODE_BAR_CHECK && !(*is_checked)) {
                Shape2D::Cross(1.5 * bar_circle.r, 6, Vec2{bar_circle.x, bar_circle.y}).draw(ColorF(bar_circle_color, 0.5));
            } else {
                bar_circle.draw(bar_circle_color);
            }
        } else if (mode == MENU_MODE_2BARS) {
            String range_str = Format(*bar_elem1, U"~", *bar_elem2);
            font(range_str).draw(font_size, Arg::topRight(bar_sx - menu_child_offset - 4, rect.y + menu_offset_y), menu_font_color);
            Rect left_label(bar_rect.x, bar_rect.y, bar_rect.w / 2, bar_rect.h);
            Rect right_label(bar_rect.x + bar_rect.w / 2, bar_rect.y, bar_rect.w - bar_rect.w / 2, bar_rect.h);
            left_label.draw(Palette::White);
            right_label.draw(Palette::Black);
            const Circle handle_circle1(value_to_bar_x(*bar_elem1, false), bar_center_y, MENU_BAR_RADIUS);
            const Circle handle_circle2(value_to_bar_x(*bar_elem2, true), bar_center_y, MENU_BAR_RADIUS);
            double scale = static_cast<double>(MENU_BAR_HEIGHT) / static_cast<double>(arrow_left.height());
            arrow_left.scaled(scale).rotated(180_deg).drawAt(handle_circle1.x, handle_circle1.y);
            arrow_left.scaled(scale).drawAt(handle_circle2.x, handle_circle2.y);
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

    std::pair<int, int> size(std::string lang_name) {
        int h, w;
        if (use_image) {
            h = rect.h - 2 * menu_image_offset_y;
            w = (double)h * (double)image.width() / (double)image.height() + menu_image_offset_y * 2; // margin: menu_image_offset_y * 2
            int ascii_count = count_ascii(str);
            w += (str.size() - ascii_count) * font_size; // zenkaku
            w += region_ascii(str, font_size, lang_name, font); // hankaku
        } else {
            // RectF r = font(str).region(font_size, Point{ 0, 0 }); // slow
            // h = r.h;
            // w = r.w;
            h = font_size;
            int ascii_count = count_ascii(str);
            w = (str.size() - ascii_count) * font_size; // zenkaku
            w += region_ascii(str, font_size, lang_name, font); // hankaku
        }
        w += h;
        if (mode == MENU_MODE_BAR || mode == MENU_MODE_BAR_CHECK) {
            w += MENU_BAR_SIZE + bar_value_offset + bar_additional_offset;
        } else if (mode == MENU_MODE_2BARS) {
            w += MENU_BAR_SIZE + bar_value_offset * 3 + bar_additional_offset;
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
        str = U"";
        bar_active_circle = 0;
    }

    int menu_mode() const {
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

    void init_inside(int x, int y, int w, int h, int bar_value_offset, std::string lang_name) {
        rect.x = x;
        rect.y = y;
        rect.w = w;
        rect.h = h;
        int height = h - menu_offset_y * 2, width = w - menu_offset_x * 2;
        for (menu_elem &child: children) {
            child.pre_init(font_size, font, checkbox, unchecked, height, bar_value_offset);
            std::pair<int, int> child_size = child.size(lang_name);
            height = std::max(height, child_size.first);
            width = std::max(width, child_size.second);
        }
        height += menu_offset_y * 2;
        width += menu_offset_x * 2;
        int xx = rect.x;
        int yy = rect.y + rect.h;
        for (menu_elem &child: children) {
            child.init_inside(xx, yy, width, height, lang_name);
            yy += height;
        }
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

    std::pair<int, int> size(std::string lang_name) {
        // RectF rect = font(str).region(font_size, Point{ 0, 0 }); // slow
        // return std::make_pair(rect.h, rect.w);
        int h = font(U"A").region(font_size, Point{ 0, 0 }).h; //font_size;
        int ascii_count = count_ascii(str);
        int w = (str.size() - ascii_count) * font_size; // zenkaku
        w += region_ascii(str, font_size, lang_name, font); // hankaku
        return std::make_pair(h, w);
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

    void init(int x, int y, int font_size, Font font, Texture checkbox, Texture unchecked, std::string lang_name) {
        uint64_t strt = tim();
        int height = 0, width = 0;
        for (menu_title &title : menu) {
            title.pre_init(font_size, font, checkbox, unchecked);
            std::pair<int, int> title_size = title.size(lang_name);
            height = std::max(height, title_size.first);
            width = std::max(width, title_size.second);
        }
        height += menu_offset_y * 2;
        width += menu_offset_x * 2;
        int xx = x;
        int yy = y;
        int bar_value_offset = font(U"88").region(font_size, Point{ 0, 0 }).w;
        for (menu_title &title : menu) {
            title.init_inside(xx, yy, width, height, bar_value_offset, lang_name);
            xx += width;
        }
        std::cerr << "menu init elapsed " << tim() - strt << " ms" << std::endl;
        // generate_text_width_size_ratio(font);
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