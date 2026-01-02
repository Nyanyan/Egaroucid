/*
    Egaroucid Project

    @file scroll.hpp
        Scroll Utility
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "./const/gui_common.hpp"

class Scroll_manager {
private:
    Rect rect;
    Rect frame_rect;
    Rect mouseover_rect;
    int sx;
    int sy;
    int width;
    int height;
    int rect_min_height;
    int n_elem;
    int n_elem_per_window;
    int rect_height;
    int max_strt_idx;
    double strt_idx_double;
    uint64_t up_strt;
    uint64_t down_strt;
    uint64_t pup_strt;
    uint64_t pdown_strt;
    bool dragged;
    int dragged_y_offset;


public:
    void init(int x, int y, int w, int h, int rect_mh, int ne, int n_epw, int mx, int my, int mw, int mh) {
        sx = x;
        sy = y;
        width = w;
        height = h;
        rect_min_height = rect_mh;
        n_elem = ne;
        n_elem_per_window = n_epw;
        rect_height = height;
        if (n_elem > n_elem_per_window) {
            rect_height = std::max(rect_min_height, (int)round((double)n_elem_per_window / (double)n_elem * (double)height));
        }
        max_strt_idx = std::max(n_elem - n_elem_per_window, 0);
        strt_idx_double = 0.0;
        up_strt = BUTTON_NOT_PUSHED;
        down_strt = BUTTON_NOT_PUSHED;
        pup_strt = BUTTON_NOT_PUSHED;
        pdown_strt = BUTTON_NOT_PUSHED;
        dragged = false;
        rect.x = sx;
        rect.w = width;
        rect.h = rect_height;
        frame_rect.x = sx;
        frame_rect.y = sy;
        frame_rect.w = width;
        frame_rect.h = height;
        mouseover_rect.x = mx;
        mouseover_rect.y = my;
        mouseover_rect.w = mw;
        mouseover_rect.h = mh;
    }

    void draw() {
        int strt_idx_int = (int)strt_idx_double;
        double percent = 0.0;
        if (max_strt_idx > 0) {
            percent = (double)strt_idx_int / (double)max_strt_idx;
        }
        int rect_y = sy + round(percent * (double)(height - rect_height));
        frame_rect.drawFrame(1.0, Palette::White);
        rect.y = rect_y;
        rect.draw(Palette::White);
        if (dragged) {
            Cursor::RequestStyle(CursorStyle::ResizeUpDown);
        }
    }

    void update() {
        if (mouseover_rect.mouseOver()) {
            strt_idx_double = std::max(0.0, std::min((double)(n_elem - n_elem_per_window), strt_idx_double + Mouse::Wheel()));
        }
        if (!KeyUp.pressed()) {
            up_strt = BUTTON_NOT_PUSHED;
        }
        if (!KeyDown.pressed()) {
            down_strt = BUTTON_NOT_PUSHED;
        }
        if (!KeyPageUp.pressed()) {
            pup_strt = BUTTON_NOT_PUSHED;
        }
        if (!KeyPageDown.pressed()) {
            pdown_strt = BUTTON_NOT_PUSHED;
        }
        if (KeyUp.down() || (up_strt != BUTTON_NOT_PUSHED && tim() - up_strt >= BUTTON_LONG_PRESS_THRESHOLD)) {
            strt_idx_double = std::max(0.0, strt_idx_double - 1.0);
            if (KeyUp.down()) {
                up_strt = tim();
            }
        }
        if (KeyDown.down() || (down_strt != BUTTON_NOT_PUSHED && tim() - down_strt >= BUTTON_LONG_PRESS_THRESHOLD)) {
            strt_idx_double = std::max(0.0, std::min((double)(n_elem - n_elem_per_window), strt_idx_double + 1.0));
            if (KeyDown.down()) {
                down_strt = tim();
            }
        }
        if (KeyPageUp.down() || (pup_strt != BUTTON_NOT_PUSHED && tim() - pup_strt >= BUTTON_LONG_PRESS_THRESHOLD)) {
            strt_idx_double = std::max(0.0, strt_idx_double - n_elem_per_window);
            if (KeyPageUp.down()) {
                pup_strt = tim();
            }
        }
        if (KeyPageDown.down() || (pdown_strt != BUTTON_NOT_PUSHED && tim() - pdown_strt >= BUTTON_LONG_PRESS_THRESHOLD)) {
            strt_idx_double = std::max(0.0, std::min((double)(n_elem - n_elem_per_window), strt_idx_double + n_elem_per_window));
            if (KeyPageDown.down()) {
                pdown_strt = tim();
            }
        }
        if (rect.leftClicked()) {
            dragged = true;
            dragged_y_offset = Cursor::Pos().y - rect.y;
        } else if (!MouseL.pressed()) {
            dragged = false;
        }
        if (dragged) {
            double n_percent = std::max(0.0, std::min(1.0, (double)(Cursor::Pos().y - dragged_y_offset - sy) / std::max(1, (height - rect_height))));
            strt_idx_double = n_percent * (double)max_strt_idx;
        }
        if (frame_rect.leftClicked() && !rect.leftClicked()) {
            double n_percent = std::max(0.0, std::min(1.0, (double)(Cursor::Pos().y - rect.h / 2 - sy) / std::max(1, (height - rect_height))));
            strt_idx_double = n_percent * (double)max_strt_idx;
            dragged = true;
            int rect_y = sy + round(n_percent * (double)(height - rect_height));
            dragged_y_offset = Cursor::Pos().y - rect_y;
        }
    }

    int get_strt_idx_int() const {
        return (int)strt_idx_double;
    }

    double get_strt_idx_double() const {
        return strt_idx_double;
    }

    bool is_dragged() const {
        return dragged;
    }

    void set_strt_idx(double idx_double) {
        strt_idx_double = idx_double;
    }

    void set_n_elem(int ne) {
        n_elem = ne;
        rect_height = height;
        if (n_elem > n_elem_per_window) {
            rect_height = std::max(rect_min_height, (int)round((double)n_elem_per_window / (double)n_elem * (double)height));
        }
        max_strt_idx = std::max(n_elem - n_elem_per_window, 0);
    }
};


class Scroll_horizontal_manager {
private:
    Rect rect;
    Rect frame_rect;
    Rect mouseover_rect;
    int sx;
    int sy;
    int width;
    int height;
    int rect_min_width;
    int n_elem;
    int n_elem_per_window;
    int rect_width;
    int max_strt_idx;
    double strt_idx_double;
    bool dragged;
    int dragged_x_offset;


public:
    void init(int x, int y, int w, int h, int rect_mw, int ne, int n_epw, int mx, int my, int mw, int mh) {
        sx = x;
        sy = y;
        width = w;
        height = h;
        rect_min_width = rect_mw;
        n_elem = ne;
        n_elem_per_window = n_epw;
        rect_width = width;
        if (n_elem > n_elem_per_window) {
            rect_width = std::max(rect_min_width, (int)round((double)n_elem_per_window / (double)n_elem * (double)height));
        }
        max_strt_idx = std::max(n_elem - n_elem_per_window, 0);
        strt_idx_double = 0.0;
        dragged = false;
        rect.y = sy;
        rect.w = rect_width;
        rect.h = height;
        frame_rect.x = sx;
        frame_rect.y = sy;
        frame_rect.w = width;
        frame_rect.h = height;
        mouseover_rect.x = mx;
        mouseover_rect.y = my;
        mouseover_rect.w = mw;
        mouseover_rect.h = mh;
    }

    void draw() {
        int strt_idx_int = (int)strt_idx_double;
        double percent = 0.0;
        if (max_strt_idx > 0) {
            percent = (double)strt_idx_int / (double)max_strt_idx;
        }
        int rect_x = sx + round(percent * (double)(width - rect_width));
        frame_rect.drawFrame(1.0, Palette::White);
        rect.x = rect_x;
        rect.draw(Palette::White);
        if (dragged) {
            Cursor::RequestStyle(CursorStyle::ResizeLeftRight);
        }
    }

    void update() {
        if (mouseover_rect.mouseOver()) {
            strt_idx_double = std::max(0.0, std::min((double)(n_elem - n_elem_per_window), strt_idx_double + Mouse::Wheel()));
        }
        if (rect.leftClicked()) {
            dragged = true;
            dragged_x_offset = Cursor::Pos().x - rect.x;
        } else if (!MouseL.pressed()) {
            dragged = false;
        }
        if (dragged) {
            double n_percent = std::max(0.0, std::min(1.0, (double)(Cursor::Pos().x - dragged_x_offset - sx) / std::max(1, (width - rect_width))));
            strt_idx_double = n_percent * (double)max_strt_idx;
        }
        if (frame_rect.leftClicked() && !rect.leftClicked()) {
            double n_percent = std::max(0.0, std::min(1.0, (double)(Cursor::Pos().x - rect.w / 2 - sx) / std::max(1, (width - rect_width))));
            strt_idx_double = n_percent * (double)max_strt_idx;
            dragged = true;
            int rect_x = sx + round(n_percent * (double)(width - rect_width));
            dragged_x_offset = Cursor::Pos().x - rect_x;
        }
    }

    int get_strt_idx_int() const{
        return (int)strt_idx_double;
    }

    bool is_dragged() const{
        return dragged;
    }
};