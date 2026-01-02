/*
    Egaroucid Project

    @file slidebar.hpp
        Bar for Egaroucid's GUI
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <Siv3D.hpp>
#include <iostream>

//constexpr Color bar_color = Palette::Lightskyblue;
//constexpr Color bar_circle_color = Palette::Deepskyblue;

class Slidebar {
public:
    Circle circle;
    Rect rect;
    String str;
    Font font;
    Color font_color;
    int font_size;
    int *value;
    int min_value;
    int max_value;
private:
    bool changeable;
    int sx;
    int ex;
    int cy;
    int str_ex;

public:
    void init(int x, int y, int w, int h, String s, int fs, Color fc, Font f, int mnv, int mxv, int *v) {
        rect.x = x;
        rect.y = y;
        rect.w = w;
        rect.h = h;
        circle.y = y + h / 2;
        circle.r = h * 1.1 / 2;
        str = s;
        font_size = fs;
        font_color = fc;
        font = f;
        min_value = mnv;
        max_value = mxv;
        value = v;
        changeable = false;
        sx = rect.x + circle.r * 1.1;
        ex = rect.x + rect.w - circle.r * 1.1;
        cy = rect.y + rect.h / 2;
        str_ex = rect.x - font(U"88").region(font_size, Point{ 0, 0 }).w - 5;
        circle.x = round((double)sx + (double)(ex - sx) * (double)(*value - min_value) / (double)(max_value - min_value));
    }

    void draw() {
        if (rect.leftClicked()) {
            changeable = true;
        } else if (!MouseL.pressed()) {
            changeable = false;
        }
        if (changeable) {
            Cursor::RequestStyle(CursorStyle::ResizeLeftRight);
            int min_error = INF;
            int cursor_x = Cursor::Pos().x;
            for (int i = min_value; i <= max_value; ++i) {
                int x = round((double)sx + (double)(ex - sx) * (double)(i - min_value) / (double)(max_value - min_value));
                if (abs(cursor_x - x) < min_error) {
                    min_error = abs(cursor_x - x);
                    *value = i;
                    circle.x = x;
                }
            }
        }
        rect.draw(Palette::Lightskyblue);
        circle.draw(Palette::Deepskyblue);
        font(*value).draw(font_size, Arg::rightCenter(rect.x - 5, cy), font_color);
        font(str).draw(font_size, Arg::rightCenter(str_ex, cy), font_color);
    }

    bool is_changeable() const{
        return changeable;
    }

private:
};