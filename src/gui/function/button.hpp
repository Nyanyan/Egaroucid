/*
    Egaroucid Project

    @file button.hpp
        Button for Egaroucid's GUI
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <Siv3D.hpp>
#include <iostream>
#include "click.hpp"

class Button {
public:
    RoundRect rect;
    String str;
    Font font;
    int font_size;
    Color button_color;
    Color font_color;
private:
    bool enabled;
    bool transparent;
    Click_supporter click_supporter;

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
        enabled = true;
        transparent = false;
        click_supporter.init();
    }

    void move(int x, int y) {
        rect.x = x;
        rect.y = y;
    }

    void draw() {
        if (enabled) {
            rect.draw(button_color);
            font(str).drawAt(font_size, rect.x + rect.w / 2, rect.y + rect.h / 2, font_color);
            if (rect.mouseOver()) {
                Cursor::RequestStyle(CursorStyle::Hand);
            }
        } else {
            if (transparent) {
                rect.draw(ColorF(button_color, 0.7));
            } else {
                rect.draw(button_color);
            }
            font(str).drawAt(font_size, rect.x + rect.w / 2, rect.y + rect.h / 2, font_color);
        }
        click_supporter.update(rect);
    }

    bool clicked() {
        return enabled && click_supporter.clicked();
    }

    void enable() {
        enabled = true;
    }

    void disable() {
        enabled = false;
        transparent = true;
    }

    void disable_notransparent() {
        enabled = false;
        transparent = false;
    }

    bool is_enabled() const{
        return enabled;
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
        if (rect.mouseOver()) {
            Cursor::RequestStyle(CursorStyle::Hand);
        }
    }

    void draw(double transparency) {
        rect.draw(ColorF(button_color, transparency)).drawFrame(frame_width, ColorF(frame_color, transparency));
        font(str).drawAt(font_size, rect.x + rect.w / 2, rect.y + rect.h / 2, ColorF(font_color, transparency));
        if (rect.mouseOver()) {
            Cursor::RequestStyle(CursorStyle::Hand);
        }
    }

    bool clicked() {
        return rect.leftClicked();
    }
};



class ImageButton {
public:
    int x;
    int y;
    int w;
    Texture texture;
private:
    Rect texture_rect;
    bool enabled;
    bool transparent;
    Click_supporter click_supporter;

public:
    void init(int x_, int y_, int w_, Texture texture_) {
        x = x_;
        y = y_;
        w = w_;
        texture = texture_;
        enabled = true;
        transparent = false;
        click_supporter.init();
        texture_rect.x = x;
        texture_rect.y = y;
        texture_rect.w = w_;
        texture_rect.h = w_ * texture.height() / texture.width();
    }

    void move(int x_, int y_) {
        x = x_;
        y = y_;
    }

    void draw() {
        if (enabled) {
            texture.scaled(w / texture.width()).draw(x, y);
            if (texture_rect.mouseOver()) {
                Cursor::RequestStyle(CursorStyle::Hand);
            }
        } else {
            if (transparent) {
                texture.scaled(w / texture.width()).draw(x, y, ColorF{ 1.0, 0.7 });
            } else {
                texture.scaled(w / texture.width()).draw(x, y);
            }
        }
        click_supporter.update(texture_rect);
    }

    bool clicked() {
        return enabled && click_supporter.clicked();
    }

    void enable() {
        enabled = true;
    }

    void disable() {
        enabled = false;
        transparent = true;
    }

    void disable_notransparent() {
        enabled = false;
        transparent = false;
    }

    bool is_enabled() const{
        return enabled;
    }
};