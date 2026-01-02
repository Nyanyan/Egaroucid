/*
    Egaroucid Project

    @file click.hpp
        Structures for clicking button
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <Siv3D.hpp>


struct Click_supporter {
    bool clicked_down;
    bool pressed;
    bool f_pressed;

    void init() {
        clicked_down = false;
        pressed = false;
        f_pressed = false;
    }

    void update(Rect rect) {
        f_pressed = pressed;
        pressed = rect.leftPressed();
        if (rect.leftClicked()) {
            clicked_down = true;
        } else if (!MouseL.down() && !f_pressed) {
            clicked_down = false;
        }
    }

    void update(RoundRect rect) {
        f_pressed = pressed;
        pressed = rect.leftPressed();
        if (rect.leftClicked()) {
            clicked_down = true;
        } else if (!MouseL.down() && !f_pressed) {
            clicked_down = false;
        }
    }

    bool clicked() {
        return clicked_down && f_pressed && !MouseL.pressed();
    }
};