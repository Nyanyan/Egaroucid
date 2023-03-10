/*
    Egaroucid Project

    @file screen_shot.hpp
        Screen shot
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include "function/function_all.hpp"
#include "draw.hpp"

class Screen_shot : public App::Scene {
public:
    Screen_shot(const InitData& init) : IScene{ init } {
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        Scene::SetBackground(getData().colors.green);
        draw_board(getData().fonts, getData().colors, getData().history_elem);
    }

    void draw() const override {

    }
};