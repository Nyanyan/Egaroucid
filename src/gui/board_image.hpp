/*
    Egaroucid Project

    @file input.hpp
        Input scenes
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"
#include "draw.hpp"
#include "screen_shot.hpp"

#define BOARD_IMAGE_COLOR_DEFAULT 0
#define BOARD_IMAGE_COLOR_MONOCHROME 1
#define BOARD_IMAGE_BRECT 0
#define BOARD_IMAGE_BSTAR 1
#define BOARD_IMAGE_WRECT 2
#define BOARD_IMAGE_WSTAR 3
#define BOARD_IMAGE_NOMARK 4
#define BOARD_IMAGE_MARK_DELETED -1
#define BOARD_IMAGE_NOT_CLICKED 2
#define BOARD_IMAGE_RECT_SIZE 20
#define BOARD_IMAGE_STAR_SIZE 15
#define BOARD_IMAGE_FRAME_WIDTH 3

class Board_image : public App::Scene {
private:
    Button back_button;
    Button save_image_button;
    Radio_button mark_radio;
    Radio_button color_radio;
    int marks[HW2];
    int last_marked[HW2];
    bool taking_screen_shot;

public:
    Board_image(const InitData& init) : IScene{ init } {
        back_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_1_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("common", "back"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        save_image_button.init(BUTTON2_VERTICAL_SX, BUTTON2_VERTICAL_2_SY, BUTTON2_VERTICAL_WIDTH, BUTTON2_VERTICAL_HEIGHT, BUTTON2_VERTICAL_RADIUS, language.get("board_image", "save_image"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        Radio_button_element radio_button_elem;

        mark_radio.init();
        radio_button_elem.init(480, 120, getData().fonts.font, 15, language.get("board_image", "rect"), true, getData().colors.black);
        mark_radio.push(radio_button_elem);
        radio_button_elem.init(480, 140, getData().fonts.font, 15, language.get("board_image", "star"), false, getData().colors.black);
        mark_radio.push(radio_button_elem);
        radio_button_elem.init(480, 160, getData().fonts.font, 15, language.get("board_image", "rect"), false, getData().colors.white);
        mark_radio.push(radio_button_elem);
        radio_button_elem.init(480, 180, getData().fonts.font, 15, language.get("board_image", "star"), false, getData().colors.white);
        mark_radio.push(radio_button_elem);
        radio_button_elem.init(480, 200, getData().fonts.font, 15, language.get("board_image", "nomark"), false, getData().colors.white);
        mark_radio.push(radio_button_elem);

        color_radio.init();
        radio_button_elem.init(480, 280, getData().fonts.font, 15, language.get("board_image", "default"), true);
        color_radio.push(radio_button_elem);
        radio_button_elem.init(480, 300, getData().fonts.font, 15, language.get("board_image", "monochrome"), false);
        color_radio.push(radio_button_elem);

        for (int i = 0; i < HW2; ++i) {
            marks[i] = BOARD_IMAGE_NOMARK;
            last_marked[i] = BOARD_IMAGE_NOT_CLICKED;
        }
        taking_screen_shot = false;
    }

    void update() override {
        if (taking_screen_shot) {
            take_screen_shot(getData().window_state.window_scale, getData().directories.document_dir);
            taking_screen_shot = false;
        }

        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
        }
        for (int cell = 0; cell < HW2; ++cell) {
            int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE;
            int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE;
            Rect cell_region(x, y, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
            if (cell_region.leftPressed()) {
                if (marks[cell] == mark_radio.checked && last_marked[cell] != mark_radio.checked) {
                    marks[cell] = BOARD_IMAGE_NOMARK;
                    last_marked[cell] = BOARD_IMAGE_MARK_DELETED;
                }
                else if (last_marked[cell] != BOARD_IMAGE_MARK_DELETED) {
                    marks[cell] = mark_radio.checked;
                    last_marked[cell] = mark_radio.checked;
                }
            }
            else {
                last_marked[cell] = BOARD_IMAGE_NOT_CLICKED;
            }
        }

        Scene::SetBackground(getData().colors.green);
        getData().fonts.font(language.get("in_out", "board_image")).draw(25, 480, 20, getData().colors.white);
        getData().fonts.font(language.get("board_image", "mark")).draw(20, 480, 80, getData().colors.white);
        getData().fonts.font(language.get("board_image", "color")).draw(20, 480, 240, getData().colors.white);
        mark_radio.draw();
        color_radio.draw();
        if (color_radio.checked == BOARD_IMAGE_COLOR_MONOCHROME) {
            const int clip_sx = BOARD_SX - BOARD_ROUND_FRAME_WIDTH - BOARD_COORD_SIZE;
            const int clip_sy = BOARD_SY - BOARD_ROUND_FRAME_WIDTH - BOARD_COORD_SIZE;
            const int clip_size_x = BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2 + BOARD_COORD_SIZE + 7;
            const int clip_size_y = BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2 + BOARD_COORD_SIZE + 7;
            Rect(clip_sx, clip_sy, clip_size_x, clip_size_y).draw(getData().colors.white);
        }
        draw_board(getData().fonts, getData().colors, getData().history_elem, color_radio.checked == BOARD_IMAGE_COLOR_MONOCHROME);
        if (color_radio.checked == BOARD_IMAGE_COLOR_MONOCHROME) {
            RoundRect(BOARD_SX, BOARD_SY, BOARD_CELL_SIZE * HW, BOARD_CELL_SIZE * HW, BOARD_ROUND_DIAMETER).drawFrame(0, BOARD_ROUND_FRAME_WIDTH, getData().colors.black);
        }

        int board_arr[HW2];
        getData().history_elem.board.translate_to_arr(board_arr, getData().history_elem.player);
        for (int cell = 0; cell < HW2; ++cell) {
            int x_center = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
            int y_center = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
            if (marks[cell] == BOARD_IMAGE_BRECT || marks[cell] == BOARD_IMAGE_WRECT) {
                Rect rect(Arg::center(x_center, y_center), BOARD_IMAGE_RECT_SIZE, BOARD_IMAGE_RECT_SIZE);
                if (marks[cell] == BOARD_IMAGE_BRECT) {
                    rect.draw(getData().colors.black);
                    if (board_arr[cell] == BLACK)
                        rect.drawFrame(0, BOARD_IMAGE_FRAME_WIDTH, getData().colors.white);
                }
                else if (marks[cell] == BOARD_IMAGE_WRECT) {
                    rect.draw(getData().colors.white);
                    if (board_arr[cell] == WHITE)
                        rect.drawFrame(0, BOARD_IMAGE_FRAME_WIDTH, getData().colors.black);
                }
            }
            else if (marks[cell] == BOARD_IMAGE_BSTAR || marks[cell] == BOARD_IMAGE_WSTAR) {
                if (marks[cell] == BOARD_IMAGE_BSTAR) {
                    Shape2D::Star(BOARD_IMAGE_STAR_SIZE, Vec2{ x_center, y_center }).draw(getData().colors.black);
                    if (board_arr[cell] == BLACK)
                        Shape2D::Star(BOARD_IMAGE_STAR_SIZE, Vec2{ x_center, y_center }).drawFrame(BOARD_IMAGE_FRAME_WIDTH, getData().colors.white);
                }
                else if (marks[cell] == BOARD_IMAGE_WSTAR) {
                    Shape2D::Star(BOARD_IMAGE_STAR_SIZE, Vec2{ x_center, y_center }).draw(getData().colors.white);
                    if (board_arr[cell] == WHITE)
                        Shape2D::Star(BOARD_IMAGE_STAR_SIZE, Vec2{ x_center, y_center }).drawFrame(BOARD_IMAGE_FRAME_WIDTH, getData().colors.black);
                }
            }
        }

        save_image_button.draw();
        back_button.draw();

        if (save_image_button.clicked()){
            taking_screen_shot = true;
            ScreenCapture::RequestCurrentFrame();
        }
        if (back_button.clicked() || KeyEscape.pressed()) {
            changeScene(U"Main_scene", SCENE_FADE_TIME);
        }
    }

    void draw() const override {

    }
};