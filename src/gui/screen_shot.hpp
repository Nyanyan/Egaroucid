#pragma once
#include <iostream>
#include "function/function_all.hpp"

void take_screen_shot(double window_scale, std::string screenshot_dir, std::string transcript) {
    Image image = ScreenCapture::GetFrame();
    const int clip_sx = BOARD_SX - BOARD_ROUND_FRAME_WIDTH - BOARD_COORD_SIZE;
    const int clip_sy = BOARD_SY - BOARD_ROUND_FRAME_WIDTH - BOARD_COORD_SIZE;
    const int clip_size_x = BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2 + BOARD_COORD_SIZE + 7;
    const int clip_size_y = BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2 + BOARD_COORD_SIZE + 7;
    const Rect clip_rect(clip_sx * window_scale, clip_sy * window_scale, clip_size_x * window_scale, clip_size_y * window_scale);
    Image image_clip = image.clipped(clip_rect);
    Clipboard::SetImage(image_clip);
    String img_date = Unicode::Widen(calc_date());
    String save_path = Unicode::Widen(screenshot_dir) + img_date + U"_" + Unicode::Widen(transcript) + U".png";
    image_clip.save(save_path);
    std::cerr << "screen shot saved to " << save_path.narrow() << " and copied to clipboard" << std::endl;
}

void take_board_image_screen_shot(double window_scale, std::string screenshot_dir, std::string transcript, bool include_coordinate) {
    Image image = ScreenCapture::GetFrame();

    const int board_outer_size = BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2;
    const int clip_sx = BOARD_SX - BOARD_ROUND_FRAME_WIDTH - (include_coordinate ? BOARD_COORD_SIZE : 0);
    const int clip_sy = BOARD_SY - BOARD_ROUND_FRAME_WIDTH - (include_coordinate ? BOARD_COORD_SIZE : 0);
    const int clip_size_x = board_outer_size + (include_coordinate ? (BOARD_COORD_SIZE + 7) : 0);
    const int clip_size_y = board_outer_size + (include_coordinate ? (BOARD_COORD_SIZE + 7) : 0);

    const Rect clip_rect(clip_sx * window_scale, clip_sy * window_scale, clip_size_x * window_scale, clip_size_y * window_scale);
    Image image_clip = image.clipped(clip_rect);

    const int coord_offset = include_coordinate ? static_cast<int>(BOARD_COORD_SIZE * window_scale) : 0;
    const int outer_size = static_cast<int>(board_outer_size * window_scale);
    const int outer_left = coord_offset;
    const int outer_top = coord_offset;
    const int radius = static_cast<int>((BOARD_ROUND_DIAMETER + BOARD_ROUND_FRAME_WIDTH) * window_scale);

    auto is_in_rounded_rect = [outer_left, outer_top, outer_size, radius](int x, int y) {
        const int left = outer_left;
        const int top = outer_top;
        const int right = outer_left + outer_size - 1;
        const int bottom = outer_top + outer_size - 1;

        if (x < left || x > right || y < top || y > bottom) {
            return false;
        }
        if (radius <= 0) {
            return true;
        }

        const int inner_left = left + radius;
        const int inner_right = right - radius;
        const int inner_top = top + radius;
        const int inner_bottom = bottom - radius;

        if (x >= inner_left && x <= inner_right) {
            return true;
        }
        if (y >= inner_top && y <= inner_bottom) {
            return true;
        }

        auto in_corner_circle = [radius](int px, int py, int cx, int cy) {
            const int dx = px - cx;
            const int dy = py - cy;
            return (dx * dx + dy * dy) <= radius * radius;
        };

        if (x < inner_left && y < inner_top) {
            return in_corner_circle(x, y, inner_left, inner_top);
        }
        if (x > inner_right && y < inner_top) {
            return in_corner_circle(x, y, inner_right, inner_top);
        }
        if (x < inner_left && y > inner_bottom) {
            return in_corner_circle(x, y, inner_left, inner_bottom);
        }
        if (x > inner_right && y > inner_bottom) {
            return in_corner_circle(x, y, inner_right, inner_bottom);
        }
        return true;
    };

    for (int y = 0; y < image_clip.height(); ++y) {
        for (int x = 0; x < image_clip.width(); ++x) {
            const bool in_board_outer = is_in_rounded_rect(x, y);
            bool keep_pixel = in_board_outer;

            if (include_coordinate) {
                const bool in_left_coord_band = (x < outer_left && y >= outer_top && y < outer_top + outer_size);
                const bool in_top_coord_band = (y < outer_top && x >= outer_left && x < outer_left + outer_size);
                keep_pixel = keep_pixel || in_left_coord_band || in_top_coord_band;
            }

            if (!keep_pixel) {
                image_clip[y][x].a = 0;
            }
        }
    }

    Clipboard::SetImage(image_clip);
    String img_date = Unicode::Widen(calc_date());
    String save_path = Unicode::Widen(screenshot_dir) + img_date + U"_" + Unicode::Widen(transcript) + U".png";
    image_clip.save(save_path);
    std::cerr << "screen shot saved to " << save_path.narrow() << " and copied to clipboard" << std::endl;
}