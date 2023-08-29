#pragma once
#include <iostream>
#include "function/function_all.hpp"

void take_screen_shot(double window_scale, std::string document_dir){
    Image image = ScreenCapture::GetFrame();
    const int clip_sx = BOARD_SX - BOARD_ROUND_FRAME_WIDTH - BOARD_COORD_SIZE;
    const int clip_sy = BOARD_SY - BOARD_ROUND_FRAME_WIDTH - BOARD_COORD_SIZE;
    const int clip_size_x = BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2 + BOARD_COORD_SIZE + 7;
    const int clip_size_y = BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2 + BOARD_COORD_SIZE + 7;
    const Rect clip_rect(clip_sx * window_scale, clip_sy * window_scale, clip_size_x * window_scale, clip_size_y * window_scale);
    Image image_clip = image.clipped(clip_rect);
    Clipboard::SetImage(image_clip);
    String img_date = Unicode::Widen(calc_date());
    String save_path = Unicode::Widen(document_dir) + U"screenshots/" + img_date + U".png";
    image_clip.save(save_path);
    std::cerr << "screen shot saved to " << save_path.narrow() << " and copied to clipboard" << std::endl;
}