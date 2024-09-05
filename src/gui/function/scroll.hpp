/*
    Egaroucid Project

    @file scroll.hpp
        Scroll Utility
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include "const/gui_common.hpp"

class Scroll_manager{
private:
    int sx;
    int sy;
    int width;
    int height;
    int rect_min_height;
    int n_elem;
    int n_elem_per_window;
    int rect_height;
    int max_strt_idx;

public:
    void init(int x, int y, int w, int h, int rect_mh, int ne, int n_epw){
        sx = x;
        sy = y;
        width = w;
        height = h;
        rect_min_height = rect_mh;
        n_elem = ne;
        n_elem_per_window = n_epw;
        rect_height = std::max(rect_min_height, (int)round((double)n_elem_per_window / (double)n_elem * (double)height));
        max_strt_idx = std::max(n_elem - n_elem_per_window, 0);
    }

    void draw(int strt_idx){
        double percent = (double)strt_idx / (double)max_strt_idx;
        int rect_y = sy + round(percent * (double)(height - rect_height));
        Rect frame_rect(sx, sy, width, height);
        frame_rect.drawFrame(1.0, Palette::White);
        Rect rect(sx, rect_y, width, rect_height);
        rect.draw(Palette::White);
        std::cerr << strt_idx << " " << percent << "  " << sx << " " << sy << " " << rect_y << " " << rect_height << std::endl;
    }
};
