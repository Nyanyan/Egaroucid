#pragma once
#include <iostream>
#include <cstring>
#include "function/function_all.hpp"

#ifdef _WIN32
#include <Windows.h>
#endif

bool set_image_to_clipboard_preserve_alpha(const Image& image) {
#ifdef _WIN32
    const int width = image.width();
    const int height = image.height();
    if (width <= 0 || height <= 0) {
        return false;
    }

    const size_t pixel_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
    const size_t total_bytes = sizeof(BITMAPV5HEADER) + pixel_bytes;

    HGLOBAL hmem = GlobalAlloc(GMEM_MOVEABLE, total_bytes);
    if (!hmem) {
        return false;
    }

    auto* mem = static_cast<unsigned char*>(GlobalLock(hmem));
    if (!mem) {
        GlobalFree(hmem);
        return false;
    }

    BITMAPV5HEADER header{};
    header.bV5Size = sizeof(BITMAPV5HEADER);
    header.bV5Width = width;
    header.bV5Height = -height;
    header.bV5Planes = 1;
    header.bV5BitCount = 32;
    header.bV5Compression = BI_BITFIELDS;
    header.bV5SizeImage = static_cast<DWORD>(pixel_bytes);
    header.bV5RedMask = 0x00FF0000;
    header.bV5GreenMask = 0x0000FF00;
    header.bV5BlueMask = 0x000000FF;
    header.bV5AlphaMask = 0xFF000000;
    header.bV5CSType = LCS_WINDOWS_COLOR_SPACE;
    header.bV5Intent = LCS_GM_IMAGES;
    std::memcpy(mem, &header, sizeof(BITMAPV5HEADER));

    auto* dst = mem + sizeof(BITMAPV5HEADER);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto& src = image[y][x];
            const size_t i = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4;
            const unsigned int a = src.a;
            dst[i + 0] = static_cast<unsigned char>((static_cast<unsigned int>(src.b) * a + 127) / 255);
            dst[i + 1] = static_cast<unsigned char>((static_cast<unsigned int>(src.g) * a + 127) / 255);
            dst[i + 2] = static_cast<unsigned char>((static_cast<unsigned int>(src.r) * a + 127) / 255);
            dst[i + 3] = static_cast<unsigned char>(a);
        }
    }

    GlobalUnlock(hmem);

    HGLOBAL hpng = nullptr;
    UINT png_format = 0;
    Blob png_blob = image.encodePNG();
    if (!png_blob.empty()) {
        png_format = RegisterClipboardFormatW(L"PNG");
        if (png_format != 0) {
            hpng = GlobalAlloc(GMEM_MOVEABLE, png_blob.size());
            if (hpng) {
                auto* png_mem = static_cast<unsigned char*>(GlobalLock(hpng));
                if (png_mem) {
                    std::memcpy(png_mem, png_blob.data(), png_blob.size());
                    GlobalUnlock(hpng);
                } else {
                    GlobalFree(hpng);
                    hpng = nullptr;
                }
            }
        }
    }

    if (!OpenClipboard(nullptr)) {
        GlobalFree(hmem);
        if (hpng) {
            GlobalFree(hpng);
        }
        return false;
    }

    EmptyClipboard();
    bool has_data = false;

    if (hpng) {
        const HANDLE png_result = SetClipboardData(png_format, hpng);
        if (png_result) {
            has_data = true;
            hpng = nullptr;
        }
    }

    const HANDLE dib_result = SetClipboardData(CF_DIBV5, hmem);
    if (dib_result) {
        has_data = true;
        hmem = nullptr;
    }

    CloseClipboard();

    if (hmem) {
        GlobalFree(hmem);
    }
    if (hpng) {
        GlobalFree(hpng);
    }
    return has_data;
#else
    Clipboard::SetImage(image);
    return true;
#endif
}

void take_screen_shot(double window_scale, std::string screenshot_dir, std::string transcript) {
    Image image = ScreenCapture::GetFrame();
    const int clip_sx = BOARD_SX - BOARD_ROUND_FRAME_WIDTH - BOARD_COORD_SIZE;
    const int clip_sy = BOARD_SY - BOARD_ROUND_FRAME_WIDTH - BOARD_COORD_SIZE;
    const int clip_size_x = BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2 + BOARD_COORD_SIZE + 7;
    const int clip_size_y = BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2 + BOARD_COORD_SIZE + 7;
    const Rect clip_rect(clip_sx * window_scale, clip_sy * window_scale, clip_size_x * window_scale, clip_size_y * window_scale);
    Image image_clip = image.clipped(clip_rect);
    if (!set_image_to_clipboard_preserve_alpha(image_clip)) {
        Clipboard::SetImage(image_clip);
    }
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

    if (include_coordinate) {
        if (!set_image_to_clipboard_preserve_alpha(image_clip)) {
            Clipboard::SetImage(image_clip);
        }
        String img_date = Unicode::Widen(calc_date());
        String save_path = Unicode::Widen(screenshot_dir) + img_date + U"_" + Unicode::Widen(transcript) + U".png";
        image_clip.save(save_path);
        std::cerr << "screen shot saved to " << save_path.narrow() << " and copied to clipboard" << std::endl;
        return;
    }

    const int coord_offset = 0;
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
            const bool keep_pixel = in_board_outer;

            if (!keep_pixel) {
                image_clip[y][x] = Color{ 0, 0, 0, 0 };
            }
        }
    }

    if (!set_image_to_clipboard_preserve_alpha(image_clip)) {
        Clipboard::SetImage(image_clip);
    }
    String img_date = Unicode::Widen(calc_date());
    String save_path = Unicode::Widen(screenshot_dir) + img_date + U"_" + Unicode::Widen(transcript) + U".png";
    image_clip.save(save_path);
    std::cerr << "screen shot saved to " << save_path.narrow() << " and copied to clipboard" << std::endl;
}