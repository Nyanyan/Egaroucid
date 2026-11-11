#pragma once
#include <iostream>
#include <cstring>
#include "function/function_all.hpp"
#include "function/board_image_vector.hpp"

#ifdef _WIN32
#include <Windows.h>
#include <ShlObj.h>
#endif

// The clipboard refers to the saved file, so file managers can paste it with its original format.
inline bool set_file_to_clipboard(const String& path) {
#ifdef _WIN32
    if (!FileSystem::IsFile(path)) {
        return false;
    }
    const std::wstring full_path = FileSystem::FullPath(path).toWstr();
    const size_t path_bytes = (full_path.size() + 2) * sizeof(wchar_t);
    HGLOBAL files = GlobalAlloc(GMEM_MOVEABLE | GMEM_ZEROINIT, sizeof(DROPFILES) + path_bytes);
    if (!files) {
        return false;
    }
    auto* drop = static_cast<DROPFILES*>(GlobalLock(files));
    if (!drop) {
        GlobalFree(files);
        return false;
    }
    drop->pFiles = sizeof(DROPFILES);
    drop->fWide = TRUE;
    std::memcpy(reinterpret_cast<unsigned char*>(drop) + sizeof(DROPFILES), full_path.c_str(),
        (full_path.size() + 1) * sizeof(wchar_t));
    GlobalUnlock(files);

    const UINT effect_format = RegisterClipboardFormatW(L"Preferred DropEffect");
    HGLOBAL effect = GlobalAlloc(GMEM_MOVEABLE, sizeof(DWORD));
    auto* effect_value = effect ? static_cast<DWORD*>(GlobalLock(effect)) : nullptr;
    if (!effect_format || !effect_value) {
        if (effect_value) GlobalUnlock(effect);
        GlobalFree(files);
        if (effect) GlobalFree(effect);
        return false;
    }
    *effect_value = DROPEFFECT_COPY;
    GlobalUnlock(effect);

    // A real window must own the clipboard after EmptyClipboard(). Retry brief clipboard contention.
    const HWND owner = static_cast<HWND>(Platform::Windows::Window::GetHWND());
    bool opened = false;
    for (int attempt = 0; owner && attempt < 5; ++attempt) {
        if (OpenClipboard(owner)) {
            opened = true;
            break;
        }
        Sleep(10);
    }
    bool copied = false;
    if (opened) {
        if (EmptyClipboard() && SetClipboardData(CF_HDROP, files)) {
            files = nullptr; // Windows owns this allocation now.
            copied = true;
            if (SetClipboardData(effect_format, effect)) effect = nullptr;
        }
        CloseClipboard();
    }
    if (files) GlobalFree(files);
    if (effect) GlobalFree(effect);
    return copied;
#else
    (void)path; // Siv3D has no file-copy API on the other platforms.
    return false;
#endif
}

inline void copy_saved_image_file(const String& path) {
    std::cerr << "board image saved to " << path.narrow();
    if (set_file_to_clipboard(path)) {
        std::cerr << " and copied to clipboard as a file" << std::endl;
    } else {
        std::cerr << "; failed to copy the file to clipboard" << std::endl;
    }
}

void take_screen_shot(double window_scale, std::string screenshot_dir, std::string transcript) {
    Image image = ScreenCapture::GetFrame();
    const int clip_sx = BOARD_SX - BOARD_ROUND_FRAME_WIDTH - BOARD_COORD_SIZE;
    const int clip_sy = BOARD_SY - BOARD_ROUND_FRAME_WIDTH - BOARD_COORD_SIZE;
    const int clip_size_x = BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2 + BOARD_COORD_SIZE + 7;
    const int clip_size_y = BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2 + BOARD_COORD_SIZE + 7;
    const Rect clip_rect(clip_sx * window_scale, clip_sy * window_scale, clip_size_x * window_scale, clip_size_y * window_scale);
    Image image_clip = image.clipped(clip_rect);
    String img_date = Unicode::Widen(calc_date());
    String save_path = Unicode::Widen(screenshot_dir) + img_date + U"_" + Unicode::Widen(transcript) + U".png";
    if (image_clip.save(save_path)) {
        copy_saved_image_file(save_path);
    } else {
        std::cerr << "failed to save board image to " << save_path.narrow() << std::endl;
    }
}

// The saved area of the board image editor in the captured frame. Without coordinates, pixels
// outside the outer round frame are transparent.
Image capture_board_image(double window_scale, bool include_coordinate) {
    Image image = ScreenCapture::GetFrame();

    const int board_outer_size = BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2;
    const int clip_sx = BOARD_SX - BOARD_ROUND_FRAME_WIDTH - (include_coordinate ? BOARD_COORD_SIZE : 0);
    const int clip_sy = BOARD_SY - BOARD_ROUND_FRAME_WIDTH - (include_coordinate ? BOARD_COORD_SIZE : 0);
    const int clip_size_x = board_outer_size + (include_coordinate ? (BOARD_COORD_SIZE + 7) : 0);
    const int clip_size_y = board_outer_size + (include_coordinate ? (BOARD_COORD_SIZE + 7) : 0);

    const Rect clip_rect(clip_sx * window_scale, clip_sy * window_scale, clip_size_x * window_scale, clip_size_y * window_scale);
    Image image_clip = image.clipped(clip_rect);

    if (include_coordinate) {
        return image_clip;
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
    return image_clip;
}

void take_board_image_screen_shot(double window_scale, std::string screenshot_dir, std::string transcript, bool include_coordinate) {
    Image image_clip = capture_board_image(window_scale, include_coordinate);
    String img_date = Unicode::Widen(calc_date());
    String save_path = Unicode::Widen(screenshot_dir) + img_date + U"_" + Unicode::Widen(transcript) + U".png";
    if (image_clip.save(save_path)) {
        copy_saved_image_file(save_path);
    } else {
        std::cerr << "failed to save board image to " << save_path.narrow() << std::endl;
    }
}

// The vector export receives the engine's disc values unchanged.
static_assert(BOARD_IMAGE_DISC_BLACK == BLACK && BOARD_IMAGE_DISC_WHITE == WHITE && BOARD_IMAGE_DISC_VACANT == VACANT,
    "Board image disc values must match the engine");

inline Vector_color to_vector_color(const Color& color) {
    return {color.r, color.g, color.b};
}

// Outlines of the glyphs that DrawableText draws; board_image_text_path() places them like drawAt().
inline Board_image_text_outline board_image_text_outline(const Font& font, const String& text) {
    Board_image_text_outline outline;
    outline.base_size = font.fontSize();
    outline.line_height = font.height();
    for (const OutlineGlyph& glyph : font.renderOutlines(text, CloseRing::No)) {
        Board_image_glyph result;
        result.x_advance = glyph.xAdvance;
        const Vec2 offset = glyph.getOffset();
        result.offset = {offset.x, offset.y};
        for (const LineString& ring : glyph.rings) {
            std::vector<Vector_point> points;
            points.reserve(ring.size());
            for (const Vec2& point : ring) {
                points.push_back({point.x, point.y});
            }
            result.rings.push_back(std::move(points));
        }
        outline.glyphs.push_back(std::move(result));
    }
    return outline;
}

// The picture of the board image editor as vector shapes, with the GUI's geometry, colors, and font.
inline Vector_image create_board_image_vector(const History_elem& history_elem, const Board_image_edit& edit,
    const Board_image_vector_options& options, const Fonts& fonts, const Colors& colors, const std::string& transcript) {
    Board board = history_elem.board;
    int board_arr[HW2];
    board.translate_to_arr(board_arr, history_elem.player);
    std::array<int, HW2> discs;
    std::copy(std::begin(board_arr), std::end(board_arr), discs.begin());
    Board_image_vector_layout layout;
    layout.board_x = BOARD_SX;
    layout.board_y = BOARD_SY;
    layout.cell_size = BOARD_CELL_SIZE;
    layout.coordinate_size = BOARD_COORD_SIZE;
    layout.frame_width = BOARD_ROUND_FRAME_WIDTH;
    layout.frame_radius = BOARD_ROUND_DIAMETER;
    layout.grid_width = BOARD_CELL_FRAME_WIDTH;
    layout.dot_radius = BOARD_DOT_SIZE;
    layout.disc_radius = DISC_SIZE;
    layout.disc_frame_width = BOARD_DISC_FRAME_WIDTH;
    Board_image_vector_palette palette;
    palette.green = to_vector_color(colors.green);
    palette.black = to_vector_color(colors.black);
    palette.white = to_vector_color(colors.white);
    palette.dark_gray = to_vector_color(colors.dark_gray);
    const Font font = fonts.font_bold;
    const auto text_outline = [font](const std::string& text) {
        return board_image_text_outline(font, Unicode::FromUTF8(text));
    };
    Vector_image image = build_board_image_vector(discs, edit, options, text_outline, layout, palette);
    image.set_title(transcript.empty() ? "Egaroucid board image" : "Egaroucid board image " + transcript);
    return image;
}

// Writes a vector file next to the PNG images. Returns the path, or an empty string on failure.
inline String save_board_image_vector(const std::string& screenshot_dir, const std::string& transcript, int format,
    const Vector_image& image) {
    std::string data;
    String extension;
    switch (format) {
    case BOARD_IMAGE_FORMAT_SVG: data = vector_image_to_svg(image); extension = U".svg"; break;
    case BOARD_IMAGE_FORMAT_EPS: data = vector_image_to_eps(image); extension = U".eps"; break;
    case BOARD_IMAGE_FORMAT_PDF: data = vector_image_to_pdf(image); extension = U".pdf"; break;
    default: return U"";
    }
    const String img_date = Unicode::Widen(calc_date());
    const String save_path = Unicode::Widen(screenshot_dir) + img_date + U"_" + Unicode::Widen(transcript) + extension;
    const int64 size = static_cast<int64>(data.size());
    BinaryWriter writer{save_path};
    if (not writer || writer.write(data.data(), size) != size) {
        std::cerr << "failed to save board image to " << save_path.narrow() << std::endl;
        return U"";
    }
    writer.close();
    return save_path;
}

// Save the selected vector format before placing its file on the clipboard.
inline void take_board_image_vector(const std::string& screenshot_dir, const std::string& transcript,
    int format, const History_elem& history_elem, const Board_image_edit& edit, const Board_image_vector_options& options,
    const Fonts& fonts, const Colors& colors) {
    const Vector_image image = create_board_image_vector(history_elem, edit, options, fonts, colors, transcript);
    const String save_path = save_board_image_vector(screenshot_dir, transcript, format, image);
    if (!save_path.isEmpty()) {
        copy_saved_image_file(save_path);
    }
}
