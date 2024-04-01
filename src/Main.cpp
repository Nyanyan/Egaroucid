/*
    Egaroucid Project

    @file Main.cpp
        Main file for GUI application
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#include <iostream>
#include "gui/gui_all.hpp"

/*
    @brief used for scaling
*/
double CalculateScale(const Vec2& baseSize, const Vec2& currentSize) {
    return Min((currentSize.x / baseSize.x), (currentSize.y / baseSize.y));
}

#if !GUI_OPEN_CONSOLE
    class Logger_cerr{
        private:
            std::stringstream logger_stream;
            std::mutex logger_mtx;
            String logger_String;

        public:
            Logger_cerr(){
                std::cerr.rdbuf(logger_stream.rdbuf());
            }

            String get_last_line(){
                std::unique_lock{logger_mtx};
                std::string log_line;
                while (getline(logger_stream, log_line)){
                    logger_String = Unicode::Widen(log_line);
                }
                logger_stream.clear(std::stringstream::goodbit);
                return logger_String;
            }
    };
#endif 

/*
    @brief main function
*/
void Main() {
    Size window_size = Size(WINDOW_SIZE_X, WINDOW_SIZE_Y);
    Window::Resize(window_size);
    Window::SetStyle(WindowStyle::Sizable);
    Scene::SetResizeMode(ResizeMode::Virtual);
    Window::SetTitle(U"Egaroucid {}"_fmt(EGAROUCID_VERSION));
    System::SetTerminationTriggers(UserAction::NoAction);
    #if GUI_OPEN_CONSOLE
        Console.open();
    #else
        Logger_cerr logger;
        String logger_String;
    #endif
    
    App scene_manager;
    scene_manager.add <Silent_load> (U"Silent_load");
    scene_manager.add <Load>(U"Load");
    scene_manager.add <Main_scene>(U"Main_scene");
    scene_manager.add <Merge_book>(U"Merge_book");
    scene_manager.add <Refer_book>(U"Refer_book");
    scene_manager.add <Enhance_book>(U"Enhance_book");
    scene_manager.add <Fix_book>(U"Fix_book");
    scene_manager.add <Reduce_book>(U"Reduce_book");
    scene_manager.add <Leaf_recalculate_book>(U"Leaf_recalculate_book");
    scene_manager.add <Import_book>(U"Import_book");
    scene_manager.add <Export_book>(U"Export_book");
    scene_manager.add <Show_book_info>(U"Show_book_info");
    scene_manager.add <Import_transcript>(U"Import_transcript");
    scene_manager.add <Import_board>(U"Import_board");
    scene_manager.add <Edit_board>(U"Edit_board");
    scene_manager.add <Import_game>(U"Import_game");
    scene_manager.add <Export_game>(U"Export_game");
    scene_manager.add <Board_image>(U"Board_image");
    scene_manager.add <Import_bitboard>(U"Import_bitboard");
    scene_manager.add <Close>(U"Close");
    scene_manager.setFadeColor(Palette::Black);
    scene_manager.init(U"Silent_load");

    while (System::Update()) {

        // scale
        double scale = CalculateScale(window_size, Scene::Size());
        const Transformer2D screenScaling{ Mat3x2::Scale(scale), TransformCursor::Yes };

        // scene
        scene_manager.update();
        scene_manager.get()->window_state.window_scale = scale;

        // log
        #if !GUI_OPEN_CONSOLE
            logger_String = logger.get_last_line();
            if (scene_manager.get()->menu_elements.show_log || scene_manager.get()->window_state.loading) {
                scene_manager.get()->fonts.font(logger_String).draw(12, Arg::bottomLeft(8, WINDOW_SIZE_Y - 5), scene_manager.get()->colors.white);
            }
        #endif
    }
}
