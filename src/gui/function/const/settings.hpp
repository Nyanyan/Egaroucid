/*
    Egaroucid Project

    @file settings.hpp
        Settings for GUI
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once

/* Compile Options
    -DGUI_PORTABLE      : Portable Mode
*/

/*
    @brief GUI settings
*/
// GUI portable mode
#ifdef __APPLE__
    // always false for Mac
    #define GUI_PORTABLE_MODE false
#else
    #ifdef GUI_PORTABLE
        #define GUI_PORTABLE_MODE true
    #else
        #define GUI_PORTABLE_MODE false
    #endif
#endif
// GUI Open Console?
#define GUI_OPEN_CONSOLE true