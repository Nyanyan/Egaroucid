/*
    Egaroucid Project

    @file settings.hpp
        Settings for GUI
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once

/*
    @brief GUI settings
*/
// GUI portable mode
#ifdef __APPLE__
    #define GUI_PORTABLE_MODE false
#else
    #define GUI_PORTABLE_MODE true
#endif
// GUI Open Console?
#define GUI_OPEN_CONSOLE false