# Common
cmake_minimum_required(VERSION 3.25)

# GUI build option for Linux (requires Siv3D to be installed)
option(BUILD_GUI "Build GUI version (requires Siv3D)" OFF)
if(BUILD_GUI)
    project(Egaroucid CXX)
else()
    project(Egaroucid_for_Console CXX)
endif()

if(APPLE)
    #add_compile_options(-O2 -mtune=native -pthread -std=c++17 -Wall -Wextra)
    add_compile_options(-O2 -mtune=native -pthread -std=c++20)
else()
    #add_compile_options(-O2 -mtune=native -march=native -mfpmath=both -pthread -std=c++17 -Wall -Wextra)
    add_compile_options(-O2 -mtune=native -march=native -pthread -std=c++20)
endif()

# No AVX2 option
option(HAS_NO_AVX2 "turn on Generic build" OFF)
if (HAS_NO_AVX2)
    add_compile_options(-DHAS_NO_AVX2)
endif(HAS_NO_AVX2)

# ARM processors option
option(HAS_ARM_PROCESSOR "turn on build for ARM processors" OFF)
if (HAS_ARM_PROCESSOR)
    add_compile_options(-DHAS_ARM_PROCESSOR)
endif(HAS_ARM_PROCESSOR)

# AMD Optimization
option(HAS_AMD_PROCESSOR "turn on AMD Optimization" OFF)
if (HAS_AMD_PROCESSOR)
    add_compile_options(-DHAS_AMD_PROCESSOR)
endif(HAS_AMD_PROCESSOR)

# 32bit option
option(HAS_32_BIT_OS "turn on build for 32bit OS" OFF)
if (HAS_32_BIT_OS)
    add_compile_options(-DHAS_32_BIT_OS)
endif(HAS_32_BIT_OS)

# AVX512 option
option(HAS_AVX512 "turn on AVX-512 build" OFF)
if (HAS_AVX512)
    add_compile_options(-DHAS_AVX512)
endif(HAS_AVX512)

# Executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# GUI build for Linux
if(BUILD_GUI)
    find_package(Siv3D REQUIRED)
    add_compile_definitions(GUI_BUILD=1)
    add_executable(Egaroucid.out ./src/Egaroucid.cpp)
    target_link_libraries(Egaroucid.out PRIVATE Siv3D::Siv3D)
    add_custom_command(TARGET Egaroucid.out POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:Egaroucid.out>/resources/engine
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/app_resources/engine
            $<TARGET_FILE_DIR:Egaroucid.out>/resources/engine
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE
            $<TARGET_FILE_DIR:Egaroucid.out>/LICENSE
    )
else()
    add_executable(Egaroucid_for_Console.out ./src/Egaroucid_for_Console.cpp)
endif()
