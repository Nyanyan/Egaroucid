# Common
cmake_minimum_required(VERSION 3.25)
project(Egaroucid_for_Console CXX)
if(APPLE)
    add_compile_options(-mtune=native -pthread -std=c++17 -Wall -Wextra)
else()
    add_compile_options(-mtune=native -march=native -mfpmath=both -pthread -std=c++17 -Wall -Wextra)
endif()

# AVX2 option
option(HAS_NO_AVX2 "turn on Generic build" OFF)
if (HAS_NO_AVX2)
    add_compile_options(-DHAS_NO_AVX2)
endif(HAS_NO_AVX2)

# ARM processors option
option(HAS_ARM_PROCESSOR "turn on build for ARM processors" OFF)
if (HAS_ARM_PROCESSOR)
    add_compile_options(-DHAS_ARM_PROCESSOR)
endif(HAS_ARM_PROCESSOR)

# 32bit option
option(HAS_32_BIT_OS "turn on build for 32bit OS" OFF)
if (HAS_32_BIT_OS)
    add_compile_options(-DHAS_32_BIT_OS)
endif(HAS_32_BIT_OS)

#Executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
add_executable(Egaroucid_for_Console.out ./src/Egaroucid_console.cpp)
