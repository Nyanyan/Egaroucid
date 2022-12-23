# Common
cmake_minimum_required(VERSION 3.13)
project(Egaroucid_for_Console CXX)
add_compile_options(-mtune=native -march=native -mfpmath=both -pthread -std=c++17 -Wall -Wextra)

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
add_executable(Egaroucid_for_Console.out Egaroucid_console.cpp)

# Resources
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources/eval.egev DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/resources)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources/book.egbk DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/resources)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources/empty_book.egbk DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/resources)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources/hash DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/resources)