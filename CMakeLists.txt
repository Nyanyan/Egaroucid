# Common
cmake_minimum_required(VERSION 3.25)
project(Egaroucid LANGUAGES CXX VERSION 7.8.1)

option(BUILD_ENGINE_LIB "Build Egaroucid engine library" OFF)
option(BUILD_CONSOLE "Build Egaroucid console executable" ON)
option(BUILD_GUI "Build Egaroucid GUI executable (requires Siv3D)" OFF)
option(BUILD_SHARED_LIBS "Build shared not static engine library" OFF)

if(NOT BUILD_ENGINE_LIB AND NOT BUILD_CONSOLE AND NOT BUILD_GUI)
    message(FATAL_ERROR "At least one of BUILD_ENGINE_LIB, BUILD_CONSOLE, or BUILD_GUI must be ON.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build-type policy:
# - Single-config generators (Ninja/Makefiles): default to Release when unspecified.
# - Multi-config generators (Visual Studio/Xcode): configuration is selected at build time.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# No AVX2 option
option(HAS_NO_AVX2 "turn on Generic build" OFF)
if(HAS_NO_AVX2)
    add_compile_definitions(HAS_NO_AVX2)
endif()

# ARM processors option
option(HAS_ARM_PROCESSOR "turn on build for ARM processors" OFF)
if(HAS_ARM_PROCESSOR)
    add_compile_definitions(HAS_ARM_PROCESSOR)
endif()

# AMD Optimization
option(HAS_AMD_PROCESSOR "turn on AMD Optimization" OFF)
if(HAS_AMD_PROCESSOR)
    add_compile_definitions(HAS_AMD_PROCESSOR)
endif()

# 32bit option
option(HAS_32_BIT_OS "turn on build for 32bit OS" OFF)
if(HAS_32_BIT_OS)
    add_compile_definitions(HAS_32_BIT_OS)
endif()

# AVX512 option
option(HAS_AVX512 "turn on AVX-512 build" OFF)
if(HAS_AVX512)
    add_compile_definitions(HAS_AVX512)
endif()

if(HAS_NO_AVX2 AND HAS_AVX512)
    message(FATAL_ERROR "HAS_AVX512=ON requires SIMD build. Set HAS_NO_AVX2=OFF.")
endif()

if(HAS_NO_AVX2)
    set(EGAROUCID_ENGINE_VARIANT "Generic")
elseif(HAS_AVX512)
    set(EGAROUCID_ENGINE_VARIANT "AVX512")
else()
    set(EGAROUCID_ENGINE_VARIANT "SIMD")
endif()
message(STATUS "Egaroucid engine variant: ${EGAROUCID_ENGINE_VARIANT}")

if(MSVC)
    # Do not force /O2 for all configurations.
    # Visual Studio Debug uses /RTC1 and conflicts with /O2.
    add_compile_options(/EHsc)
    if(HAS_AVX512)
        add_compile_options(/arch:AVX512)
    elseif(NOT HAS_NO_AVX2 AND NOT HAS_ARM_PROCESSOR)
        add_compile_options(/arch:AVX2)
    endif()
    if(CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Multi-config generator detected. Use '--config Release' to avoid Debug builds.")
    endif()
elseif(APPLE)
    # Keep the historical local optimization defaults for non-MSVC builds.
    add_compile_options(-O2 -mtune=native -pthread)
else()
    # Keep the historical -march=native behavior by default.
    # For a library-only build, avoid forcing it so the artifact is more portable.
    if(BUILD_ENGINE_LIB AND NOT BUILD_CONSOLE AND NOT BUILD_GUI)
        add_compile_options(-O2 -mtune=native -pthread)
    else()
        add_compile_options(-O2 -mtune=native -march=native -pthread)
    endif()
endif()

# Executable
if(NOT DEFINED EXECUTABLE_OUTPUT_PATH)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
endif()

if(BUILD_GUI)
    find_package(Siv3D REQUIRED)
    add_executable(Egaroucid.out ./src/Egaroucid.cpp)
    target_compile_definitions(Egaroucid.out PRIVATE GUI_BUILD=1)
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
endif()

if(BUILD_CONSOLE)
    add_executable(Egaroucid_for_Console.out ./src/Egaroucid_for_Console.cpp)
endif()

if(BUILD_ENGINE_LIB)
    add_library(egaroucid ./src/lib/egaroucid_c_api.cpp)
    target_include_directories(egaroucid
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(egaroucid PRIVATE EGAROUCID_BUILDING_DLL)
    if(NOT BUILD_SHARED_LIBS)
        target_compile_definitions(egaroucid PUBLIC EGAROUCID_STATIC)
    else()
        target_compile_definitions(egaroucid PUBLIC EGAROUCID_SHARED)
        set_target_properties(egaroucid PROPERTIES
                VERSION ${PROJECT_VERSION}
                SOVERSION ${PROJECT_VERSION_MAJOR}
        )        
    endif()

    add_executable(egaroucid_cpp_example EXCLUDE_FROM_ALL ./examples/cpp/simple.cpp)
    target_link_libraries(egaroucid_cpp_example PRIVATE egaroucid)
    # Keep the sample binary in the build tree so it can be built/run even when
    # source-tree output directories are read-only.
    set_target_properties(egaroucid_cpp_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
    )
endif()
