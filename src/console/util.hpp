/*
    Egaroucid Project

    @file util.hpp
        Utility
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/
#pragma once

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>
#include "./../engine/common.hpp"

std::string get_current_datetime() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    get_localtime(&tm, &in_time_t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %X");
    return ss.str();
}

std::string get_current_datetime_for_file() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    get_localtime(&tm, &in_time_t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d-%H=%M=%S");
    return ss.str();
}

std::vector<std::string> split_by_space(const std::string &str) {
    std::vector<std::string> tokens;
    std::istringstream iss(str);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> split_by_delimiter(const std::string &str, const std::string &delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = str.find(delimiter);
    
    while (end != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }
    
    tokens.push_back(str.substr(start));
    return tokens;
}

std::string remove_spaces(const std::string &str) {
    std::string result = str;
    result.erase(std::remove(result.begin(), result.end(), ' '), result.end());
    return result;
}