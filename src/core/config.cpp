#include "rvs/core/config.hpp"
#include "rvs/core/error.hpp"

#include <fstream>
#include <sstream>

namespace rvs {
namespace {
std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}
}  // namespace

Config Config::load(const std::filesystem::path& file) {
    std::ifstream stream(file);
    if (!stream) throw Error(ErrorCode::invalid_input, "cannot open config: " + file.string());
    Config config;
    config.directory_ = std::filesystem::absolute(file).parent_path();
    std::string line, section;
    std::size_t number = 0;
    while (std::getline(stream, line)) {
        ++number;
        line = trim(line);
        if (line.empty() || line.front() == '#' || line.front() == ';') continue;
        if (line.front() == '[') {
            if (line.back() != ']' || line.size() < 3)
                throw Error(ErrorCode::invalid_input, "invalid config section at line " + std::to_string(number));
            section = trim(line.substr(1, line.size() - 2));
            if (section.empty() || section.find_first_of("[]= \t") != std::string::npos)
                throw Error(ErrorCode::invalid_input, "invalid section name at line " + std::to_string(number));
            continue;
        }
        const auto equal = line.find('=');
        if (equal == std::string::npos)
            throw Error(ErrorCode::invalid_input, "expected key=value at line " + std::to_string(number));
        auto key = trim(line.substr(0, equal));
        if (key.empty() || key.find_first_of("[] \t") != std::string::npos)
            throw Error(ErrorCode::invalid_input, "invalid key at line " + std::to_string(number));
        if (!section.empty()) key = section + "." + key;
        if (!config.values_.emplace(key, trim(line.substr(equal + 1))).second)
            throw Error(ErrorCode::invalid_input, "duplicate config key: " + key);
    }
    if (stream.bad()) throw Error(ErrorCode::invalid_input, "cannot read config: " + file.string());
    return config;
}

std::string Config::get(const std::string& key, const std::string& fallback) const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
}

int Config::integer(const std::string& key, int fallback, int minimum, int maximum) const {
    const auto raw = get(key, std::to_string(fallback));
    try {
        std::size_t consumed = 0;
        const auto value = std::stoi(raw, &consumed);
        if (consumed != raw.size() || value < minimum || value > maximum) throw std::invalid_argument("range");
        return value;
    } catch (const std::exception&) {
        throw Error(ErrorCode::invalid_input, "invalid integer for " + key + ": " + raw);
    }
}

bool Config::boolean(const std::string& key, bool fallback) const {
    const auto raw = get(key, fallback ? "true" : "false");
    if (raw == "true") return true;
    if (raw == "false") return false;
    throw Error(ErrorCode::invalid_input, "expected true or false for " + key);
}

std::vector<std::string> Config::list(const std::string& key) const {
    const auto raw = get(key);
    if (raw.empty()) throw Error(ErrorCode::invalid_input, "empty list: " + key);
    std::vector<std::string> result;
    std::size_t start = 0;
    while (true) {
        const auto end = raw.find(',', start);
        auto item = trim(raw.substr(start, end == std::string::npos ? end : end - start));
        if (item.empty()) throw Error(ErrorCode::invalid_input, "empty list entry: " + key);
        result.push_back(std::move(item));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return result;
}

std::filesystem::path Config::path(const std::string& key, const std::string& fallback) const {
    std::filesystem::path value(get(key, fallback));
    if (value.empty()) throw Error(ErrorCode::invalid_input, "empty path: " + key);
    if (value.is_relative()) value = directory_ / value;
    return value.lexically_normal();
}

}  // namespace rvs

