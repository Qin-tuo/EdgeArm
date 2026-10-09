#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace rvs {

class Config {
public:
    static Config load(const std::filesystem::path& file);
    std::string get(const std::string& key, const std::string& fallback = "") const;
    int integer(const std::string& key, int fallback, int minimum, int maximum) const;
    bool boolean(const std::string& key, bool fallback) const;
    std::vector<std::string> list(const std::string& key) const;
    std::filesystem::path path(const std::string& key, const std::string& fallback) const;
private:
    std::map<std::string, std::string> values_;
    std::filesystem::path directory_;
};

}  // namespace rvs

