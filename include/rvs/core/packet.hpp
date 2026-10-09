#pragma once

#include "rvs/core/error.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <utility>
#include <vector>

namespace rvs {

// Business-specific field names belong to modules, not the executor.
using Value = std::variant<std::string, std::int64_t, double, bool,
                           std::vector<float>, std::vector<std::uint8_t>>;

class Packet {
public:
    void set(std::string key, Value value) { values_.insert_or_assign(std::move(key), std::move(value)); }
    void set(std::string key, const char* value) { set(std::move(key), Value(std::string(value))); }
    bool contains(const std::string& key) const { return values_.count(key) != 0; }
    template<class T> const T& get(const std::string& key) const {
        const auto it = values_.find(key);
        if (it == values_.end()) throw Error(ErrorCode::invalid_input, "missing field: " + key);
        const auto* value = std::get_if<T>(&it->second);
        if (!value) throw Error(ErrorCode::invalid_input, "wrong field type: " + key);
        return *value;
    }
private:
    std::map<std::string, Value> values_;
};

}  // namespace rvs

