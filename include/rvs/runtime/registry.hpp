#pragma once

#include "rvs/core/config.hpp"
#include "rvs/runtime/processor.hpp"

#include <filesystem>
#include <map>

namespace rvs {

struct BuildContext {
    const Config& config;
    std::filesystem::path artifact_directory;
};

class Registry {
public:
    using Factory = std::function<ProcessorPtr(const BuildContext&)>;
    void add(std::string name, Factory factory);
    ProcessorPtr create(const std::string& name, const BuildContext& context) const;
    std::vector<std::string> names() const;
private:
    std::map<std::string, Factory> factories_;
};

}  // namespace rvs

