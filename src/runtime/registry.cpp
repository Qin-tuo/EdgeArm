#include "rvs/runtime/registry.hpp"

namespace rvs {

void Registry::add(std::string name, Factory factory) {
    if (name.empty() || !factory) throw Error(ErrorCode::invalid_input, "invalid processor registration");
    if (!factories_.emplace(std::move(name), std::move(factory)).second)
        throw Error(ErrorCode::invalid_input, "duplicate processor registration");
}

ProcessorPtr Registry::create(const std::string& name, const BuildContext& context) const {
    const auto it = factories_.find(name);
    if (it == factories_.end()) throw Error(ErrorCode::unsupported, "unregistered processor: " + name);
    auto processor = it->second(context);
    if (!processor) throw Error(ErrorCode::internal, "factory returned null: " + name);
    return processor;
}

std::vector<std::string> Registry::names() const {
    std::vector<std::string> result;
    for (const auto& item : factories_) result.push_back(item.first);
    return result;
}

}  // namespace rvs

