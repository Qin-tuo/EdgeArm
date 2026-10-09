#pragma once

#include "rvs/runtime/processor.hpp"

#include <vector>

namespace rvs {

// A sequential pipeline is the first scheduler. Modules can replace it later.
class Pipeline {
public:
    explicit Pipeline(std::vector<ProcessorPtr> processors);
    void run(Packet& packet, const Context& context);
    bool simulated() const noexcept;
private:
    std::vector<ProcessorPtr> processors_;
    bool simulated_ = false;
};

}  // namespace rvs

