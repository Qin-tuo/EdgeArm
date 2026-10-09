#pragma once

#include "rvs/core/context.hpp"
#include "rvs/core/packet.hpp"

#include <functional>
#include <memory>

namespace rvs {

class Processor {
public:
    virtual ~Processor() = default;
    virtual Descriptor describe() const = 0;
    virtual void process(Packet& packet, const Context& context) = 0;
};

class FunctionProcessor final : public Processor {
public:
    using Function = std::function<void(Packet&, const Context&)>;
    FunctionProcessor(Descriptor descriptor, Function function)
        : descriptor_(std::move(descriptor)), function_(std::move(function)) {}
    Descriptor describe() const override { return descriptor_; }
    void process(Packet& packet, const Context& context) override { function_(packet, context); }
private:
    Descriptor descriptor_;
    Function function_;
};

using ProcessorPtr = std::unique_ptr<Processor>;

}  // namespace rvs

