#include "rvs/runtime/pipeline.hpp"

#include <set>

namespace rvs {

Pipeline::Pipeline(std::vector<ProcessorPtr> processors) : processors_(std::move(processors)) {
    if (processors_.empty()) throw Error(ErrorCode::invalid_input, "pipeline must contain a processor");
    std::set<std::string> names;
    for (const auto& processor : processors_) {
        if (!processor) throw Error(ErrorCode::invalid_input, "null processor");
        const auto descriptor = processor->describe();
        simulated_ = simulated_ || descriptor.simulated;
        const auto name = descriptor.name;
        if (name.empty() || !names.insert(name).second)
            throw Error(ErrorCode::invalid_input, "empty or duplicate processor name: " + name);
    }
}

void Pipeline::run(Packet& packet, const Context& context) {
    for (const auto& processor : processors_) {
        context.checkpoint();
        const auto descriptor = processor->describe();
        context.observer.record({context.task_id, descriptor, "begin", "running", 0});
        const auto start = Clock::now();
        try {
            processor->process(packet, context);
            context.checkpoint();
            context.report(descriptor, "end", elapsed_ms(start));
        } catch (const Error& error) {
            context.observer.record({context.task_id, descriptor, "end", to_string(error.code()), elapsed_ms(start)});
            throw;
        } catch (const std::exception& error) {
            context.observer.record({context.task_id, descriptor, "end", "failed", elapsed_ms(start)});
            throw Error(ErrorCode::internal, error.what());
        } catch (...) {
            context.observer.record({context.task_id, descriptor, "end", "failed", elapsed_ms(start)});
            throw Error(ErrorCode::internal, "unknown processor exception");
        }
    }
}

bool Pipeline::simulated() const noexcept { return simulated_; }

}  // namespace rvs

