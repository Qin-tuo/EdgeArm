#include "rvs/runtime/executor.hpp"

#include <iostream>
#include <numeric>

int main() {
    // A numeric workload uses only rvs_core, without any speech module.
    std::vector<rvs::ProcessorPtr> stages;
    stages.push_back(std::make_unique<rvs::FunctionProcessor>(rvs::Descriptor{"sum", "builtin"},
        [](rvs::Packet& packet, const rvs::Context& context) {
            context.checkpoint();
            const auto& samples = packet.get<std::vector<float>>("values");
            packet.set("sum", std::accumulate(samples.begin(), samples.end(), 0.0));
        }));
    rvs::NullObserver observer;
    rvs::Executor executor(rvs::Pipeline(std::move(stages)), 2, observer);
    rvs::Packet packet;
    packet.set("values", std::vector<float>{1, 2, 3});
    const auto result = executor.submit(std::move(packet), std::chrono::seconds(1)).result.get();
    if (result.status != rvs::Status::completed) return 1;
    std::cout << "sum=" << result.output.get<double>("sum") << '\n';
    return 0;
}

