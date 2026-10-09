#include "rvs/modules/examples.hpp"

namespace rvs::examples {

void register_echo(Registry& registry) {
    registry.add("echo", [](const BuildContext& build) -> ProcessorPtr {
        const auto backend = build.config.get("echo.backend", "builtin");
        if (backend != "builtin") throw Error(ErrorCode::unsupported, "echo backend not implemented: " + backend);
        const auto prefix = build.config.get("echo.prefix", "echo: ");
        const auto delay = build.config.integer("echo.delay_ms", 0, 0, 60000);
        return std::make_unique<FunctionProcessor>(Descriptor{"echo", "builtin"},
            [prefix, delay](Packet& packet, const Context& context) {
                context.sleep_for(std::chrono::milliseconds(delay));
                packet.set("output", prefix + packet.get<std::string>("input"));
            });
    });
}

}  // namespace rvs::examples

