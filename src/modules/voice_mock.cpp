#include "rvs/modules/examples.hpp"
#include "rvs/audio/wav.hpp"

#include <algorithm>

namespace rvs::examples {
namespace {
void require_backend(const Config& config, const std::string& stage, const std::string& supported) {
    const auto requested = config.get(stage + ".backend", supported);
    if (requested != supported)
        throw Error(ErrorCode::unsupported, stage + " backend not implemented: " + requested);
}

Descriptor mock(const std::string& name) { return {name, "mock", "none", true}; }
}  // namespace

void register_voice_mock(Registry& registry) {
    registry.add("asr", [](const BuildContext& build) -> ProcessorPtr {
        require_backend(build.config, "asr", "mock");
        const auto delay = build.config.integer("asr.delay_ms", 2, 0, 60000);
        return std::make_unique<FunctionProcessor>(mock("asr"),
            [delay](Packet& packet, const Context& context) {
                // A transcript fixture exercises the contract without a model or microphone.
                context.sleep_for(std::chrono::milliseconds(delay));
                packet.set("transcript", packet.get<std::string>("input"));
                packet.set("input_kind", std::string("transcript_fixture"));
            });
    });
    registry.add("llm", [](const BuildContext& build) -> ProcessorPtr {
        require_backend(build.config, "llm", "mock");
        const auto delay = build.config.integer("llm.token_delay_ms", 2, 0, 60000);
        const auto descriptor = mock("llm");
        return std::make_unique<FunctionProcessor>(descriptor,
            [delay, descriptor](Packet& packet, const Context& context) {
                const auto start = Clock::now();
                const std::vector<std::string> chunks{"模拟回答：", packet.get<std::string>("transcript")};
                std::string output;
                bool first = true;
                for (const auto& chunk : chunks) {
                    context.sleep_for(std::chrono::milliseconds(delay));
                    if (first) { context.report(descriptor, "first_output", elapsed_ms(start)); first = false; }
                    output += chunk;
                }
                packet.set("output", std::move(output));
                packet.set("generated_chunks", static_cast<std::int64_t>(chunks.size()));
            });
    });
    registry.add("tts", [](const BuildContext& build) -> ProcessorPtr {
        require_backend(build.config, "tts", "mock");
        const auto rate = build.config.integer("tts.sample_rate", 16000, 8000, 192000);
        const auto delay = build.config.integer("tts.chunk_delay_ms", 1, 0, 60000);
        const auto descriptor = mock("tts");
        return std::make_unique<FunctionProcessor>(descriptor,
            [rate, delay, descriptor](Packet& packet, const Context& context) {
                const auto start = Clock::now();
                // Silence is deliberately used: this is not intelligible synthesized speech.
                const auto count = std::min<std::size_t>(16000,
                    800 + packet.get<std::string>("output").size() * 20);
                std::vector<float> samples;
                bool first = true;
                while (samples.size() < count) {
                    context.sleep_for(std::chrono::milliseconds(delay));
                    if (first) { context.report(descriptor, "first_output", elapsed_ms(start)); first = false; }
                    samples.insert(samples.end(), std::min<std::size_t>(512, count - samples.size()), 0.0f);
                }
                packet.set("audio.samples", std::move(samples));
                packet.set("audio.sample_rate", static_cast<std::int64_t>(rate));
            });
    });
    registry.add("file_sink", [](const BuildContext& build) -> ProcessorPtr {
        require_backend(build.config, "file_sink", "wav");
        const auto directory = build.artifact_directory;
        return std::make_unique<FunctionProcessor>(Descriptor{"file_sink", "wav", "none", true},
            [directory](Packet& packet, const Context& context) {
                context.checkpoint();
                const auto target = directory / ("task-" + std::to_string(context.task_id) + ".wav");
                const auto temporary = std::filesystem::path(target.string() + ".tmp");
                try {
                    const auto rate = packet.get<std::int64_t>("audio.sample_rate");
                    if (rate < 1 || rate > 192000) throw Error(ErrorCode::invalid_input, "invalid audio sample rate");
                    audio::write_wav(temporary, packet.get<std::vector<float>>("audio.samples"),
                                     static_cast<std::uint32_t>(rate));
                    context.checkpoint();
                    std::filesystem::rename(temporary, target);
                    packet.set("artifact", target.string());
                } catch (...) {
                    std::error_code ignored;
                    std::filesystem::remove(temporary, ignored);
                    throw;
                }
            });
    });
}

}  // namespace rvs::examples

