#include "rvs/audio/wav.hpp"
#include "rvs/core/error.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

namespace rvs::audio {
namespace {
void little_endian(std::ostream& stream, std::uint32_t value, unsigned int bytes) {
    for (unsigned int i = 0; i < bytes; ++i) stream.put(static_cast<char>((value >> (8 * i)) & 0xff));
}
}  // namespace

void write_wav(const std::filesystem::path& path, const std::vector<float>& samples,
               std::uint32_t sample_rate) {
    if (sample_rate == 0 || sample_rate > 192000 ||
        samples.size() > (std::numeric_limits<std::uint32_t>::max() - 36) / 2)
        throw Error(ErrorCode::invalid_input, "invalid WAV dimensions");
    for (auto sample : samples)
        if (!std::isfinite(sample)) throw Error(ErrorCode::invalid_input, "non-finite audio sample");
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw Error(ErrorCode::invalid_input, "cannot write WAV: " + path.string());
    const auto size = static_cast<std::uint32_t>(samples.size() * 2);
    stream.write("RIFF", 4); little_endian(stream, 36 + size, 4);
    stream.write("WAVEfmt ", 8); little_endian(stream, 16, 4);
    little_endian(stream, 1, 2); little_endian(stream, 1, 2);
    little_endian(stream, sample_rate, 4); little_endian(stream, sample_rate * 2, 4);
    little_endian(stream, 2, 2); little_endian(stream, 16, 2);
    stream.write("data", 4); little_endian(stream, size, 4);
    for (auto sample : samples) {
        const auto pcm = static_cast<std::int16_t>(std::lround(std::clamp(sample, -1.0f, 1.0f) * 32767.0f));
        little_endian(stream, static_cast<std::uint16_t>(pcm), 2);
    }
    stream.close();
    if (!stream) throw Error(ErrorCode::internal, "WAV write failed: " + path.string());
}

}  // namespace rvs::audio

