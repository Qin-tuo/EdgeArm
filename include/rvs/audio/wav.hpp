#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace rvs::audio {

// Mono float samples in [-1, 1] are encoded as PCM16. No hardware playback.
void write_wav(const std::filesystem::path& path, const std::vector<float>& samples,
               std::uint32_t sample_rate);

}  // namespace rvs::audio
