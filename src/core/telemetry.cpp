#include "rvs/core/telemetry.hpp"
#include "rvs/core/error.hpp"

#include <chrono>
#include <iomanip>
#include <locale>
#include <sstream>

namespace rvs {

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const auto raw : value) {
        const auto c = static_cast<unsigned char>(raw);
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                                  << static_cast<int>(c) << std::dec;
                else out << raw;
        }
    }
    return out.str();
}

JsonlObserver::JsonlObserver(const std::filesystem::path& file, std::string run_id)
    : stream_(file), run_id_(std::move(run_id)) {
    if (!stream_) throw Error(ErrorCode::invalid_input, "cannot open metrics: " + file.string());
    stream_.imbue(std::locale::classic());
}

void JsonlObserver::record(const Event& event) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        stream_ << "{\"schema_version\":1,\"run_id\":\"" << json_escape(run_id_)
                << "\",\"task_id\":" << event.task_id << ",\"monotonic_us\":" << timestamp
                << ",\"stage\":\"" << json_escape(event.processor.name)
                << "\",\"backend\":\"" << json_escape(event.processor.backend)
                << "\",\"model\":\"" << json_escape(event.processor.model)
                << "\",\"simulated\":" << (event.processor.simulated ? "true" : "false")
                << ",\"phase\":\"" << json_escape(event.phase)
                << "\",\"status\":\"" << json_escape(event.status)
                << "\",\"elapsed_ms\":" << std::fixed << std::setprecision(6) << event.elapsed_ms << "}\n";
        stream_.flush();
        if (!stream_) failed_.store(true);
    } catch (...) {
        failed_.store(true);
    }
}

}  // namespace rvs

