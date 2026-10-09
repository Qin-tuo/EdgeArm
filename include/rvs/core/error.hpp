#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace rvs {

enum class ErrorCode { invalid_input, unsupported, cancelled, timeout, queue_full, stopped, internal };

inline const char* to_string(ErrorCode code) {
    switch (code) {
        case ErrorCode::invalid_input: return "invalid_input";
        case ErrorCode::unsupported: return "unsupported";
        case ErrorCode::cancelled: return "cancelled";
        case ErrorCode::timeout: return "timeout";
        case ErrorCode::queue_full: return "queue_full";
        case ErrorCode::stopped: return "stopped";
        case ErrorCode::internal: return "internal";
    }
    return "internal";
}

class Error : public std::runtime_error {
public:
    Error(ErrorCode code, std::string message) : std::runtime_error(std::move(message)), code_(code) {}
    ErrorCode code() const noexcept { return code_; }
private:
    ErrorCode code_;
};

}  // namespace rvs

