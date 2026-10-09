#pragma once

#include "rvs/core/error.hpp"
#include "rvs/core/telemetry.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

namespace rvs {

using Clock = std::chrono::steady_clock;

class Cancellation {
public:
    Cancellation() : flag_(std::make_shared<std::atomic<bool>>(false)) {}
    void cancel() const noexcept { flag_->store(true); }
    bool cancelled() const noexcept { return flag_->load(); }
private:
    std::shared_ptr<std::atomic<bool>> flag_;
};

struct Context {
    std::uint64_t task_id;
    Cancellation cancellation;
    Clock::time_point deadline;
    Observer& observer;

    void checkpoint() const {
        if (cancellation.cancelled()) throw Error(ErrorCode::cancelled, "task cancelled");
        if (Clock::now() >= deadline) throw Error(ErrorCode::timeout, "task deadline exceeded");
    }
    void sleep_for(std::chrono::milliseconds duration) const {
        const auto until = Clock::now() + duration;
        do {
            checkpoint();
            const auto now = Clock::now();
            if (now >= until) break;
            std::this_thread::sleep_until(std::min(until, now + std::chrono::milliseconds(1)));
        } while (true);
    }
    void report(const Descriptor& descriptor, std::string phase, double milliseconds) const {
        observer.record({task_id, descriptor, std::move(phase), "completed", milliseconds});
    }
};

inline double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace rvs

