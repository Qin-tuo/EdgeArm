#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace rvs {

struct Descriptor {
    std::string name;
    std::string backend;
    std::string model = "none";
    bool simulated = false;
};

struct Event {
    std::uint64_t task_id;
    Descriptor processor;
    std::string phase;
    std::string status;
    double elapsed_ms;
};

class Observer {
public:
    virtual ~Observer() = default;
    // Observation must not prevent task futures from being resolved.
    virtual void record(const Event& event) noexcept = 0;
};

class NullObserver final : public Observer {
public:
    void record(const Event&) noexcept override {}
};

class JsonlObserver final : public Observer {
public:
    JsonlObserver(const std::filesystem::path& file, std::string run_id);
    void record(const Event& event) noexcept override;
    bool healthy() const noexcept { return !failed_.load(); }
private:
    std::ofstream stream_;
    std::string run_id_;
    std::mutex mutex_;
    std::atomic<bool> failed_{false};
};

std::string json_escape(const std::string& value);

}  // namespace rvs

