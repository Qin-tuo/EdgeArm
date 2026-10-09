#pragma once

#include "rvs/runtime/pipeline.hpp"

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <optional>
#include <thread>

namespace rvs {

enum class Status { completed, cancelled, timeout, failed };
const char* to_string(Status status);

struct Result {
    std::uint64_t task_id;
    Status status;
    Packet output;
    std::optional<ErrorCode> error_code;
    std::string message;
};

struct Ticket {
    std::uint64_t task_id;
    std::shared_future<Result> result;
};

class Executor {
public:
    // The observer must outlive this executor. Capacity counts waiting tasks.
    Executor(Pipeline pipeline, std::size_t queue_capacity, Observer& observer);
    ~Executor();
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    Ticket submit(Packet input, std::chrono::milliseconds timeout);
    bool cancel(std::uint64_t task_id);
    void shutdown();
private:
    struct Job {
        std::uint64_t id;
        Packet input;
        Cancellation cancellation;
        Clock::time_point submitted;
        Clock::time_point deadline;
        std::promise<Result> promise;
    };
    void work();
    void finish_pending(const std::shared_ptr<Job>& job);
    void report_task(const Job& job, Status status);
    Pipeline pipeline_;
    std::size_t capacity_;
    Observer& observer_;
    std::mutex mutex_;
    std::mutex shutdown_mutex_;
    std::condition_variable ready_;
    std::deque<std::shared_ptr<Job>> pending_;
    std::shared_ptr<Job> active_;
    std::uint64_t next_id_ = 1;
    bool stopping_ = false;
    std::thread worker_;
};

}  // namespace rvs

