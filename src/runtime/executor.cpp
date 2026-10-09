#include "rvs/runtime/executor.hpp"

#include <algorithm>

namespace rvs {
namespace {
Status error_status(ErrorCode code) {
    if (code == ErrorCode::cancelled) return Status::cancelled;
    if (code == ErrorCode::timeout) return Status::timeout;
    return Status::failed;
}
}  // namespace

const char* to_string(Status status) {
    switch (status) {
        case Status::completed: return "completed";
        case Status::cancelled: return "cancelled";
        case Status::timeout: return "timeout";
        case Status::failed: return "failed";
    }
    return "failed";
}

Executor::Executor(Pipeline pipeline, std::size_t queue_capacity, Observer& observer)
    : pipeline_(std::move(pipeline)), capacity_(queue_capacity), observer_(observer) {
    if (capacity_ == 0) throw Error(ErrorCode::invalid_input, "queue capacity must be positive");
    worker_ = std::thread(&Executor::work, this);
}

Executor::~Executor() { shutdown(); }

Ticket Executor::submit(Packet input, std::chrono::milliseconds timeout) {
    if (timeout.count() <= 0) throw Error(ErrorCode::invalid_input, "timeout must be positive");
    const auto submitted = Clock::now();
    auto job = std::make_shared<Job>();
    job->input = std::move(input);
    job->submitted = submitted;
    job->deadline = submitted + timeout;
    auto result = job->promise.get_future().share();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) throw Error(ErrorCode::stopped, "executor has stopped");
        if (pending_.size() >= capacity_) throw Error(ErrorCode::queue_full, "waiting queue is full");
        job->id = next_id_++;
        pending_.push_back(job);
    }
    ready_.notify_one();
    return {job->id, std::move(result)};
}

bool Executor::cancel(std::uint64_t task_id) {
    std::shared_ptr<Job> pending;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_ && active_->id == task_id) {
            active_->cancellation.cancel();
            return true;
        }
        const auto it = std::find_if(pending_.begin(), pending_.end(),
            [task_id](const auto& job) { return job->id == task_id; });
        if (it == pending_.end()) return false;
        pending = *it;
        pending->cancellation.cancel();
        pending_.erase(it);
    }
    finish_pending(pending);
    return true;
}

void Executor::report_task(const Job& job, Status status) {
    observer_.record({job.id, {"task", "runtime", "none", pipeline_.simulated()},
                      "end", to_string(status), elapsed_ms(job.submitted)});
}

void Executor::finish_pending(const std::shared_ptr<Job>& job) {
    report_task(*job, Status::cancelled);
    job->promise.set_value({job->id, Status::cancelled, {}, ErrorCode::cancelled,
                            "cancelled before execution"});
}

void Executor::shutdown() {
    std::lock_guard<std::mutex> lifecycle(shutdown_mutex_);
    std::deque<std::shared_ptr<Job>> cancelled;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        if (active_) active_->cancellation.cancel();
        cancelled.swap(pending_);
    }
    for (const auto& job : cancelled) finish_pending(job);
    ready_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void Executor::work() {
    while (true) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
            if (stopping_) return;
            job = pending_.front();
            pending_.pop_front();
            active_ = job;
        }
        Context context{job->id, job->cancellation, job->deadline, observer_};
        observer_.record({job->id, {"queue", "runtime", "none", pipeline_.simulated()},
                          "end", "completed", elapsed_ms(job->submitted)});
        Result result{job->id, Status::completed, std::move(job->input), std::nullopt, {}};
        try {
            context.checkpoint();
            pipeline_.run(result.output, context);
        } catch (const Error& error) {
            result.status = error_status(error.code());
            result.error_code = error.code();
            result.message = error.what();
        } catch (const std::exception& error) {
            result.status = Status::failed;
            result.error_code = ErrorCode::internal;
            result.message = error.what();
        } catch (...) {
            result.status = Status::failed;
            result.error_code = ErrorCode::internal;
            result.message = "unknown execution error";
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Completion and cancellation are ordered by the same lock.
            if (result.status == Status::completed) {
                try { context.checkpoint(); }
                catch (const Error& error) {
                    result.status = error_status(error.code());
                    result.error_code = error.code();
                    result.message = error.what();
                }
            }
            active_.reset();
        }
        report_task(*job, result.status);
        job->promise.set_value(std::move(result));
    }
}

}  // namespace rvs

