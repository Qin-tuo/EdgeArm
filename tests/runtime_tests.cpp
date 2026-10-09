#include "rvs/runtime/executor.hpp"
#include "rvs/runtime/registry.hpp"

#include <atomic>
#include <iostream>
#include <stdexcept>

using namespace std::chrono_literals;

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function> void expect_error(rvs::ErrorCode expected, Function function) {
    try { function(); }
    catch (const rvs::Error& error) { require(error.code() == expected, "wrong error code"); return; }
    throw std::runtime_error("expected error was not raised");
}

rvs::Pipeline one(rvs::FunctionProcessor::Function function) {
    std::vector<rvs::ProcessorPtr> processors;
    processors.push_back(std::make_unique<rvs::FunctionProcessor>(rvs::Descriptor{"test", "builtin"}, std::move(function)));
    return rvs::Pipeline(std::move(processors));
}

rvs::Result await(const rvs::Ticket& ticket) {
    require(ticket.result.wait_for(2s) == std::future_status::ready, "task future did not resolve");
    return ticket.result.get();
}

void success_and_failure_recovery() {
    rvs::NullObserver observer;
    rvs::Executor executor(one([](rvs::Packet& packet, const rvs::Context&) {
        const auto& text = packet.get<std::string>("input");
        if (text == "fail") throw std::runtime_error("injected failure");
        packet.set("output", text);
    }), 4, observer);
    rvs::Packet bad;
    bad.set("input", "fail");
    const auto failed = await(executor.submit(std::move(bad), 1s));
    require(failed.status == rvs::Status::failed && failed.error_code == rvs::ErrorCode::internal, "failure not reported");
    rvs::Packet good;
    good.set("input", "next task");
    const auto completed = await(executor.submit(std::move(good), 1s));
    require(completed.status == rvs::Status::completed, "executor did not recover");
    require(completed.output.get<std::string>("output") == "next task", "packet output lost");
    require(!executor.cancel(completed.task_id), "completed task was cancelled");
}

void cancel_blocks_downstream_and_recovers() {
    rvs::NullObserver observer;
    std::promise<void> entered;
    auto started = entered.get_future();
    std::atomic<int> downstream{0};
    std::vector<rvs::ProcessorPtr> stages;
    stages.push_back(std::make_unique<rvs::FunctionProcessor>(rvs::Descriptor{"wait", "builtin"},
        [&](rvs::Packet& packet, const rvs::Context& context) {
            if (packet.get<std::string>("input") == "wait") {
                entered.set_value();
                context.sleep_for(1s);
            }
        }));
    stages.push_back(std::make_unique<rvs::FunctionProcessor>(rvs::Descriptor{"effect", "builtin"},
        [&](rvs::Packet&, const rvs::Context&) { ++downstream; }));
    rvs::Executor executor(rvs::Pipeline(std::move(stages)), 2, observer);
    rvs::Packet packet;
    packet.set("input", "wait");
    const auto ticket = executor.submit(std::move(packet), 2s);
    require(started.wait_for(1s) == std::future_status::ready, "processor not entered");
    require(executor.cancel(ticket.task_id), "active cancellation failed");
    require(await(ticket).status == rvs::Status::cancelled, "cancelled task completed");
    require(downstream == 0, "downstream processor ran after cancellation");
    rvs::Packet next;
    next.set("input", "continue");
    require(await(executor.submit(std::move(next), 1s)).status == rvs::Status::completed, "no recovery after cancellation");
    require(downstream == 1, "new task did not reach downstream processor");
}

void queue_limit_pending_cancel_and_shutdown() {
    rvs::NullObserver observer;
    std::promise<void> entered;
    auto started = entered.get_future();
    rvs::Executor executor(one([&](rvs::Packet&, const rvs::Context& context) {
        entered.set_value();
        context.sleep_for(2s);
    }), 1, observer);
    const auto active = executor.submit({}, 3s);
    require(started.wait_for(1s) == std::future_status::ready, "processor not entered");
    const auto queued = executor.submit({}, 3s);
    expect_error(rvs::ErrorCode::queue_full, [&] { executor.submit({}, 3s); });
    require(executor.cancel(queued.task_id), "pending cancellation failed");
    require(await(queued).status == rvs::Status::cancelled, "pending future not cancelled");
    const auto another = executor.submit({}, 3s);
    executor.shutdown();
    require(await(active).status == rvs::Status::cancelled, "active task not cancelled on shutdown");
    require(await(another).status == rvs::Status::cancelled, "queued task not cancelled on shutdown");
    expect_error(rvs::ErrorCode::stopped, [&] { executor.submit({}, 1s); });
    executor.shutdown();
}

void deadline_includes_queue_time() {
    rvs::NullObserver observer;
    std::promise<void> entered;
    auto started = entered.get_future();
    std::atomic<int> calls{0};
    rvs::Executor executor(one([&](rvs::Packet&, const rvs::Context& context) {
        if (++calls == 1) { entered.set_value(); context.sleep_for(50ms); }
    }), 2, observer);
    const auto first = executor.submit({}, 1s);
    require(started.wait_for(1s) == std::future_status::ready, "processor not entered");
    const auto expired = executor.submit({}, 5ms);
    require(await(first).status == rvs::Status::completed, "first task failed");
    require(await(expired).status == rvs::Status::timeout, "queue deadline ignored");
    require(calls == 1, "expired pending task executed");
}

void active_timeout_and_validation() {
    rvs::NullObserver observer;
    rvs::Executor executor(one([](rvs::Packet&, const rvs::Context& context) { context.sleep_for(1s); }), 1, observer);
    require(await(executor.submit({}, 5ms)).status == rvs::Status::timeout, "active timeout ignored");
    expect_error(rvs::ErrorCode::invalid_input, [&] { executor.submit({}, 0ms); });
    expect_error(rvs::ErrorCode::invalid_input, [] { rvs::Pipeline pipeline({}); });
    rvs::Packet packet;
    packet.set("number", std::int64_t{1});
    expect_error(rvs::ErrorCode::invalid_input, [&] { packet.get<std::string>("number"); });
    expect_error(rvs::ErrorCode::invalid_input, [&] { packet.get<std::string>("missing"); });
}
}  // namespace

int main() {
    try {
        success_and_failure_recovery();
        cancel_blocks_downstream_and_recovers();
        queue_limit_pending_cancel_and_shutdown();
        deadline_includes_queue_time();
        active_timeout_and_validation();
        std::cout << "5 runtime behavior suites passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

