#include "rvs/core/config.hpp"
#include "rvs/core/telemetry.hpp"
#include "rvs/modules/examples.hpp"
#include "rvs/runtime/executor.hpp"

#include <fstream>
#include <iostream>
#include <limits>

namespace {
struct Options {
    std::filesystem::path config = "configs/echo.ini";
    std::filesystem::path output_root;
    std::string input = "你好";
    int repeat = 1;
    int cancel_after_ms = -1;
    bool list = false;
    bool help = false;
};

int number(const std::string& raw, int minimum, int maximum) {
    try {
        std::size_t used;
        const auto value = std::stoi(raw, &used);
        if (used != raw.size() || value < minimum || value > maximum) throw std::invalid_argument("range");
        return value;
    } catch (const std::exception&) {
        throw rvs::Error(rvs::ErrorCode::invalid_input, "invalid command line number: " + raw);
    }
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") { options.help = true; continue; }
        if (arg == "--list-processors") { options.list = true; continue; }
        if (arg != "--config" && arg != "--input" && arg != "--output-root" &&
            arg != "--repeat" && arg != "--cancel-after-ms")
            throw rvs::Error(rvs::ErrorCode::invalid_input, "unknown option: " + arg);
        if (++i == argc) throw rvs::Error(rvs::ErrorCode::invalid_input, "missing value for " + arg);
        const std::string value = argv[i];
        if (arg == "--config") options.config = value;
        else if (arg == "--input") options.input = value;
        else if (arg == "--output-root") {
            if (value.empty()) throw rvs::Error(rvs::ErrorCode::invalid_input, "empty output root");
            options.output_root = std::filesystem::absolute(value);
        } else if (arg == "--repeat") options.repeat = number(value, 1, 10000);
        else options.cancel_after_ms = number(value, 0, 60000);
    }
    return options;
}

std::filesystem::path plan_run_directory(const std::filesystem::path& root) {
    const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
    for (int i = 0; i < 1000; ++i) {
        auto path = root / ("run-" + std::to_string(stamp) + "-" + std::to_string(i));
        if (!std::filesystem::exists(path)) return path;
    }
    throw rvs::Error(rvs::ErrorCode::internal, "cannot allocate unique run directory");
}

void print_result(const rvs::Result& result, bool simulated) {
    std::cout << "{\"task_id\":" << result.task_id << ",\"status\":\"" << rvs::to_string(result.status)
              << "\",\"simulated\":" << (simulated ? "true" : "false");
    if (result.status == rvs::Status::completed) {
        for (const auto& key : {"output", "artifact"}) {
            if (result.output.contains(key))
                std::cout << ",\"" << key << "\":\"" << rvs::json_escape(result.output.get<std::string>(key)) << "\"";
        }
    } else {
        std::cout << ",\"error\":\"" << (result.error_code ? rvs::to_string(*result.error_code) : "internal")
                  << "\",\"message\":\"" << rvs::json_escape(result.message) << "\"";
    }
    std::cout << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        rvs::Registry registry;
        rvs::examples::register_echo(registry);
        rvs::examples::register_voice_mock(registry);
        if (options.help) {
            std::cout << "rvs-run --config FILE --input TEXT [--repeat N] [--output-root DIR]\n"
                         "        [--cancel-after-ms N] [--list-processors]\n"
                         "Voice mock uses transcript fixtures and silent WAV files; no hardware inference.\n";
            return 0;
        }
        if (options.list) {
            for (const auto& name : registry.names()) std::cout << name << '\n';
            return 0;
        }
        const auto config = rvs::Config::load(options.config);
        const auto capacity = config.integer("runtime.queue_capacity", 4, 1, 1024);
        const auto timeout = config.integer("runtime.task_timeout_ms", 5000, 1, 3600000);
        const auto root = options.output_root.empty() ? config.path("runtime.output_root", "../runs") : options.output_root;
        const auto directory = plan_run_directory(root);
        std::vector<rvs::ProcessorPtr> processors;
        for (const auto& name : config.list("pipeline.stages")) processors.push_back(registry.create(name, {config, directory}));
        rvs::Pipeline pipeline(std::move(processors));
        const auto simulated = pipeline.simulated();
        // Factories and pipeline structure are validated before any run artifacts.
        std::filesystem::create_directories(root);
        if (!std::filesystem::create_directory(directory))
            throw rvs::Error(rvs::ErrorCode::internal, "run directory already exists; retry the command");
        std::filesystem::copy_file(options.config, directory / "config.ini");
        std::ofstream manifest(directory / "run.json");
        manifest << "{\"schema_version\":1,\"run_id\":\"" << rvs::json_escape(directory.filename().string())
                 << "\",\"build_version\":\"" << RVS_BUILD_VERSION
                 << "\",\"simulated\":" << (simulated ? "true" : "false")
                 << ",\"hardware_metrics_collected\":false,\"repeat\":" << options.repeat << "}\n";
        manifest.close();
        if (!manifest) throw rvs::Error(rvs::ErrorCode::internal, "cannot write run manifest");
        rvs::JsonlObserver observer(directory / "metrics.jsonl", directory.filename().string());
        rvs::Executor executor(std::move(pipeline), static_cast<std::size_t>(capacity), observer);
        int exit_code = 0;
        for (int i = 0; i < options.repeat; ++i) {
            rvs::Packet input;
            input.set("input", options.input);
            const auto ticket = executor.submit(std::move(input), std::chrono::milliseconds(timeout));
            if (options.cancel_after_ms >= 0 &&
                ticket.result.wait_for(std::chrono::milliseconds(options.cancel_after_ms)) != std::future_status::ready)
                executor.cancel(ticket.task_id);
            const auto result = ticket.result.get();
            print_result(result, simulated);
            if (result.status != rvs::Status::completed) exit_code = 1;
        }
        executor.shutdown();
        if (!observer.healthy()) throw rvs::Error(rvs::ErrorCode::internal, "metrics write failed");
        std::cout << "{\"run_directory\":\"" << rvs::json_escape(directory.string()) << "\"}\n";
        return exit_code;
    } catch (const rvs::Error& error) {
        std::cerr << rvs::to_string(error.code()) << ": " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "internal: " << error.what() << '\n';
        return 2;
    }
}

