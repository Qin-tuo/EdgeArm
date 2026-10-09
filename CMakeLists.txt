cmake_minimum_required(VERSION 3.16)
project(robot_voice_stack VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)

add_library(rvs_core
    src/core/config.cpp
    src/core/telemetry.cpp
    src/runtime/pipeline.cpp
    src/runtime/executor.cpp
    src/runtime/registry.cpp)
target_include_directories(rvs_core PUBLIC include)
target_link_libraries(rvs_core PUBLIC Threads::Threads)
target_compile_options(rvs_core PRIVATE -Wall -Wextra -Wpedantic)

option(RVS_BUILD_EXAMPLES "Build CLI and application examples" ON)
if(RVS_BUILD_EXAMPLES)
    add_library(rvs_examples src/modules/echo.cpp src/modules/voice_mock.cpp src/audio/wav.cpp)
    target_link_libraries(rvs_examples PUBLIC rvs_core)
    target_compile_options(rvs_examples PRIVATE -Wall -Wextra -Wpedantic)
    add_executable(rvs-run apps/rvs_run.cpp)
    target_link_libraries(rvs-run PRIVATE rvs_examples)
    target_compile_options(rvs-run PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_definitions(rvs-run PRIVATE RVS_BUILD_VERSION="${PROJECT_VERSION}")
    add_executable(rvs-custom-demo examples/custom_pipeline.cpp)
    target_link_libraries(rvs-custom-demo PRIVATE rvs_core)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(rvs-tests tests/runtime_tests.cpp)
    target_link_libraries(rvs-tests PRIVATE rvs_core)
    target_compile_options(rvs-tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME runtime-contracts COMMAND rvs-tests)
    set_tests_properties(runtime-contracts PROPERTIES TIMEOUT 15)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    if(RVS_BUILD_EXAMPLES)
        add_test(NAME cli-contracts
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_tests.py
            $<TARGET_FILE:rvs-run> ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(cli-contracts PROPERTIES TIMEOUT 20)
        add_test(NAME custom-workload COMMAND rvs-custom-demo)
        set_tests_properties(custom-workload PROPERTIES PASS_REGULAR_EXPRESSION "sum=6" TIMEOUT 5)
    endif()
    add_test(NAME tool-contracts
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/tool_tests.py ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(tool-contracts PROPERTIES TIMEOUT 10)
endif()

