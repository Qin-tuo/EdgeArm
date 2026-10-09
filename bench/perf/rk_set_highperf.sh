#!/bin/sh
# Lock RK3576/RK3588 CPU/NPU/DDR clocks for repeatable RKLLM benchmarks.
# This is a benchmark/perf-mode helper, not a default production policy.

set -u

write_if() {
    path="$1"
    value="$2"
    if [ -e "$path" ]; then
        echo "$value" > "$path" 2>/dev/null || true
    fi
}

set_cpu_policy() {
    policy="$1"
    freq="$2"
    write_if "/sys/devices/system/cpu/cpufreq/${policy}/scaling_governor" userspace
    write_if "/sys/devices/system/cpu/cpufreq/${policy}/scaling_setspeed" "$freq"
}

for idle in /sys/devices/system/cpu/cpu*/cpuidle/state1/disable; do
    [ -e "$idle" ] && write_if "$idle" 1
done

if [ -d /sys/class/devfreq/fdab0000.npu ]; then
    # RK3588 / ROCK 5T observed max clocks.
    write_if /sys/class/devfreq/fdab0000.npu/governor userspace
    write_if /sys/class/devfreq/fdab0000.npu/userspace/set_freq 1000000000
    write_if /sys/class/devfreq/dmc/governor userspace
    write_if /sys/class/devfreq/dmc/userspace/set_freq 2400000000
    set_cpu_policy policy0 1800000
    set_cpu_policy policy4 2352000
    set_cpu_policy policy6 2352000
elif [ -d /sys/class/devfreq/27700000.npu ]; then
    # RK3576 / LubanCat-3 observed max clocks.
    write_if /sys/class/devfreq/27700000.npu/governor userspace
    write_if /sys/class/devfreq/27700000.npu/userspace/set_freq 950000000
    write_if /sys/class/devfreq/dmc/governor userspace
    write_if /sys/class/devfreq/dmc/userspace/set_freq 1848000000
    set_cpu_policy policy0 2016000
    set_cpu_policy policy4 2208000
else
    echo "unsupported RK devfreq layout" >&2
    exit 2
fi

grep -H . \
    /sys/class/devfreq/*npu*/governor \
    /sys/class/devfreq/*npu*/cur_freq \
    /sys/class/devfreq/dmc/governor \
    /sys/class/devfreq/dmc/cur_freq \
    /sys/devices/system/cpu/cpufreq/policy*/scaling_governor \
    /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq 2>/dev/null || true
