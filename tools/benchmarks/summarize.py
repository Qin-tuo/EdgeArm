#!/usr/bin/env python3
"""Summarize stage durations without mixing backends, models or simulation modes."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(files):
    groups = defaultdict(list)
    task_ids = set()
    for file in files:
        with Path(file).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("schema_version") != 1:
                    raise ValueError(f"{file}:{line_number}: unsupported event schema")
                if event["phase"] not in {"end", "first_output"}:
                    continue
                if type(event["simulated"]) is not bool:
                    raise ValueError(f"{file}:{line_number}: simulated must be boolean")
                duration = event["elapsed_ms"]
                if type(duration) not in {int, float} or not math.isfinite(duration) or duration < 0:
                    raise ValueError(f"{file}:{line_number}: invalid duration")
                if event["stage"] == "task" and event["phase"] == "end":
                    identity = (event["run_id"], event["task_id"])
                    if identity in task_ids:
                        raise ValueError(f"duplicate terminal task event: {identity}")
                    task_ids.add(identity)
                key = tuple(event[k] for k in ("stage", "backend", "model", "simulated", "phase"))
                groups[key].append(event)
    rows = []
    for key, events in sorted(groups.items()):
        statuses = Counter(event["status"] for event in events)
        successful = [event["elapsed_ms"] for event in events if event["status"] == "completed"]
        row = dict(zip(("stage", "backend", "model", "simulated", "phase"), key))
        row.update({
            "samples": len(events), "successful_samples": len(successful),
            "status_counts": dict(sorted(statuses.items())),
            "non_success_rate": 1 - len(successful) / len(events),
            "p50_ms": percentile(successful, 0.50), "p95_ms": percentile(successful, 0.95),
        })
        rows.append(row)
    return {
        "schema_version": 1,
        "percentile_method": "linear interpolation over successful samples",
        "sources": [str(file) for file in files],
        "note": "Simulated durations are framework checks, not model or hardware performance.",
        "groups": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        data = summarize(args.metrics)
        text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

