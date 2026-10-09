#!/usr/bin/env python3
"""Build a dialogue-style RKLLM quantization dataset for Qwen3 ASR decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GENERIC_PROMPTS = (
    "Human: Please transcribe the audio.\nAssistant: ",
    "Human: Provide the speech-to-text result.\nAssistant: ",
    "Human: Convert speech to text.\nAssistant: ",
    "Human: Return only the transcript without explanation.\nAssistant: ",
)


ZH_PROMPTS = (
    "Human: Transcribe in Chinese.\nAssistant: ",
    "Human: 请将语音转写成中文，只返回转写文本。\nAssistant: ",
    "Human: Return the Chinese transcript only, without quotes or explanation.\nAssistant: ",
)


EN_PROMPTS = (
    "Human: Transcribe in English.\nAssistant: ",
    "Human: Return the English transcript only, without quotes or explanation.\nAssistant: ",
    "Human: Convert this English speech to text.\nAssistant: ",
)


ZH_EXTRA_TARGETS = (
    "好的，已经为您安排好了。",
    "你好，今天天气真不错。",
    "请问还有其他需要帮助的吗？",
    "明天上午十点的会议已经提醒您。",
    "音量已经调到合适的大小。",
    "美国海军还表示，他们正在调查这起事件。",
    "传统上，王位继承人在完成学业后会直接入伍。",
    "请打开客厅的灯，然后把音量调低一点。",
    "不用解释，直接告诉我识别结果。",
    "我刚才说的是明天下午三点，不是今天下午三点。",
)


EN_EXTRA_TARGETS = (
    "Hello, how can I help you today?",
    "Your order has been confirmed.",
    "The timer is set for ten minutes.",
    "Lights in the living room are now off.",
    "Please say that again, I did not catch it.",
    "Television reports show white smoke coming from the plant.",
    "He referred to the rumors as unfounded.",
    "Fellow wrestlers also paid tribute to his career.",
    "Please turn off the kitchen lights and set a reminder.",
    "Return only the transcript without any extra words.",
)


def _load_targets(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    targets: dict[str, list[str]] = {"zh": [], "en": []}
    for entry in data.get("files", []):
        lang = entry.get("lang")
        if lang not in targets:
            continue
        text = (entry.get("eval_transcript") or entry.get("transcript") or "").strip()
        if text:
            targets[lang].append(text)
    return targets


def _take_balanced(targets: dict[str, list[str]]) -> list[tuple[str, str]]:
    zh = targets["zh"]
    en = targets["en"]
    limit = min(len(zh), len(en))
    rows: list[tuple[str, str]] = []
    for index in range(limit):
        rows.append(("zh", zh[index]))
        rows.append(("en", en[index]))
    return rows


def _mode_targets(mode: str, manifest_targets: dict[str, list[str]]) -> list[tuple[str, str]]:
    zh = [("zh", text) for text in manifest_targets["zh"] + list(ZH_EXTRA_TARGETS)]
    en = [("en", text) for text in manifest_targets["en"] + list(EN_EXTRA_TARGETS)]
    if mode == "zh":
        return zh
    if mode == "en":
        return en
    if mode == "balanced":
        return _take_balanced(
            {
                "zh": manifest_targets["zh"] + list(ZH_EXTRA_TARGETS),
                "en": manifest_targets["en"] + list(EN_EXTRA_TARGETS),
            }
        )
    return zh + en


def _prompts_for_lang(lang: str) -> tuple[str, ...]:
    if lang == "zh":
        return GENERIC_PROMPTS + ZH_PROMPTS
    if lang == "en":
        return GENERIC_PROMPTS + EN_PROMPTS
    return GENERIC_PROMPTS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=("mixed", "zh", "en", "balanced"),
        default="mixed",
        help="Calibration distribution to generate.",
    )
    args = parser.parse_args()

    targets = _mode_targets(args.mode, _load_targets(args.manifest))

    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, str]] = []
    counts = {"zh": 0, "en": 0}
    for target_index, (lang, target) in enumerate(targets):
        prompts = _prompts_for_lang(lang)
        for offset in range(args.repeat):
            prompt = prompts[(target_index + offset) % len(prompts)]
            key = (prompt, target)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"input": prompt, "target": target})
            if lang in counts:
                counts[lang] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {args.output} ({counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
