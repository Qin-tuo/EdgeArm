#!/usr/bin/env python3
"""Check selected model files and run inference through local EdgeArm services."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def verify_files(data_root, llm_manifest):
    manifest = json.loads((ROOT / 'deploy/artifacts/rk_manifest.json').read_text())
    selected = manifest['artifact_sets'][manifest['default_set']]
    entries = []
    for item in selected['files']:
        parts = Path(item['path']).parts
        if parts[:3] not in [('opt', 'asr', 'models'), ('opt', 'tts', 'models')]:
            raise ValueError(f'Unexpected model destination: {item["path"]}')
        entries.append((data_root / parts[1] / Path(*parts[3:]), item['size_bytes'], item['sha256']))
    frontend = json.loads((ROOT / 'server/core/matcha_artifacts.json').read_text())
    for name, item in frontend['model_bundle']['required_files'].items():
        entries.append((data_root / 'tts/matcha-icefall-zh-en' / name, item['size'], item['sha256']))
    errors = []
    for path, size, expected in entries:
        try:
            if path.stat().st_size != size:
                raise ValueError('size mismatch')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError('SHA256 mismatch')
            print(f'PASS file: {path}', flush=True)
        except (OSError, ValueError) as error:
            errors.append(f'{path}: {error}')
    # Reuse the service's matched four-file RK1828 export validation.
    spec = importlib.util.spec_from_file_location('edgearm_verify_artifacts', ROOT / 'services/rk1828-llm/verify_artifacts.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    try:
        verifier.verify(llm_manifest, data_root / 'llm')
        print(f'PASS Qwen3-4B export: {llm_manifest}', flush=True)
    except (OSError, ValueError, KeyError) as error:
        errors.append(f'LLM export: {error}')
    if errors:
        raise RuntimeError('\n'.join(errors))
    return {'status': 'passed', 'verified_files': len(entries) + 4, 'llm_manifest': str(llm_manifest), 'inference_tested': False}


def normalize(text):
    return ''.join(c for c in text.casefold() if c.isalnum())


def character_error_rate(expected, actual):
    expected, actual = normalize(expected), normalize(actual)
    if not expected:
        raise ValueError('Expected text must contain letters or digits')
    row = list(range(len(actual) + 1))
    for i, a in enumerate(expected, 1):
        next_row = [i]
        for j, b in enumerate(actual, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = next_row
    return row[-1] / len(expected)


def test_services(args, output, report):
    import httpx
    import numpy as np
    import soundfile as sf

    with httpx.Client(timeout=args.timeout, trust_env=False) as client:
        for name, base in [('speech', args.speech_url), ('llm', args.llm_url)]:
            response = client.get(base.rstrip('/') + '/health')
            response.raise_for_status()
            health = response.json()
            report[f'{name}_health'] = health
            if name == 'speech' and not (health.get('asr') is True and health.get('tts') is True):
                raise RuntimeError(f'ASR/TTS not ready: {health}')
            if name == 'llm' and health.get('status') != 'ok':
                raise RuntimeError(f'LLM worker not ready: {health}')
            print(f'PASS {name} health: {json.dumps(health, ensure_ascii=False)}', flush=True)

        start = time.monotonic()
        response = client.post(args.llm_url.rstrip('/') + '/v1/chat/completions', json={
            'model': 'Qwen3-4B', 'messages': [{'role': 'user', 'content': '请用一句简短的中文介绍你自己。'}],
            'max_tokens': 128, 'temperature': 0, 'stream': False,
        })
        response.raise_for_status()
        result = response.json()
        (output / 'llm-response.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        answer = result['choices'][0]['message']['content']
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError('LLM returned no text')
        report['llm'] = {'status': 'passed', 'text': answer, 'request_s': round(time.monotonic() - start, 3)}
        print(f'PASS LLM reply: {answer}', flush=True)

        start = time.monotonic()
        response = client.post(args.speech_url.rstrip('/') + '/tts', json={'text': args.text, 'language': 'zh'})
        response.raise_for_status()
        wav_bytes = response.content
        samples, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype='float32')
        if samples.size == 0 or not np.isfinite(samples).all():
            raise RuntimeError('TTS produced empty or non-finite audio')
        rms = float(np.sqrt(np.mean(samples.astype('float64') ** 2)))
        duration = len(samples) / sample_rate
        if duration < 0.1 or rms < 1e-4:
            raise RuntimeError(f'TTS audio too short or silent: {duration:.3f}s, RMS={rms}')
        wav_path = output / 'tts.wav'
        wav_path.write_bytes(wav_bytes)
        report['tts'] = {'status': 'passed', 'wav': str(wav_path), 'sample_rate': sample_rate,
                         'duration_s': round(duration, 3), 'rms': rms, 'request_s': round(time.monotonic() - start, 3)}
        print(f'PASS TTS: {wav_path} ({duration:.2f}s, {sample_rate} Hz)', flush=True)

        start = time.monotonic()
        response = client.post(args.speech_url.rstrip('/') + '/asr', params={'language': 'zh'},
                               files={'file': ('tts.wav', wav_bytes, 'audio/wav')})
        response.raise_for_status()
        result = response.json()
        (output / 'asr-response.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        actual = result.get('text', '')
        cer = character_error_rate(args.text, actual)
        report['tts_asr_roundtrip'] = {'status': 'passed' if cer <= args.max_cer else 'failed',
                                     'expected': args.text, 'actual': actual, 'cer': cer,
                                     'max_cer': args.max_cer, 'request_s': round(time.monotonic() - start, 3)}
        print(f'ASR: {actual} (CER={cer:.3f}, limit={args.max_cer:.3f})', flush=True)
        if cer > args.max_cer:
            raise RuntimeError('TTS-to-ASR roundtrip exceeded the character error limit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true', help='check model sizes and hashes without loading devices')
    mode.add_argument('--services-only', action='store_true', help='run HTTP inference without repeating file checks')
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get('EDGEARM_DATA_ROOT', ROOT / '.local/data')))
    parser.add_argument('--llm-manifest', type=Path, help='override the matched RK1828 model manifest')
    parser.add_argument('--speech-url', default='http://127.0.0.1:8621')
    parser.add_argument('--llm-url', default='http://127.0.0.1:1828')
    parser.add_argument('--text', default='你好，今天天气真不错。')
    parser.add_argument('--max-cer', type=float, default=0.3, help='roundtrip smoke threshold, not a corpus quality benchmark')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if args.timeout <= 0 or not 0 <= args.max_cer <= 1 or not normalize(args.text):
        parser.error('timeout must be positive, max-cer must be 0..1, and text must be nonempty')
    local_manifest = args.data_root / 'llm/artifacts.local.json'
    llm_manifest = args.llm_manifest or (local_manifest if local_manifest.exists() else ROOT / 'services/rk1828-llm/artifacts.json')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / '.local/reports/model-smoke' / stamp
    output.mkdir(parents=True)
    report = {'started_at': stamp, 'mode': 'verify' if args.verify_only else 'services' if args.services_only else 'all',
              'speech_url': args.speech_url, 'llm_url': args.llm_url}
    try:
        if not args.services_only:
            report['artifacts'] = verify_files(args.data_root, llm_manifest)
        if not args.verify_only:
            test_services(args, output, report)
        report['status'] = 'passed'
        print('PASS model file verification.' if args.verify_only else 'PASS model inference smoke; listen to tts.wav and review the LLM reply.', flush=True)
        return 0
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        print(f'FAIL: {error}', file=sys.stderr, flush=True)
        return 1
    finally:
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(f'Report: {output / "report.json"}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
