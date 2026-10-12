#!/usr/bin/env python3
"""Replay ordinary dialogue for ONE manufacturer log case; stop on failure."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:1828'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def now():
    return datetime.now(timezone.utc).isoformat()

def proxy():
    return subprocess.run(['pgrep', '-af', '^rknn3_transfer_proxy_'],
        capture_output=True, text=True).stdout.splitlines()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_directory', type=Path)
    parser.add_argument('--ready-timeout', type=float, default=210)
    parser.add_argument('--request-timeout', type=float, default=60)
    args = parser.parse_args()
    case = args.case_directory.resolve()
    case.relative_to(ROOT / '.local/reports/inference-check/vendor-logs')
    metadata = json.loads((case / 'case.json').read_text())
    output = case / 'requests.json'
    if output.exists():
        raise SystemExit('Case already tested; use a fresh case directory.')
    report = {'status': 'running', 'started_utc': now(), 'case': metadata,
              'cases': [], 'tools': False, 'auto_reset': False}

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    def health():
        with OPENER.open(BASE + '/health', timeout=3) as response:
            return response.status, json.load(response)

    try:
        save()
        deadline = time.monotonic() + args.ready_timeout
        while time.monotonic() < deadline:
            if not proxy():
                raise RuntimeError('Device proxy absent during initialization')
            if (case / 'host.exit-code').exists():
                raise RuntimeError('Application exited during initialization: ' +
                                   (case / 'host.exit-code').read_text().strip())
            try:
                status, body = health()
                if status == 200 and body.get('status') == 'ok':
                    report['initial_health'] = body
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError('Application did not become ready')
        messages = [{'role': 'system', 'content': '你是本地语音助手，回答简短中文。'}]
        prompts = ['你好，请用一句简短中文打招呼。', '你好', '你是谁', '什么模型',
                   '请用一句话说明语音助手如何工作。', '我叫小林，请记住我的名字。',
                   '我的名字是什么？', '请解释本地运行模型的好处。',
                   '请用一句话解释麦克风的作用。', '请用一句话解释扬声器的作用。',
                   '请解释语音识别和语音合成的区别。', '什么模型',
                   '请介绍你可以帮助我做什么。', '我的名字是什么？',
                   '请用一句简短中文问候我。', '请确认你收到了本轮消息。']
        for index, text in enumerate(prompts, 1):
            messages.append({'role': 'user', 'content': text})
            body = {'model': 'Qwen3-4B', 'messages': list(messages),
                    'max_tokens': 512, 'temperature': 1, 'stream': False}
            row = {'turn': index, 'started_utc': now(), 'request': body,
                   'proxy_before': proxy()}
            report['cases'].append(row)
            save()
            if not row['proxy_before']:
                raise RuntimeError('Device proxy absent; no new inference sent')
            request = urllib.request.Request(BASE + '/v1/chat/completions',
                data=json.dumps(body, ensure_ascii=False).encode(),
                headers={'Content-Type': 'application/json'})
            start = time.monotonic()
            try:
                with OPENER.open(request, timeout=args.request_timeout) as response:
                    row['http_status'] = response.status
                    row['response'] = json.load(response)
            except urllib.error.HTTPError as error:
                row['http_status'] = error.code
                row['error_body'] = error.read().decode(errors='replace')
                raise
            finally:
                row['request_s'] = round(time.monotonic() - start, 3)
                row['ended_utc'] = now()
                row['proxy_after'] = proxy()
                save()
            if not row['proxy_after']:
                raise RuntimeError('Device proxy disappeared after request')
            if row['response'].get('error'):
                raise RuntimeError(str(row['response']['error']))
            answer = row['response']['choices'][0]['message'].get('content') or ''
            if not answer.strip():
                raise RuntimeError('Model returned empty content')
            row['text'] = answer
            row['transport_passed'] = True
            messages.append({'role': 'assistant', 'content': answer})
            # Wait briefly for the stderr reader/tee to flush this request's
            # statistics. A previous request's count must not guide the bound.
            metric_deadline = time.monotonic() + 2
            while time.monotonic() < metric_deadline:
                log = (case / 'rk3588-host.log').read_text(errors='replace')
                matches = list(re.finditer(r'req#(\d+) done prefill_tokens=(\d+).*?decode_tokens=(\d+)', log))
                if matches and int(matches[-1][1]) == index - 1:
                    match = matches[-1]
                    break
                time.sleep(.05)
            else:
                raise RuntimeError('Could not associate worker metrics; stopped before growing history further')
            if match:
                row['worker_metrics'] = {'request_id': int(match[1]),
                                        'prefill_tokens': int(match[2]),
                                        'decode_tokens': int(match[3])}
            print(json.dumps({'turn': index, 'text': text, 'answer': answer,
                              'worker_metrics': row.get('worker_metrics'),
                              'proxy_after': row['proxy_after']}, ensure_ascii=False), flush=True)
            save()
            # Bound growing history independently of placeholder API usage.
            if row.get('worker_metrics', {}).get('prefill_tokens', 0) >= 1000:
                report['stop_reason'] = 'Reached 1000-token history sample; no context-limit test intended'
                break
        report['status'] = 'sample_passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        print(f'STOP: {error}', flush=True)
    finally:
        report['ended_utc'] = now()
        report['final_proxy'] = proxy()
        save()
        print(f'Report: {output}', flush=True)
    return 0 if report['status'] == 'sample_passed' else 1

if __name__ == '__main__':
    raise SystemExit(main())
