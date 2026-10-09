"""2-client concurrent /tts/stream load test for Phase 3b-B-4 verification.

Fires N parallel POSTs to /tts/stream and measures TTFA (first PCM byte
past the 4-byte SR header) per client. Run with N=1 first (baseline) then
N=2 (concurrent) and compare. Acceptance: concurrent TTFA p50 <= 1.5x
single-client baseline → N=2 actually parallelizes; if blown out, the
worker / Code2Wav-mutex / engine pool is still serializing.
"""
import sys, time, requests
from concurrent.futures import ThreadPoolExecutor

HOST = sys.argv[1] if len(sys.argv) > 1 else "100.82.225.102:8621"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
URL = f"http://{HOST}/tts/stream"
TEXTS = [
    "我们都非常震惊。这位母亲表示。",
    "今天天气真不错，适合出门散步。",
    "人工智能正在改变我们的生活方式。",
    "请问您需要什么帮助吗？",
]

def one(text):
    t0 = time.perf_counter()
    r = requests.post(URL, json={"text": text}, stream=True, timeout=60)
    r.raise_for_status()
    header_seen = False
    first_audio_t = None
    for chunk in r.iter_content(chunk_size=4096):
        if not chunk: continue
        if not header_seen:
            if len(chunk) > 4:
                first_audio_t = time.perf_counter()
                header_seen = True
                break
            else:
                header_seen = True
        else:
            first_audio_t = time.perf_counter()
            break
    r.close()
    if first_audio_t is None:
        first_audio_t = time.perf_counter()
    return (first_audio_t - t0) * 1000

print(f"=== N={N} concurrent @ {URL} ===")
with ThreadPoolExecutor(max_workers=N) as ex:
    t0 = time.perf_counter()
    results = list(ex.map(one, TEXTS[:N]))
    wall = (time.perf_counter() - t0) * 1000
for i, ttfa in enumerate(results):
    print(f"  client {i+1}: ttfa={ttfa:.1f} ms")
print(f"  wall_clock={wall:.1f} ms (max-TTFA across clients)")
print(f"  ttfa_min={min(results):.1f}  ttfa_max={max(results):.1f}  ttfa_p50={sorted(results)[len(results)//2]:.1f}")
