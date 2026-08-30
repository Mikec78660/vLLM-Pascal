import json, time, urllib.request

PROMPT = ("Explain the concept of entropy in thermodynamics in clear detail. "
          "Cover the statistical and macroscopic views. Then list 5 everyday "
          "examples where entropy clearly increases.")
MAXTOK = 250

def hit(base, model, label, timeout=900):
    payload = {"model":model,"prompt":PROMPT,"max_tokens":MAXTOK,"temperature":0.0}
    req = urllib.request.Request(base, data=json.dumps(payload).encode(),
            headers={"Content-Type":"application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    dt = time.time() - t0
    ch = data.get("choices",[])
    n = 0
    if ch:
        n = ch[0].get("token_count") or 0
    if not n:
        n = data.get("usage",{}).get("completion_tokens",0)
    print("  %-14s tokens=%-4d wall=%6.2fs => %5.2f tok/s  finish=%s" % (
        label, n, dt, (n/dt if dt>0 else 0), ch[0].get("finish_reason") if ch else "?"))
    return n/dt if dt>0 else 0

print("=== A/B: SAME prompt, max_tokens=%d, temp=0 (sequential) ===" % MAXTOK)
print("  [warmup] vLLM 25tok (JIT/graph warm)")
hit("http://localhost:8901/v1/completions", "qwen3.8-27b-awq-int4-mtp", "vllm-warmup", timeout=300)
print("  [1] vLLM  AWQ-INT4+MTP  2xP40 TP=2  mode=0 + FULL_DECODE_ONLY graphs  :8901")
v = hit("http://localhost:8901/v1/completions", "qwen3.8-27b-awq-int4-mtp", "vllm+graphs")
print("  [2] llama Q4_0+MTP  (router :8080)")
l = hit("http://localhost:8080/v1/completions", "Qwen3.8-27B-Q4_0-mtp1", "llama-ref")
print()
print("  ratio llama/vllm = %.2fx   (llama %.2f / vllm %.2f t/s)" % (
    (l/v) if v>0 else 0, l, v))
