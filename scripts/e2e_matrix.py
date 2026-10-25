#!/usr/bin/env python3
"""End-to-end API matrix for the StoikovKV dashboard.

Usage:
    python scripts/e2e_matrix.py                      # via frontend proxy (:3000)
    python scripts/e2e_matrix.py http://127.0.0.1:8000  # backend directly
"""
import json
import sys
import time
import urllib.request

ROOT = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:3000"
BASE = ROOT.rstrip("/") + "/api/v1"


def call(path, body=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"},
                                 method=method or ("POST" if body is not None else "GET"))
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status, r.read().decode()


def stream(path, body):
    t0 = time.time()
    status, payload = call(path, body)
    events = [json.loads(l[6:]) for l in payload.splitlines() if l.startswith("data: ")]
    errors = [e for e in events if e.get("type") == "error"]
    done = next((e for e in reversed(events) if e.get("type") == "done"), None)
    assert status == 200 and done and not errors, {"status": status, "errors": errors}
    toks = [e for e in events if e.get("type") == "token"]
    return {"tokens": len(toks), "secs": round(time.time() - t0, 2),
            "text": "".join(e["text"] for e in toks)[:60], "metrics": done["metrics"]}


def summarize(m):
    return {k: m.get(k) for k in ("seq_length", "active_atoms", "memory_ratio_vs_fp16",
                                  "memory_ratio_incl_dictionary", "hmm_shift_events")}


seg = ("The rescue report says Maya's friend fell in a ditch near the old bridge. "
       "A later unrelated CSS discussion mentions grid columns, blur filters, and theme colors. "
       "The factual anchor remains: the friend fell in a ditch, and the emergency contact is 555-0147. ")
PROMPT = "System long-context validation.\n" + seg * 95 + "\nQuestion: where did Maya's friend fall?"
results, failures = [], []


def run(name, fn):
    try:
        out = fn()
        results.append((name, "PASS", out))
        print(f"PASS  {name:42s} {json.dumps(out)[:160]}")
    except Exception as exc:  # noqa: BLE001
        failures.append(name)
        results.append((name, "FAIL", str(exc)))
        print(f"FAIL  {name:42s} {exc}")


run("GET /health", lambda: json.loads(call("/health")[1]))

for i, p in enumerate([{"temperature": 0.0, "top_k": 1, "top_p": 1.0},
                       {"temperature": 0.7, "top_k": 32, "top_p": 0.9},
                       {"temperature": 1.0, "top_k": 50, "top_p": 0.5},
                       {"temperature": 1.3, "top_k": 0, "top_p": 1.0}]):
    def std(p=p, i=i):
        r = stream("/generate", {"prompt": PROMPT, "session_id": f"std-{i}", "max_new_tokens": 16,
                                 "use_matryoshka": False, **p})
        assert r["tokens"] > 0
        m = json.loads(call(f"/metrics?session_id=std-{i}")[1])
        assert m["seq_length"] > 2000 and m["head_roles"]
        return {**p, "tokens": r["tokens"], "secs": r["secs"], **summarize(m)}
    run(f"generate standard {p}", std)

# Every option exposed by the UI dropdowns (draft 2/4/8, verify 8/12/16, gamma 2/4/8)
for d, v, g in [(2, 16, 4), (4, 16, 4), (8, 16, 4), (4, 8, 2), (4, 12, 8), (8, 8, 2), (2, 12, 8)]:
    def mat(d=d, v=v, g=g):
        sid = f"mat-{d}-{v}-{g}"
        r = stream("/generate", {"prompt": PROMPT, "session_id": sid, "max_new_tokens": 16,
                                 "temperature": 0.7, "top_k": 32, "top_p": 0.9, "use_matryoshka": True,
                                 "draft_atoms": d, "verify_atoms": v, "gamma_spec": g})
        assert r["tokens"] > 0
        return {"tokens": r["tokens"], "secs": r["secs"], **summarize(r["metrics"])}
    run(f"generate matryoshka d={d} v={v} g={g}", mat)


def multi_turn():
    sid = "multi"
    a = stream("/generate", {"prompt": PROMPT, "session_id": sid, "max_new_tokens": 8})
    b = stream("/generate", {"prompt": "\nAnd the emergency contact?", "session_id": sid, "max_new_tokens": 8})
    assert b["metrics"]["seq_length"] > a["metrics"]["seq_length"]
    return {"turn1_seq": a["metrics"]["seq_length"], "turn2_seq": b["metrics"]["seq_length"]}


run("generate multi-turn (same session)", multi_turn)


def system():
    s = json.loads(call("/system?session_id=multi")[1])
    assert s["sessions"] > 0 and s["kv_fp16_bytes_all_sessions"] > s["kv_compressed_bytes_all_sessions"] > 0
    assert s["session"]["seq_length"] > 2000
    g = s["gpu"]
    return {"gpu": g.get("name"), "vram_used_mb": round(g.get("used_bytes", 0) / 2**20),
            "kv_ratio_all": round(s["kv_compressed_bytes_all_sessions"] / s["kv_fp16_bytes_all_sessions"], 4)}


run("GET /system (VRAM gauge)", system)
run("chat (top_k/top_p)", lambda: summarize(stream("/chat", {
    "session_id": "chat", "max_new_tokens": 12, "temperature": 0.7, "top_k": 20, "top_p": 0.8,
    "messages": [{"role": "user", "content": PROMPT},
                 {"role": "assistant", "content": "Noted."},
                 {"role": "user", "content": "What happened to Maya's friend?"}]})["metrics"]))


def delete():
    stream("/generate", {"prompt": "test delete", "session_id": "test-del", "max_new_tokens": 4})
    out = json.loads(call("/session/test-del", method="DELETE")[1])
    assert out["deleted"] and json.loads(call("/metrics?session_id=test-del")[1]).get("seq_length", 0) == 0
    return out


run("DELETE /session/{id}", delete)
run("POST /calibrate", lambda: json.loads(call("/calibrate", body={})[1]))
run("generate after calibrate", lambda: summarize(stream("/generate", {
    "prompt": PROMPT, "session_id": "post-cal", "max_new_tokens": 8})["metrics"]))

print(f"\n{len(results) - len(failures)}/{len(results)} passed")
sys.exit(1 if failures else 0)
