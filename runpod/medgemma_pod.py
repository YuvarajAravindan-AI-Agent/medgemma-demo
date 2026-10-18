#!/usr/bin/env python3
"""Start/stop a short-lived RunPod GPU pod serving MedGemma, under a hard monthly budget.

Runs on the Contabo box (root). Stdlib only.

  medgemma_pod.py spend            month-to-date RunPod spend (all pods, incl. idle disk)
  medgemma_pod.py up [MINUTES]     start a pod if the budget allows; auto-terminates after MINUTES (default 15)
  medgemma_pod.py down             terminate every medgemma-* pod and clear state
  medgemma_pod.py status           pods on the account + current demo state

Budget guard (three layers, any one of them stops the spend):
  1. `up` refuses unless month spend + this session's worst case stays under BUDGET_INR.
  2. the pod terminates itself after MINUTES using its own pod-scoped RUNPOD_API_KEY.
  3. a systemd timer on this box runs `down` two minutes after the deadline.
The pod has no volume (volumeInGb=0), so nothing is billed once it is gone.
"""
import datetime as dt
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"
KEY_FILE = "/root/.config/runpod/api_key"
HF_FILE = "/root/.config/huggingface/token"
STATE = "/var/lib/medgemma-demo/pod.json"

BUDGET_INR = float(os.environ.get("MEDGEMMA_BUDGET_INR", "100"))
INR_PER_USD = float(os.environ.get("MEDGEMMA_INR_PER_USD", "89"))
MAX_USD_PER_HR = 0.50          # never accept a pod pricier than this
SAFETY_MINUTES = 5             # boot + teardown slack added to the worst case

MODEL = "google/medgemma-1.5-4b-it"
IMAGE = "vllm/vllm-openai:v0.10.1.1"
# 16 GB+ Ampere-or-newer cards (bf16), cheapest first.
UA = "medgemma-pod/1.0 (+https://ai-agentic-enterprises.com)"  # RunPod Cloudflare 403s (1010) the default urllib UA
GPUS = ["NVIDIA RTX A4000", "NVIDIA RTX A4500", "NVIDIA RTX 4000 Ada Generation",
        "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA L4"]


def _key(path):
    with open(path) as f:
        return f.read().strip()


def api(method, path, body=None, query=""):
    req = urllib.request.Request(
        f"{API}{path}{query}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {_key(KEY_FILE)}", "Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raise SystemExit(f"RunPod API {method} {path}: HTTP {e.code} {e.read().decode()[:400]}")


def _parse_ts(s):
    """RunPod timestamps come as ISO 8601 or as '2026-10-07 05:12:03.123 +0000 UTC'."""
    s = s.replace(" UTC", "").strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z"):
        try:
            return dt.datetime.strptime(s.replace("Z", "+0000"), fmt)
        except ValueError:
            pass
    return dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)  # unknown format: assume a full hour


def month_spend_usd():
    """Billed this month across all pods, plus running pods' cost not yet billed."""
    now = dt.datetime.now(dt.timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    q = f"?bucketSize=month&grouping=podId&startTime={start:%Y-%m-%dT%H:%M:%SZ}&endTime={now:%Y-%m-%dT%H:%M:%SZ}"
    billed = sum(float(r.get("amount") or 0) for r in (api("GET", "/billing/pods", query=q) or []))
    unbilled = 0.0
    for p in api("GET", "/pods") or []:
        if p.get("desiredStatus") == "RUNNING" and p.get("lastStartedAt"):
            started = _parse_ts(p["lastStartedAt"])
            # billing lags by up to an hour; count the last hour of each running pod again to stay conservative
            unbilled += float(p.get("costPerHr") or 0) * min(1.0, (now - started).total_seconds() / 3600)
    return billed, unbilled


def cmd_spend():
    billed, unbilled = month_spend_usd()
    total = billed + unbilled
    print(f"month-to-date: ${billed:.3f} billed + ${unbilled:.3f} running (≈ ₹{total * INR_PER_USD:.0f})"
          f" of ₹{BUDGET_INR:.0f} budget")
    return total


def _write_state(d):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, STATE)


def _read_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def cmd_up(minutes=15):
    minutes = max(5, min(int(minutes), 30))
    if (s := _read_state()) and s.get("status") in ("booting", "ready"):
        raise SystemExit(f"a demo pod is already up: {s['pod_id']} (run `down` first)")
    spent = cmd_spend()
    budget_usd = BUDGET_INR / INR_PER_USD
    worst = MAX_USD_PER_HR * (minutes + SAFETY_MINUTES) / 60
    if spent + worst > budget_usd:
        raise SystemExit(f"REFUSED: ${spent:.3f} spent + ${worst:.3f} worst case > ${budget_usd:.2f} (₹{BUDGET_INR:.0f})")

    vllm_key = secrets.token_urlsafe(24)
    max_seconds = minutes * 60
    killer = ("import os,time,urllib.request as u\n"
              f"time.sleep({max_seconds})\n"
              "pid=os.environ['RUNPOD_POD_ID'];h={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'],'User-Agent':'medgemma-pod-killer/1.0'}\n"
              "for m,p in (('DELETE',''),('POST','/stop')):\n"
              "  try: u.urlopen(u.Request('https://rest.runpod.io/v1/pods/'+pid+p,method=m,headers=h),timeout=30); break\n"
              "  except Exception as e: print('self-terminate',m,e,flush=True)\n")
    start = (f"python3 -c \"$KILLER\" & "
             f"exec vllm serve {MODEL} --host 0.0.0.0 --port 8000 --api-key \"$VLLM_API_KEY\" "
             "--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 "
             "--limit-mm-per-prompt '{\"image\":1}' --served-model-name medgemma")
    body = {
        "name": f"medgemma-{dt.datetime.now():%m%d-%H%M}",
        "imageName": IMAGE,
        "cloudType": "COMMUNITY",
        "gpuTypeIds": GPUS, "gpuTypePriority": "custom", "gpuCount": 1,
        "allowedCudaVersions": ["12.8", "12.9", "13.0"],
        "containerDiskInGb": 40, "volumeInGb": 0,
        "ports": ["8000/http"],
        "env": {"HF_TOKEN": _key(HF_FILE), "VLLM_API_KEY": vllm_key, "KILLER": killer},
        "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [start],
    }
    pod = api("POST", "/pods", body)
    pid, rate = pod["id"], float(pod.get("costPerHr") or pod.get("adjustedCostPerHr") or 0)
    gpu = (pod.get("machine") or {}).get("gpuTypeId") or (pod.get("gpu") or {}).get("id") or "?"
    if rate > MAX_USD_PER_HR:
        api("DELETE", f"/pods/{pid}")
        raise SystemExit(f"REFUSED: got {gpu} at ${rate}/hr > ${MAX_USD_PER_HR}; pod terminated")
    deadline = time.time() + max_seconds
    _write_state({"pod_id": pid, "gpu": gpu, "cost_per_hr": rate, "status": "booting",
                  "url": f"https://{pid}-8000.proxy.runpod.net", "api_key": vllm_key,
                  "started": time.time(), "deadline": deadline})
    # layer 3: this box tears it down even if the pod's self-terminate fails
    subprocess.run(["systemd-run", "--unit", f"medgemma-down-{pid}", f"--on-active={max_seconds + 120}",
                    sys.executable, os.path.abspath(__file__), "down"], check=False, capture_output=True)
    print(f"pod {pid} on {gpu} at ${rate}/hr (₹{rate * INR_PER_USD:.1f}/hr); auto-off in {minutes} min")

    t0 = time.time()
    last = 0
    while time.time() < deadline:
        code = "-"
        try:
            req = urllib.request.Request(f"https://{pid}-8000.proxy.runpod.net/v1/models",
                                         headers={"Authorization": f"Bearer {vllm_key}", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=15) as r:
                if r.status == 200:
                    s = _read_state(); s["status"] = "ready"; s["ready_after_s"] = round(time.time() - t0); _write_state(s)
                    print(f"READY after {time.time() - t0:.0f}s: {s['url']}", flush=True)
                    return
        except urllib.error.HTTPError as e:
            code = e.code
        except Exception as e:
            code = type(e).__name__
        if time.time() - last >= 30:
            last = time.time()
            print(f"[{time.time() - t0:4.0f}s] proxy={code} {_runtime(pid)}", flush=True)
        time.sleep(10)
    print(f"not ready within the {minutes}-min session; tearing down", flush=True)
    cmd_down()


def _runtime(pid):
    """Boot phase from RunPod GraphQL: no runtime yet = pulling image; GPU memory climbing = loading model."""
    q = ('query { pod(input: {podId: "%s"}) { desiredStatus runtime { uptimeInSeconds '
         'container { cpuPercent memoryPercent } gpus { gpuUtilPercent memoryUtilPercent } } } }' % pid)
    try:
        req = urllib.request.Request("https://api.runpod.io/graphql", method="POST",
                                     data=json.dumps({"query": q}).encode(),
                                     headers={"Authorization": f"Bearer {_key(KEY_FILE)}",
                                              "Content-Type": "application/json", "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            pod = (json.loads(r.read()).get("data") or {}).get("pod") or {}
    except Exception as e:
        return f"runtime=? ({type(e).__name__})"
    rt = pod.get("runtime")
    if not rt:
        return f"{pod.get('desiredStatus')} no runtime yet (pulling image / waiting for host)"
    c, g = rt.get("container") or {}, (rt.get("gpus") or [{}])[0]
    return (f"{pod.get('desiredStatus')} up={rt.get('uptimeInSeconds')}s cpu={c.get('cpuPercent')}% "
            f"mem={c.get('memoryPercent')}% gpu={g.get('gpuUtilPercent')}% vram={g.get('memoryUtilPercent')}%")


def cmd_down():
    gone = []
    for p in api("GET", "/pods") or []:
        if p.get("name", "").startswith("medgemma-"):
            api("DELETE", f"/pods/{p['id']}")
            gone.append(p["id"])
    if (s := _read_state()):
        s.update(status="off", stopped=time.time(), api_key=None)
        _write_state(s)
    print("terminated:", ", ".join(gone) or "nothing running")


def cmd_status():
    for p in api("GET", "/pods") or []:
        print(f"{p['id']}  {p.get('name')!r:28} {p.get('desiredStatus'):10} ${p.get('costPerHr')}/hr "
              f"volume={p.get('volumeInGb')}GB disk={p.get('containerDiskInGb')}GB")
    s = _read_state()
    print("demo state:", {k: v for k, v in (s or {}).items() if k != "api_key"})
    cmd_spend()


if __name__ == "__main__":
    a = sys.argv[1:] or ["status"]
    {"spend": lambda: cmd_spend(), "up": lambda: cmd_up(*a[1:2]), "down": cmd_down,
     "status": cmd_status}[a[0]]()
