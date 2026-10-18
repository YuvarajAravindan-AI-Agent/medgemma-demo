#!/usr/bin/env python3
"""MedGemma demo web app (Contabo). Stdlib only.

Serves the page, sample images, live infra status, and proxies one image +
prompt to the RunPod vLLM endpoint described in /var/lib/medgemma-demo/pod.json
(written by medgemma_pod.py). The GPU key never reaches the browser.
"""
import base64
import json
import mimetypes
import os
import socket
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = "/var/lib/medgemma-demo/pod.json"
PORT = int(os.environ.get("PORT", "8310"))
MAX_UPLOAD = 4 * 1024 * 1024

CASES = {
    "cxr_normal": ("Chest X-ray", "Normal PA chest X-ray (CC0)", "You are an expert radiologist.",
                   "Describe this chest X-ray. Give findings and an impression, written as a draft report for a radiologist to review."),
    "cxr_pneumonia": ("Chest X-ray", "Lobar pneumonia (CC0)", "You are an expert radiologist.",
                      "Describe this chest X-ray. Give findings and an impression, written as a draft report for a radiologist to review."),
    "fundus_normal": ("Retina", "Normal left fundus (CC0)", "You are an expert ophthalmologist.",
                      "Describe this retinal fundus photograph. Note any signs of diabetic retinopathy or other abnormality, and state whether referral is suggested."),
    "fundus_retinopathy": ("Retina", "Diabetic retinopathy, NEI (public domain)", "You are an expert ophthalmologist.",
                           "Describe this retinal fundus photograph. Note any signs of diabetic retinopathy or other abnormality, and state whether referral is suggested."),
    "derm_melanoma": ("Dermatology", "Melanoma, NCI (public domain)", "You are an expert dermatologist.",
                      "Describe this skin lesion using the ABCDE criteria and state whether dermatology referral is suggested."),
}


def pod_state():
    try:
        with open(STATE) as f:
            s = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"status": "off"}
    if s.get("status") in ("booting", "ready") and time.time() > s.get("deadline", 0):
        s["status"] = "off"
    return s


def public_status():
    s = pod_state()
    out = {"status": s.get("status", "off"), "host": socket.gethostname(), "model": "google/medgemma-1.5-4b-it"}
    if out["status"] in ("booting", "ready"):
        out.update(pod_id=s["pod_id"], gpu=s.get("gpu"), cost_per_hr=s.get("cost_per_hr"),
                   seconds_left=max(0, int(s["deadline"] - time.time())),
                   up_for=int(time.time() - s["started"]), ready_after_s=s.get("ready_after_s"))
    return out


class H(BaseHTTPRequestHandler):
    server_version = "medgemma-demo"

    def log_message(self, fmt, *a):
        print(f"{self.address_string()} {fmt % a}", flush=True)

    def _send(self, code, body, ctype="application/json", cache="no-store"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8", "no-cache")
        if p == "/api/status":
            return self._send(200, public_status())
        if p == "/api/cases":
            return self._send(200, [{"id": k, "domain": v[0], "label": v[1], "prompt": v[3]} for k, v in CASES.items()])
        if p.startswith("/samples/"):
            name = os.path.basename(p)
            path = os.path.join(HERE, "samples", name)
            if name.endswith(".jpg") and os.path.isfile(path):
                with open(path, "rb") as f:
                    return self._send(200, f.read(), "image/jpeg", "public, max-age=86400")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/analyze":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_UPLOAD * 2:
            return self._send(413, {"error": "request too large"})
        req = json.loads(self.rfile.read(n))
        s = pod_state()
        if s.get("status") != "ready":
            return self._send(503, {"error": "The GPU is switched off right now. It runs only during live demos."})

        case = CASES.get(req.get("case_id"))
        if case:
            with open(os.path.join(HERE, "samples", f"{req['case_id']}.jpg"), "rb") as f:
                img, mime = f.read(), "image/jpeg"
            role, prompt = case[2], case[3]
        else:
            data = req.get("image_b64") or ""
            img = base64.b64decode(data.split(",")[-1]) if data else b""
            if not img or len(img) > MAX_UPLOAD:
                return self._send(400, {"error": "choose a sample or upload an image under 4 MB"})
            mime = mimetypes.guess_type(req.get("filename") or "x.png")[0] or "image/png"
            role = "You are an expert medical imaging assistant."
            prompt = (req.get("prompt") or "Describe this medical image.")[:1000]

        body = {"model": "medgemma", "max_tokens": 500, "temperature": 0,
                "messages": [{"role": "system", "content": role},
                             {"role": "user", "content": [
                                 {"type": "text", "text": prompt},
                                 {"type": "image_url", "image_url": {
                                     "url": f"data:{mime};base64,{base64.b64encode(img).decode()}"}}]}]}
        t0 = time.time()
        try:
            r = urllib.request.Request(f"{s['url']}/v1/chat/completions", data=json.dumps(body).encode(),
                                       headers={"Authorization": f"Bearer {s['api_key']}",
                                                "Content-Type": "application/json",
                                                "User-Agent": "medgemma-demo/1.0"})
            with urllib.request.urlopen(r, timeout=180) as resp:
                out = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as e:
            return self._send(502, {"error": f"GPU endpoint error: {e}"})
        u = out.get("usage") or {}
        self._send(200, {"output": out["choices"][0]["message"]["content"], "seconds": round(time.time() - t0, 1),
                         "prompt": prompt, "gpu": s.get("gpu"), "pod_id": s.get("pod_id"),
                         "tokens_in": u.get("prompt_tokens"), "tokens_out": u.get("completion_tokens")})


if __name__ == "__main__":
    print(f"medgemma-demo on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
