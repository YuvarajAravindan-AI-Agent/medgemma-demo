"""Flagship MedGemma demo video: real model outputs from the Colab run (results.json), drawn in code.

    cd /opt/ai-digest && venv/bin/python medgemma/medgemma_render.py   -> medgemma/out/medgemma-flagship.mp4 (+ .srt)

Reuses the weekly pipeline's look (game_render chrome, intro, outro, captions, Puck narration).
Pathology is left out on purpose: that image has the diagnosis printed on it, so it was not a fair test.
"""
import array, json, math, os, subprocess, sys, wave
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio_ffmpeg
import game_render as g
from game_render import (W, H, FPS, BASE, CYAN, MAGENTA, LIME, AMBER, ORANGE, VIOLET, WHITE, SOFT, DIM, PANEL, URL,
                         ease, pill, text_block, put_glow, check_mark, chime, mix_in, sc_intro, sc_outro, chrome)
import render
from render import F, wrap, tts, read_wav, chunks, srt_time

HERE = os.path.dirname(os.path.abspath(__file__))
R = {r["id"]: r for r in json.load(open(os.path.join(HERE, "results", "results.json")))}
RED = (248, 72, 72)
_img = {}
def photo(cid, bw, bh):
    k = (cid, bw, bh)
    if k not in _img:
        im = Image.open(os.path.join(HERE, "results", "images", f"{cid}.png")).convert("RGB")
        s = min(bw / im.width, bh / im.height) * 1.12  # headroom for the slow zoom
        _img[k] = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return _img[k]

def framed(img, d, cid, x, y, bw, bh, t, T, col):
    """Image in a rounded frame with a slow push-in and a scan line sweeping down in the first 40%."""
    src = photo(cid, bw, bh); z = 1 / 1.12 * (1 + 0.1 * ease(t / T))
    cw, ch = int(src.width / (1.12 * z)), int(src.height / (1.12 * z))
    ox, oy = (src.width - cw) // 2, (src.height - ch) // 2
    s = min(bw / cw, bh / ch)
    crop = src.crop((ox, oy, ox + cw, oy + ch)).resize((int(cw * s), int(ch * s)))
    px, py = x + (bw - crop.width) // 2, y + (bh - crop.height) // 2
    d.rounded_rectangle((px - 10, py - 10, px + crop.width + 10, py + crop.height + 10), 20, fill=PANEL + (255,), outline=col + (255,), width=4)
    img.paste(crop, (px, py))
    sp = t / (0.4 * T)
    if sp < 1:
        sy = py + int(crop.height * ease(sp))
        d.rectangle((px, sy - 3, px + crop.width, sy + 3), fill=CYAN + (220,))
        d.rectangle((px, sy - 40, px + crop.width, sy), fill=CYAN + (35,))
    return py + crop.height

def stamp(img, d, x, y, ok, t, T, at=0.62):
    a = ease((t - at * T) / (0.08 * T))
    if a <= 0: return
    col = LIME if ok else RED; word = "MATCH" if ok else "MISSED"
    sc = 1 + 0.6 * (1 - a); f = F("Black", int(54 * sc)); tw = d.textlength(word, font=f)
    put_glow(img, col, x, y, 60)
    d.rounded_rectangle((x - tw / 2 - 70, y - 44 * sc, x + tw / 2 + 30, y + 44 * sc), 18, fill=PANEL + (235,), outline=col + (255,), width=6)
    if ok: check_mark(d, x - tw / 2 - 34, y, 18, col, 7)
    else: d.line((x - tw / 2 - 52, y - 18, x - tw / 2 - 16, y + 18), fill=col, width=7); d.line((x - tw / 2 - 52, y + 18, x - tw / 2 - 16, y - 18), fill=col, width=7)
    d.text((x + 20 - 0, y), word, font=f, fill=col, anchor="mm")

def typed(d, x, y, text, t, T, size, col, width, start=0.3, end=0.6):
    n = int(len(text) * ease((t - start * T) / ((end - start) * T)))
    return text_block(d, x, y, "“" + text[:n] + ("”" if n >= len(text) else ""), size, col, width, "Bold", 1.2)

FOOT = "Research demo · not for clinical use · Images: Wikimedia Commons (CC0 / public domain)"
def foot(d): d.text((W // 2, 1062), FOOT, font=F("Regular", 21), fill=DIM, anchor="ma")

def case(cid, said, ok, col):
    r = R[cid]
    def fn(img, d, t, T, s):
        yb = framed(img, d, cid, 90, 190, 900, 470, t, T, col)
        y = yb + 34
        d.text((90, y), "SOURCE LABEL", font=F("Bold", 24), fill=SOFT); d.text((330, y - 4), r["source_label"], font=F("Black", 32), fill=WHITE)
        y += 56
        d.text((90, y), "MEDGEMMA SAID", font=F("Bold", 24), fill=col)
        typed(d, 90, y + 36, said, t, T, 34, WHITE, 900)
        pill(d, 990, yb + 26, f"{r['seconds']:.0f} s on a T4", VIOLET, 24, anchor="r")
        stamp(img, d, 780, 300, ok, t, T)
        foot(d)
    return fn

def sc_setup(img, d, t, T, s):
    put_glow(img, CYAN, 540, 330, 110)
    d.text((540, 250), "MedGemma 1.5", font=F("Black", 96), fill=WHITE, anchor="mm")
    d.text((540, 340), "Google's open medical model", font=F("Bold", 38), fill=CYAN, anchor="mm")
    chips = [("4 billion parameters", LIME), ("One free cloud GPU", AMBER), ("5 public images", MAGENTA), ("Known answers", ORANGE)]
    for i, (c, col) in enumerate(chips):
        if t > T * (0.15 + 0.15 * i): pill(d, 540, 450 + 100 * i, c, col, 36, 70, anchor="m")
    if t > 0.8 * T: d.text((540, 900), "Unedited outputs. Here is exactly what it said.", font=F("Bold", 32), fill=SOFT, anchor="ma")

def sc_fundus(img, d, t, T, s):
    for k, (cid, ok_text) in enumerate((("fundus_normal", "No signs of diabetic retinopathy. No referral."),
                                        ("fundus_retinopathy", "Microaneurysms, haemorrhages: diabetic retinopathy. Refer."))):
        x = 70 + k * 480; tt = t - k * 0.45 * T; TT = 0.55 * T
        if tt < 0: continue
        framed(img, d, cid, x, 200, 460, 420, tt, TT, CYAN); yb = 620
        d.text((x, yb + 30), "SOURCE", font=F("Bold", 22), fill=SOFT)
        d.text((x, yb + 58), R[cid]["source_label"], font=F("Black", 28), fill=WHITE)
        d.text((x, yb + 110), "MEDGEMMA", font=F("Bold", 22), fill=CYAN)
        typed(d, x, yb + 140, ok_text, tt, TT, 28, WHITE, 450, 0.25, 0.65)
        stamp(img, d, x + 250, 900, True, tt, TT, 0.7)
    foot(d)

def sc_lesson(img, d, t, T, s):
    p = ease(t / (0.3 * T))
    d.text((540, 230), f"{round(4 * p)} of 5", font=F("Black", 130), fill=WHITE, anchor="mm")
    d.text((540, 330), "A fast first reader. Not a doctor.", font=F("Black", 44), fill=AMBER, anchor="mm")
    nodes = [("MedGemma drafts", CYAN), ("Clinician signs off", MAGENTA), ("Audit trail", AMBER)]
    for i, (n, col) in enumerate(nodes):
        a = ease((t - (0.3 + 0.12 * i) * T) / (0.1 * T))
        if a <= 0: continue
        y = 450 + 150 * i
        d.rounded_rectangle((200, y, 880, y + 100), 26, fill=col + (int(60 * a),), outline=col + (int(255 * a),), width=4)
        d.text((540, y + 50), n, font=F("Black", 44), fill=WHITE + (int(255 * a),), anchor="mm")
        if i < 2: d.line((540, y + 104, 540, y + 146), fill=col + (int(255 * a),), width=6)
    if t > 0.75 * T:
        pill(d, 540, 920, "Patient images stay in your own cloud", LIME, 32, 80, anchor="m")

SCENES = [
    ("intro", None, sc_intro, ["A I Agentic Enterprises. Founder Yuvaraj Aravindan has been solving enterprise IT for three decades. "
                               "Today: a real test of Google's open medical A I model."]),
    ("setup", "THE TEST", sc_setup, ["MedGemma. Four billion parameters, running on one free cloud GPU.",
                                     "Five public medical images with known answers. Here is exactly what it said."]),
    ("cxr", "CASE 1 · CHEST X-RAY", case("cxr_normal", "The lungs are clear. The heart size is normal. Impression: normal chest X-ray.", True, LIME),
     ["A normal chest X-ray. Lungs clear, heart normal. Nine seconds. That's a match."]),
    ("fundus", "CASES 2–3 · RETINA", sc_fundus, ["Two retina photos. The healthy eye: no diabetic retinopathy, no referral.",
                                                "The diseased eye: microaneurysms and haemorrhages. Refer to a retina specialist. Both match."]),
    ("derm", "CASE 4 · SKIN", case("derm_melanoma", "Asymmetric, irregular border, varied colour, over 6 mm. Concerning for melanoma: dermatology review and biopsy.", True, LIME),
     ["A skin lesion. It walked the A B C D E checklist and put melanoma first, with a biopsy recommended. Match."]),
    ("miss", "CASE 5 · CHEST X-RAY", case("cxr_pneumonia", "The lungs are clear. Impression: no acute cardiopulmonary process.", False, RED),
     ["Now the one that matters. The source labels this X-ray lobar pneumonia.",
      "MedGemma called it clear. A clean, confident, wrong report."]),
    ("lesson", "THE LESSON", sc_lesson, ["Four out of five. So it is a fast first reader, not a doctor.",
                                         "The model drafts, a clinician signs off, every decision lands in an audit trail, and patient images never leave your own cloud."]),
    ("outro", None, sc_outro, ["Want governed medical A I built for your enterprise? Visit A I agentic enterprises dot com."]),
]
VEO_BG = {  # cinematic opening: Veo clips behind the first two scenes (no people, per render.STYLE)
    "intro": "Slow dolly through an empty, dark modern hospital radiology reading room at night, a wall of glowing backlit chest X-ray films and teal monitors, light haze.",
    "setup": "Slow macro push-in on a glowing translucent hologram of a retina and a chest X-ray floating above a dark glass desk, drifting teal and amber light particles.",
}
class Bg:
    """Streams a scene-fitted Veo clip as RGB frames; darkened so the drawn text stays readable."""
    def __init__(self, clip, dur, name):
        path = render.scene_background(clip, dur, os.path.join(HERE, "out", f"bg-{name}.mp4"))
        self.p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", path, "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.last = BASE
    def next(self):
        b = self.p.stdout.read(W * H * 3)
        if len(b) == W * H * 3: self.last = Image.blend(Image.frombytes("RGB", (W, H), b), BASE, 0.5)
        return self.last.copy()

SAY = [("A I agentic enterprises dot com", URL), ("A I Agentic Enterprises", "AI Agentic Enterprises"), ("medical A I", "medical AI"),
       ("A B C D E", "ABCDE")]

def main():
    g.SCENE_COL[:] = [LIME, CYAN, LIME, CYAN, LIME, RED, AMBER, VIOLET]
    params = None; audio = array.array("h"); caps = []; spans = []; t = 0.0
    for key, label, fn, segs in SCENES:
        t0 = t
        for gi, seg in enumerate(segs):
            pr, fr = read_wav(tts(seg)); params = params or pr
            gap = g.P_SCENE + (0.5 if key in ("miss", "lesson") else 0) if gi == len(segs) - 1 else g.P_SEG
            audio.frombytes(fr); audio.extend([0] * int(pr.framerate * gap))
            sp = len(fr) / (pr.framerate * 2); cap = seg
            for a, b in SAY: cap = cap.replace(a, b)
            cs = chunks(cap); tot = sum(len(c) for c in cs)
            for k, c in enumerate(cs):
                dur = sp * len(c) / tot + (gap if k == len(cs) - 1 else 0)
                caps.append((t, t + dur, c)); t += dur
        spans.append((t0, t))
    rate = params.framerate
    for i, (key, *_r) in enumerate(SCENES):  # chime on each MATCH stamp, low buzz on the miss
        a, b = spans[i]; T = b - a
        if key in ("cxr", "derm"): mix_in(audio, rate, a + 0.62 * T, chime(rate, 880, 0.25))
        if key == "fundus":
            for st in (0.7 * 0.55, 0.45 + 0.7 * 0.55): mix_in(audio, rate, a + st * T, chime(rate, 880, 0.25))
        if key == "miss": mix_in(audio, rate, a + 0.62 * T, chime(rate, 140, 0.5, 4200))
    mix_in(audio, rate, spans[0][0] + 0.6 * (spans[0][1] - spans[0][0]), chime(rate, 1046, 0.3))
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    bgs = {}
    for i, (key, *_r) in enumerate(SCENES):
        clip = render.veo_clip(VEO_BG.get(key))
        if clip: bgs[i] = Bg(clip, spans[i][1] - spans[i][0], key)
    print("veo backgrounds:", sorted(SCENES[i][0] for i in bgs), flush=True)
    base = os.path.join(HERE, "out", "medgemma-flagship")
    with wave.open(base + ".wav", "wb") as w: w.setparams(params); w.writeframes(audio.tobytes())
    with open(base + ".srt", "w") as f:
        for i, (a, b, x) in enumerate(caps, 1): f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{x}\n\n")
    total = int(t * FPS) + 1
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", base + ".wav", "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest",
                           "-movflags", "+faststart", base + ".mp4"], stdin=subprocess.PIPE)
    only = os.environ.get("STILLS")  # STILLS=1 -> just write one PNG per scene for review
    ci = 0
    for fno in range(total):
        tau = fno / FPS
        si = next((k for k, (a, b) in enumerate(spans) if tau < b), len(spans) - 1)
        while ci < len(caps) - 1 and tau >= caps[ci][1]: ci += 1
        a, b = spans[si]; key, label, fn, _ = SCENES[si]
        bg = bgs[si].next() if si in bgs else None
        if only and abs((tau - a) - 0.85 * (b - a)) > 0.5 / FPS: continue
        img = bg if bg is not None else BASE.copy(); d = ImageDraw.Draw(img, "RGBA")
        fn(img, d, tau - a, b - a, {})
        chrome(img, d, tau, label, si, len(SCENES), caps[ci][2])
        if only: img.save(os.path.join(HERE, "out", f"still-{si}-{key}.png")); continue
        ff.stdin.write(img.tobytes())
        if fno % (FPS * 10) == 0: print(f"  frame {fno}/{total}", flush=True)
    ff.stdin.close(); ff.wait()
    os.remove(base + ".wav")
    print(f"done {base}.mp4  {t:.1f} s")

if __name__ == "__main__":
    main()
