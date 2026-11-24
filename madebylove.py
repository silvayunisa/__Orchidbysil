import math
import os
import sys

import numpy as np
import pygame

VIEW_W, VIEW_H = 562, 1000
FPS = 60
LOOP = 10.0
W = math.tau / LOOP

T_LEAVES = 5.0
T_STEM = (5.3, 7.3)
T_BLOOM, BLOOM_STEP, BLOOM_DUR = 6.4, 0.42, 1.6
T_IDLE = (10.5, 13.0)
T_LOOP = 13.0

STEM = [
    ((452, -40), (446, 90), (380, 190), (350, 330)),
    ((350, 330), (318, 470), (340, 610), (300, 800)),
]

# positions & design
FLOWERS = [
    dict(at=0.04, pos=(470, 40), r=112, roll=20, yaw=0.8, open=0.85),
    dict(at=0.10, pos=(282, 92), r=126, roll=-8, yaw=0.92, open=1.0),
    dict(at=0.28, pos=(468, 190), r=115, roll=24, yaw=0.78, open=0.9),
    dict(at=0.45, pos=(206, 276), r=124, roll=-16, yaw=0.86, open=1.0),
    dict(at=0.70, pos=(400, 372), r=126, roll=12, yaw=0.8, open=1.0),
    dict(at=0.95, pos=(236, 520), r=115, roll=-12, yaw=0.80, open=0.55),
    dict(at=1.18, pos=(420, 576), r=130, roll=6, yaw=0.95, open=1.0),
    dict(at=1.42, pos=(318, 706), r=119, roll=-30, yaw=0.74, open=0.62),
    dict(at=1.78, pos=(218, 852), r=135, roll=-4, yaw=0.95, open=1.0),
    dict(at=1.95, pos=(372, 930), r=36, bud=True),
]

LEAVES = [  # long 
    (((620, 0), (520, 160), (500, 360), (540, 520)), 28),
    (((-30, 120), (90, 210), (120, 380), (70, 540)), 26),
]

FLOWER_PARTS = [  # length dll deh
    ("tepal", 92, 1.0, 90),
    ("tepal", 212, 0.95, 250),
    ("tepal", 328, 0.95, 290),
    ("tepal", 142, 0.9, 100),
    ("tepal", 38, 0.9, 80),
    ("lip", 270, 0.56, 270),
    ("column", 90, 0.36, 90),
]

BLUES = [(18, 52, 160), (35, 105, 215), (80, 170, 240), (160, 215, 250), (228, 244, 255)]
DIGIT_COLORS = [(14, 40, 130), (30, 90, 200), (60, 150, 235), (110, 200, 250), (170, 225, 255),
                (225, 245, 255), (80, 220, 240), (40, 110, 80), (70, 150, 105), (110, 150, 110)]
RAIN_COLORS = [6, 3, 4, 5]


def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(a, b, x):
    t = clamp01((x - a) / (b - a))
    return t * t * (3 - 2 * t)


def smoother(t):
    t = clamp01(t)
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_out_cubic(t):
    return 1 - (1 - clamp01(t)) ** 3


def ease_out_back(t, s=1.3):
    t = clamp01(t) - 1
    return 1 + (s + 1) * t ** 3 + s * t ** 2


def progress(t, start, dur):
    return clamp01((t - start) / dur)


def np_smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def gradient(p, colors):
    pos = np.linspace(0, 1, len(colors))
    out = np.empty(np.shape(p) + (3,))
    for c in range(3):
        out[..., c] = np.interp(p, pos, [col[c] for col in colors])
    return out


def mix(a, b, t):
    b = np.asarray(b, float)
    if np.ndim(t):
        t = np.asarray(t)[..., None]
    return a + (b - a) * t


def cubic(p0, p1, p2, p3, t):
    m = 1 - t
    a, b, c, d = m ** 3, 3 * m * m * t, 3 * m * t * t, t ** 3
    return (a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
            a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1])


def profile(segments, n=300):
    xs, ys = [], []
    for seg in segments:
        x, y = cubic(*seg, np.linspace(0, 1, n))
        xs.append(x)
        ys.append(y)
    x, y = np.concatenate(xs), np.concatenate(ys)
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


TEPAL = profile([((0, .16), (.14, .52), (.32, 1.0), (.52, 1.0)), ((.52, 1.0), (.76, 1.0), (.95, .38), (1.0, 0))])
BUD = profile([((0, .25), (.10, .70), (.35, 1.0), (.55, 1.0)), ((.55, 1.0), (.80, 1.0), (1.0, .60), (1.0, 0))])


def grid(w, h, cx, cy):
    return np.meshgrid(np.arange(w) - cx + 0.5, np.arange(h) - cy + 0.5, indexing="ij")


def to_surface(rgb, alpha):
    w, h = alpha.shape
    surf = pygame.Surface((w, h), pygame.SRCALPHA, 32)
    px = pygame.surfarray.pixels3d(surf)
    px[...] = np.clip(rgb, 0, 255).astype(np.uint8)
    del px
    pa = pygame.surfarray.pixels_alpha(surf)
    pa[...] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    del pa
    return surf.convert_alpha()


def blit_center(dst, src, pos, alpha=None):
    if alpha is not None:
        if alpha < 2:
            return
        src.set_alpha(int(min(255, alpha)))
    dst.blit(src, (int(pos[0] - src.get_width() / 2), int(pos[1] - src.get_height() / 2)))


def ribbon(surf, pts, w0, w1, color, edge=None):
    n = len(pts)
    if n < 2:
        return
    left, right = [], []
    for i, (x, y) in enumerate(pts):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, n - 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy / ln, dx / ln
        hw = lerp(w0, w1, i / (n - 1)) / 2
        left.append((x + nx * hw, y + ny * hw))
        right.append((x - nx * hw, y - ny * hw))
    poly = left + right[::-1]
    pygame.draw.polygon(surf, color, poly)
    pygame.draw.aalines(surf, edge or color, True, poly)


def shape(length, half_w, prof, bend=0.0, extra=1.0):
    h = int(2 * (half_w + abs(bend) + 4))
    w = int(2 * length * extra)
    X, Y = grid(w, h, w / 2, h / 2)
    u = X / length
    uc = np.clip(u, 0, 1)
    half = np.maximum(np.interp(uc, *prof) * half_w, 0.01)
    dy = Y - bend * uc ** 2
    r = dy / half
    edge = half - np.abs(dy)
    alpha = np.clip(edge / 1.2 + 0.5, 0, 1) * np.clip(X / 1.5, 0, 1) * (u <= 1)
    return X, Y, u, uc, dy, r, np.minimum(np.abs(r), 1.5), edge, alpha


def tepal_texture(length, seed, bend):
    rng = np.random.default_rng(seed)
    X, Y, u, uc, dy, r, a, edge, alpha = shape(length, length * 0.36, TEPAL, bend)
    a1 = np.minimum(a, 1)
    col = gradient(uc, [(18, 58, 175), (40, 125, 235), (70, 180, 252), (120, 215, 255), (175, 232, 255)])
    col = mix(col, (90, 215, 240), 0.22 * np.exp(-((uc - 0.45) / 0.18) ** 2))
    col = mix(col, (150, 220, 255), 0.35 * a1 ** 3)
    vein = (0.5 + 0.5 * np.cos(r * math.pi * 11)) ** 6
    col = mix(col, (28, 82, 190), 0.22 * vein * (1 - 0.5 * uc))
    streak = 0.5 + 0.5 * np.cos(r * 37 + rng.uniform(0, 6))
    col *= (1 - 0.05 * streak)[..., None]
    col *= (1 - 0.16 * a1 ** 2 - 0.12 * (1 - np_smooth(0, 0.25, uc)))[..., None]
    sheen = np.exp(-((r + 0.35) / 0.25) ** 2 - ((uc - 0.55) / 0.3) ** 2)
    col = mix(col, (200, 240, 255), 0.2 * sheen)
    half_px = length * 0.36
    col = mix(col, (16, 50, 160), 0.75 * (1 - np_smooth(0.0, 0.38, uc)))
    col = mix(col, (40, 105, 205), 0.45 * np.exp(-(dy / (0.07 * half_px)) ** 2) * (uc < 0.85))
    col = mix(col, (6, 18, 60), 0.95 * np.clip(1.4 - edge / (0.16 * half_px), 0, 1))
    return to_surface(col, alpha), length


def lip_texture(length, seed):
    rng = np.random.default_rng(seed)
    w, h = int(2 * length * 1.08), int(length * 1.1)
    X, Y = grid(w, h, w / 2, h / 2)
    soft = 0.04 * length

    def blob(cx, cy, rx, ry, wave=0.0):
        ang = np.arctan2(Y - cy, X - cx)
        d = np.hypot((X - cx) / rx, (Y - cy) / ry) / (1 + wave * np.sin(ang * 9))
        return np.clip((1 - d) * min(rx, ry) / soft, 0, 1), d

    side_a, side_d = blob(0.30 * length, 0.25 * length, 0.30 * length, 0.17 * length)
    side_b, side_e = blob(0.30 * length, -0.25 * length, 0.30 * length, 0.17 * length)
    mid, mid_d = blob(0.70 * length, 0, 0.30 * length, 0.40 * length, 0.05)
    neck, _ = blob(0.18 * length, 0, 0.22 * length, 0.18 * length)
    alpha = np.maximum.reduce([side_a, side_b, mid, neck]) * (X > 0)

    col = np.empty(X.shape + (3,))
    col[...] = (150, 200, 245)
    sides = np.maximum(side_a, side_b) * (1 - mid)
    stripes = (0.5 + 0.5 * np.cos(Y / length * 42)) ** 3
    col = mix(col, (24, 60, 160), 0.55 * stripes * sides * np_smooth(0.15, 0.45, X / length))
    patch = np.clip((0.78 - mid_d) / 0.12, 0, 1) * mid
    col = mix(col, (12, 28, 100), 0.92 * patch)
    throat = np.exp(-((X - 0.2 * length) / (0.16 * length)) ** 2 - (Y / (0.13 * length)) ** 2)
    col = mix(col, (18, 48, 140), 0.8 * throat)
    spots = np.zeros_like(X)
    for _ in range(70):
        cx = rng.uniform(0.15, 0.85) * length
        cy = rng.normal(0, 0.2) * length
        rad = rng.uniform(0.012, 0.03) * length + 0.6
        spots = np.maximum(spots, np.clip((1 - np.hypot(X - cx, Y - cy) / rad) * 2, 0, 1))
    col = mix(col, (10, 26, 95), 0.85 * spots * (1 - patch))
    col = mix(col, (60, 110, 200), 0.35 * spots * patch)
    for s in (-1, 1):
        ridge = np.exp(-((Y - s * 0.05 * length) / (0.03 * length)) ** 2) * np_smooth(0.18, 0.28, X / length) * (1 - np_smooth(0.5, 0.6, X / length))
        col = mix(col, (190, 245, 255), 0.95 * ridge)
    col = mix(col, (245, 250, 255), 0.6 * np.clip((mid_d - 0.8) / 0.15, 0, 1) * mid)
    col *= (1 - 0.15 * np.clip(np.abs(Y) / (0.45 * length), 0, 1) ** 2)[..., None]
    return to_surface(col, alpha), length


def column_texture(length):
    X, Y, u, uc, dy, r, a, edge, alpha = shape(length, length * 0.36, BUD, length * 0.08)
    col = gradient(uc, [(150, 195, 245), (205, 228, 252), (225, 240, 255)])
    col = mix(col, (30, 70, 170), 0.5 * (0.5 + 0.5 * np.cos(r * 9 + uc * 20)) ** 6 * (uc < 0.6) * (r > 0))
    col *= (1 - 0.3 * np.minimum(a, 1) ** 2)[..., None]
    cap = np.exp(-((uc - 0.88) / 0.1) ** 2 - (r / 0.6) ** 2)
    col = mix(col, (250, 253, 255), 0.85 * cap)
    col = mix(col, (255, 255, 255), 0.4 * np.exp(-((r + 0.35) / 0.3) ** 2 - ((uc - 0.6) / 0.25) ** 2))
    return to_surface(col, alpha), length


def bud_texture(length):
    X, Y, u, uc, dy, r, a, edge, alpha = shape(length, length * 0.46, BUD)
    a1 = np.minimum(a, 1)
    col = gradient(uc, [(70, 140, 130), (110, 185, 215), (175, 220, 250), (215, 238, 255)])
    for s in (-0.34, 0.34):
        col *= (1 - 0.12 * np.exp(-((r - s) / 0.06) ** 2) * (uc > 0.2))[..., None]
    col *= (1 - 0.38 * a1 ** 2)[..., None]
    col = mix(col, (250, 253, 255), 0.5 * np.exp(-((r + 0.35) / 0.25) ** 2 - ((uc - 0.55) / 0.25) ** 2))
    return to_surface(col, alpha), length


def glow_sprite(rx, ry, color, power=2.0, peak=1.0):
    rx, ry = max(2, int(rx)), max(2, int(ry))
    X, Y = grid(2 * rx, 2 * ry, rx, ry)
    d = np.hypot(X / rx, Y / ry)
    rgb = np.empty(d.shape + (3,))
    rgb[...] = color
    return to_surface(rgb, np.clip(1 - d, 0, 1) ** power * peak)


class Orchid:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.k = k = h / VIEW_H
        size = int(150 * k)
        self.sets = []
        for v in range(3):
            right = tepal_texture(size, 10 + v, size * 0.07 * (1 + v * 0.4))
            left = (pygame.transform.flip(right[0], False, True), right[1])
            self.sets.append({
                "right": right, "left": left,
                "lip": lip_texture(int(size * 0.56), 30 + v),
                "column": column_texture(int(size * 0.36)),
            })
        self.bud = bud_texture(int(60 * k))
        self.glow = glow_sprite(140 * k, 140 * k, (60, 150, 255), 2.0, 0.3)
        self.flash = glow_sprite(160 * k, 160 * k, (190, 240, 255), 1.6, 0.9)
        self.order = sorted(range(len(FLOWERS)), key=lambda i: FLOWERS[i]["at"])
        for n, i in enumerate(self.order):
            FLOWERS[i]["start"] = T_BLOOM + n * BLOOM_STEP
        self.centers = []

    def sway_at(self, y, sway):
        return sway * clamp01(y / VIEW_H) ** 1.3

    def stem_point(self, s, sway):
        i = min(int(s), len(STEM) - 1)
        x, y = cubic(*STEM[i], s - i)
        return (x + self.sway_at(y, sway)) * self.k, y * self.k

    def draw_part(self, surf, tex, size, length, angle, cx, cy, roll, yaw, wmul, offset=(0, 0), alpha=255):
        img, tex_len = tex
        ra = math.radians(angle)
        dx, dy = math.cos(ra) * yaw, math.sin(ra)
        wx, wy = -math.sin(ra) * yaw, math.cos(ra)
        cr, sr = math.cos(roll), math.sin(roll)
        dx, dy = dx * cr - dy * sr, dx * sr + dy * cr
        wx, wy = wx * cr - wy * sr, wx * sr + wy * cr
        lf = max(math.hypot(dx, dy), 1e-3)
        wf = max(abs(dx * wy - dy * wx) / lf, 0.08)
        sc = length * size / tex_len
        tw, th = img.get_size()
        scaled = pygame.transform.smoothscale(img, (max(2, int(tw * sc * lf)), max(2, int(th * sc * wf * wmul))))
        rot = pygame.transform.rotozoom(scaled, math.degrees(math.atan2(dy, dx)), 1)
        ox, oy = offset[0] * yaw, offset[1]
        ox, oy = ox * cr - oy * sr, ox * sr + oy * cr
        blit_center(surf, rot, (cx + ox, cy - oy), alpha)

    def draw_flower(self, surf, f, cx, cy, o, t, idle, idx):
        r = f["r"] * self.k
        roll = math.radians(f["roll"] + idle * 2.5 * math.sin(W * t + idx))
        yaw = f["yaw"] + idle * 0.04 * math.sin(2 * W * t + idx * 1.7)
        tex = self.sets[idx % 3]
        alpha = 255 * smoothstep(0.0, 0.18, o)
        for n, (kind, ang, length, pull) in enumerate(FLOWER_PARTS):
            a = ang + (pull - ang) * (1 - o) * 0.5
            a += idle * 1.8 * math.sin(2 * W * t + idx + n * 1.3)
            ln = length * lerp(0.7, 1.0, o) * (1 + idle * 0.02 * math.sin(3 * W * t + n * 2 + idx))
            wm = lerp(0.6, 1.0, o)
            if kind == "tepal":
                part = tex["left"] if 90 < a % 360 < 270 else tex["right"]
                self.draw_part(surf, part, r, ln, a, cx, cy, roll, yaw, wm, alpha=alpha)
            elif kind == "lip":
                self.draw_part(surf, tex["lip"], r, ln, a, cx, cy, roll, yaw, wm, (0, -0.1 * r), alpha)
            else:
                self.draw_part(surf, tex["column"], r, ln, a, cx, cy, roll, yaw, 1.0, (0, -0.06 * r), alpha)

    def draw(self, surf, t, idle, glow=True):
        k = self.k
        sway = idle * (16 * math.sin(W * t) + 6 * math.sin(2 * W * t + 0.9))

        for i, (pts, width) in enumerate(LEAVES):
            g = ease_out_cubic(progress(t, T_LEAVES + i * 0.25, 1.6))
            if g <= 0.01:
                continue
            wave = idle * 10 * math.sin(2 * W * t + i * 2)
            ctrl = [(x + wave * j / 3 + self.sway_at(y, sway) * 0.4, y) for j, (x, y) in enumerate(pts)]
            n = max(2, int(50 * g))
            line = [(x * k, y * k) for x, y in (cubic(*ctrl, g * j / (n - 1)) for j in range(n))]
            ribbon(surf, line, width * k, lerp(width, 2, g) * k, (24, 66, 54), (14, 40, 36))
            ribbon(surf, [(x - 2 * k, y) for x, y in line], width * 0.25 * k, 1 * k, (64, 128, 98))

        g = smoother(progress(t, *T_STEM))
        if g > 0:
            n = max(2, int(120 * g))
            pts = [self.stem_point(2 * g * j / (n - 1), sway) for j in range(n)]
            ribbon(surf, pts, 9 * k, lerp(9, 5, g) * k, (70, 116, 104), (44, 80, 74))
            ribbon(surf, [(x - 1.6 * k, y) for x, y in pts], 2.4 * k, 1.2 * k, (120, 168, 150))

        self.centers = []
        items = []
        for idx in self.order:
            f = FLOWERS[idx]
            if 2 * g < f["at"]:
                continue
            grow = ease_out_cubic(progress(t, f["start"] - 0.9, 0.9))
            if grow <= 0.01:
                continue
            sx, sy = self.stem_point(f["at"], sway)
            px, py = f["pos"]
            bob = idle * 3 * math.sin(2 * W * t + idx * 2.1)
            cx, cy = (px + self.sway_at(py, sway)) * k, (py + bob) * k
            mid = ((sx + cx) / 2 + 18 * k, (sy + cy) / 2 + 14 * k)
            line = [cubic((sx, sy), mid, mid, (cx, cy), j / 12) for j in range(13)]
            ribbon(surf, line, 5 * k * grow, 3.5 * k * grow, (76, 122, 110), (46, 82, 76))
            items.append((idx, f, cx, cy, sx, sy, grow))
            self.centers.append((cx, cy, f["r"] * k, f["start"]))

        items.sort(key=lambda it: it[3])
        if glow:
            for idx, f, cx, cy, sx, sy, grow in items:
                if not f.get("bud"):
                    o = progress(t, f["start"], BLOOM_DUR)
                    blit_center(surf, self.glow, (cx, cy), 255 * smoothstep(0, 0.5, o) * f["r"] / 145)
                    fl = math.sin(math.pi * progress(t, f["start"] - 0.15, 1.0))
                    blit_center(surf, self.flash, (cx, cy), 110 * fl)

        for idx, f, cx, cy, sx, sy, grow in items:
            o = f.get("open", 0) * ease_out_back(progress(t, f["start"], BLOOM_DUR), 1.1)
            bud_alpha = 255 * (1 - smoothstep(0.08, 0.3, o))
            if bud_alpha > 2:
                ang = math.degrees(math.atan2(-(cy - sy), cx - sx)) + idle * 4 * math.sin(2 * W * t + idx)
                size = (f["r"] * 1.5 if f.get("bud") else 46) * k * grow / self.bud[1]
                blit_center(surf, pygame.transform.rotozoom(self.bud[0], ang, size), (cx, cy), bud_alpha)
            if o > 0.02:
                self.draw_flower(surf, f, cx, cy, o, t, idle, idx)


def background(w, h):
    rng = np.random.default_rng(5)
    X, Y = grid(w, h, 0, 0)
    col = gradient(Y / h, [(2, 5, 14), (4, 10, 26), (5, 14, 34)])
    d = np.hypot((X - w * 0.55) / (w * 0.75), (Y - h * 0.48) / (h * 0.5))
    col = mix(col, (14, 40, 86), 0.45 * np.exp(-d * d * 2.5))
    col += rng.normal(0, 1.0, col.shape)
    surf = pygame.surfarray.make_surface(np.clip(col, 0, 255).astype(np.uint8)).convert()
    for _ in range(140):
        c = int(rng.uniform(60, 170))
        pygame.draw.circle(surf, (c // 2, c, min(255, c + 60)), (int(rng.uniform(0, w)), int(rng.uniform(0, h))), 1)
    return surf


def mono_font(size):
    for name in ("consolas", "cascadiamono", "couriernew", "dejavusansmono"):
        path = pygame.font.match_font(name, bold=True)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, int(size * 1.3))


def make_palette():
    cols = []
    ramps = [
        ([(4, 14, 50), (16, 56, 175), (40, 120, 240), (70, 175, 255), (150, 220, 255), (205, 240, 255)], 20),
        ([(10, 60, 90), (20, 170, 230), (110, 235, 255)], 7),
        ([(26, 74, 58), (60, 130, 100), (140, 205, 170)], 6),
    ]
    green = []
    for r, (stops, n) in enumerate(ramps):
        for p in np.linspace(0, 1, n):
            cols.append(tuple(int(v) for v in gradient(np.array(p), stops)))
            green.append(r == 2)
    cols.append((255, 255, 255))
    pal = np.array(cols, float)
    q = (np.arange(32) * 8 + 4).astype(float)
    cube = np.stack(np.meshgrid(q, q, q, indexing="ij"), -1).reshape(-1, 3)
    dist = ((cube[:, None, :] - pal[None]) ** 2).sum(-1)
    bluish = cube[:, 2] >= cube[:, 1] - 6
    dist[np.ix_(bluish, np.array(green + [False]))] += 1e6
    lut = np.argmin(dist, axis=1).reshape(32, 32, 32)
    return cols, lut


class Binary:
    def __init__(self, w, h):
        self.ch = max(8, round(h / 100))
        self.cw = max(5, round(self.ch * 0.66))
        self.cols, self.rows = w // self.cw, h // self.ch
        ox, oy = (w - self.cols * self.cw) // 2, (h - self.rows * self.ch) // 2
        self.colors, self.lut = make_palette()
        self.white = len(self.colors) - 1
        font = mono_font(int(self.ch * 0.98))
        self.glyphs = []
        for ch in "01":
            for c in self.colors:
                img = font.render(ch, True, c).convert_alpha()
                self.glyphs.append(img)
        self.n = len(self.colors)
        gw, gh = self.glyphs[0].get_size()
        xs = ox + np.arange(self.cols) * self.cw + (self.cw - gw) / 2
        ys = oy + np.arange(self.rows) * self.ch + (self.ch - gh) / 2
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self.pos = list(zip(gx.ravel().astype(int).tolist(), gy.ravel().astype(int).tolist()))
        self.cx = gx.ravel() + gw / 2
        self.cy = gy.ravel() + gh / 2
        rng = np.random.default_rng(1)
        size = self.cols * self.rows
        self.seed = rng.integers(0, 1000, size)
        self.rate = rng.choice([10, 20, 30, 40], size)

    def sample(self, small):
        rgb = pygame.surfarray.array3d(small).reshape(-1, 3).astype(float)
        a = pygame.surfarray.array_alpha(small).reshape(-1).astype(float)
        c = rgb * (a[:, None] / 255) / 255
        c = np.clip(255 * c ** 0.9 * 1.08, 0, 255).astype(int) >> 3
        idx = self.lut[c[:, 0], c[:, 1], c[:, 2]]
        return idx, a

    def draw(self, surf, small, t):
        idx, a = self.sample(small)
        on = np.nonzero(a > 38)[0]
        if len(on) == 0:
            return
        bits = (self.seed[on] + np.floor(t * self.rate[on] / LOOP)).astype(int) & 1
        keys = idx[on]
        spark = (((self.seed[on] + np.floor(t * 4)).astype(int) % 40) == 0) & (self.seed[on] % 2 == 0)
        keys = np.where(spark & (a[on] > 150), self.white, keys) + bits * self.n
        g, p = self.glyphs, self.pos
        surf.blits([(g[k], p[i]) for k, i in zip(keys.tolist(), on.tolist())], doreturn=False)


class Rain:
    def __init__(self, binary, targets, colors, reveal, w, h):
        rng = np.random.default_rng(3)
        self.b = binary
        self.w, self.h = w, h
        n = len(targets)
        self.tx, self.ty = binary.cx[targets] - binary.glyphs[0].get_width() / 2, binary.cy[targets] - binary.glyphs[0].get_height() / 2
        self.color = colors
        self.reveal = reveal
        self.x0 = np.round(rng.uniform(0, w, n) / binary.cw) * binary.cw
        self.y0 = rng.uniform(0, h, n)
        self.speed = rng.uniform(200, 460, n) * h / VIEW_H
        self.start = rng.uniform(1.4, 3.0, n)
        self.dur = rng.uniform(1.6, 2.2, n)
        self.swirl = rng.uniform(-70, 70, n) * h / VIEW_H
        self.seed = rng.integers(0, 1000, n)
        self.flip = rng.uniform(4, 10, n)
        self.rain = rng.choice([23, 25, 15, 13], n)
        m = 120
        self.ax = np.round(rng.uniform(0, w, m) / binary.cw) * binary.cw
        self.ay = rng.uniform(0, h, m)
        self.ak = rng.integers(1, 3, m)
        self.aseed = rng.integers(0, 1000, m)
        self.acol = rng.choice([6, 9, 12, 23], m)

    def draw_ambient(self, surf, t):
        bright = 0.16 + 0.5 * (1 - smoothstep(4.0, 9.0, t))
        span = self.h + 40
        y = (self.ay + self.ak * span * t / LOOP) % span - 20
        bits = (self.aseed + np.floor(t * 20 / LOOP)).astype(int) & 1
        g = self.b.glyphs
        for i in range(len(self.ax)):
            img = g[self.acol[i] + bits[i] * self.b.n]
            img.set_alpha(int(255 * bright * smoothstep(0, 1, t)))
            surf.blit(img, (self.ax[i], y[i]))
            img.set_alpha(255)

    def draw(self, surf, t):
        if t > self.reveal.max() + 1:
            return
        ry = (self.y0 + self.speed * t) % (self.h + 40) - 20
        p = np.clip((t - self.start) / self.dur, 0, 1)
        e = p * p * p * (p * (p * 6 - 15) + 10)
        x = self.x0 + (self.tx - self.x0) * e + self.swirl * np.sin(np.pi * e)
        y = ry + (self.ty - ry) * e
        fade = np_smooth(self.reveal, self.reveal + 0.5, t)
        alpha = 255 * (1 - fade) * min(1, t / 0.6)
        bits = (self.seed + np.floor(t * self.flip * (1 - 0.8 * e))).astype(int) & 1
        cols = np.where(e > 0.6, self.color, self.rain)
        flash = (e > 0.98) & (fade > 0) & (fade < 0.6)
        cols = np.where(flash, self.b.white, cols)
        g = self.b.glyphs
        for i in np.nonzero(alpha > 4)[0]:
            img = g[cols[i] + bits[i] * self.b.n]
            img.set_alpha(int(alpha[i]))
            surf.blit(img, (x[i], y[i]))
            img.set_alpha(255)


class Scene:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.bg = background(w, h)
        self.binary = Binary(w, h)
        b = self.binary
        self.art_h = b.rows * 4
        self.art_w = int(self.art_h * VIEW_W / VIEW_H)
        self.art = pygame.Surface((self.art_w, self.art_h), pygame.SRCALPHA)
        self.orchid = Orchid(self.art_w, self.art_h)

        self.art.fill((0, 0, 0, 0))
        self.orchid.draw(self.art, 40.0, 0.0, glow=False)
        small = pygame.transform.smoothscale(self.art, (b.cols, b.rows))
        idx, a = b.sample(small)
        cells = np.nonzero(a > 60)[0]
        rng = np.random.default_rng(2)
        targets = rng.choice(cells, min(len(cells), 1400), replace=False)
        sx, sy = w / self.art_w, h / self.art_h
        tx, ty = b.cx[targets] / sx, b.cy[targets] / sy
        reveal = T_STEM[0] + (T_STEM[1] - T_STEM[0]) * np.clip(ty / self.art_h, 0, 1)
        best = np.full(len(targets), 1e9)
        for cx, cy, r, start in self.orchid.centers:
            d = np.hypot(tx - cx, ty - cy) / (r * 1.1)
            hit = (d < 1) & (d < best)
            reveal[hit] = start + 0.25 + 0.4 * d[hit]
            best[hit] = d[hit]
        self.rain = Rain(b, targets, idx[targets], reveal, w, h)
        self.flash = glow_sprite(h * 0.12, h * 0.12, (120, 210, 255), 1.8, 0.7)
        self.scale = (sx, sy)

    def draw(self, surf, t):
        surf.blit(self.bg, (0, 0))
        idle = smoothstep(*T_IDLE, t)
        self.rain.draw_ambient(surf, t)

        self.art.fill((0, 0, 0, 0))
        self.orchid.draw(self.art, t, idle, glow=False)
        show = smoothstep(T_STEM[0], T_BLOOM + 3, t)
        halo = pygame.transform.smoothscale(self.art, (self.w // 10, self.h // 10))
        halo = pygame.transform.smoothscale(halo, (self.w, self.h))
        halo.set_alpha(int(55 * show * (0.85 + 0.15 * math.sin(2 * W * t))))
        surf.blit(halo, (0, 0))
        body = pygame.transform.smoothscale(self.art, (self.w, self.h))
        body.set_alpha(int(62 * show))
        surf.blit(body, (0, 0))

        small = pygame.transform.smoothscale(self.art, (self.binary.cols * 2, self.binary.rows * 2))
        small = pygame.transform.scale(small, (self.binary.cols, self.binary.rows))
        self.binary.draw(surf, small, t)
        self.rain.draw(surf, t)

        sx, sy = self.scale
        for idx, f in enumerate(FLOWERS):
            if "start" in f and not f.get("bud"):
                fl = math.sin(math.pi * progress(t, f["start"] - 0.1, 0.9))
                if fl > 0:
                    px, py = f["pos"]
                    blit_center(surf, self.flash, (px * self.art_w / VIEW_W * sx, py * self.art_h / VIEW_H * sy), 90 * fl)


def fit(win_w, win_h):
    h = min(win_h, int(win_w / (VIEW_W / VIEW_H)))
    return int(h * VIEW_W / VIEW_H), h


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    pygame.init()
    info = pygame.display.Info()
    win_h = info.current_h - 90
    win_w = int(win_h * VIEW_W / VIEW_H)
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{info.current_w - win_w - 8},32"
    pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
    pygame.display.set_caption("Blue Orchid")

    size = fit(win_w, win_h)
    scene = Scene(*size)
    canvas = pygame.Surface(size).convert()
    clock = pygame.time.Clock()
    t = 0.0
    pending = None
    running = True

    while running:
        dt = min(clock.tick(FPS) / 1000, 1 / 20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_r, pygame.K_SPACE):
                    t = 0.0
            elif event.type == pygame.VIDEORESIZE:
                pending = (event.w, event.h, pygame.time.get_ticks())

        if pending and pygame.time.get_ticks() - pending[2] > 300:
            size = fit(pending[0], pending[1])
            if size != (scene.w, scene.h):
                scene = Scene(*size)
                canvas = pygame.Surface(size).convert()
            pending = None

        t += dt
        if t > T_LOOP + LOOP:
            t -= LOOP

        scene.draw(canvas, t)
        window = pygame.display.get_surface()
        window.fill((2, 5, 14))
        window.blit(canvas, ((window.get_width() - scene.w) // 2, (window.get_height() - scene.h) // 2))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()