# -*- coding: utf-8 -*-
"""出作品集用图：侧视、斜视、姿态网格、腕部特写。

两个关键做法：

1. **取景自适应** —— 用每个 geom 的包围球（geom_rbound + geom_xpos）求整机 AABB，
   反算 lookat 和 dist。改一行 arm_params.py 重跑，构图自动跟着变，不用手调相机。
2. **标注位置取自前向运动学** —— d.xpos 是 MuJoCo 算出来的关节世界坐标，
   再用同一个相机参数投影成像素，所以关节名永远贴在真实位置上。

用法：
    python tools/render_gallery.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import mujoco
from PIL import Image, ImageDraw, ImageFont

SHOTS = os.path.join(ROOT, "shots")
os.makedirs(SHOTS, exist_ok=True)

BG = (24, 26, 32)
FOVY = 45.0                      # MuJoCo Renderer 默认垂直视场
FILL = 0.88                      # 物体占视锥的比例

# (body 名, 标签, 世界系微调) —— M1 的 body 就是 M2 支点，标注要下移到舵机舱中心
JOINTS = [
    ("L1_base_m2", "M1", (-0.012, 0.0, -0.058)),
    ("L2_upper_a", "M2", (0, 0, 0)),
    ("L3_fore_a", "M3", (0, 0, 0)),
    ("L4_wr1", "M4", (0, 0, 0)),
    ("L5_wr2", "M5", (0, 0, 0)),
    ("L6_wr3", "M6", (0, 0, 0)),
    ("L7_tool", "M7", (0, 0, 0)),
]

FONT_PATH = None
for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
          r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(p):
        FONT_PATH = p
        break


def font(sz):
    return ImageFont.truetype(FONT_PATH, sz) if FONT_PATH else ImageFont.load_default()


# ---------------------------------------------------------------- 取景
def auto_frame(m, d, az, el, w, h, pad=1.0, keep=None):
    """按当前姿态的 AABB 反算 lookat / dist，保证物体居中且不裁边。

    做法：把 AABB 的 8 个角点转到相机坐标系，对每个角点解
    「要让它落在视锥内，相机至少要退到多远」，取最大值。
    比用包围球保守估计准得多（包围球对细长物体几乎总是退太远）。
    """
    pts = []
    for g in range(m.ngeom):
        if m.geom_rgba[g][3] < 0.05:      # 已隐藏（桌面）
            continue
        if keep is not None:
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
            if b not in keep:
                continue
        r = float(m.geom_rbound[g])
        if r <= 0:
            continue
        c = d.geom_xpos[g]
        pts.append(c - r)
        pts.append(c + r)
    pts = np.array(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2.0

    el_r, az_r = np.deg2rad(el), np.deg2rad(az)
    fwd = np.array([np.cos(el_r) * np.cos(az_r), np.cos(el_r) * np.sin(az_r), np.sin(el_r)])
    right = np.cross(fwd, np.array([0, 0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    tan_v = math.tan(math.radians(FOVY / 2.0)) * FILL
    tan_h = tan_v * (w / float(h))

    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    dist = 0.0
    for p in corners:
        v = p - center
        need = max(abs(v @ right) / tan_h, abs(v @ up) / tan_v) + (v @ fwd)
        dist = max(dist, need)
    return dict(az=az, el=el, dist=dist * pad, lookat=tuple(center))


def make_projector(cam, w, h, fovy=FOVY):
    el, az = np.deg2rad(cam["el"]), np.deg2rad(cam["az"])
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = np.asarray(cam["lookat"], float) - fwd * cam["dist"]
    right = np.cross(fwd, np.array([0, 0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = (h / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)

    def project(pt):
        v = np.asarray(pt, float) - eye
        zc = v @ fwd
        if zc <= 1e-9:
            return None
        return np.array([w / 2 + (v @ right) / zc * f,
                         h / 2 - (v @ up) / zc * f])
    return project


# ---------------------------------------------------------------- 渲染
def load_model():
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "arm_real.xml"))
    d = mujoco.MjData(m)
    # 桌面隐藏；顺带把整体环境光提亮，免得灰杆在黑底上糊成一团
    for i in range(m.ngeom):
        if (m.geom(i).name or "") == "table":
            m.geom_rgba[i] = [0, 0, 0, 0]
    m.vis.headlight.ambient[:] = [0.42, 0.44, 0.50]
    m.vis.headlight.diffuse[:] = [0.55, 0.55, 0.58]
    m.vis.headlight.specular[:] = [0.18, 0.18, 0.20]
    m.vis.map.znear = 0.01
    mujoco.mj_forward(m, d)
    return m, d


def render(m, d, cam, w, h):
    r = mujoco.Renderer(m, h, w)
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = cam["lookat"]
    c.distance = cam["dist"]
    c.azimuth = cam["az"]
    c.elevation = cam["el"]
    r.update_scene(d, camera=c)
    img = Image.fromarray(r.render()).convert("RGB")
    r.close()
    return img


def annotate(img, m, d, cam, title=None, subtitle=None, labels=True, keep=None):
    w, h = img.size
    dr = ImageDraw.Draw(img, "RGBA")
    pj = make_projector(cam, w, h)

    if labels:
        for body, tag, off in JOINTS:
            if keep is not None and body not in keep:
                continue
            try:
                p = d.xpos[m.body(body).id] + np.asarray(off, float)
            except Exception:
                continue
            uv = pj(p)
            if uv is None:
                continue
            x, y = float(uv[0]), float(uv[1])
            if not (10 < x < w - 10 and 10 < y < h - 10):
                continue
            dr.ellipse([x - 6, y - 6, x + 6, y + 6],
                       fill=(6, 8, 12, 170), outline=(0, 226, 255, 240), width=2)
            dr.text((x + 10, y - 11), tag, fill=(0, 226, 255, 255),
                    font=font(max(15, int(h * 0.026))))

    if title:
        f = font(max(20, int(h * 0.038)))
        dr.rectangle([0, 0, w, int(h * 0.072)], fill=(10, 11, 15, 200))
        dr.text((18, int(h * 0.014)), title, fill=(238, 241, 246, 255), font=f)
    if subtitle:
        dr.text((18, int(h * 0.072) + 8 if title else 14), subtitle,
                fill=(146, 155, 170, 255), font=font(max(14, int(h * 0.024))))
    return img



def trim(img, pad=34, keep_top=0):
    """裁掉背景留白。背景色取左右边列中位数，亮度差超过阈值的算内容。

    MuJoCo 渲染的背景是渐变，所以不用"等于某个颜色"判断，用偏离中位背景色的程度。
    keep_top：顶部标题栏高度，不参与内容判定（否则整幅图都算内容）。
    """
    a = np.asarray(img.convert("RGB")).astype(int)
    h, w = a.shape[:2]
    edges = np.concatenate([a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
    bg = np.median(edges, axis=0)
    mask = np.abs(a - bg).sum(axis=2) > 30
    if keep_top:
        mask[:keep_top, :] = False
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    if keep_top:
        # 有标题栏时只裁底部：顶部标题必须留，左右也不能裁（否则标题被切掉）
        return img.crop((0, 0, w, min(h, rows[-1] + pad)))
    t = max(0, rows[0] - pad)
    b = min(h, rows[-1] + pad)
    l = max(0, cols[0] - pad)
    r = min(w, cols[-1] + pad)
    return img.crop((l, t, r, b))


POSES = {
    "零位（装配基准）": {},
    "前伸（肩 -25° 肘 +35°）": {"M2_should": -25, "M3_elbow": 35},
    "折叠（收拢）": {"M2_should": -75, "M3_elbow": 110, "M4_wp": 60, "M5_wy": 40},
    "上扬（肩 +55° 肘 -60°）": {"M2_should": 55, "M3_elbow": -60, "M4_wp": 30},
}


def set_pose(m, d, pose):
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    d.qpos[:] = 0.0
    for jn, ang in pose.items():
        d.qpos[names.index(jn)] = math.radians(ang)
    mujoco.mj_forward(m, d)


def main():
    m, d = load_model()

    # ---- 01 侧视（az=90 视线沿 +y，看到的是臂的侧面轮廓）
    set_pose(m, d, {})
    cam = auto_frame(m, d, az=90, el=0, w=1200, h=760)
    img = render(m, d, cam, 1200, 760)
    annotate(img, m, d, cam,
             title="侧视 · 零位构型",
             subtitle="7 关节串链；M2→M3、M3→M4 是拐折段。杆的深浅表示尺寸来源：浅=实测，深=仍为估计值")
    img = trim(img, keep_top=56)
    img.save(os.path.join(SHOTS, "01-side.png"))
    print("-> shots/01-side.png  dist=%.3f lookat=%s" % (cam["dist"], np.round(cam["lookat"], 3)))

    # ---- 02 斜视（底板 160x160 在斜视下会把臂推到画面上半，先隐藏）
    set_pose(m, d, {})
    plate = [g for g in range(m.ngeom) if (m.geom(g).name or "") == "base_plate"]
    for g in plate:
        m.geom_rgba[g] = [0, 0, 0, 0]
    cam = auto_frame(m, d, az=128, el=20, w=1200, h=760)
    img = render(m, d, cam, 1200, 760)
    annotate(img, m, d, cam,
             title="斜视 · 零位构型",
             subtitle="每个关节处是真实舵机盒（大舵机 40×40×20 / 飞特 STS3215 45×25×35），不是「细杆 + 球」")
    img = trim(img, keep_top=56)
    img.save(os.path.join(SHOTS, "02-iso.png"))
    for g in plate:
        m.geom_rgba[g] = [0.35, 0.35, 0.38, 1]
    print("-> shots/02-iso.png")

    # ---- 03 姿态网格（每格独立取景；底板 160x160 太吃画面，隐藏后臂才居中）
    TW, TH = 620, 500
    plate = [g for g in range(m.ngeom)
             if (m.geom(g).name or "") == "base_plate"]
    for g in plate:
        m.geom_rgba[g] = [0, 0, 0, 0]
    tiles = []
    for desc, pose in POSES.items():
        set_pose(m, d, pose)
        cam = auto_frame(m, d, az=126, el=18, w=TW, h=TH)
        im = render(m, d, cam, TW, TH)
        im = annotate(im, m, d, cam)
        dr = ImageDraw.Draw(im, "RGBA")
        dr.rectangle([0, 0, TW, 34], fill=(10, 11, 15, 205))
        dr.text((12, 5), desc, fill=(232, 236, 243, 255), font=font(18))
        tiles.append(im)
    grid = Image.new("RGB", (TW * 2 + 8, TH * 2 + 8), BG)
    for k, im in enumerate(tiles):
        grid.paste(im, ((k % 2) * (TW + 8), (k // 2) * (TH + 8)))
    for g in plate:
        m.geom_rgba[g] = [0.35, 0.35, 0.38, 1]
    grid.save(os.path.join(SHOTS, "03-pose-grid.png"))
    print("-> shots/03-pose-grid.png")

    # ---- 06 腕部特写：只留 M3 之后的部分，底板/舱体隐藏，画面干净
    set_pose(m, d, {})
    keep = {"L3_fore_b", "L4_wr1", "L5_wr2", "L6_wr3", "L7_tool"}
    saved = {}
    for g in range(m.ngeom):
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        if b not in keep and m.geom_rgba[g][3] > 0.05:
            saved[g] = m.geom_rgba[g].copy()
            m.geom_rgba[g] = [0, 0, 0, 0]
    cam = auto_frame(m, d, az=116, el=16, w=1000, h=700, keep=keep)
    img = render(m, d, cam, 1000, 700)
    annotate(img, m, d, cam,
             title="腕部特写 · M4→M7 四颗 STS3215 串联",
             subtitle="橙色件是 U 型延长臂（外宽 41 / 内宽 37 / 单边间隙 1.0mm）；关节轴必须与舵机输出轴重合",
             keep=keep)
    img = trim(img, keep_top=56)
    img.save(os.path.join(SHOTS, "06-wrist.png"))
    for g, c in saved.items():
        m.geom_rgba[g] = c
    print("-> shots/06-wrist.png")


if __name__ == "__main__":
    main()
