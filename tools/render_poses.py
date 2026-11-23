# -*- coding: utf-8 -*-
"""离屏渲染 + 碰撞高亮：出 2x2 大图和一段演示 mp4。

碰撞的连杆会被染红，一眼能看出撞在哪。

用法：
    python render_shots.py          # 出 4 张 PNG + 一张 2x2 拼图
    python render_shots.py --video  # 额外出一段 mp4
"""
import argparse
import math
import os

import numpy as np

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")

POSES = {
    "01_zero": ("零位：立柱 + 水平伸出", {}),
    "02_reach": ("前伸：肩 -25° 肘 +35°", {"M2_should": -25, "M3_elbow": 35}),
    "03_fold": ("折叠：肩 -75° 肘 +110°", {"M2_should": -75, "M3_elbow": 110,
                                          "M4_wp": 60, "M5_wy": 40}),
    "04_pillar": ("撞立柱：肩 +80° 肘 +100°", {"M2_should": 80, "M3_elbow": 100}),
}

RED = np.array([0.85, 0.15, 0.15, 1.0], dtype=np.float32)
GRAY = np.array([0.28, 0.31, 0.36, 1.0], dtype=np.float32)


def _load_font(size):
    """Windows 上找个能显示中文的字体；找不到就退回默认位图字体。"""
    from PIL import ImageFont
    for p in (r"C:\Windows\Fonts\msyh.ttc",        # 微软雅黑
              r"C:\Windows\Fonts\msyhbd.ttc",
              r"C:\Windows\Fonts\simhei.ttf",      # 黑体
              r"C:\Windows\Fonts\simsun.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_cam(az=142.0, el=-20.0, dist=0.95, look=(0.17, 0.03, 0.19)):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = az
    cam.elevation = el
    cam.distance = dist
    cam.lookat[:] = look
    return cam


def highlight(m, d):
    """把当前参与碰撞的连杆 geom 染红，返回撞到的名字列表。"""
    base = {g: m.geom_rgba[g].copy() for g in range(m.ngeom)}
    hit = []
    for i in range(d.ncon):
        c = d.contact[i]
        for g in (c.geom1, c.geom2):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "?"
            if nm.startswith("g_"):
                m.geom_rgba[g] = RED
                hit.append(nm[2:])
    return hit, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "arm_real.xml"))
    d = mujoco.MjData(m)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    renderer = mujoco.Renderer(m, 600, 800)
    cam = make_cam()

    from PIL import Image, ImageDraw

    tiles = []
    for tag, (desc, pose) in POSES.items():
        d.qpos[:] = 0.0
        for jn, ang in pose.items():
            d.qpos[names.index(jn)] = math.radians(ang)
        mujoco.mj_forward(m, d)
        hit, base = highlight(m, d)
        renderer.update_scene(d, camera=cam)
        img = renderer.render().copy()
        for g, rgba in base.items():
            m.geom_rgba[g] = rgba

        fn = os.path.join(OUT, "shot_%s.png" % tag)
        Image.fromarray(img).save(fn)
        tiles.append((img, desc, hit))
        print("%-12s %-26s 撞: %s" % (tag, desc, ", ".join(hit) if hit else "无"))

    # 2x2 拼图
    W, H = tiles[0][0].shape[1], tiles[0][0].shape[0]
    sheet = Image.new("RGB", (W * 2, H * 2), (22, 24, 28))
    dr = ImageDraw.Draw(sheet)
    font = _load_font(17)
    for k, (img, desc, hit) in enumerate(tiles):
        x, y = (k % 2) * W, (k // 2) * H
        sheet.paste(Image.fromarray(img), (x, y))
        txt = desc + ("   撞: " + ", ".join(hit) if hit else "   安全")
        tw = dr.textbbox((0, 0), txt, font=font)[2]
        dr.rectangle([x + 8, y + 8, x + 24 + tw, y + 40], fill=(16, 18, 22))
        dr.text((x + 16, y + 14), txt, font=font,
                fill=(255, 118, 108) if hit else (170, 235, 175))
    sheet_fn = os.path.join(OUT, "sheet_poses.png")
    sheet.save(sheet_fn)
    print("\n拼图 -> %s" % sheet_fn)

    if args.video:
        video_demo(m, d, names, renderer, cam)

    renderer.close()


def video_demo(m, d, names, renderer, cam):
    import imageio.v2 as imageio
    frames = []
    # 扫 M2/M3 组合，覆盖从伸直、折叠到撞立柱
    seq = []
    for t in np.linspace(0, 1, 90):
        seq.append({"M2_should": -80 + 170 * t, "M3_elbow": 20 + 90 * math.sin(math.pi * t)})
    for pose in seq:
        d.qpos[:] = 0.0
        for jn, ang in pose.items():
            d.qpos[names.index(jn)] = math.radians(ang)
        mujoco.mj_forward(m, d)
        hit, base = highlight(m, d)
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render().copy())
        for g, rgba in base.items():
            m.geom_rgba[g] = rgba
    fn = os.path.join(OUT, "arm_scan.mp4")
    imageio.mimsave(fn, frames, fps=30, quality=8)
    print("视频 -> %s (%d 帧)" % (fn, len(frames)))


if __name__ == "__main__":
    main()
