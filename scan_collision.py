# -*- coding: utf-8 -*-
"""自碰撞扫描：把 7 个关节的可行空间扫一遍，找出会撞自己的姿态。

两个层次的扫描：
  1) 随机全局采样 —— 快速估出「整个关节空间里有多大比例是危险的」
  2) 成对关节 2D 网格 —— 对最容易撞的关节对画热力图，能直接看出禁区形状

「撞」分三类，分别记：
  * self  —— 两根不相邻连杆互撞（你说要查的）
  * table —— 连杆砸到桌面
  * base  —— 撞底座圆盘

用法：
    python scan_collision.py                 # 默认 20 万次随机 + 3 对热力图
    python scan_collision.py --n 500000
"""
import argparse
import itertools
import math
import os

import numpy as np

import arm_params as P

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))


def load(backlash=None):
    path = os.path.join(HERE, "arm_real.xml")
    if not os.path.exists(path):
        raise SystemExit("先跑 build_mjcf.py 生成模型")
    return mujoco.MjModel.from_xml_path(path)


def classify(m, d):
    """看当前姿态撞了谁。返回 (类别, 说明) 或 None。"""
    if d.ncon == 0:
        return None
    hits = []
    for i in range(d.ncon):
        c = d.contact[i]
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?"
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?"
        # 只关心连杆胶囊之间、或连杆与桌面/底座
        if "g_" not in n1 and "g_" not in n2:
            continue
        pair = tuple(sorted([n1, n2]))
        if n1 in ("table", "floor", "_") or n2 in ("table", "floor", "_"):
            hits.append(("table", pair))
        elif n1.startswith("base_") or n2.startswith("base_"):
            # base_plate（方形底板）/ base_drum（舵机舱圆柱）/ base_turntable（转盘）
            hits.append(("base", pair))
        else:
            hits.append(("self", pair))
    if not hits:
        return None
    order = {"self": 0, "base": 1, "table": 2}
    hits.sort(key=lambda h: order[h[0]])
    return hits[0]


def random_sweep(m, n=200000, seed=0):
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    lo = m.jnt_range[:, 0]
    hi = m.jnt_range[:, 1]
    nq = m.nq
    stats = {"self": 0, "base": 0, "table": 0, "free": 0}
    pairs = {}
    worst = []
    for k in range(n):
        d.qpos[:nq] = rng.uniform(lo, hi)
        mujoco.mj_forward(m, d)
        r = classify(m, d)
        if r is None:
            stats["free"] += 1
            continue
        kind, pair = r
        stats[kind] += 1
        key = "%s | %s" % pair
        pairs[key] = pairs.get(key, 0) + 1
        if len(worst) < 400:
            worst.append((kind, key, d.qpos[:nq].copy()))
    return stats, pairs, worst


def pair_grid(m, ia, ib, ns=61):
    """固定其它关节在 0，扫 ia×ib 的 2D 网格，返回危险掩码。"""
    d = mujoco.MjData(m)
    lo = m.jnt_range[:, 0]
    hi = m.jnt_range[:, 1]
    ga = np.linspace(lo[ia], hi[ia], ns)
    gb = np.linspace(lo[ib], hi[ib], ns)
    mask = np.zeros((ns, ns), dtype=np.int8)
    for yi, vb in enumerate(gb):
        for xi, va in enumerate(ga):
            d.qpos[:] = 0.0
            d.qpos[ia] = va
            d.qpos[ib] = vb
            mujoco.mj_forward(m, d)
            r = classify(m, d)
            if r is not None:
                mask[yi, xi] = 1 if r[0] == "self" else 2
    return ga, gb, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--grids", type=int, default=3)
    ap.add_argument("--outdir", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    m = load()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    print("关节: %s" % ", ".join(names))
    print("随机采样 %d 个姿态..." % args.n)

    stats, pairs, worst = random_sweep(m, args.n)
    tot = sum(stats.values())
    print("-" * 60)
    print("安全姿态   %7d  (%.1f%%)" % (stats["free"], stats["free"] / tot * 100))
    print("自碰撞     %7d  (%.1f%%)" % (stats["self"], stats["self"] / tot * 100))
    print("撞底座     %7d  (%.1f%%)" % (stats["base"], stats["base"] / tot * 100))
    print("撞桌面     %7d  (%.1f%%)" % (stats["table"], stats["table"] / tot * 100))
    print("-" * 60)
    print("最常见的碰撞对（前 12）：")
    for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])[:12]:
        print("  %-34s %6d 次  (%.1f%%)" % (k, v, v / tot * 100))

    # 关节对热力图：优先扫随机结果里出现最多的关节对
    print("-" * 60)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = []
    for key, _cnt in sorted(pairs.items(), key=lambda kv: -kv[1]):
        b1, b2 = [s.strip() for s in key.split("|")]
        if b1.startswith("g_") and b2.startswith("g_"):
            i1 = int(b1[2:].split("_")[0][1:]) - 1   # g_L3_fore -> 2
            i2 = int(b2[2:].split("_")[0][1:]) - 1
            if 0 <= i1 < m.njnt and 0 <= i2 < m.njnt and i1 != i2:
                p = tuple(sorted((i1, i2)))
                if p not in picks:
                    picks.append(p)
        if len(picks) >= args.grids:
            break
    if not picks:
        picks = [(1, 2), (2, 3), (3, 5)]

    for ia, ib in picks[: args.grids]:
        ga, gb, mask = pair_grid(m, ia, ib)
        fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=110)
        ax.imshow(mask, origin="lower", aspect="auto",
                  extent=[math.degrees(ga[0]), math.degrees(ga[-1]),
                          math.degrees(gb[0]), math.degrees(gb[-1])],
                  cmap=matplotlib.colors.ListedColormap(["#2f7d3a", "#c0392b", "#c9a227"]),
                  vmin=0, vmax=2, interpolation="nearest")
        ax.set_xlabel("%s (deg)" % names[ia])
        ax.set_ylabel("%s (deg)" % names[ib])
        ax.set_title("%s vs %s   green=safe  red=self-hit  gold=table" % (names[ia], names[ib]),
                     fontsize=9)
        fn = os.path.join(args.outdir, "collide_%s_%s.png" % (names[ia], names[ib]))
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)
        print("热力图 -> %s" % fn)

    # 保存几个具体危险姿态，方便复现
    fn = os.path.join(args.outdir, "danger_poses.txt")
    with open(fn, "w", encoding="utf-8") as f:
        f.write("# 危险姿态样例（qpos 弧度，顺序同关节表）\n")
        for kind, key, q in worst[:200]:
            f.write("%s\t%s\t%s\n" % (kind, key, " ".join("%.4f" % v for v in q)))
    print("危险姿态样例 -> %s (%d 条)" % (fn, min(200, len(worst))))


if __name__ == "__main__":
    main()
