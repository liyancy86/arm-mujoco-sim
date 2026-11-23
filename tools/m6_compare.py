# -*- coding: utf-8 -*-
"""M6 尺寸排布三方案对比：法向 z 的硬约束下，45/25/35 各放哪个轴。

不改 arm_params.py 源文件，只在内存里改 P 的字典，逐方案 build + 反查实测。
输出：out/m6_case_{A,B,C}.xml  +  out/m6_compare.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import mujoco
import arm_params as P
import build_mjcf as B

OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

# 学长约束：M6 输出轴沿 z；输出轴穿过 45x25 面 ⇒ 该面法向必须是 z。
# 25 维的偏心为 0（输出轴在 25 方向居中 12.5），所以只有 45 维的轴/符号影响盒位置。
CANDS = [
    ("C 现状", (0.045, 0.035, 0.025), ("x", -1, "y", +1),
     "45在x 25在z -> 45x25面法向 y  ≠ 关节轴 z  【矛盾】"),
    ("A", (0.045, 0.025, 0.035), ("x", -1, "y", +1),
     "45在x 25在y -> 45x25面法向 z  = 关节轴 z  【满足】"),
    ("B", (0.025, 0.045, 0.035), ("y", -1, "x", +1),
     "45在y 25在x -> 45x25面法向 z  = 关节轴 z  【满足】"),
]


def dist(m, d, a, b):
    try:
        ga, gb = m.geom(a).id, m.geom(b).id
    except Exception:
        return None
    return mujoco.mj_geomDistance(m, d, ga, gb, 1.0, np.zeros(6)) * 1000.0


print("=" * 96)
print("%-8s %-18s %-6s %-24s %-26s" % ("方案", "ORIENT(x,y,z)", "法向", "M6盒中心(世界mm)", "盒相对关节原点偏移mm"))
print("-" * 96)

rows = []
for tag, orient, mp, note in CANDS:
    P.SERVO_ORIENT["M6_wr"] = orient
    P.SHAFT_AXIS_MAP["M6_wr"] = mp
    xml, chains, report = B.build(None)
    path = os.path.join(OUT, "m6_case_%s.xml" % tag.split()[0])
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    mm = [round(v * 1000) for v in orient]
    a45 = "xyz"[mm.index(45)]
    a25 = "xyz"[mm.index(25)]
    normal = [a for a in "xyz" if a not in (a45, a25)][0]

    bx = m.geom("servo_M6_wr").id
    bc = d.geom_xpos[bx] * 1000
    half = m.geom_size[bx] * 1000
    jp = d.xpos[m.body("L6_wr3").id] * 1000
    off = bc - jp

    print("%-8s %-18s %-6s (%7.2f,%7.2f,%7.2f) %6s(%+6.2f,%+6.2f,%+6.2f)  半(%4.1f,%4.1f,%4.1f)"
          % (tag, str(tuple(mm)), normal, bc[0], bc[1], bc[2], "",
             off[0], off[1], off[2], half[0], half[1], half[2]))
    print("         %s" % note)

    # 腕部各关节世界坐标
    for jn, bn in (("M5", "L5_wr2"), ("M6", "L6_wr3"), ("M7", "L7_tool")):
        p = d.xpos[m.body(bn).id] * 1000
        print("           %s 关节轴心 (%7.2f,%7.2f,%7.2f)" % (jn, p[0], p[1], p[2]))

    # 相邻件距离
    d_m5 = dist(m, d, "servo_M5_wy", "servo_M6_wr")
    d_m7 = dist(m, d, "servo_M6_wr", "servo_M7_tool")
    d_r2 = dist(m, d, "g_L5_wr2", "servo_M6_wr")
    d_r3 = dist(m, d, "servo_M6_wr", "g_L6_wr3")
    print("           距离：M5盒<->M6盒 %s | M6盒<->M7盒 %s | 杆L5<->M6 %s | M6<->杆L6 %s"
          % tuple("  -.  " if v is None else "%6.2f" % v for v in (d_m5, d_m7, d_r2, d_r3)))
    print("-" * 96)
    rows.append((tag, orient, normal, bc, off, half, (d_m5, d_m7, d_r2, d_r3)))

# 还原源文件状态
import importlib
importlib.reload(P)
print("（arm_params.py 未被修改，以上均为内存试算）")
