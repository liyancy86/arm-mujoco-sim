# -*- coding: utf-8 -*-
"""从 arm_params.py 生成 MuJoCo 模型。

用法：
    python build_mjcf.py            # 生成 arm_real.xml 并自检
    python build_mjcf.py --backlash 0.87   # 开启齿隙，参数为总回差度数

设计要点：
  * 胶囊体当碰撞体 —— 自碰撞是厘米级问题，不需要精确外形，也就不用导 STL / 做凸包分解
  * fromto 端点间距自动扣掉两个半球，保证胶囊总长 = 连杆轴心距（否则每段虚长 2r，七段累加会严重误报）
  * 相邻连杆 MuJoCo 默认不检测碰撞，正好只查「不相邻的两段」
"""
import argparse
import math
import os
import sys

import arm_params as P

AXIS = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}


def deg(d):
    return d * math.pi / 180.0


def capsule_geom(vec, radius, name, rgba=None, mass=None):
    """生成一个从 body 原点到 vec 的胶囊。总长度 = |vec|，半径收进两端半球里。"""
    L = math.sqrt(sum(c * c for c in vec))
    # MuJoCo 的 capsule: fromto 定义两端球心，实际长度 = 球心距 + 2r
    # 想让总长 = L，球心距就得是 L - 2r；太短的段把半径压小，否则退化
    r = min(radius, 0.45 * L)
    half = L / 2.0 - r
    if half <= 1e-6:
        half = 0.0
    u = [c / L for c in vec]
    a = [u[i] * (L / 2.0 - half) for i in range(3)]
    b = [u[i] * (L / 2.0 + half) for i in range(3)]
    attrs = [
        'type="capsule"',
        'fromto="%.5f %.5f %.5f %.5f %.5f %.5f"' % (*a, *b),
        'size="%.5f"' % r,
        'name="%s"' % name,
    ]
    if rgba:
        attrs.append('rgba="%s"' % rgba)
    if mass:
        attrs.append('mass="%.4f"' % mass)
    return "<geom %s/>" % " ".join(attrs)


def build(backlash_deg=None, collision=True):
    bl_joints = set(P.BACKLASH_JOINTS) if backlash_deg else set()
    L = []
    w = L.append

    w('<?xml version="1.0" encoding="utf-8"?>')
    w('<mujoco model="arm_real">')
    w('  <compiler angle="radian" autolimits="true"/>')
    w('  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>')
    w('  <visual>')
    w('    <global offwidth="1280" offheight="960"/>')
    w('    <quality shadowsize="2048"/>')
    w('    <headlight ambient="0.38 0.38 0.40" diffuse="0.62 0.62 0.60" '
      'specular="0.12 0.12 0.12"/>')
    w('  </visual>')
    w('')
    w('  <default>')
    w('    <default class="servo">')
    w('      <joint damping="0.60" frictionloss="0.052" armature="0.028"/>')
    w('      <geom contype="1" conaffinity="1" rgba="0.28 0.31 0.36 1"/>')
    w('    </default>')
    if backlash_deg:
        half = deg(backlash_deg) / 2.0
        w('    <default class="bl">')
        w('      <joint damping="0.01" frictionloss="0" armature="0.01" limited="true" '
          'range="%.6f %.6f"/>' % (-half, half))
        w('      <geom contype="0" conaffinity="0" rgba="0.9 0.3 0.2 1"/>')
        w('    </default>')
    w('  </default>')
    w('')

    # ---------------- world ----------------
    w('  <asset>')
    w('    <texture name="skybox" type="skybox" builtin="gradient" '
      'rgb1="0.16 0.17 0.20" rgb2="0.06 0.06 0.08" width="256" height="1536"/>')
    w('  </asset>')
    w('  <worldbody>')
    if P.SHOW_TABLE:
        # 只留桌面，不留无限大地板 —— 两个都在 z=0 会互相抢碰撞，判定会串到 floor 上
        w('    <geom name="table" type="box" size="%.3f %.3f 0.005" pos="0 0 -0.005" '
          'rgba="0.72 0.68 0.60 1" contype="1" conaffinity="1"/>'
          % (P.TABLE_HALF, P.TABLE_HALF))
    q = '  <body name="base" pos="0 0 0">'
    w('    <body name="base" pos="0 0 0">')
    w('      <geom name="dish" type="cylinder" size="%.4f %.4f" pos="0 0 %.4f" '
      'rgba="0.20 0.20 0.22 1" contype="1" conaffinity="1"/>'
      % (P.BASE_DISH_R, P.BASE_DISH_H / 2.0, P.BASE_DISH_H / 2.0))

    indent = '      '
    # ★ L1 起点要抬到底座圆盘顶面之上，否则立柱胶囊插在圆盘里 -> 100% 撞底座（实测踩过）
    prev_vec = (0.0, 0.0, P.BASE_DISH_H + P.L1_LIFT)
    for i, (name, vec, radius, mass, src) in enumerate(P.LINKS):
        jname, jaxis, lo, hi, model, torque, speed, jsrc = P.JOINTS[i]
        is_bl = jname in bl_joints
        w('%s<body name="%s" pos="%.5f %.5f %.5f">' % (indent, name, *prev_vec))
        w('%s  <joint name="%s" axis="%s" range="%.5f %.5f" class="servo"/>'
          % (indent, jname, AXIS[jaxis], deg(lo), deg(hi)))
        if is_bl:
            w('%s  <joint name="%s_bl" axis="%s" class="bl"/>' % (indent, jname, AXIS[jaxis]))
        w('%s  %s' % (indent, capsule_geom(vec, radius, "g_" + name, mass=mass)))
        # 舵机壳（只做视觉，不参与碰撞）
        w('%s  <geom type="box" size="0.022 0.011 0.022" pos="0 0 0" '
          'rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0"/>' % indent)
        indent += '  '
        prev_vec = vec

    # 关闭最深一层（L7 后面没有子 body）
    for _ in range(len(P.LINKS) + 1):
        indent = indent[:-2]
        w('%s</body>' % indent)

    w('  </worldbody>')
    w('')

    # ---------------- actuators ----------------
    w('  <actuator>')
    for i, (jname, jaxis, lo, hi, model, torque, speed, jsrc) in enumerate(P.JOINTS):
        kp = 998.22 if "STS3215" in model else 500.0
        w('    <position name="a_%s" joint="%s" kp="%.1f" kv="%.2f" '
          'ctrlrange="%.5f %.5f" forcerange="%.4f %.4f"/>'
          % (jname, jname, kp, 2.731, deg(lo), deg(hi), -torque, torque))
    w('  </actuator>')
    w('</mujoco>')
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backlash", type=float, default=P.BACKLASH_DEG,
                    help="总回差度数，比如 0.87；不填则关闭")
    ap.add_argument("--out", default=P.XML_OUT)
    args = ap.parse_args()

    xml = build(args.backlash)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    print("已生成 %s  (%.1f KB)" % (path, len(xml) / 1024.0))

    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print("自检：nbody=%d  nq=%d  nu=%d  ngeom=%d" % (m.nbody, m.nq, m.nu, m.ngeom))

    # 关节行程
    print("-" * 58)
    for i in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        lo, hi = m.jnt_range[i]
        print("  %-10s  %7.1f° ~ %7.1f°   (%.0f° 行程)"
              % (nm, math.degrees(lo), math.degrees(hi), math.degrees(hi - lo)))

    # 零位末端位置
    tip = m.body("L7_tool").id
    p0 = d.xpos[tip].copy()
    print("-" * 58)
    print("零位末端 L7_tool 位置: x=%.3f y=%.3f z=%.3f m" % tuple(p0))
    # 各段轴心
    for i, (name, *_rest) in enumerate(P.LINKS):
        b = m.body(name).id
        print("  %-10s 轴心 (%.3f, %.3f, %.3f)" % (name, *d.xpos[b]))
    print("-" * 58)
    if args.backlash:
        print("齿隙已开启：%.2f°（加在 %s）" % (args.backlash, ", ".join(P.BACKLASH_JOINTS)))
    else:
        print("齿隙关闭（刚性关节）")


if __name__ == "__main__":
    main()
