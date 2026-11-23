# -*- coding: utf-8 -*-
"""从 arm_params.py 生成 MuJoCo 模型。

用法：
    python build_mjcf.py            # 生成 arm_real.xml 并自检
    python build_mjcf.py --backlash 0.87   # 开启齿隙，参数为总回差度数

设计要点：
  * 胶囊体当碰撞体 —— 自碰撞是厘米级问题，不需要精确外形，也就不用导 STL / 做凸包分解
  * fromto 端点间距自动扣掉两个半球，保证胶囊总长 = 连杆轴心距（否则每段虚长 2r，七段累加会严重误报）
  * 相邻连杆 MuJoCo 默认不检测碰撞，正好只查「不相邻的两段」
  * 一节多段（拐弯）：LINKS 里插 None 当折点占位，后继段自动变成上一段的**子体**，
    共用同一个关节，几何上连成折线。见 arm_params.py 的「拐弯拆分」注释。
"""
import argparse
import math
import os
import sys

import arm_params as P

AXIS = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}


def deg(d):
    return d * math.pi / 180.0


def _norm(vec):
    return math.sqrt(sum(c * c for c in vec))


def _sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def capsule_endpoints(L, r):
    """胶囊两个球心，沿连杆方向、相对 body 原点。球心距 = L - 2r（r 已夹过）。"""
    if L <= 1e-9:
        return None
    u = None
    return (L / 2.0 - r, L / 2.0 + r)


def capsule_geom(vec, radius, name, mass, rgba=None):
    """生成一个从 body 原点到 vec 的胶囊。总长度 = |vec|，半径收进两端半球里。

    ★ 半径夹取改为「按真实截面收，收到就大声报警」——
      旧版静默 r = min(radius, 0.45L)，把轴心距 43.86mm 的腕部段砍成 19.74mm（填入 26），
    碰撞率被悄悄放大。注意这里的 L 是未取整的轴心距，不是文档里显示的 44mm。
    """
    L = _norm(vec)
    r = radius
    warn = None
    if L > 1e-9 and r > 0.45 * L:
        warn = "L 太短(r 被夹到 0.45L)"
        r = 0.45 * L
    if L <= 1e-9:
        # 退化成长度为 0 的球，仍保留 r
        attrs = ['type="sphere"', 'pos="0 0 0"', 'size="%.5f"' % r, 'name="%s"' % name]
        if rgba:
            attrs.append('rgba="%s"' % rgba)
        attrs.append('mass="%.4f"' % mass)
        return "<geom %s/>" % " ".join(attrs), r, warn or "L=0 退化成球"
    u = [c / L for c in vec]
    # ★ 修 bug（2026-10-22）：原来写成 half = L/2 - r，导致球心距 = L-2r 而不是
    #   想要的 L-2r... 实际算出来球心距 = 2*(L/2-half) = 2r，几何缩成 r 附近一小段，
    #   整条链看起来"断开"。正确关系：
    #     胶囊总长 = 球心距 + 2r，要总长 = L  =>  球心距 = L - 2r  =>  球心偏移 ±(L/2 - r)
    half = L / 2.0 - r            # 球心相对 body 原点的偏移量
    a = [u[i] * (L / 2.0 - half) for i in range(3)]   # = u * r
    b = [u[i] * (L / 2.0 + half) for i in range(3)]   # = u * (L - r)
    attrs = [
        'type="capsule"',
        'contype="%d"' % (1 if getattr(P, "COLLISION_ON", True) else 0),
        'conaffinity="%d"' % (1 if getattr(P, "COLLISION_ON", True) else 0),
        'fromto="%.5f %.5f %.5f %.5f %.5f %.5f"' % (*a, *b),
        'size="%.5f"' % r,
        'name="%s"' % name,
    ]
    if rgba:
        attrs.append('rgba="%s"' % rgba)
    attrs.append('mass="%.4f"' % mass)
    return "<geom %s/>" % " ".join(attrs), r, warn


def apply_link_basis(vec):
    """把【侧视平面坐标】(x=沿臂伸出, y=向上, z=面外) 转到 MuJoCo 连杆坐标 (y=向上)。

    学长描述尺寸的方式是看侧视图："先水平走 150，再向上走 20"。
    而机械臂的关节轴统一是 y（pitch），所以"向上"必须是连杆局部系的 +y，
    "面外"才是 z。为了让他可以直接按侧视图写数据，这里统一做一次坐标映射：

        (x, y_up, z_out)  ->  (x, -z_out, y_up)

    对应旋转 Rx(-90deg)：向上映射到 y，面外映射到 -z。
    P.LINK_PLATE 控制开关，改成 "xy" 就退回旧的"直接当 MuJoCo 坐标写"。
    """
    if getattr(P, "LINK_PLATE", "xz") != "xz":
        return vec
    x, y_up, z_out = vec
    return (x, -z_out, y_up)


def apply_plate_rot(vec, rot):
    """在【侧视平面坐标】里再做一次绕 z 轴的旋转，用于把舵机朝向对齐到 M4 局部系。

    rot 是四元数；(x=沿臂, y=向上, z=面外) 的中心轴 = 世界 y。
    先绕世界 y 转一个角度，再走 apply_link_basis 的映射。
    """
    if not rot or abs(rot[0] - 1.0) < 1e-12:
        return vec
    w, qx, qy, qz = rot
    # 世界系向量
    wx, wy, wz = vec[0], -vec[2], vec[1]
    # 四元数旋转
    def cross(a, bb):
        return (a[1]*bb[2]-a[2]*bb[1], a[2]*bb[0]-a[0]*bb[2], a[0]*bb[1]-a[1]*bb[0])
    q = (qx, qy, qz)
    t = cross(q, (wx, wy, wz))
    t = (2*t[0], 2*t[1], 2*t[2])
    qx_t = cross(q, t)
    rx, ry, rz = wx+t[0]+w*qx_t[0], wy+t[1]+w*qx_t[1], wz+t[2]+w*qx_t[2]
    # 转回侧视平面坐标
    return (rx, rz, -ry)


def servo_rot_for(jname):
    rot = getattr(P, "SERVO_BASIS_ROT", {})
    return rot.get(jname, None)


def parse_chain(links):
    """把 LINKS 解析成「节 -> 若干段」。

    规则：LINKS 里 None 表示折点占位 —— 下一段不新增关节，接在上一段末尾。
    返回 [{'joint_idx': i, 'segs': [(name, vec, r, m, src), ...]}, ...]
      joint_idx 指向 JOINTS 的下标；折点段沿用上一段的 joint_idx。
    """
    chains = []
    pending_join = False       # True = 下一段属于当前节，不新开关节
    for item in links:
        if item is None:
            if not chains:
                raise ValueError("LINKS 第一个元素不能是 None（没有可以接的上一段）")
            pending_join = True
            continue
        name, vec, radius, mass, src = item
        if pending_join and chains:
            chains[-1]["segs"].append((name, vec, radius, mass, src))
        else:
            chains.append({"joint_idx": len(chains),
                           "segs": [(name, vec, radius, mass, src)]})
        pending_join = False
    return chains


# 渲染配色（学长要求：杆=灰银、舵机=深色、延长臂=橙红）
EXT_RGBA = "0.90 0.35 0.10 1"       # 延长臂：橙红，醒目
SERVO_RGBA_PLATE = "0.16 0.17 0.20 1"  # 舵机外壳：深色
ROD_RGBA = "0.72 0.74 0.78 1"       # 铝杆：银灰
ROD_RGBA_EST = "0.55 0.57 0.62 1"   # 还是估计值的杆：稍暗，一眼能看出哪些没量


def ext_arm_geoms(jname, hs, jaxis, indent, servo_off=(0.0, 0.0, 0.0)):
    """舵机延长臂 = 直角 U 型件：1 条底 + 2 条叉臂，共 3 个 box。

    学长最终口径：
      "41mm 是 U 的宽度，26mm 是 U 的深度，两条 20mm 的边夹住舵机的两个 45x25 的面。"
      "舵机那两个面之间是35毫米的距离。U 的外宽是41毫米，U 的内宽是37毫米。"

    推导（全部自洽）：
      臂厚 = (41-37)/2 = 2mm
      舵机 35 塞进内宽 37 → 每边 1mm 间隙

    body 局部坐标（关节原点 = 舵机输出轴中心）：
        x = 沿臂伸出，y = 左右（U 的夹持方向），z = 舵机输出轴

    俯视 x-y：
             y+  外宽 41
        +----------------------+  上叉臂（厚 2，高 20）
        |  +------------+      |  内宽 37
        |  |  舵机 35   | 1mm  |
        |  +------------+      |
        +----------------------+  下叉臂
        ^底部(贴杆)   叉臂沿 x 深 26
    """
    L = []
    if jaxis != "y":
        return L
    th = P.EXT_ARM_T              # 臂厚 2mm（沿 y）
    arm_w = P.EXT_ARM_W           # 臂高 20mm（沿 z）
    depth = P.EXT_DEPTH           # 深 26mm（沿 x）
    outer = P.EXT_OUTER_W         # 外宽 41mm（沿 y）
    inner = P.EXT_INNER_W         # 内宽 37mm（沿 y）
    z_half = arm_w / 2.0
    x0 = -depth                   # 底部在 -x 侧（贴杆那边）
    base_t = 0.003                # 底部板厚 3mm（学长没给，假设）

    ox, oy, oz = servo_off     # 输出轴在面外的位置被学长定位后，U 型臂要跟着舵机本体走
    # ① 底部：沿 y 覆盖整个外宽，沿 z 高 20，沿 x 厚 base_t
    L.append('%s<geom name="ext_%s_base" type="box" size="%.5f %.5f %.5f" '
             'pos="%.5f %.5f %.5f" rgba="%s" contype="0" conaffinity="0"/>'
             % (indent, jname, base_t / 2.0, outer / 2.0, z_half,
                x0 + base_t / 2.0 + ox, oy, oz, EXT_RGBA))
    # ② 两条叉臂：外表面在 +-outer/2，内表面在 +-inner/2，臂厚 2mm
    for sign, tag in ((1, "upper"), (-1, "lower")):
        cy = sign * (inner / 2.0 + th / 2.0)
        L.append('%s<geom name="ext_%s_%s" type="box" size="%.5f %.5f %.5f" '
                 'pos="%.5f %.5f %.5f" rgba="%s" contype="0" conaffinity="0"/>'
                 % (indent, jname, tag, depth / 2.0, th / 2.0, z_half,
                    x0 + base_t + depth / 2.0 + ox, cy + oy, oz, EXT_RGBA))
    return L



def servo_offset(jname, jaxis):
    """舵机本体就在关节原点上（关节 = 舵机输出轴中心）。延长臂挂在舵机外侧。"""
    return (0.0, 0.0, 0.0)


def mass_of(chain):
    """一节的总质量（不含舵机本体，舵机质量另算）。"""
    return sum(seg[3] for seg in chain["segs"])


def servo_clamped_dim(jname, jaxis):
    """舵机在「U 夹持方向」上的实际尺寸。

    SERVO_AIM 里 M4 = (长 45, 宽 25, 厚 35)。
    build 里 jaxis=="y" 的排布是 hs = (长/2, 厚/2, 宽/2)，
    所以 y 方向（U 的夹持方向）占的是"厚"= 35mm。
    """
    if jname in P.SERVO_AIM:
        sl, sw, st = P.SERVO_AIM[jname]
    else:
        sl, sw, st = P.SERVO_L, P.SERVO_W, P.SERVO_T
    return st if jaxis == "y" else sw



def build(backlash_deg=None, collision=True):
    bl_joints = set(P.BACKLASH_JOINTS) if backlash_deg else set()
    chains = parse_chain(P.LINKS)
    if len(chains) != len(P.JOINTS):
        raise ValueError("连杆节数(%d) != 关节数(%d)：检查 arm_params.LINKS 里的 None 折点是否插错位置"
                         % (len(chains), len(P.JOINTS)))
    report = []   # (名字, 填的 r, 生效 r, 长度, 警告)
    L = []
    w = L.append

    w('<?xml version="1.0" encoding="utf-8"?>')
    w('<mujoco model="arm_real">')
    w('  <compiler angle="radian" autolimits="true"/>')
    w('  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>')
    w('  <visual>')
    w('    <global offwidth="1600" offheight="1100"/>')
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
    # COLLISION_ON=False 时全模型关碰撞：只建骨架，不判撞
    _on = P.COLLISION_ON
    w('  <worldbody>')
    if P.SHOW_TABLE:
        # 只留桌面，不留无限大地板 —— 两个都在 z=0 会互相抢碰撞，判定会串到 floor 上
        w('    <geom name="table" type="box" size="%.3f %.3f 0.005" pos="0 0 -0.005" '
          'rgba="0.72 0.68 0.60 1" contype="%s" conaffinity="%s"/>'
          % (P.TABLE_HALF, P.TABLE_HALF,
             '1' if _on else '0', '1' if _on else '0'))
    ca = '1' if (P.BASE_COLLIDE and _on) else '0'
    pa = '1' if (P.PLATE_COLLIDE and _on) else '0'
    w('    <body name="base" pos="0 0 0">')
    # 方形底板：160 x 160 x 7mm，躺在桌面上
    w('      <geom name="base_plate" type="box" size="%.4f %.4f %.4f" pos="0 0 %.4f" '
      'rgba="0.24 0.24 0.26 1" contype="%s" conaffinity="%s"/>'
      % (P.BASE_PLATE_L / 2.0, P.BASE_PLATE_W / 2.0, P.BASE_PLATE_T / 2.0,
         P.BASE_PLATE_T / 2.0, pa, pa))
    # 舵机舱圆柱：坐在底板上，M1 舵机整个躺在这个筒子里
    w('      <geom name="base_drum" type="cylinder" size="%.4f %.4f" pos="0 0 %.4f" '
      'rgba="0.20 0.20 0.22 1" contype="%s" conaffinity="%s"/>'
      % (P.BASE_R, P.BASE_H / 2.0, P.BASE_PLATE_T + P.BASE_H / 2.0, ca, ca))

    # ★ 第一节起点要抬到底座圆盘顶面之上，否则立柱胶囊插在圆盘里 -> 100% 撞底座
    # M2 支点：圆柱顶面之上 M2_LIFT_Z，再横向偏 M2_OFFSET
    _z = P.BASE_PLATE_T + P.BASE_H + (P.TURNTABLE_T if P.TURNTABLE_ON else 0.0) + P.M2_LIFT_Z
    _ox = P.M2_OFFSET if P.M2_OFFSET_DIR == "x" else 0.0
    _oy = P.M2_OFFSET if P.M2_OFFSET_DIR == "y" else 0.0
    body_pos = (_ox, _oy, _z)
    indent = '      '
    servo_rgba = SERVO_RGBA_PLATE
    for chain in chains:
        ji = chain["joint_idx"]
        jname, jaxis, lo, hi, model, torque, speed, jsrc = P.JOINTS[ji]
        is_bl = jname in bl_joints
        first_name = chain["segs"][0][0]
        w('%s<body name="%s" pos="%.5f %.5f %.5f">' % (indent, first_name, *body_pos))
        if chain is chains[0] and P.TURNTABLE_ON:
            ta = '1' if P.TURNTABLE_COLLIDE else '0'
            w('%s  <geom name="base_turntable" type="cylinder" size="%.4f %.4f" pos="0 0 %.4f" '
              'rgba="0.26 0.27 0.30 1" contype="%s" conaffinity="%s"/>'
              % (indent, P.TURNTABLE_R, P.TURNTABLE_T / 2.0, -P.TURNTABLE_T / 2.0, ta, ta))
        w('%s  <joint name="%s" axis="%s" range="%.5f %.5f" class="servo"/>'
          % (indent, jname, AXIS[jaxis], deg(lo), deg(hi)))
        if is_bl:
            w('%s  <joint name="%s_bl" axis="%s" class="bl"/>' % (indent, jname, AXIS[jaxis]))
        if all(seg[2] <= 0.0 for seg in chain["segs"]) and \
           not (P.SERVO_GEOM_ON and jname not in P.SERVO_SKIP):
            # 整节没有几何，body 会没有质量/惯量 → MuJoCo 直接报错，补一个惯性张量
            w('%s  <inertial pos="0 0 0" mass="%.4f" diaginertia="1e-4 1e-4 1e-4"/>'
              % (indent, mass_of(chain)))
        # 舵机本体：每个关节都是一个 40x40x20 的方盒（学长实测）。
        # 厚度方向沿关节轴 —— M4/M5/M6 的轴是 y，所以 y 半长用 SERVO_T。
        if P.SERVO_GEOM_ON and jname not in P.SERVO_SKIP:
            if jname in getattr(P, "SERVO_ORIENT", {}):
                # 已经在 SERVO_ORIENT 里按"哪个面朝哪"写死了世界轴上的尺寸
                sx, sy, sz = P.SERVO_ORIENT[jname]
                hs = (sx / 2.0, sy / 2.0, sz / 2.0)
            else:
                sl, sw, st = P.SERVO_AIM.get(jname, (P.SERVO_L, P.SERVO_W, P.SERVO_T))
                if jaxis == "y":
                    hs = (sl / 2.0, st / 2.0, sw / 2.0)
                elif jaxis == "x":
                    hs = (st / 2.0, sl / 2.0, sw / 2.0)
                else:
                    hs = (sl / 2.0, sw / 2.0, st / 2.0)
            # 输出轴位置由学长用【面局部坐标系】给出：(10, 12.5) 在 45x25 面上。
            # 关节原点 = 输出轴，所以舵机盒中心要相对关节原点偏回去：
            #   盒中心在面局部坐标里是 (45/2, 25/2) = (22.5, 12.5)
            #   偏移(面局部) = 盒中心 - 输出轴 = (22.5-10, 12.5-12.5) = (+12.5, 0)
            _off = (0.0, 0.0, 0.0)
            if getattr(P, "SERVO_SHAFT_OFFSET_ON", False):
                _m = getattr(P, "SHAFT_AXIS_MAP", {}).get(jname)
                if _m:
                    _a45, _s45, _a25, _s25 = _m
                    _l45 = 0.045; _l25 = 0.025        # 45x25 面的两条边
                    _d45 = (_l45 / 2.0 - P.SHAFT_UV_45) * _s45
                    _d25 = (_l25 / 2.0 - P.SHAFT_UV_25) * _s25
                    _v = [0.0, 0.0, 0.0]
                    _v["xyz".index(_a45)] += _d45
                    _v["xyz".index(_a25)] += _d25
                    _off = tuple(_v)
            # 大舵机（宽40 x 厚20 x 高40）：输出轴在两端的 40x20 面上，
            # 沿"长 40"方向距 0 边 BIG_SHAFT_UV_40，沿"厚 20"方向居中。
            # 关节原点 = 输出轴，所以盒中心要朝"远离 0 边"一侧偏 (40/2 - 10)。
            if getattr(P, "BIG_SHAFT_OFFSET_ON", False):
                _bm = getattr(P, "BIG_SHAFT_MAP", {}).get(jname)
                if _bm:
                    _a40, _s40 = _bm
                    _d40 = (0.040 / 2.0 - P.BIG_SHAFT_UV_40) * _s40
                    _v = [0.0, 0.0, 0.0]
                    _v["xyz".index(_a40)] += _d40
                    _off = (_off[0] + _v[0], _off[1] + _v[1], _off[2] + _v[2])
            _so = servo_offset(jname, jaxis)
            _so = (_so[0] + _off[0], _so[1] + _off[1], _so[2] + _off[2])
            # 舵机盒的世界朝向直接由 SERVO_ORIENT 的 (沿x,沿y,沿z) 决定，
            # 所以这里【不加 quat】——加了会和 body 自己的朝向叠加，把盒子转到不知哪去。
            w('%s  <geom name="servo_%s" type="box" size="%.4f %.4f %.4f" '
              'pos="%.4f %.4f %.4f" rgba="%s" contype="0" conaffinity="0"/>'
              % (indent, jname, hs[0], hs[1], hs[2], _so[0], _so[1], _so[2],
                 servo_rgba))
            if P.EXT_ON and jname in P.EXT_JOINTS:
                for line in ext_arm_geoms(jname, hs, jaxis, indent, _off):
                    w(line)
            # 双轴标记：飞特 STS3215 官方全名就写着 "dual-axis"，两个输出轴在相对的两个面上同轴贯穿。
            # 在舵机盒两端各画一个小圆柱，位置 = 该方向的半尺寸外表面，方便学长确认轴向。
            if getattr(P, "SERVO_DUAL_AXIS_ON", True):
                _sa = getattr(P, "SERVO_AXIS", {}).get(jname)
                if _sa:
                    # 双轴：一根细圆柱【贯穿】舵机本体，两端各凸出 SHAFT_LEN
                    _ai = {"x": 0, "y": 1, "z": 2}[_sa]
                    _half = hs[_ai]
                    _hlen = _half + P.SHAFT_LEN          # 半长 = 舵机半厚 + 凸出
                    _eul = {0: "0 1.5708 0", 1: "-1.5708 0 0", 2: "0 0 0"}[_ai]
                    w('%s  <geom name="shaft_%s" type="cylinder" size="%.5f %.5f" '
                      'pos="0 0 0" euler="%s" rgba="0.95 0.80 0.20 1" '
                      'contype="0" conaffinity="0"/>'
                      % (indent, jname, P.SHAFT_R, _hlen, _eul))
            # 关节 body 至少要有点质量/惯量，否则 MuJoCo 报 inertia 太小
            w('%s  <inertial pos="0 0 0" mass="%.4f" '
              'diaginertia="%.6f %.6f %.6f"/>'
              % (indent, max(mass_of(chain), 0.01),
                 1e-5, 1e-5, 1e-5))

        # M4 -> M5 的连接件：学长说 45x35 面和 45x25 面之间相距 18mm，
        # 真实结构里这里必须有个过渡件（支架/转接块），否则舵机悬空。
        # 方向：沿关节轴方向（世界 y）看是两根立柱，竖直方向跨过那 18mm 间隙。
        if jname == "M5_wy" and getattr(P, "M45_LINK_ON", True):
            # M5 body 原点 = M5 舵机中心（世界 z = M5中心）。
            # M4 舵机中心相对 M5 是 (+9, +2.5, +43)mm —— 即"上 43、右 9、面外 -2.5"。
            # 连接件要跨越这 18mm 竖直间隙：从 M5 顶面(z=+12.5) 到 M4 底面(z=+30.5)。
            _gap = P.M45_GAP
            _htop = hs[2]                      # M5 半厚 12.5 -> 顶面
            _z0 = _htop                        # M5 顶面
            _zc = _z0 + _gap / 2.0             # 跨间隙的中点
            # M4 中心在 M5 局部系里是 +9mm(x)。立柱放在两者中间 = +4.5mm
            # 立柱位置跟着 M5 相对 M4 的偏移走（学长改过 M5 位置后自动适应）
            _dx = P.M45_LINK_X
            _dy = P.M45_LINK_Y
            for _s in (-1, 1):
                w('%s  <geom name="link_M4M5_%d" type="box" size="%.5f %.5f %.5f" '
                  'pos="%.5f %.5f %.5f" rgba="0.62 0.65 0.70 1" '
                  'contype="0" conaffinity="0"/>'
                  % (indent, _s,
                     P.ROD_R / 2.0, P.ROD_R / 2.0, (_gap / 2.0 + 0.004),
                     _dx + _s * P.ROD_R * 0.8, _dy, _z0 + _gap / 2.0))

        # 本节内的折线段：第 1 段从本节原点出发，后续段接上一段末尾
        seg_indent = indent + '  '
        cursor = (0.0, 0.0, 0.0)      # 当前 body 局部坐标下的段起点

        _rot = servo_rot_for(jname)
        # 第 0 段是该节自己的连杆（从本 body 原点出发），永远不转；
        # 后续段按该关节舵机的朝向旋转（学长口径：舵机朝向随关节一起翻）。
        for si, (name, vec, radius, mass, src) in enumerate(chain["segs"]):
            if si > 0 and _rot:
                vec = apply_link_basis(apply_plate_rot(vec, _rot))
            else:
                vec = apply_link_basis(vec)
            if si > 0:
                # 折点：新开一个 body，位置 = 上一段末端，无关节
                w('%s<body name="%s" pos="%.5f %.5f %.5f">' % (seg_indent, name, *cursor))
                seg_indent += '  '
            if radius <= 0.0:
                # 半径 0 = 这一节只是关节占位，不画碰撞体也不画外形
                r_eff, warn = 0.0, "r=0 不建几何（仅关节占位）"
                w('%s<!-- %s: 半径 0，仅承载关节，不建几何 -->' % (seg_indent, name))
            else:
                _src = chain["segs"][si][4]
                _col = ROD_RGBA if _src == "实测" else ROD_RGBA_EST
                geom, r_eff, warn = capsule_geom(vec, radius, "g_" + name, mass,
                                                 rgba=_col)
                w('%s%s' % (seg_indent, geom))
            report.append((name, radius, r_eff, _norm(vec), warn))
            cursor = tuple(cursor[i] + vec[i] for i in range(3))

        # ★ 修 bug（2026-10-22，学长反馈"手臂断开"的根因）：
        #   下一节的 body 挂在【本节最后那个 body】下面，所以它的 pos 应该只是
        #   "最后一段的向量"，而不是整节累积的 cursor。
        #   原来写 body_pos = cursor，多段节会把前面的段重复加一遍 ——
        #   上臂 150mm 被加了两次，M3 直接飞到 312mm，中间空出 160mm。
        body_pos = apply_link_basis(chain["segs"][-1][1]) if _rot is None else \
            apply_link_basis(apply_plate_rot(chain["segs"][-1][1], _rot))
        indent = seg_indent + '  '

    # 递归关闭所有 body（base + 每节 1 + 每节内折点）
    n_bodies = 1 + sum(len(c["segs"]) for c in chains)
    for _ in range(n_bodies):
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
    return "\n".join(L) + "\n", chains, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backlash", type=float, default=P.BACKLASH_DEG,
                    help="总回差度数，比如 0.87；不填则关闭")
    ap.add_argument("--out", default=P.XML_OUT)
    ap.add_argument("--collision", choices=["on", "off"], default=None,
                    help="覆盖 arm_params.COLLISION_ON：on=几何参与碰撞，off=全模型关闭")
    args = ap.parse_args()

    if args.collision is not None:
        P.COLLISION_ON = (args.collision == "on")

    print("配置：COLLISION_ON=%s  backlash=%s°  SHOW_TABLE=%s  SERVO_GEOM_ON=%s"
          % (P.COLLISION_ON, args.backlash, getattr(P, "SHOW_TABLE", True),
             getattr(P, "SERVO_GEOM_ON", True)))

    xml, chains, report = build(args.backlash)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    print("已生成 %s  (%.1f KB)" % (path, len(xml) / 1024.0))

    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print("自检：nbody=%d  nq=%d  nu=%d  ngeom=%d" % (m.nbody, m.nq, m.nu, m.ngeom))

    # ---------- 分段报告：每段的填充半径 / 生效半径 / 长度 ----------
    print("=" * 66)
    print("%-14s %8s %8s %8s   %s" % ("段名", "填r", "生效r", "长度", "备注"))
    print("-" * 66)
    for name, r_in, r_eff, L, warn in report:
        tag = ""
        if warn:
            tag = "<< " + warn
        elif abs(r_eff - r_in) > 1e-9:
            tag = "<< r 被夹"
        print("  %-14s %6.1fmm %6.1fmm %6.1fmm   %s"
              % (name, r_in * 1000, r_eff * 1000, L * 1000, tag))
    n_seg = len(report)
    n_chain = len(chains)
    bended = [c["segs"][0][0].split("_")[0] for c in chains if len(c["segs"]) > 1]
    print("-" * 66)
    print("共 %d 节 / %d 段；拐弯的节：%s" % (n_chain, n_seg, ", ".join(bended) if bended else "无（全直杆）"))

    # ---------- 舵机本体自检 ----------
    n_srv = 0
    for i in range(m.ngeom):
        nm = m.geom(i).name or ""
        if nm.startswith("servo_"):
            s = m.geom_size[i]
            n_srv += 1
            print("  %-20s %.0f x %.0f x %.0f mm" % (nm, s[0] * 2000, s[1] * 2000, s[2] * 2000))
    print("舵机实体共 %d 个（跳过：%s；M1 舵机体积由底座圆柱代表）"
          % (n_srv, ", ".join(sorted(P.SERVO_SKIP)) or "无"))

    # ---------- 延长臂自检 ----------
    if P.EXT_ON:
        for jn in P.EXT_JOINTS:
            ji = [jj[0] for jj in P.JOINTS].index(jn)
            jax = P.JOINTS[ji][1]
            servo_dim = servo_clamped_dim(jn, jax)
            inner = P.EXT_INNER_W
            gap = inner - servo_dim
            print("延长臂 %s 自检：" % jn)
            print("    U 外宽 = %.0f mm，内宽 = %.0f mm，臂厚 = (41-37)/2 = %.0f mm"
                  % (P.EXT_OUTER_W * 1000, inner * 1000, P.EXT_ARM_T * 1000))
            print("    舵机夹持维 = %.0f mm（45x25 两个面之间的距离）" % (servo_dim * 1000))
            print("    单边间隙 = (%.0f - %.0f)/2 = %.1f mm"
                  % (inner * 1000, servo_dim * 1000, gap / 2.0 * 1000))
            print("    深度 = %.0f mm（叉臂沿臂方向套住舵机的长度）" % (P.EXT_DEPTH * 1000))

    # ---------- 关节行程 ----------
    print("-" * 66)
    for i in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm.endswith("_bl"):
            continue
        lo, hi = m.jnt_range[i]
        print("  %-10s  %7.1f° ~ %7.1f°   (%.0f° 行程)"
              % (nm, math.degrees(lo), math.degrees(hi), math.degrees(hi - lo)))

    # ---------- 底座总成自检（学长给的底座尺寸对不对，一眼能看出来）----------
    print("-" * 66)
    z_plate = P.BASE_PLATE_T
    z_drum_top = P.BASE_PLATE_T + P.BASE_H
    print("底座  方形底板 %.0f x %.0f x %.0f mm，顶面 z=%.1f mm"
          % (P.BASE_PLATE_L * 1000, P.BASE_PLATE_W * 1000, P.BASE_PLATE_T * 1000, z_plate * 1000))
    print("      舵机舱圆柱 R=%.1f mm  H=%.0f mm，z=[%.1f, %.1f] mm"
          % (P.BASE_R * 1000, P.BASE_H * 1000, z_plate * 1000, z_drum_top * 1000))
    print("      M2 支点 = 圆柱顶 + %.1f mm，横向偏 %.1f mm(%s) -> z=%.1f mm"
          % (P.M2_LIFT_Z * 1000, P.M2_OFFSET * 1000, P.M2_OFFSET_DIR,
             (z_drum_top + P.M2_LIFT_Z) * 1000))

    # ---------- 零位几何 ----------
    tip_name = chains[-1]["segs"][-1][0]
    tip = m.body(tip_name).id
    p0 = d.xpos[tip].copy()
    print("-" * 66)
    print("零位末端 %s 位置: x=%.3f y=%.3f z=%.3f m" % (tip_name, *p0))
    for chain in chains:
        for name, vec, *_rest in chain["segs"]:
            b = m.body(name).id
            print("  %-14s 轴心 (%.3f, %.3f, %.3f)" % (name, *d.xpos[b]))
    print("-" * 66)
    print("零位身体总高（末端 z）: %.1f mm" % (p0[2] * 1000))
    if args.backlash:
        print("齿隙已开启：%.2f°（加在 %s）" % (args.backlash, ", ".join(P.BACKLASH_JOINTS)))
    else:
        print("齿隙关闭（刚性关节）")


if __name__ == "__main__":
    main()