# -*- coding: utf-8 -*-
"""精确的零件连接自检：用 mujoco 自己的距离函数 mj_geomDistance。"""
import numpy as np, mujoco, itertools

m = mujoco.MjModel.from_xml_path("arm_real.xml")
d = mujoco.MjData(m); mujoco.mj_forward(m, d)

def gid(nm):
    try: return m.geom(nm).id
    except Exception: return None

def dist(a, b):
    ga, gb = gid(a), gid(b)
    if ga is None or gb is None: return None
    fromto = np.zeros(6)
    dd = mujoco.mj_geomDistance(m, d, ga, gb, 1.0, fromto)
    return dd*1000

pairs = [
 ("base_plate","base_drum"),
 ("base_drum","servo_M2_should"),
 ("servo_M2_should","g_L2_upper_a"),
 ("g_L2_upper_a","servo_M3_elbow"),
 ("servo_M3_elbow","g_L3_fore_a"),
 ("g_L3_fore_a","g_L3_fore_b"),
 ("g_L3_fore_b","servo_M4_wp"),
 ("servo_M4_wp","ext_M4_wp_base"),
 ("ext_M4_wp_base","ext_M4_wp_upper"),
 ("ext_M4_wp_base","ext_M4_wp_lower"),
 ("ext_M4_wp_upper","servo_M4_wp"),
 ("ext_M4_wp_lower","servo_M4_wp"),
 # M4/M5 之间隔着学长说的 18mm 间隙，真实结构由过渡件连接，不直接接触
 ("servo_M4_wp","link_M4M5_1"),
 ("link_M4M5_1","servo_M5_wy"),
 ("servo_M4_wp","link_M4M5_-1"),
 ("link_M4M5_-1","servo_M5_wy"),
 ("servo_M4_wp","g_L4_wr1"),
 ("g_L4_wr1","servo_M5_wy"),
 ("servo_M5_wy","g_L5_wr2"),
 ("g_L5_wr2","servo_M6_wr"),
 ("servo_M6_wr","g_L6_wr3"),
 ("g_L6_wr3","servo_M7_tool"),
 ("servo_M7_tool","g_L7_tool"),
]
print("%-22s <-> %-22s %10s  %s" % ("A","B","距离mm","状态"))
print("-"*78)
bad = []
for a,b in pairs:
    dd = dist(a,b)
    if dd is None:
        print("%-22s <-> %-22s %10s  缺失" % (a,b,"-")); continue
    ok = dd < 1.0
    if not ok: bad.append((a,b,dd))
    print("%-22s <-> %-22s %10.2f  %s" % (a,b,dd,"OK" if ok else "!! 断开"))
print()
if bad:
    print("断开的有 %d 处：" % len(bad))
    for a,b,dd in bad: print("   %s <-> %s   %.1f mm" % (a,b,dd))
else:
    print("全部接触 ✓")
