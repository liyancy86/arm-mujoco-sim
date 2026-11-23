# -*- coding: utf-8 -*-
"""装配链自检：用 mujoco 自己的距离函数 mj_geomDistance 量 24 对相邻零件的表面间距。

判定口径（写在输出头部，避免"全绿"被当成"装配正确"）：
    负值  = 两个包围体互相穿透，|值| 是穿透深度
    0 附近 = 贴合
    正值  = 有缝，值就是缝宽

本脚本只回答一个问题：**这条装配链有没有断开。** 它不回答"装配是否正确"——
连杆用胶囊、舵机用长方体，都是简化体，穿透是这类简化必然产生的结果，
不代表真机干涉，也不代表真机不干涉。
"""
import numpy as np, mujoco

TOL_GAP = 1.0     # 缝宽超过 1mm 视为断开
TOL_PEN = -1.0    # 穿透深于 1mm 单独归为一类

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

print("模型: arm_real.xml   mujoco=%s" % mujoco.__version__)
print("口径: 负=穿透深度  0=贴合  正=缝宽   断开阈值=%.1fmm  深穿透阈值=%.1fmm" % (TOL_GAP, TOL_PEN))
print("边界: 连杆=胶囊  舵机=长方体，均为简化体；穿透是简化的产物，不等于真机干涉")
print()
print("%-22s <-> %-22s %10s  %s" % ("A","B","距离mm","分类"))
print("-"*80)

pen, touch, gap, missing = [], [], [], []
for a,b in pairs:
    dd = dist(a,b)
    if dd is None:
        missing.append((a,b)); print("%-22s <-> %-22s %10s  缺失" % (a,b,"-")); continue
    if dd < TOL_PEN:   cls, pen   = "穿透", pen + [(a,b,dd)]
    elif dd < TOL_GAP: cls, touch = "贴合", touch + [(a,b,dd)]
    else:              cls, gap   = "断开", gap + [(a,b,dd)]
    print("%-22s <-> %-22s %10.2f  %s" % (a,b,dd,cls))

print()
print("合计 %d 对：贴合 %d  穿透 %d  断开 %d  缺失 %d"
      % (len(pairs), len(touch), len(pen), len(gap), len(missing)))
if pen:
    worst = min(pen, key=lambda t: t[2])
    print("最深穿透 %.2f mm：%s <-> %s" % (worst[2], worst[0], worst[1]))
if gap:
    print("断开处：")
    for a,b,dd in gap: print("   %s <-> %s   %.2f mm" % (a,b,dd))
print()
print("结论只有一条：装配链%s。" % ("无断开" if not gap and not missing else "有断开，见上"))
print("穿透的 %d 对不构成\"装配正确\"的证据，也不构成\"真机干涉\"的证据。" % len(pen))
