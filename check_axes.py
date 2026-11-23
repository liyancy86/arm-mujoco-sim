# -*- coding: utf-8 -*-
"""全关节一致性自检：舵机输出轴 vs 关节轴。学长规矩：不重合就是 bug。"""
import arm_params as P

print("%-12s %-8s %-8s %-10s %s" % ("关节", "关节轴", "输出轴", "是否重合", "舵机实体"))
print("-" * 66)
bad = []
for jn, jax, lo, hi, *_ in P.JOINTS:
    sa = P.SERVO_AXIS.get(jn)
    built = "不建" if jn in P.SERVO_SKIP else "建"
    if sa is None:
        status = "—（无舵机实体）"
    elif sa == jax:
        status = "OK"
    else:
        status = "NO !! 不重合"
        bad.append(jn)
    print("%-12s %-8s %-8s %-10s %s" % (jn, jax, sa or "-", status, built))
print("-" * 66)
if bad:
    print("不重合：%s" % ", ".join(bad))
else:
    print("全部重合 ✓")

# ---- 附加检查：飞特舵机的 45x25 面法向 是否等于关节轴 ----
# 输出轴穿过 45x25 面，所以该面法向必须就是关节轴
print()
print("飞特舵机 45x25 面法向 vs 关节轴：")
for jn, jax, lo, hi, *_ in P.JOINTS:
    o = P.SERVO_ORIENT.get(jn)
    if not o:
        continue
    mm = [round(v * 1000) for v in o]
    if 45 not in mm or 25 not in mm:
        continue
    a45 = "xyz"[mm.index(45)]
    a25 = "xyz"[mm.index(25)]
    normal = [a for a in "xyz" if a not in (a45, a25)][0]
    ok = "OK" if normal == jax else "!! 法向 %s != 关节轴 %s" % (normal, jax)
    print("  %-10s ORIENT=%-14s 45在%s 25在%s -> 法向=%s  关节轴=%s  %s"
          % (jn, str(tuple(mm)), a45, a25, normal, jax, ok))
