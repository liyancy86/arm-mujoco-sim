#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从编译后的 MuJoCo 模型里，把「哪些 geom 真的能互碰」算出来。

为什么需要这个脚本：README 里那句"即使打开碰撞也只有 11 个 geom 参与"
原来只有 `contype=0` 一条依据，不够。MuJoCo 的碰撞过滤是四条一起判的
（官方文档 Computation -> Collision detection）：

  1. 掩码相容： (contype_i & conaffinity_j) || (contype_j & conaffinity_i)
  2. 不同 body
  3. 互不为对方的 parent body（world 例外）
  4. 没被 <contact><exclude> 排除；或者反过来被 <contact><pair> 显式指定

所以"contype=0 就不参与碰撞"只在 conaffinity 也是 0 时才成立。
本脚本把四条都算一遍，输出四个**不同**的数：

  ① 掩码活跃（contype 或 conaffinity 非 0）的 geom 数
  ② 掩码相容的 geom 对数（未含显式 pair）
  ③ 四条件全过、真正可能互碰的 geom 对数
  ④ 出现在 ③ 里的 geom 数

"只有 N 个 geom 参与碰撞"这句里的 N，必须说清是 ① 还是 ④。

用法（从仓根）：
    python build_mjcf.py --collision on --out out/arm_collision.xml
    python check_collision_set.py                       # 默认读上面那个
    python check_collision_set.py --xml arm_real.xml     # 读碰撞关闭版做对比

退出码恒为 0：它只报告事实，不做合格判定。
"""
import argparse
import os
import sys

import mujoco
import numpy as np


def name_of(m, objtype, i):
    return mujoco.mj_id2name(m, objtype, i) or "<unnamed%d>" % i


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.path.join("out", "arm_collision.xml"),
                    help="要读的模型（默认 out/arm_collision.xml，即碰撞开启版）")
    args = ap.parse_args()

    path = args.xml if os.path.isabs(args.xml) else os.path.join(here, args.xml)
    if not os.path.exists(path):
        sys.exit("找不到 %s\n先跑：python build_mjcf.py --collision on --out out/arm_collision.xml" % path)

    m = mujoco.MjModel.from_xml_path(path)
    print("模型：%s   ngeom=%d  nbody=%d  npair=%d  nexclude=%d"
          % (os.path.relpath(path, here), m.ngeom, m.nbody, m.npair, m.nexclude))

    n = m.ngeom
    ct = np.array([m.geom_contype[i] for i in range(n)], dtype=int)
    ca = np.array([m.geom_conaffinity[i] for i in range(n)], dtype=int)
    gb = np.array([m.geom_bodyid[i] for i in range(n)], dtype=int)
    par = np.array([m.body_parentid[i] for i in range(m.nbody)], dtype=int)

    pairs = set()
    for i in range(m.npair):
        g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
        pairs.add((min(g1, g2), max(g1, g2)))
    pair_geoms = set()
    for g1, g2 in pairs:
        pair_geoms.add(g1)
        pair_geoms.add(g2)

    excl = set()
    for i in range(m.nexclude):
        s = int(m.exclude_signature[i])
        b1, b2 = s & 0xFFFF, (s >> 16) & 0xFFFF
        excl.add((min(b1, b2), max(b1, b2)))

    GEO = mujoco.mjtObj.mjOBJ_GEOM
    BODY = mujoco.mjtObj.mjOBJ_BODY

    print("=" * 84)
    print("%-3s %-24s %-16s %8s %12s %8s" % ("id", "geom", "body", "contype", "conaffinity", "pair"))
    print("-" * 84)
    for g in range(n):
        print("%-3d %-24s %-16s %8d %12d %8s"
              % (g, name_of(m, GEO, g)[:24], name_of(m, BODY, int(gb[g]))[:16],
                 ct[g], ca[g], "有" if g in pair_geoms else ""))

    active = [g for g in range(n) if ct[g] != 0 or ca[g] != 0]
    only_ct = [g for g in range(n) if ct[g] != 0]

    possible = []
    mask_ok = same_body = parent_skip = excluded = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not ((ct[i] & ca[j]) or (ct[j] & ca[i])):
                continue
            if (i, j) in pairs:
                possible.append((i, j, "pair"))
                continue
            mask_ok += 1
            if gb[i] == gb[j]:
                same_body += 1
                continue
            if par[gb[i]] == gb[j] or par[gb[j]] == gb[i]:
                parent_skip += 1
                continue
            bp = (min(int(gb[i]), int(gb[j])), max(int(gb[i]), int(gb[j])))
            if bp in excl:
                excluded += 1
                continue
            possible.append((i, j, "mask"))

    inv = set()
    for i, j, _ in possible:
        inv.add(i)
        inv.add(j)

    print()
    print("-" * 84)
    print("① 掩码活跃（contype!=0 或 conaffinity!=0）= %d 个；其中 contype!=0 的 = %d 个"
          % (len(active), len(only_ct)))
    print("② 掩码相容的 geom 对 = %d 对（不含显式 pair）" % mask_ok)
    print("     同 body 被排除     = %d 对" % same_body)
    print("     父子 body 被排除   = %d 对" % parent_skip)
    print("     被 exclude 排除    = %d 对" % excluded)
    print("③ 四条件全过、可能互碰的 geom 对 = %d 对（含显式 pair %d 个）"
          % (len(possible), len(pairs)))
    print("④ 出现在 ③ 里的 geom = %d 个" % len(inv))

    left = sorted(set(active) - inv)
    if left:
        print()
        print("   掩码活跃却配不出任何一个对（被同 body / 父子 / exclude 全部吃掉）：")
        for g in left:
            print("     %-24s  body=%s" % (name_of(m, GEO, g), name_of(m, BODY, int(gb[g]))))

    print()
    print("口径提醒：说「只有 N 个 geom 参与碰撞」时，N 必须指明是 ① 还是 ④。")
    print("          两个数不一样，而且都不是「③ 的对数」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
