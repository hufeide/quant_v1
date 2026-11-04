#!/usr/bin/env python3
"""一键测试：运行全部单元测试并输出摘要

用法：
    python tests/run_all.py
"""
import sys, os, subprocess, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TESTS = [
    ("核心模块（风控/闸门/撮合/组合/冷却期）", "tests/test_core.py"),
    ("形态检测器 + 指标计算", "tests/test_detectors.py"),
]

results = []
for name, path in TESTS:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    r = subprocess.run([sys.executable, path], capture_output=True, text=True)
    elapsed = time.time() - t0
    # 提取 "通过 N 项，失败 M 项" 摘要
    counts = [l.strip() for l in r.stdout.split("\n") if "通过" in l and "项，失败" in l]
    tail = counts[0] if counts else "无摘要"
    status = "✓ 通过" if r.returncode == 0 else "✗ 失败"
    print(f"  结果: {status}  耗时 {elapsed:.1f}s  ({tail})")
    results.append((name, status, tail))

print(f"\n{'='*60}")
print("测试汇总")
print(f"{'='*60}")
for name, status, tail in results:
    print(f"  {status}  {name}  → {tail}")

failed = [r for r in results if "失败" in r[1]]
sys.exit(1 if failed else 0)
