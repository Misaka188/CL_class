#!/usr/bin/env python3
"""
清理 terminal_*.log 中的 tqdm 进度条行，保留有效日志。

用法：
  python clean_terminal_log.py --in logs/terminal_20251217_205719.log \
                               --out logs/terminal_20251217_205719.clean.log
"""
import argparse
import os
import re
import sys
from typing import List


# 识别 tqdm / 进度条的常见行模式
PROGRESS_PATTERNS: List[re.Pattern] = [
    # 如 "  1%|          | 1/118 [00:00<01:28,  1.32it/s]"
    re.compile(r"^\s*\d+%?\|.+it/s\]"),
    # 如 "0it [00:00, ?it/s]" 或 "23it [00:04,  5.70it/s]"
    re.compile(r"^\s*\d+it\s*\[.+it/s\]"),
    # tqdm 残留的 carriage return 行
    re.compile(r"^\s*\r"),
]


def is_progress_line(line: str) -> bool:
    s = line.strip("\n")
    for pat in PROGRESS_PATTERNS:
        if pat.search(s):
            return True
    return False


def clean_file(inp: str, out: str) -> None:
    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    kept = []
    skipped = 0
    for ln in lines:
        if is_progress_line(ln):
            skipped += 1
            continue
        kept.append(ln)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(kept)

    print(f"[done] input={inp}")
    print(f"       output={out}")
    print(f"       kept={len(kept)}, skipped(progress)={skipped}")


def main():
    ap = argparse.ArgumentParser(description="去掉 terminal log 中的 tqdm 进度条行")
    ap.add_argument("--in", dest="inp", required=True, help="输入 log 路径")
    ap.add_argument("--out", dest="out", default=None, help="输出路径（默认 <in>.clean.log）")
    args = ap.parse_args()

    inp = os.path.abspath(args.inp)
    out = os.path.abspath(args.out) if args.out else inp + ".clean.log"

    if not os.path.isfile(inp):
        print(f"[error] 输入文件不存在: {inp}", file=sys.stderr)
        sys.exit(1)

    clean_file(inp, out)


if __name__ == "__main__":
    main()

