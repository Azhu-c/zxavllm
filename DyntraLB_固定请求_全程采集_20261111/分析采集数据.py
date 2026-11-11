#!/usr/bin/env python3
"""解析自动采集得到的 Ascend profiler 数据。"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="解析 profiling 目录下的 Ascend 数据")
    parser.add_argument("path", nargs="?", default="./profiling", help="采集数据目录，默认 ./profiling")
    args = parser.parse_args()
    root = Path(args.path)
    folders = sorted(path for path in root.rglob("*ascend_pt") if path.is_dir())
    if not folders:
        raise SystemExit(f"没有找到 *_ascend_pt 数据目录：{root}")

    try:
        from torch_npu.profiler.profiler import analyse
    except ImportError as exc:
        raise SystemExit("请在采集服务器的 torch_npu 环境运行此脚本。") from exc

    for folder in folders:
        if any(folder.rglob("kernel_details.csv")):
            print(f"已有 kernel 结果，跳过解析: {folder}", flush=True)
        else:
            print(f"正在解析: {folder}", flush=True)
            analyse(str(folder))

    print("\n结果文件：")
    for name in ("kernel_details.csv", "operator_details.csv", "op_statistic.csv", "trace_view.json"):
        for path in sorted(root.rglob(name)):
            print(path)


if __name__ == "__main__":
    main()
