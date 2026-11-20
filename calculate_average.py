#!/usr/bin/env python3
"""Average the printed metrics from complete vLLM benchmark rounds."""
import argparse
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
HEADER = re.compile(r"^=+\s*Serving Benchmark Result\s*=+$", re.I)
FOOTER = re.compile(r"^={10,}$")
SUMMARY = re.compile(r"^=+\s*\d+\s*轮结果平均值\s*=+$")
SECTION = re.compile(r"^-{2,}.*-{2,}$")
METRIC = re.compile(r"^([^:]+):\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$")


def parse_reports(text):
    reports = []
    current = None
    incomplete = 0
    for raw in text.splitlines():
        line = ANSI.sub("", raw).strip()
        if SUMMARY.fullmatch(line):
            break  # Ignore a summary previously appended by this utility.
        if HEADER.fullmatch(line):
            if current is not None:
                incomplete += 1
            current = []
        elif current is not None and FOOTER.fullmatch(line):
            if not any(item[0] == "metric" for item in current):
                raise ValueError("发现没有数值指标的结果块")
            reports.append(current)
            current = None
        elif current is not None and SECTION.fullmatch(line):
            current.append(("section", line))
        elif current is not None:
            match = METRIC.fullmatch(line)
            if match:
                value = Decimal(match.group(2))
                if not value.is_finite():
                    raise ValueError("发现非有限数值")
                current.append(("metric", match.group(1).strip(), value))
    if current is not None:
        incomplete += 1
    if incomplete:
        raise ValueError(f"发现 {incomplete} 个不完整的结果块，未计算平均值")
    if not reports:
        raise ValueError("没有找到完整的 Serving Benchmark Result 结果块")
    expected = [(item[0], item[1]) for item in reports[0]]
    for i, report in enumerate(reports[1:], 2):
        if [(item[0], item[1]) for item in report] != expected:
            raise ValueError(f"第 {i} 轮的指标或顺序与第 1 轮不同，未计算平均值")
    if len({(item[0], item[1]) for item in reports[0]}) != len(reports[0]):
        raise ValueError("同一轮出现重复指标或标题，未计算平均值")
    return reports


def render(reports):
    count = Decimal(len(reports))
    lines = [f"========== {len(reports)} 轮结果平均值 =========="]
    lines.append("============ Serving Benchmark Result ============")
    for index, item in enumerate(reports[0]):
        if item[0] == "section":
            lines.append(item[1])
        else:
            value = sum((report[index][2] for report in reports), Decimal(0)) / count
            lines.append(f"{item[1] + ':':<40} {value:.2f}")
    lines.append("==================================================")
    lines.append("说明：每项均为各轮结果的算术平均；百分位数的平均不是合并所有请求后的百分位数。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="计算 TXT 中各轮 Serving Benchmark Result 的平均值")
    parser.add_argument("file", type=Path, help="一个模式的 benchmark 结果 TXT")
    parser.add_argument("--append", action="store_true", help="同时追加到 TXT 文件末尾")
    parser.add_argument("--expected", type=int, help="预期的完整结果轮数")
    args = parser.parse_args()
    try:
        original = args.file.read_text(encoding="utf-8", errors="replace")
        reports = parse_reports(original)
        if args.expected is not None and len(reports) != args.expected:
            raise ValueError(f"找到 {len(reports)} 轮完整结果，预期 {args.expected} 轮，未计算")
        summary = render(reports)
        if args.append:
            if any(SUMMARY.fullmatch(ANSI.sub("", line).strip()) for line in original.splitlines()):
                raise ValueError("文件已包含平均值汇总；为避免重复，未追加")
            with args.file.open("a", encoding="utf-8") as result:
                result.write("\n" + summary + "\n")
        print(summary)
    except (OSError, ValueError, InvalidOperation) as exc:
        print(f"计算失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
