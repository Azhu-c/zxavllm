#!/usr/bin/env python3

import argparse
import os
import re
import sys


PROGRESS_RE = re.compile(
    rb"(?P<percent>\d{1,3})%\|[^\r\n]*\|\s*"
    rb"(?P<completed>\d+)/(?P<total>\d+)\s*\["
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pass benchmark output through while emitting tqdm progress events."
    )
    parser.add_argument("--progress-file", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("--total-rounds", required=True, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    progress_fd = os.open(
        args.progress_file,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        0o644,
    )
    parse_buffer = b""
    last_progress = None

    def process_line(line):
        nonlocal last_progress
        matches = list(PROGRESS_RE.finditer(line))
        if not matches:
            return

        match = matches[-1]
        completed = int(match.group("completed"))
        total = int(match.group("total"))
        if total <= 0 or completed < 0 or completed > total:
            return

        progress = (completed, total)
        if progress == last_progress:
            return
        last_progress = progress
        event = (
            f"ROUND_PROGRESS|{args.mode}|{args.round}|{args.total_rounds}|"
            f"{completed}|{total}\n"
        ).encode()
        os.write(progress_fd, event)

    try:
        while True:
            chunk = sys.stdin.buffer.read1(4096)
            if not chunk:
                break

            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

            parse_buffer += chunk
            lines = re.split(rb"[\r\n]", parse_buffer)
            parse_buffer = lines.pop()
            for line in lines:
                process_line(line)

            if len(parse_buffer) > 16384:
                parse_buffer = parse_buffer[-16384:]

        if parse_buffer:
            process_line(parse_buffer)
    finally:
        os.close(progress_fd)


if __name__ == "__main__":
    main()
