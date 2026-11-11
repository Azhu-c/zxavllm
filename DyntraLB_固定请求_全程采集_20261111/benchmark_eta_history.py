#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import statistics
import tempfile


MODES = ("bl", "lb", "dynam")
HISTORY_VERSION = 1
DEFAULT_HISTORY_WINDOW = 5
MAX_HISTORY_SAMPLES = 20


def profile_config(args):
    return {
        "dataset_name": args.dataset_name,
        "num_prompts": args.num_prompts,
        "max_concurrency": args.max_concurrency,
        "request_rate": args.request_rate,
        "disable_shuffle": args.disable_shuffle,
        "initial_startup_delay": args.initial_startup_delay,
        "ready_settle_delay": args.ready_settle_delay,
    }


def profile_key(config):
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def empty_history():
    return {"version": HISTORY_VERSION, "profiles": {}}


def load_history(path):
    if not os.path.exists(path):
        return empty_history()
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return empty_history()
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
        return empty_history()
    return data


def save_history(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".benchmark_eta_history.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_profile(history, config, create=False):
    key = profile_key(config)
    profiles = history["profiles"]
    if key not in profiles and create:
        profiles[key] = {
            "config": config,
            "base_startup_durations": [],
            "modes": {
                mode: {
                    "round_durations": [],
                    "decoder_startup_durations": [],
                }
                for mode in MODES
            },
        }
    return profiles.get(key)


def recent_median(samples, window=DEFAULT_HISTORY_WINDOW):
    valid = [
        int(value)
        for value in samples
        if isinstance(value, (int, float)) and value >= 0
    ]
    if not valid:
        return 0
    return int(round(statistics.median(valid[-window:])))


def record_sample(history, config, metric, value, mode=None):
    profile = get_profile(history, config, create=True)
    if metric == "base_startup":
        samples = profile["base_startup_durations"]
    else:
        if mode not in MODES:
            raise ValueError("mode is required for round and decoder_startup")
        mode_history = profile["modes"].setdefault(
            mode,
            {
                "round_durations": [],
                "decoder_startup_durations": [],
            },
        )
        if metric == "round":
            samples = mode_history["round_durations"]
        elif metric == "decoder_startup":
            samples = mode_history["decoder_startup_durations"]
        else:
            raise ValueError("unsupported metric: {}".format(metric))
    samples.append(int(value))
    del samples[:-MAX_HISTORY_SAMPLES]
    return recent_median(samples), len(samples)


def parse_mode_state(value):
    fields = value.split(":")
    if len(fields) != 13:
        raise ValueError("invalid mode state: {}".format(value))
    mode = fields[0]
    if mode not in MODES:
        raise ValueError("invalid mode in state: {}".format(mode))
    numbers = [int(field) for field in fields[1:]]
    return {
        "mode": mode,
        "completed": numbers[0],
        "round_start": numbers[1],
        "round_requests": numbers[2],
        "round_requests_total": numbers[3],
        "decoder_started": numbers[4],
        "decoder_start": numbers[5],
        "startup_done": numbers[6],
        "drain_start": numbers[7],
        "done": numbers[8],
        "gap_start": numbers[9],
        "gap_duration": numbers[10],
        "selected": numbers[11],
    }


def mode_history_estimates(profile, mode):
    if not profile:
        return 0, 0, 0, 0
    mode_history = profile.get("modes", {}).get(mode, {})
    round_samples = mode_history.get("round_durations", [])
    startup_samples = mode_history.get("decoder_startup_durations", [])
    return (
        recent_median(round_samples),
        len(round_samples),
        recent_median(startup_samples),
        len(startup_samples),
    )


def projected_round_duration(state, historical_duration, now, min_progress):
    if state["round_start"] <= 0:
        return historical_duration, False
    elapsed = max(now - state["round_start"], 0)
    completed = state["round_requests"]
    total = state["round_requests_total"]
    if total <= 0 or completed <= 0:
        return historical_duration, False
    progress = completed / total
    if progress < min_progress:
        return historical_duration, False
    progress_duration = max(int(round(elapsed / progress)), elapsed)
    if historical_duration <= 0:
        return progress_duration, True
    blended = int(
        round(
            historical_duration * (1.0 - progress)
            + progress_duration * progress
        )
    )
    return max(blended, elapsed), True


def estimate_remaining(args, history):
    config = profile_config(args)
    profile = get_profile(history, config)
    states = [parse_mode_state(value) for value in args.mode_state]
    states = [state for state in states if state["selected"]]
    if not states:
        return None

    historical_round_estimates = {}
    current_round_estimates = {}
    round_counts = {}
    startup_estimates = {}
    startup_counts = {}
    for state in states:
        mode = state["mode"]
        historical_round, round_count, startup_estimate, startup_count = (
            mode_history_estimates(profile, mode)
        )
        current_round, _ = projected_round_duration(
            state,
            historical_round,
            args.now,
            args.min_progress,
        )
        historical_round_estimates[mode] = historical_round
        current_round_estimates[mode] = current_round
        round_counts[mode] = round_count
        startup_estimates[mode] = startup_estimate
        startup_counts[mode] = startup_count

    known_rounds = [
        historical_round_estimates[mode] or current_round_estimates[mode]
        for mode in historical_round_estimates
        if historical_round_estimates[mode] > 0
        or current_round_estimates[mode] > 0
    ]
    fallback_round = (
        int(round(statistics.median(known_rounds))) if known_rounds else 0
    )
    known_startups = [
        value for value in startup_estimates.values() if value > 0
    ]
    fallback_startup = (
        int(round(statistics.median(known_startups))) if known_startups else 0
    )

    remaining = 0
    fallback_used = False
    missing_estimate = False

    if not args.base_startup_done:
        base_samples = profile.get("base_startup_durations", []) if profile else []
        base_estimate = recent_median(base_samples)
        if base_estimate > 0:
            remaining += max(
                base_estimate - (args.now - args.script_start),
                0,
            )
        else:
            missing_estimate = True

    for state in states:
        if state["done"]:
            continue
        mode = state["mode"]
        startup_estimate = startup_estimates[mode] or fallback_startup
        if startup_estimates[mode] == 0 and fallback_startup > 0:
            fallback_used = True

        if not state["decoder_started"]:
            if startup_estimate > 0:
                remaining += startup_estimate
            else:
                missing_estimate = True
        elif not state["startup_done"]:
            if startup_estimate > 0:
                startup_elapsed = max(args.now - state["decoder_start"], 0)
                remaining += max(startup_estimate - startup_elapsed, 0)
            else:
                missing_estimate = True

        mode_round_estimate = historical_round_estimates[mode] or fallback_round
        if historical_round_estimates[mode] == 0 and fallback_round > 0:
            fallback_used = True

        completed = min(max(state["completed"], 0), args.num_rounds)
        active_round = state["round_start"] > 0
        active_gap = state["gap_start"] > 0

        if active_round:
            current_estimate = current_round_estimates[mode] or fallback_round
            if current_estimate > 0:
                elapsed = max(args.now - state["round_start"], 0)
                remaining += max(current_estimate - elapsed, 0)
            else:
                missing_estimate = True
            future_rounds = max(args.num_rounds - completed - 1, 0)
            future_gaps = max(args.num_rounds - completed - 1, 0)
        elif active_gap:
            gap_elapsed = max(args.now - state["gap_start"], 0)
            remaining += max(state["gap_duration"] - gap_elapsed, 0)
            future_rounds = max(args.num_rounds - completed, 0)
            future_gaps = max(args.num_rounds - completed - 1, 0)
        else:
            future_rounds = max(args.num_rounds - completed, 0)
            future_gaps = max(args.num_rounds - completed - 1, 0)
            if completed == 0:
                future_gaps = max(args.num_rounds - 1, 0)

        if future_rounds > 0:
            if mode_round_estimate > 0:
                remaining += future_rounds * mode_round_estimate
            else:
                missing_estimate = True
        remaining += future_gaps * args.round_interval_delay

        if state["drain_start"] > 0:
            drain_elapsed = max(args.now - state["drain_start"], 0)
            remaining += max(args.drain_delay - drain_elapsed, 0)
        else:
            remaining += args.drain_delay

    transitions_left = max(
        args.transition_total - args.transition_completed,
        0,
    )
    if args.transition_start > 0 and transitions_left > 0:
        transition_elapsed = max(args.now - args.transition_start, 0)
        remaining += max(
            args.resource_release_delay - transition_elapsed,
            0,
        )
        transitions_left -= 1
    remaining += transitions_left * args.resource_release_delay

    if not known_rounds:
        return None

    all_modes_well_sampled = all(
        round_counts[state["mode"]] >= 3
        and startup_counts[state["mode"]] >= 1
        for state in states
    )
    if (
        missing_estimate
        or fallback_used
        or any(round_counts[state["mode"]] == 0 for state in states)
    ):
        confidence = "低"
    elif all_modes_well_sampled:
        confidence = "高"
    else:
        confidence = "中"
    return {
        "remaining": max(int(remaining), 0),
        "eta_epoch": args.now + max(int(remaining), 0),
        "confidence": confidence,
    }


def add_profile_args(parser):
    parser.add_argument("--history-file", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--num-prompts", required=True, type=int)
    parser.add_argument("--max-concurrency", required=True, type=int)
    parser.add_argument("--request-rate", required=True)
    parser.add_argument("--disable-shuffle", required=True)
    parser.add_argument("--initial-startup-delay", required=True, type=int)
    parser.add_argument("--ready-settle-delay", required=True, type=int)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Persist benchmark timing history and estimate pipeline ETA."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record")
    add_profile_args(record_parser)
    record_parser.add_argument(
        "--metric",
        required=True,
        choices=("base_startup", "decoder_startup", "round"),
    )
    record_parser.add_argument("--mode", choices=MODES)
    record_parser.add_argument("--value", required=True, type=int)

    estimate_parser = subparsers.add_parser("estimate")
    add_profile_args(estimate_parser)
    estimate_parser.add_argument("--now", required=True, type=int)
    estimate_parser.add_argument("--script-start", required=True, type=int)
    estimate_parser.add_argument("--base-startup-done", required=True, type=int)
    estimate_parser.add_argument("--num-rounds", required=True, type=int)
    estimate_parser.add_argument("--round-interval-delay", required=True, type=int)
    estimate_parser.add_argument("--drain-delay", required=True, type=int)
    estimate_parser.add_argument(
        "--resource-release-delay",
        required=True,
        type=int,
    )
    estimate_parser.add_argument("--transition-total", required=True, type=int)
    estimate_parser.add_argument(
        "--transition-completed",
        required=True,
        type=int,
    )
    estimate_parser.add_argument("--transition-start", required=True, type=int)
    estimate_parser.add_argument(
        "--min-progress",
        type=float,
        default=0.10,
    )
    estimate_parser.add_argument(
        "--mode-state",
        action="append",
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = profile_config(args)
    history = load_history(args.history_file)

    if args.command == "record":
        median, count = record_sample(
            history,
            config,
            args.metric,
            args.value,
            args.mode,
        )
        save_history(args.history_file, history)
        print("{}|{}".format(median, count))
        return

    estimate = estimate_remaining(args, history)
    if estimate is None:
        print("0|0|低")
        return
    print(
        "{}|{}|{}".format(
            estimate["eta_epoch"],
            estimate["remaining"],
            estimate["confidence"],
        )
    )


if __name__ == "__main__":
    main()

