# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only DyntraLB planner; no torch or vLLM runtime is required."""

from collections.abc import Sequence
from typing import Literal, TypedDict


class _RequestItem(TypedDict):
    blk_num: int
    newly_added: bool
    slot: int


class _CardData(TypedDict):
    card_idx: int
    running: list[_RequestItem]
    waiting: list[_RequestItem]
    tot_blk: int


class _Modification(TypedDict):
    out_blk: list[int]
    in_blk: list[int]
    freeze: bool


class IndexedModification(_Modification, total=False):
    out_slots: list[int]
    in_slots: list[int]


def balance_load(
    requests_by_rank: Sequence[tuple[Sequence[int], int]],
    dev_num: int,
    max_num_seqs: int | None = None,
    max_iters: int = 64,
    threshold: float = 5,
    *,
    min_throughput_gain: float = 0.02,
    step_latency_fixed_us: float = 20000.0,
    step_latency_per_block_us: float = 19.2,
    with_request_indices: bool = False,
    max_pauses_per_rank: int = 1,
    cp_min_blocks: int | None = None,
    cp_max_pairs: int = 1,
) -> list[IndexedModification]:
    if dev_num <= 0:
        raise ValueError("dev_num must be positive")
    modifications: list[IndexedModification] = [{"out_blk": [], "in_blk": [], "freeze": False} for _ in range(dev_num)]

    def calc_bubble(cards_info: Sequence[_CardData]) -> float:
        tot_blks = [c["tot_blk"] for c in cards_info]
        max_blk = max(tot_blks)
        avg_blk = sum(tot_blks) / dev_num
        if threshold < 1:
            return (max_blk - avg_blk) / max_blk if max_blk > 0 else 0.0
        return max_blk - avg_blk

    def perform_swap(
        run_item: _RequestItem,
        wait_item: _RequestItem,
        card: _CardData,
    ) -> None:
        card["tot_blk"] += wait_item["blk_num"] - run_item["blk_num"]
        card["running"].remove(run_item)
        card["waiting"].remove(wait_item)
        card["running"].append(wait_item)
        card["waiting"].append(run_item)

    def perform_pull_in(
        wait_item: _RequestItem,
        card: _CardData,
    ) -> None:
        card["tot_blk"] += wait_item["blk_num"]
        card["waiting"].remove(wait_item)
        card["running"].append(wait_item)

    def fill_running_from_waiting(card: _CardData) -> bool:
        if max_num_seqs is None:
            return False
        pulled_any = False
        while len(card["running"]) < max_num_seqs and card["waiting"]:
            item = card["waiting"][0]
            item["newly_added"] = True
            perform_pull_in(item, card)
            pulled_any = True
        return pulled_any

    def find_best_swap(
        card: _CardData,
        mode: Literal["increase", "decrease"],
    ) -> tuple[_RequestItem, _RequestItem] | None:
        target_avg = sum(c["tot_blk"] for c in cards_data) / dev_num
        best_swap: tuple[_RequestItem, _RequestItem] | None = None
        best_dist = abs(card["tot_blk"] - target_avg)
        best_newly_added = False
        originals = original_running[card["card_idx"]]
        selected = {item["slot"] for item in card["running"]}
        paused_count = len(originals.keys() - selected)
        old_max = max(c["tot_blk"] for c in cards_data)
        other_max = max((c["tot_blk"] for c in cards_data if c is not card), default=0)
        old_latency = step_latency_fixed_us + step_latency_per_block_us * old_max

        for run_item in card["running"]:
            if (card["card_idx"], run_item["slot"]) in cp_protected:
                continue
            for wait_item in card["waiting"]:
                new_paused_count = paused_count + (run_item["slot"] in originals) - (wait_item["slot"] in originals)
                if new_paused_count > max_pauses_per_rank:
                    continue
                delta = wait_item["blk_num"] - run_item["blk_num"]
                if mode == "increase" and delta <= 0:
                    continue
                if mode == "decrease" and delta >= 0:
                    continue
                new_tot = card["tot_blk"] + delta
                # Reducing distance to the average is insufficient: increasing
                # the global maximum makes every decoding request slower.
                new_max = max(new_tot, other_max)
                if new_max > old_max:
                    continue
                if not run_item["newly_added"]:
                    new_latency = step_latency_fixed_us + step_latency_per_block_us * new_max
                    if old_latency / new_latency - 1 < min_throughput_gain:
                        continue
                new_avg = target_avg + delta / dev_num
                new_dist = abs(new_tot - new_avg)
                run_newly_added = run_item.get("newly_added", False)
                if new_dist < best_dist or (new_dist == best_dist and run_newly_added and not best_newly_added):
                    best_dist = new_dist
                    best_swap = (run_item, wait_item)
                    best_newly_added = run_newly_added
        return best_swap

    def try_drop_for_throughput(max_card: _CardData) -> bool:
        card_idx = max_card["card_idx"]
        total_reqs = sum(len(c["running"]) for c in cards_data)
        if total_reqs <= 1:
            return False

        max_blk = max_card["tot_blk"]
        second_max_blk = max([c["tot_blk"] for c in cards_data if c["card_idx"] != card_idx] + [0])
        current_latency = step_latency_fixed_us + step_latency_per_block_us * max_blk
        current_tp = total_reqs / current_latency
        best_drop: _RequestItem | None = None
        best_tp = current_tp * (1 + min_throughput_gain)
        best_newly_added = False
        originals = original_running[card_idx]
        selected = {item["slot"] for item in max_card["running"]}
        paused_count = len(originals.keys() - selected)

        for run_item in max_card["running"]:
            if (card_idx, run_item["slot"]) in cp_protected:
                continue
            if paused_count + (run_item["slot"] in originals) > max_pauses_per_rank:
                continue
            new_max_blk = max(
                max_blk - run_item["blk_num"],
                second_max_blk,
            )
            new_latency = step_latency_fixed_us + step_latency_per_block_us * new_max_blk
            new_tp = (total_reqs - 1) / new_latency
            run_newly_added = run_item.get("newly_added", False)
            if new_tp > best_tp or (new_tp == best_tp and run_newly_added and not best_newly_added):
                best_tp = new_tp
                best_drop = run_item
                best_newly_added = run_newly_added

        if best_drop:
            max_card["tot_blk"] -= best_drop["blk_num"]
            max_card["running"].remove(best_drop)
            max_card["waiting"].append(best_drop)
            return True
        return False

    cards_data: list[_CardData] = []
    for rank in range(dev_num):
        blks, split = requests_by_rank[rank] if rank < len(requests_by_rank) else ([], 0)
        running: list[_RequestItem] = []
        waiting: list[_RequestItem] = []
        tot_run_blk = 0
        for index, blk_num in enumerate(blks):
            if blk_num == 0:
                break
            req_obj: _RequestItem = {
                "blk_num": blk_num,
                "newly_added": False,
                "slot": index,
            }
            if index < split:
                running.append(req_obj)
                tot_run_blk += blk_num
            else:
                waiting.append(req_obj)
        cards_data.append(
            {
                "card_idx": rank,
                "running": running,
                "waiting": waiting,
                "tot_blk": tot_run_blk,
            }
        )

    original_running = [{item["slot"]: item["blk_num"] for item in c["running"]} for c in cards_data]
    original_waiting = [{item["slot"]: item["blk_num"] for item in c["waiting"]} for c in cards_data]
    for card in cards_data:
        fill_running_from_waiting(card)

    # Let the worker-side CP path run long requests when spare DP workers
    # exist, instead of pausing exactly the work those helpers could split.
    cp_protected: set[tuple[int, int]] = set()
    if cp_min_blocks is not None:
        spare = sum(not card["running"] for card in cards_data)
        used_owners: set[int] = set()
        candidates = sorted(
            (
                (item["blk_num"], card["card_idx"], item["slot"])
                for card in cards_data
                for item in card["running"]
                if item["blk_num"] >= cp_min_blocks
            ),
            reverse=True,
        )
        for _, rank, slot in candidates:
            if len(cp_protected) >= min(spare, cp_max_pairs):
                break
            if rank not in used_owners:
                cp_protected.add((rank, slot))
                used_owners.add(rank)

    iteration = 0
    while iteration < max_iters:
        cards_data.sort(key=lambda x: x["tot_blk"], reverse=True)
        if calc_bubble(cards_data) < threshold:
            break

        swapped_this_round = False
        dropped_this_round = False
        min_card = cards_data[-1]
        best_min_swap = find_best_swap(min_card, mode="increase")
        if best_min_swap:
            perform_swap(best_min_swap[0], best_min_swap[1], min_card)
            swapped_this_round = True

        if swapped_this_round:
            if calc_bubble(cards_data) < threshold:
                break
            cards_data.sort(key=lambda x: x["tot_blk"], reverse=True)

        max_card = cards_data[0]
        best_max_swap = find_best_swap(max_card, mode="decrease")
        if best_max_swap:
            perform_swap(best_max_swap[0], best_max_swap[1], max_card)
            swapped_this_round = True

        if not swapped_this_round and try_drop_for_throughput(max_card):
            dropped_this_round = True

        if calc_bubble(cards_data) < threshold:
            break
        if not swapped_this_round and not dropped_this_round:
            break
        iteration += 1

    for card in cards_data:
        rank = card["card_idx"]
        # Derive the final diff by identity. Equal block counts do not identify
        # equal requests, and intermediate swaps can cancel each other.
        selected = {item["slot"] for item in card["running"]}
        out_slots = [slot for slot in original_running[rank] if slot not in selected]
        in_slots = [slot for slot in original_waiting[rank] if slot in selected]
        mod = modifications[rank]
        mod["out_blk"] = [original_running[rank][slot] for slot in out_slots]
        mod["in_blk"] = [original_waiting[rank][slot] for slot in in_slots]
        mod["freeze"] = not in_slots
        if with_request_indices:
            mod["out_slots"] = out_slots
            mod["in_slots"] = in_slots

    return modifications
