# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compact, lossless encoding of a DyntraLB rank's planning snapshot."""


def encode_snapshot(blocks: list[int], running_count: int, pending_long: bool) -> bytes:
    if not 0 <= running_count <= len(blocks):
        raise ValueError("running count exceeds snapshot length")
    data = bytearray()
    for value in (running_count, int(pending_long), *blocks):
        if value < 0:
            raise ValueError("snapshot values must be nonnegative")
        while value >= 128:
            data.append((value & 127) | 128)
            value >>= 7
        data.append(value)
    return bytes(data)


def decode_snapshot(data: bytes) -> tuple[list[int], int, bool]:
    values: list[int] = []
    value = shift = 0
    for byte in data:
        value |= (byte & 127) << shift
        if byte & 128:
            shift += 7
            if shift > 63:
                raise ValueError("snapshot varint is too large")
        else:
            values.append(value)
            value = shift = 0
    if shift or len(values) < 2 or values[1] not in (0, 1):
        raise ValueError("invalid snapshot")
    running_count, pending_long = values[:2]
    blocks = values[2:]
    if running_count > len(blocks):
        raise ValueError("running count exceeds snapshot length")
    return blocks, running_count, bool(pending_long)
