# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone reproduction test for GitHub issue #39734.

Demonstrates the scheduler head-of-line blocking bug and its fix using
minimal mock objects — no torch or GPU required.

Run directly:
    python3 tests/v1/core/test_reproduce_39734.py

To reproduce the BUG (before fix), replace can_ever_fit_full_sequence
with a version that always returns True, and confirm the loop stalls.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

# ---------------------------------------------------------------------------
# Minimal mock objects (no torch/vllm imports)
# ---------------------------------------------------------------------------


class MockBlockPool:
    def __init__(self, num_gpu_blocks: int, initial_free: int | None = None):
        self.num_gpu_blocks = num_gpu_blocks
        # Null block excluded; initially all usable blocks are free
        self._free = initial_free if initial_free is not None else num_gpu_blocks - 1

    def get_num_free_blocks(self) -> int:
        return self._free


class MockCoordinator:
    def __init__(self, block_size: int):
        self.block_size = block_size

    def get_num_blocks_to_allocate(self, num_tokens: int, **_: Any) -> int:
        return math.ceil(num_tokens / self.block_size)


class MockRequest:
    def __init__(self, request_id: str, num_tokens: int):
        self.request_id = request_id
        self.num_tokens = num_tokens
        self.num_computed_tokens = 0
        self.status = "WAITING"

    def is_finished(self) -> bool:
        return self.status in ("FINISHED_ABORTED", "FINISHED_STOPPED")


class MockKVCacheManager:
    """Reproduces the logic of KVCacheManager relevant to issue #39734."""

    def __init__(self, num_gpu_blocks: int, block_size: int, max_model_len: int):
        self.block_pool = MockBlockPool(num_gpu_blocks)
        self.coordinator = MockCoordinator(block_size)
        self.max_model_len = max_model_len

    def _blocks_needed(self, request: MockRequest) -> int:
        full_tokens = min(request.num_tokens, self.max_model_len)
        return self.coordinator.get_num_blocks_to_allocate(
            num_tokens=full_tokens,
            request_id=request.request_id,
        )

    def can_fit_full_sequence(self, request: MockRequest) -> bool:
        """Checks against current free blocks (may fail due to pressure)."""
        return self._blocks_needed(request) <= self.block_pool.get_num_free_blocks()

    def can_ever_fit_full_sequence(self, request: MockRequest) -> bool:
        """Checks against total pool capacity (permanent impossibility check)."""
        return self._blocks_needed(request) <= (self.block_pool.num_gpu_blocks - 1)


# ---------------------------------------------------------------------------
# Two scheduler simulations: BUGGY and FIXED
# ---------------------------------------------------------------------------


def schedule_buggy(
    kv: MockKVCacheManager,
    waiting: deque[MockRequest],
) -> tuple[list[str], list[str]]:
    """Simulate the BUGGY scheduler loop (before fix).

    When can_fit_full_sequence() returns False, just `break` without
    removing the request from the queue.

    Returns (scheduled_ids, finished_ids).
    """
    scheduled: list[str] = []
    finished: list[str] = []

    while waiting:
        req = waiting[0]  # peek

        if not kv.can_fit_full_sequence(req):
            # BUG: break without popping — request stays at head forever
            break

        waiting.popleft()  # pop
        scheduled.append(req.request_id)

    return scheduled, finished


def schedule_fixed(
    kv: MockKVCacheManager,
    waiting: deque[MockRequest],
) -> tuple[list[str], list[str]]:
    """Simulate the FIXED scheduler loop (after fix).

    When can_fit_full_sequence() returns False:
      - If can_ever_fit_full_sequence() is also False → abort immediately.
      - Otherwise → break (memory pressure, retry next step).

    Returns (scheduled_ids, finished_ids).
    """
    scheduled: list[str] = []
    finished: list[str] = []

    while waiting:
        req = waiting[0]  # peek

        if not kv.can_fit_full_sequence(req):
            if not kv.can_ever_fit_full_sequence(req):
                # FIX: abort and continue — unblocks the queue
                waiting.popleft()
                finished.append(req.request_id)
                continue
            break

        waiting.popleft()
        scheduled.append(req.request_id)

    return scheduled, finished


# ---------------------------------------------------------------------------
# Reproduction tests
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 5  # 4 usable × 16 = 64-token capacity
MAX_MODEL_LEN = 512


def test_bug_demonstration():
    """Show that the buggy scheduler blocks on an oversized request."""
    kv = MockKVCacheManager(NUM_GPU_BLOCKS, BLOCK_SIZE, MAX_MODEL_LEN)
    capacity = (NUM_GPU_BLOCKS - 1) * BLOCK_SIZE  # 64 tokens

    oversized = MockRequest("oversized", num_tokens=capacity + BLOCK_SIZE)  # 80
    normal0 = MockRequest("n0", num_tokens=BLOCK_SIZE)
    normal1 = MockRequest("n1", num_tokens=BLOCK_SIZE)

    waiting: deque[MockRequest] = deque([oversized, normal0, normal1])

    for step in range(3):
        scheduled, finished = schedule_buggy(kv, waiting)
        # BUG: every step schedules nothing — completely stalled
        assert scheduled == [], (
            f"Step {step}: expected no scheduling but got {scheduled}"
        )
        assert finished == [], f"Step {step}: nothing should be finished"
        # The oversized request is still at the head — never removed
        assert waiting[0].request_id == "oversized", (
            f"Step {step}: oversized request should still be at head"
        )

    print("BUG confirmed: oversized request blocks queue for all 3 steps")


def test_fix_unblocks_subsequent_requests():
    """Show that the fix aborts the oversized request and allows normal requests."""
    kv = MockKVCacheManager(NUM_GPU_BLOCKS, BLOCK_SIZE, MAX_MODEL_LEN)
    capacity = (NUM_GPU_BLOCKS - 1) * BLOCK_SIZE  # 64 tokens

    oversized = MockRequest("oversized", num_tokens=capacity + BLOCK_SIZE)  # 80
    normal0 = MockRequest("n0", num_tokens=BLOCK_SIZE)
    normal1 = MockRequest("n1", num_tokens=BLOCK_SIZE)

    waiting: deque[MockRequest] = deque([oversized, normal0, normal1])

    scheduled, finished = schedule_fixed(kv, waiting)

    assert "oversized" in finished, "Oversized request should be aborted"
    assert "n0" in scheduled and "n1" in scheduled, (
        f"Normal requests should be scheduled, got: {scheduled}"
    )
    assert len(waiting) == 0, "Queue should be empty after scheduling"

    print("FIX confirmed: oversized request aborted, normal requests scheduled")


def test_fix_multiple_oversized():
    """Multiple oversized requests are each aborted in one scheduling step."""
    kv = MockKVCacheManager(NUM_GPU_BLOCKS, BLOCK_SIZE, MAX_MODEL_LEN)
    capacity = (NUM_GPU_BLOCKS - 1) * BLOCK_SIZE
    oversized_tokens = capacity + BLOCK_SIZE

    waiting: deque[MockRequest] = deque(
        [
            MockRequest("big0", oversized_tokens),
            MockRequest("big1", oversized_tokens),
            MockRequest("big2", oversized_tokens),
            MockRequest("ok", BLOCK_SIZE),
        ]
    )

    scheduled, finished = schedule_fixed(kv, waiting)

    assert set(finished) == {"big0", "big1", "big2"}, (
        f"Expected all oversized aborted, got finished={finished}"
    )
    assert scheduled == ["ok"], f"Expected ok scheduled, got {scheduled}"

    print("FIX confirmed: all oversized requests aborted, normal request scheduled")


def test_fix_memory_pressure_not_aborted():
    """A request that fits in total capacity but not currently should NOT be aborted."""
    # Fill most blocks so only 1 is free, but total capacity = 9 blocks
    num_blocks = 10
    kv = MockKVCacheManager(num_blocks, BLOCK_SIZE, MAX_MODEL_LEN)
    # Simulate memory pressure: only 2 free blocks but 9 total usable
    kv.block_pool._free = 2

    # Request needs 5 blocks — doesn't fit in 2 free but fits in 9 total
    req = MockRequest("pressure_req", num_tokens=5 * BLOCK_SIZE)  # 80 tokens

    assert not kv.can_fit_full_sequence(req), "Should not fit in 2 free blocks"
    assert kv.can_ever_fit_full_sequence(req), "Should fit in 9 total blocks"

    waiting: deque[MockRequest] = deque([req])
    scheduled, finished = schedule_fixed(kv, waiting)

    # Must NOT be aborted — it should stay in queue (break, not continue)
    assert finished == [], "Memory-pressure request should NOT be aborted"
    assert scheduled == [], "Memory-pressure request should not be scheduled now"
    assert req in waiting, "Memory-pressure request should remain in queue"

    print("FIX confirmed: memory-pressure request stays in queue (not aborted)")


def test_capacity_boundary():
    """Request exactly at capacity should be schedulable."""
    kv = MockKVCacheManager(NUM_GPU_BLOCKS, BLOCK_SIZE, MAX_MODEL_LEN)
    capacity = (NUM_GPU_BLOCKS - 1) * BLOCK_SIZE  # 64 tokens

    req = MockRequest("at_capacity", num_tokens=capacity)
    assert kv.can_fit_full_sequence(req), "At-capacity request should fit"
    assert kv.can_ever_fit_full_sequence(req), "At-capacity request should ever fit"

    print("BOUNDARY confirmed: at-capacity request fits")


if __name__ == "__main__":
    tests = [
        test_bug_demonstration,
        test_fix_unblocks_subsequent_requests,
        test_fix_multiple_oversized,
        test_fix_memory_pressure_not_aborted,
        test_capacity_boundary,
    ]
    failures = 0
    for test_fn in tests:
        try:
            test_fn()
        except AssertionError as exc:
            print(f"FAIL {test_fn.__name__}: {exc}")
            failures += 1
        except Exception as exc:
            print(f"ERROR {test_fn.__name__}: {exc}")
            failures += 1

    if failures == 0:
        print(f"\nAll {len(tests)} tests passed.")
    else:
        print(f"\n{failures}/{len(tests)} tests FAILED.")
        raise SystemExit(1)
