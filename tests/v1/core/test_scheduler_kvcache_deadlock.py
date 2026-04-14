# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for scheduler head-of-line blocking fix when a request exceeds the
total KV cache capacity (GitHub issue #39734).

When scheduler_reserve_full_isl is enabled (the default), the scheduler checks
whether the full sequence fits in the KV cache before scheduling a request.
If a request is so large that it can NEVER fit — even when the cache is empty —
leaving it at the head of the waiting queue permanently blocks all subsequent
requests (head-of-line blocking / deadlock).

The fix: detect such requests and abort them immediately so later requests
can be scheduled.
"""

import math

import pytest

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
BLOCK_SIZE = 16
# num_gpu_blocks=5 → null_block takes 1 → 4 usable blocks × 16 = 64 token cap.
NUM_BLOCKS = 5
# max_model_len is larger than the KV cache capacity so the oversized request
# passes initial admission control but is rejected during scheduling.
MAX_MODEL_LEN = 512


def _kv_capacity_tokens(num_blocks: int, block_size: int) -> int:
    """Total KV token capacity (excludes null block)."""
    return (num_blocks - 1) * block_size


def _blocks_needed(num_tokens: int, block_size: int) -> int:
    return math.ceil(num_tokens / block_size)


# ---------------------------------------------------------------------------
# Unit tests for KVCacheManager.can_ever_fit_full_sequence
# ---------------------------------------------------------------------------


def test_can_ever_fit_returns_true_for_small_request():
    """A request well within capacity should always fit."""
    scheduler = create_scheduler(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
    )
    kv = scheduler.kv_cache_manager
    capacity = _kv_capacity_tokens(NUM_BLOCKS, BLOCK_SIZE)  # 64
    # Request using half the capacity
    [req] = create_requests(
        num_requests=1, num_tokens=capacity // 2, block_size=BLOCK_SIZE
    )
    assert kv.can_ever_fit_full_sequence(req)


def test_can_ever_fit_returns_false_for_oversized_request():
    """A request exceeding total capacity should never be able to fit."""
    scheduler = create_scheduler(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
    )
    kv = scheduler.kv_cache_manager
    capacity = _kv_capacity_tokens(NUM_BLOCKS, BLOCK_SIZE)  # 64
    # Request 50% larger than total capacity
    oversized_tokens = int(capacity * 1.5)
    [req] = create_requests(
        num_requests=1, num_tokens=oversized_tokens, block_size=BLOCK_SIZE
    )
    assert not kv.can_ever_fit_full_sequence(req)


def test_can_fit_full_sequence_vs_can_ever_fit():
    """Both methods agree when comparing against capacity boundaries.

    On an empty cache, can_fit_full_sequence and can_ever_fit_full_sequence
    return the same answer: a request within capacity returns True, a request
    exceeding capacity returns False.  The methods only diverge when some blocks
    are currently occupied (memory pressure): then can_fit_full_sequence may
    return False while can_ever_fit_full_sequence still returns True.

    This test covers the boundary cases on an empty cache.
    """
    scheduler = create_scheduler(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
    )
    kv = scheduler.kv_cache_manager
    capacity = _kv_capacity_tokens(NUM_BLOCKS, BLOCK_SIZE)  # 64 tokens

    # Exactly at capacity: both methods return True.
    [at_cap_req] = create_requests(
        num_requests=1, num_tokens=capacity, block_size=BLOCK_SIZE
    )
    assert kv.can_fit_full_sequence(at_cap_req)
    assert kv.can_ever_fit_full_sequence(at_cap_req)

    # One block over: both methods return False (truly unserviceable).
    over_tokens = capacity + BLOCK_SIZE
    [over_req] = create_requests(
        num_requests=1, num_tokens=over_tokens, block_size=BLOCK_SIZE
    )
    assert not kv.can_fit_full_sequence(over_req)
    assert not kv.can_ever_fit_full_sequence(over_req)


# ---------------------------------------------------------------------------
# Scheduler-level deadlock / abort tests
# ---------------------------------------------------------------------------


def test_oversized_request_is_aborted_not_stuck():
    """An oversized request must be aborted, not left at the head of the queue.

    Without the fix, can_fit_full_sequence returns False, break exits the
    scheduling loop, and the oversized request is never popped.  On every
    subsequent call to schedule() the same outcome repeats — permanent deadlock.

    With the fix, can_ever_fit_full_sequence also returns False so the request
    is immediately aborted (FINISHED_ABORTED), freeing the queue.
    """
    scheduler = create_scheduler(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
    )
    capacity = _kv_capacity_tokens(NUM_BLOCKS, BLOCK_SIZE)  # 64 tokens
    oversized_tokens = capacity + BLOCK_SIZE  # 80 > 64

    [oversized] = create_requests(
        num_requests=1,
        num_tokens=oversized_tokens,
        block_size=BLOCK_SIZE,
        req_ids=["oversized"],
    )
    scheduler.add_request(oversized)

    output = scheduler.schedule()

    # The oversized request must have been aborted during scheduling.
    assert "oversized" in output.finished_req_ids, (
        "Oversized request was not aborted; scheduler may be deadlocked."
    )
    # No requests should have been scheduled (the only request was aborted).
    assert len(output.scheduled_new_reqs) == 0
    # The waiting queue must now be empty (request was removed).
    assert len(scheduler.waiting) == 0


def test_oversized_request_unblocks_subsequent_normal_requests():
    """After aborting the oversized request, subsequent normal requests proceed.

    This is the core regression scenario from issue #39734: a single oversized
    request at the head of the queue was preventing ALL later requests from
    being served.
    """
    scheduler = create_scheduler(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
    )
    capacity = _kv_capacity_tokens(NUM_BLOCKS, BLOCK_SIZE)  # 64 tokens
    oversized_tokens = capacity + BLOCK_SIZE  # 80 > 64
    normal_tokens = BLOCK_SIZE  # 16 — one block, easily fits

    # The oversized request arrives first (head-of-line position).
    [oversized] = create_requests(
        num_requests=1,
        num_tokens=oversized_tokens,
        block_size=BLOCK_SIZE,
        req_ids=["oversized"],
    )
    normal_reqs = create_requests(
        num_requests=3,
        num_tokens=normal_tokens,
        block_size=BLOCK_SIZE,
        req_ids=["n0", "n1", "n2"],
    )

    scheduler.add_request(oversized)
    for req in normal_reqs:
        scheduler.add_request(req)

    output = scheduler.schedule()

    # Oversized request must be aborted.
    assert "oversized" in output.finished_req_ids

    # All three normal requests must be scheduled (they fit in the cache).
    scheduled_ids = {r.request_id for r in output.scheduled_new_reqs}
    assert scheduled_ids == {"n0", "n1", "n2"}, (
        f"Expected normal requests to be scheduled, got: {scheduled_ids}"
    )


def test_multiple_oversized_requests_all_aborted():
    """Multiple back-to-back oversized requests are each aborted in turn."""
    scheduler = create_scheduler(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
    )
    capacity = _kv_capacity_tokens(NUM_BLOCKS, BLOCK_SIZE)  # 64 tokens
    oversized_tokens = capacity + BLOCK_SIZE

    oversized_reqs = create_requests(
        num_requests=3,
        num_tokens=oversized_tokens,
        block_size=BLOCK_SIZE,
        req_ids=["big0", "big1", "big2"],
    )
    for req in oversized_reqs:
        scheduler.add_request(req)

    output = scheduler.schedule()

    # All three must be aborted in one scheduling step.
    assert output.finished_req_ids == {"big0", "big1", "big2"}, (
        f"Not all oversized requests aborted: finished={output.finished_req_ids}"
    )
    assert len(scheduler.waiting) == 0


def test_exactly_at_capacity_is_not_aborted():
    """A request exactly at the KV cache limit should be schedulable, not aborted."""
    scheduler = create_scheduler(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
    )
    # Exactly fills all usable blocks (4 blocks × 16 = 64 tokens)
    capacity = _kv_capacity_tokens(NUM_BLOCKS, BLOCK_SIZE)

    [req] = create_requests(
        num_requests=1, num_tokens=capacity, block_size=BLOCK_SIZE, req_ids=["exact"]
    )
    scheduler.add_request(req)

    output = scheduler.schedule()

    # Must be scheduled, not aborted.
    assert "exact" not in output.finished_req_ids, (
        "Request at exact capacity was incorrectly aborted."
    )
    scheduled_ids = {r.request_id for r in output.scheduled_new_reqs}
    assert "exact" in scheduled_ids


def test_schedule_called_multiple_times_no_deadlock():
    """Repeated schedule() calls with an oversized request must not deadlock.

    Pre-fix: each call would re-inspect the oversized request, get False from
    can_fit_full_sequence, break, and return with zero scheduled tokens — the
    server would appear to have stalled.

    Post-fix: first call aborts the request; subsequent calls find an empty
    queue and return normally.
    """
    scheduler = create_scheduler(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
    )
    capacity = _kv_capacity_tokens(NUM_BLOCKS, BLOCK_SIZE)
    oversized_tokens = capacity + BLOCK_SIZE

    [oversized] = create_requests(
        num_requests=1,
        num_tokens=oversized_tokens,
        block_size=BLOCK_SIZE,
        req_ids=["big"],
    )
    scheduler.add_request(oversized)

    # First call should abort the request.
    output1 = scheduler.schedule()
    assert "big" in output1.finished_req_ids

    # Second and third calls: queue is empty, no hang, no error.
    output2 = scheduler.schedule()
    assert len(output2.scheduled_new_reqs) == 0

    output3 = scheduler.schedule()
    assert len(output3.scheduled_new_reqs) == 0


def test_normal_requests_not_affected_without_overflow():
    """Baseline: normal-sized requests schedule fine with no overflow requests."""
    scheduler = create_scheduler(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
    )
    normal_tokens = BLOCK_SIZE  # 16 tokens — one block per request

    reqs = create_requests(
        num_requests=3, num_tokens=normal_tokens, block_size=BLOCK_SIZE
    )
    for req in reqs:
        scheduler.add_request(req)

    output = scheduler.schedule()

    assert len(output.finished_req_ids) == 0
    assert len(output.scheduled_new_reqs) == 3
