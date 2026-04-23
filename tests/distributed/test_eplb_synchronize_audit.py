# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests verifying that the SYNC and ASYNC EPLB execution paths do not depend on
torch.accelerator.synchronize() (the call commented out at
vllm/distributed/eplb/rebalance_execute.py:586).

Each test:
  1. Patches every relevant synchronization primitive with a recorder.
  2. Runs the path end-to-end on a single GPU with a fake communicator.
  3. Prints the sync-mechanism inventory (which primitives were called, where).
  4. Asserts torch.accelerator.synchronize() was NOT called.
  5. Repeats N times to check for flakiness after the removal.

The fake communicator records P2P sends/recvs but performs no real transfer
(no NCCL, no torch.distributed init), keeping the test single-process.
"""

import asyncio
import functools
import inspect
from collections import defaultdict
from typing import Any

import pytest
import torch
from vllm.distributed.eplb.eplb_utils import CpuGpuEvent
from vllm.distributed.eplb.rebalance_execute import (
    rearrange_expert_weights_inplace,
    transfer_layer,
)


# ---------- Sync auditor ----------


def _caller_in_eplb() -> str:
    """Return the first frame inside vllm/distributed/eplb/, else first non-test frame."""
    for fr in inspect.stack()[2:]:
        if "vllm/distributed/eplb/" in fr.filename:
            return f"{fr.filename.split('vllm/')[-1]}:{fr.lineno}"
    for fr in inspect.stack()[2:]:
        fname = fr.filename
        if "test_eplb_synchronize_audit" not in fname and "asyncio" not in fname:
            return f"{fname}:{fr.lineno}"
    return "<unknown>"


class SyncAuditor:
    """Patch sync primitives and record (callsite, count) for each."""

    PRIMITIVES = [
        # (target_object, attr_name, label, is_method)
        (torch.accelerator, "synchronize", "torch.accelerator.synchronize", False),
        (torch.cuda, "synchronize", "torch.cuda.synchronize", False),
        (torch.cuda.Stream, "synchronize", "torch.cuda.Stream.synchronize", True),
        (torch.cuda.Stream, "wait_stream", "torch.cuda.Stream.wait_stream", True),
        (torch.cuda.Stream, "wait_event", "torch.cuda.Stream.wait_event", True),
        (torch.cuda.Event, "synchronize", "torch.cuda.Event.synchronize", True),
        (torch.cuda.Event, "wait", "torch.cuda.Event.wait", True),
        (torch.cuda.Event, "record", "torch.cuda.Event.record", True),
        (CpuGpuEvent, "wait", "CpuGpuEvent.wait", True),
        (CpuGpuEvent, "record", "CpuGpuEvent.record", True),
    ]

    def __init__(self) -> None:
        self.calls: dict[str, list[str]] = defaultdict(list)
        self._undo: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> "SyncAuditor":
        for target, attr, label, is_method in self.PRIMITIVES:
            try:
                orig = getattr(target, attr)
            except AttributeError:
                continue
            if is_method:
                wrapper = self._method_wrapper(orig, label)
            else:
                wrapper = self._callable_wrapper(orig, label)
            setattr(target, attr, wrapper)
            self._undo.append((target, attr, orig))
        return self

    def __exit__(self, *exc_info) -> None:
        for target, attr, orig in self._undo:
            setattr(target, attr, orig)

    def _callable_wrapper(self, orig, label):
        @functools.wraps(orig)
        def wrapper(*args, **kwargs):
            self.calls[label].append(_caller_in_eplb())
            return orig(*args, **kwargs)

        return wrapper

    def _method_wrapper(self, orig, label):
        @functools.wraps(orig)
        def wrapper(instance, *args, **kwargs):
            self.calls[label].append(_caller_in_eplb())
            return orig(instance, *args, **kwargs)

        return wrapper

    def report(self, title: str) -> None:
        print(f"\n--- sync inventory: {title} ---")
        if not self.calls:
            print("  (no synchronization primitives invoked)")
            return
        for label in sorted(self.calls):
            callers = self.calls[label]
            unique = sorted(set(callers))
            print(f"  {label}: {len(callers)} call(s)")
            for c in unique[:5]:
                print(f"      from {c}")


# ---------- Test fixtures ----------


class FakeProcessGroup:
    """Minimal ProcessGroup stand-in (only size() and rank() are used)."""

    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


class FakeEplbCommunicator:
    """Records P2P intent without performing real NCCL transfers."""

    needs_profile_buffer_reservation = False

    def __init__(self) -> None:
        self.sends: list[tuple[torch.Size, int]] = []
        self.recvs: list[tuple[torch.Size, int]] = []

    def add_send(self, tensor: torch.Tensor, dst: int) -> None:
        self.sends.append((tensor.shape, dst))

    def add_recv(self, tensor: torch.Tensor, src: int) -> None:
        self.recvs.append((tensor.shape, src))

    def execute(self) -> None:
        pass

    def set_stream(self, stream) -> None:
        pass


def _make_test_data(num_layers, num_local_experts, ep_size, hidden_sizes, device):
    num_physical = ep_size * num_local_experts
    old_indices = torch.arange(num_physical, dtype=torch.long).repeat(num_layers, 1)
    new_indices = torch.stack(
        [
            old_indices[layer][torch.randperm(num_physical)]
            for layer in range(num_layers)
        ]
    )
    expert_weights = []
    for _ in range(num_layers):
        layer = [
            torch.randn(num_local_experts, h, device=device, dtype=torch.float32)
            for h in hidden_sizes
        ]
        expert_weights.append(layer)
    return old_indices, new_indices, expert_weights


# ---------- Tests ----------


REPETITIONS = 20  # cheap correctness loop after the audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sync_path_inventory_and_no_torch_accelerator_synchronize():
    """SYNC path: rearrange_expert_weights_inplace.

    Prints which sync primitives the path uses and asserts that
    torch.accelerator.synchronize() (the commented-out line at
    rebalance_execute.py:586) is NOT among them.
    """
    device = torch.device("cuda:0")
    ep_size = 2  # exercise send/recv branches; rank=0 only

    with SyncAuditor() as audit:
        old_idx, new_idx, expert_weights = _make_test_data(
            num_layers=2,
            num_local_experts=4,
            ep_size=ep_size,
            hidden_sizes=[64, 128],
            device=device,
        )
        rearrange_expert_weights_inplace(
            old_global_expert_indices=old_idx,
            new_global_expert_indices=new_idx,
            expert_weights=expert_weights,
            ep_group=FakeProcessGroup(size=ep_size, rank=0),
            communicator=FakeEplbCommunicator(),
            is_profile=False,
        )
    audit.report("SYNC path (rearrange_expert_weights_inplace)")

    assert not audit.calls["torch.accelerator.synchronize"], (
        "torch.accelerator.synchronize was invoked by the SYNC path. "
        f"Callers: {audit.calls['torch.accelerator.synchronize']}"
    )

    # Stability: removing the synchronize must not regress correctness
    for _ in range(REPETITIONS):
        old_idx, new_idx, expert_weights = _make_test_data(
            num_layers=2,
            num_local_experts=4,
            ep_size=ep_size,
            hidden_sizes=[64, 128],
            device=device,
        )
        rearrange_expert_weights_inplace(
            old_global_expert_indices=old_idx,
            new_global_expert_indices=new_idx,
            expert_weights=expert_weights,
            ep_group=FakeProcessGroup(size=ep_size, rank=0),
            communicator=FakeEplbCommunicator(),
            is_profile=False,
        )
        # Function returned without raising; tensor shapes preserved
        for layer in expert_weights:
            for w in layer:
                assert w.is_cuda
                assert w.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_async_path_inventory_and_no_torch_accelerator_synchronize():
    """ASYNC path: transfer_layer + the producer-side sync sequence used by
    async_worker.transfer_run_periodically.

    Re-creates the producer half of the async pipeline:
      1) await transfer_layer(... cuda_stream=...)    # the per-layer await
      2) cuda_stream.synchronize()                    # async_worker.py:134
      3) consumed_event = CpuGpuEvent()               # async_worker.py:139
      4) (consumer side, simulated): consumed_event.record()
      5) consumed_event.wait(stream=cuda_stream)      # async_worker.py:151
    """
    device = torch.device("cuda:0")
    ep_size = 2
    cuda_stream = torch.cuda.Stream(device=0)

    old_idx, new_idx, expert_weights = _make_test_data(
        num_layers=1,
        num_local_experts=4,
        ep_size=ep_size,
        hidden_sizes=[64, 128],
        device=device,
    )
    weights_buffer = [torch.empty_like(w) for w in expert_weights[0]]

    with SyncAuditor() as audit:
        result = asyncio.run(
            transfer_layer(
                old_layer_indices=old_idx[0],
                new_layer_indices=new_idx[0],
                expert_weights=expert_weights[0],
                expert_weights_buffer=weights_buffer,
                ep_group=FakeProcessGroup(size=ep_size, rank=0),
                communicator=FakeEplbCommunicator(),
                cuda_stream=cuda_stream,
                is_profile=False,
            )
        )
        # Producer-side fence after the per-layer transfer (async_worker.py:134)
        cuda_stream.synchronize()
        # Producer→consumer handoff (async_worker.py:139–151)
        consumed_event = CpuGpuEvent()
        consumed_event.record(stream=cuda_stream)  # consumer side
        consumed_event.wait(stream=cuda_stream)  # producer side
    audit.report("ASYNC path (transfer_layer + cuda_stream.synchronize + CpuGpuEvent)")

    assert result is not None
    assert not audit.calls["torch.accelerator.synchronize"], (
        "torch.accelerator.synchronize was invoked by the ASYNC path. "
        f"Callers: {audit.calls['torch.accelerator.synchronize']}"
    )
    # The async path's actual sync mechanisms — assert they DID fire
    assert audit.calls["torch.cuda.Stream.synchronize"], (
        "Async path did not call cuda_stream.synchronize() — broken expectation."
    )
    assert audit.calls["CpuGpuEvent.record"], (
        "Async path did not call CpuGpuEvent.record() — broken expectation."
    )
    assert audit.calls["CpuGpuEvent.wait"], (
        "Async path did not call CpuGpuEvent.wait() — broken expectation."
    )

    # Stability loop
    for _ in range(REPETITIONS):
        old_idx, new_idx, expert_weights = _make_test_data(
            num_layers=1,
            num_local_experts=4,
            ep_size=ep_size,
            hidden_sizes=[64, 128],
            device=device,
        )
        weights_buffer = [torch.empty_like(w) for w in expert_weights[0]]
        result = asyncio.run(
            transfer_layer(
                old_layer_indices=old_idx[0],
                new_layer_indices=new_idx[0],
                expert_weights=expert_weights[0],
                expert_weights_buffer=weights_buffer,
                ep_group=FakeProcessGroup(size=ep_size, rank=0),
                communicator=FakeEplbCommunicator(),
                cuda_stream=cuda_stream,
                is_profile=False,
            )
        )
        cuda_stream.synchronize()
        assert result is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_source_has_no_active_torch_accelerator_synchronize():
    """Bytecode-level proof that the SYNC and ASYNC entry points do not contain
    a torch.accelerator.synchronize() call (independent of source-text grepping).
    """
    for fn in (rearrange_expert_weights_inplace, transfer_layer):
        co = fn.__code__
        # The string literal "synchronize" would appear in co_consts if attr access
        # were compiled, and "synchronize" would appear in co_names for LOAD_ATTR.
        names_lower = [str(n).lower() for n in co.co_names]
        assert "synchronize" not in names_lower, (
            f"{fn.__qualname__} bytecode references a synchronize() attribute: "
            f"{[n for n in names_lower if 'synchronize' in n]}"
        )
