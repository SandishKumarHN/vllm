import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import ensure_model_parallel_initialized, get_tp_group
from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace
from .eplb_utils import distributed_run, set_env_vars_and_device
from .test_eplb_execute import (
    create_expert_indices_with_redundancy,
    create_expert_weights,
    create_redundancy_config,
    verify_expert_weights_after_shuffle,
    create_eplb_communicator_or_raise
)

def _stress_test_worker(env, world_size, iterations=50):
    set_env_vars_and_device(env)
    vllm_config = VllmConfig()
    vllm_config.parallel_config.tensor_parallel_size = world_size
    
    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1)
        ep_group_coordinator = get_tp_group()
        ep_group = ep_group_coordinator.cpu_group
        ep_rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{ep_rank}")

        num_layers = 4
        num_local_experts = 8
        total_physical_experts = world_size * num_local_experts
        num_logical_experts = 16
        hidden_sizes = [128, 256] # Larger tensors increase race probability

        for i in range(iterations):
            # Create new random mappings for each iteration
            old_redundancy = create_redundancy_config(num_logical_experts, total_physical_experts)
            old_indices = create_expert_indices_with_redundancy(num_layers, num_logical_experts, total_physical_experts, old_redundancy)
            
            new_redundancy = create_redundancy_config(num_logical_experts, total_physical_experts)
            new_indices = create_expert_indices_with_redundancy(num_layers, num_logical_experts, total_physical_experts, new_redundancy)

            expert_weights = create_expert_weights(num_layers, num_local_experts, hidden_sizes, ep_rank, device, old_indices)
            
            communicator = create_eplb_communicator_or_raise(
                group_coordinator=ep_group_coordinator,
                backend="torch_nccl",
                expert_weights=expert_weights[0],
            )

            # Execute rearrangement
            rearrange_expert_weights_inplace(
                old_indices, new_indices, expert_weights, ep_group, is_profile=False, communicator=communicator
            )

            # Verify correctness
            local_ok = verify_expert_weights_after_shuffle(expert_weights, new_indices, hidden_sizes, ep_rank, num_local_experts)
            
            ok_tensor = torch.tensor([1 if local_ok else 0], device="cuda", dtype=torch.int32)
            torch.distributed.all_reduce(ok_tensor, op=torch.distributed.ReduceOp.MIN)
            assert bool(ok_tensor.item()), f"Race condition caught on iteration {i}!"

@pytest.mark.parametrize("world_size", [2, 4])
def test_eplb_race_condition(world_size):
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    distributed_run(_stress_test_worker, world_size)
