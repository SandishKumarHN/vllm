# Min-Latency GEMM for `o_proj` — implementation spec

**Status:** SPEC ONLY (not yet implemented). Target: DeepSeek-V3 (TP=8) on Hopper (sm_90+).
**Not runnable on the current 2× A100 host** — requires PDL (Programmatic Dependent Launch, sm_90+) and the DeepSeek-V3 model (671B, won't fit in 160 GB).

## Goal

Extend `dsv3_fused_a_gemm` to support the `o_proj` shape `[num_tokens, 2048] @ [2048, 7168]` for decode (num_tokens 1–16), using PDL to overlap kernel launch/exec with the tail of the preceding attention all-reduce.

Expected per-layer savings on TP=8 H100: o_proj GEMM 13.5 → 4.3 µs (−9.2 µs/layer), TPOT 7.55 → 6.99 ms (−7.4%) for 61 DeepSeek-V3 layers.

## Code changes

### 1. Kernel: `csrc/libtorch_stable/dsv3_fused_a_gemm.cu`

Add instantiations and update dispatch:

```cpp
// New template instantiations
template void invokeFusedAGemm<__nv_bfloat16, 2048, 7168, 8>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, int num_tokens,
    cudaStream_t);
template void invokeFusedAGemm<__nv_bfloat16, 2048, 7168, 16>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, int num_tokens,
    cudaStream_t);

void dsv3_fused_a_gemm(torch::stable::Tensor& output,
                       torch::stable::Tensor const& mat_a,
                       torch::stable::Tensor const& mat_b) {
  int const hd_in = mat_a.size(1);
  int const hd_out = mat_b.size(1);

  bool is_qkv_a  = (hd_in == 7168 && hd_out == 2112);
  bool is_o_proj = (hd_in == 2048 && hd_out == 7168);

  STD_TORCH_CHECK(is_qkv_a || is_o_proj,
                  "Unsupported shapes. Supported: (7168, 2112) or (2048, 7168)");

  if (is_qkv_a) {
    if (num_tokens <= 8) invokeFusedAGemm<__nv_bfloat16, 7168, 2112, 8>(...);
    else                 invokeFusedAGemm<__nv_bfloat16, 7168, 2112, 16>(...);
  } else {
    if (num_tokens <= 8) invokeFusedAGemm<__nv_bfloat16, 2048, 7168, 8>(...);
    else                 invokeFusedAGemm<__nv_bfloat16, 2048, 7168, 16>(...);
  }
}
```

### 2. Python wrapper: `vllm/_custom_ops.py`

```python
def dsv3_fused_o_proj(
    output: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
) -> None:
    """DeepSeek V3 fused o_proj GEMM (SM 9.0+, bf16 only, 1-16 tokens).

    Computes output = mat_a @ mat_b.T where:
      mat_a: [num_tokens, 2048] row-major bf16 (hidden states)
      mat_b: [7168, 2048] column-major bf16 (weight transposed)
      output: [num_tokens, 7168] row-major bf16
    """
    torch.ops._C.dsv3_fused_a_gemm(output, mat_a, mat_b)
```

### 3. Model integration: `vllm/model_executor/models/deepseek_v2.py`

New subclass + swap in `DeepseekV2MLAAttention.__init__`:

```python
class DeepSeekV2MinLatencyOProj(RowParallelLinear):
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Fallback for large batches or unsupported shapes
        if input_.shape[0] > 16 or input_.shape[1] != 2048 or self.output_size != 7168:
            return super().forward(input_)

        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        output_parallel = torch.empty(
            (input_parallel.shape[0], self.output_size),
            dtype=input_parallel.dtype, device=input_parallel.device,
        )
        ops.dsv3_fused_o_proj(output_parallel, input_parallel, self.weight.T)

        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel
        return output
```

Replace:
```python
self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim, self.hidden_size, ...)
```
With:
```python
self.o_proj = DeepSeekV2MinLatencyOProj(self.num_heads * self.v_head_dim, self.hidden_size, ...)
```

## Testing

### Unit (`tests/kernels/test_dsv3_fused_a_gemm.py`)
- Correctness: `M ∈ [1, 16]`, `K=2048`, `N=7168`. Compare to `torch.matmul(A, B.T)`. `torch.allclose(out, ref, atol=1e-3, rtol=1e-3)`.
- Rejection: `M=17` falls back; bad shapes (e.g. `K=1024`) raise.

### Integration (`tests/models/test_deepseek_v2.py`)
- Decode: `batch_size=1, seq_len=1`. Compare logits to baseline.
- Prefill: `batch_size=32` to verify the `super().forward()` fallback.

### Manual
```bash
pip install -e .
vllm bench latency --model deepseek-ai/DeepSeek-V3 --tensor-parallel-size 8 \
  --batch-size 1 --input-len 128 --output-len 128 --num-iters 100 --enforce-eager
nsys profile -t cuda,nvtx -o o_proj_profile vllm bench latency ...
# verify fused_a_gemm_kernel launches concurrently with the attention ncclAllReduce
```

## Rollback

Revert the `DeepseekV2MLAAttention.__init__` line back to plain `RowParallelLinear`. Kernel additions can stay in-tree (only invoked when explicitly called).

---

## Notes for this fork (sandish/devvm-build)

- Cannot validate on the current 2× A100 devvm — needs sm_90+ (Hopper) for PDL and ≥640 GB GPU mem for DeepSeek-V3 671B.
- To validate the kernel additions in isolation (without DeepSeek-V3), the unit test in `tests/kernels/test_dsv3_fused_a_gemm.py` runs the bf16 GEMM directly. Even there, `invokeFusedAGemm` may have a sm_90 guard — check before running on A100.
- When migrating to an 8×H100 host, revert `sandish_scripts/00c_apply_devvm_patches.sh apply` (FA3 ON, no need for the cublas LD_PRELOAD trick on a real CUDA install).
