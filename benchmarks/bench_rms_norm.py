import torch
import triton
from flux_kernels import rms_norm_ref, rms_norm

def bench_rms_norm():
  x = torch.randn((1, 4608, 128), deivce="cuda", dtype=torch.bfloat16)
  weight = torch.randn((128,), device="cuda", dtype=torch.bfloat16)

  ms_torch = triton.testing.do_bench(lambda: rms_norm_ref(x, weight))
  ms_triton = triton.testing.do_bench(lambda: rms_norm(x, weight))

  total_bytes = 2 * x.numel() * x.element_size() # 1 read + 1 write
  gbps_torch = (total_bytes * 1e-9) / (ms_torch * 1e-3)
  gbps_triton = (total_bytes * 1e-9) / (ms_triton * 1e-3)

  print(f"PyTorch Time: {ms_torch*1000:.2f} us | Bandwidth: {gbps_torch:.2f} GB/s")
  print(f"Triton Time:  {ms_triton*1000:.2f} us | Bandwidth: {gbps_triton:.2f} GB/s")
  print(f"Speedup:      {ms_torch / ms_triton:.2f}x")

if __name__ == "__main__":
    bench_rms_norm()