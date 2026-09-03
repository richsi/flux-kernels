import pytest 
import torch 
from flux_kernels import rms_norm, rms_norm_ref

@pytest.mark.parametrize("batch_size, seq_len, dim", [
  (1, 4608, 128),
  (2, 4096, 3072),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm(batch_size, seq_len, dim, dtype):
  torch.manual_seed(42)
  x = torch.randn((batch_size, seq_len, dim), device="cuda", dtype=dtype)
  weight = torch.randn((dim,), device="cuda", dtype=dtype)

  torch_out = rms_norm_ref(x, weight)
  triton_out = rms_norm(x, weight)

  torch.testing.assert_close(torch_out, triton_out, rtol=1e-2, atol=1e-2)