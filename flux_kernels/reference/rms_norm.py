import torch
from torch import Tensor

def torch_rms_norm(
    x: Tensor, weight: Tensor, eps: float = 1e-6
) -> Tensor:
  """Computes RMSNorm using PyTorch"""

  variance = torch.mean(x**2, dim=-1, keepdim=True)
  rsqrt = torch.rsqrt(variance + eps)
  return x * rsqrt * weight