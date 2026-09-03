import torch
import triton
from torch import Tensor

from flux_kernels.kernels.rms_norm import _rms_norm_fwd

def triton_rms_norm(
  x: Tensor, weight: Tensor, eps: float = 1e-6
) -> Tensor:
  """
  Args:
    x (torch.Tensor): Input shape (*, D)
    weight (torch.Tensor): Shape (D,)
    eps (float): epislon to prevent division by zero

  Returns:
    torch.Tensor: Normalized tensor with shape matching x
  """

  if not x.is_cuda or not weight.is_cuda:
    raise ValueError("Triton RMSNorm inputs must be CUDA tensors.")

  x = x.contiguous()
  weight = weight.contiguous()

  shape = x.shape
  x_2d = x.view(-1, shape[-1])
  n_rows, n_cols = x_2d.shape

  if weight.numel() != n_cols:
    raise ValueError(f"Weight dim {weight.numel()} differs from n_cols {n_cols}.")

  y_2d = torch.empty_like(x_2d)

  BLOCK_SIZE = triton.next_power_of_2(n_cols)
  grid = (n_rows,)

  _rms_norm_fwd[grid](
    x_2d,
    y_2d,
    weight,
    x_2d.stride(0),
    y_2d.stride(0),
    N_COLS=n_cols,
    EPS=eps,
    BLOCK_SIZE=BLOCK_SIZE 
  )

  return y_2d.view(*shape)