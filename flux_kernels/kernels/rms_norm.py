import torch
import triton
from torch import Tensor
import triton.language as tl

@triton.jit
def _rms_norm_fwd(
  X,
  Y,
  W,
  stride_x_row,
  stride_y_row,
  N_COLS: tl.constexpr,
  EPS: tl.constexpr,
  BLOCK_SIZE: tl.constexpr
):
  row = tl.program_id(0)

  x_row_ptr = X + row * stride_x_row
  y_row_ptr = Y + row * stride_y_row

  # Load row into SRAM
  offs = tl.arange(0, BLOCK_SIZE)
  mask = offs < N_COLS

  x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
  weight = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)

  variance = tl.sum(x * x, axis=0) / N_COLS
  rsqrt = tl.extra.cuda.math.rsqrt(variance + EPS)
  y = x * rsqrt * weight

  tl.store(y_row_ptr + offs, y.to(X.dtype.element_ty), mask=mask)
