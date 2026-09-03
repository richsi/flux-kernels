import torch
import torch.Tensor as Tensor
import triton
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


def triton_rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
  shape = x.shape
  x_2d = x.view(-1, shape[-1])
  n_rows, n_cols = x_2d.shape

  y_2d = torch.empty_like(x_2d)

  BLOCK_SIZE = triton.next_power_of_2(n_cols)
  grid = (n_rows,)

  _rms_norm_fwd[grid](
    x_2d, y_2d, weight,
    x_2d.stride(0), y_2d.stride(0),
    N_COLS=n_cols,
    EPS=eps,
    BLOCK_SIZE=BLOCK_SIZE
  )
  return y_2d.view(*shape)