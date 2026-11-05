"""Optional CUDA dependency, imported only by a live collector kernel call."""
import triton
import triton.language as tl


@triton.jit
def row_moments_kernel(X, Out, WIDTH: tl.constexpr, STRIDE_ROW: tl.constexpr,
                       STRIDE_COL: tl.constexpr, OUT_STRIDE: tl.constexpr,
                       THRESHOLD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    active = columns < WIDTH
    value = tl.load(X + row * STRIDE_ROW + columns * STRIDE_COL,
                    mask=active, other=0).to(tl.float32)
    finite = active & (tl.abs(value) < float("inf"))
    safe = tl.where(finite, value, 0)
    count = tl.sum(finite.to(tl.int32), 0)
    denominator = tl.maximum(count, 1)
    maximum = tl.max(tl.abs(safe), 0)
    scale = tl.where(maximum > 0, maximum, 1.)
    normalized = safe / scale
    mean = scale * (tl.sum(normalized, 0) / denominator)
    rms = scale * tl.sqrt(tl.sum(normalized * normalized, 0) / denominator)
    if WIDTH == 1:
        mean = tl.load(X + row * STRIDE_ROW).to(tl.float32)
        rms = tl.abs(mean)
    tl.store(Out + row * OUT_STRIDE, tl.where(count > 0, mean, float("nan")))
    tl.store(Out + row * OUT_STRIDE + 1, tl.where(count > 0, rms, float("nan")))
    tl.store(Out + row * OUT_STRIDE + 2, tl.where(count > 0, maximum, float("nan")))
    tl.store(Out + row * OUT_STRIDE + 3, WIDTH - count)
    mean_normalized = tl.sum(normalized, 0) / denominator
    variance = tl.maximum(tl.sum(normalized * normalized, 0) / denominator
                          - mean_normalized * mean_normalized, 0.)
    std = scale * tl.sqrt(variance)
    if WIDTH == 1:
        std = 0.
    zeros = tl.sum((finite & (value == 0)).to(tl.int32), 0) / denominator
    outliers = tl.sum((finite & (tl.abs(value) > THRESHOLD * rms)).to(tl.int32), 0) / denominator
    tl.store(Out + row * OUT_STRIDE + 4, tl.where(count > 0, std, float("nan")))
    tl.store(Out + row * OUT_STRIDE + 5, tl.where(count > 0, zeros, float("nan")))
    tl.store(Out + row * OUT_STRIDE + 6, tl.where(count > 0, outliers, float("nan")))
