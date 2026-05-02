# Bolt's Journal

## 2026-05-21 - [Optimizing activation statistics with PyTorch primitives]
**Learning:** Using `torch.special.entr` and `torch.amax` instead of manual implementations significantly reduces the latency of statistics computation. `torch.special.entr` is faster and more numerically stable than adding an epsilon to avoid `log(0)`. `torch.amax` is faster than `.max()[0]` because it doesn't compute or return indices.
**Action:** Always prefer specialized PyTorch primitives like `torch.special.entr` and `torch.amax` for performance-critical tensor operations.
