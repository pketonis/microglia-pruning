## 2026-05-22 - [Autograd In-place Pitfall]
**Learning:** Using in-place operations (like `.mul_()`) on tensors that are required for gradient computation will break PyTorch's autograd. Specifically, if a tensor `A` is multiplied in-place by `B` where `B` requires grad, the backward pass for `B` cannot be computed because the original value of `A` was modified.
**Action:** Always use out-of-place operations (`A * B`) when the operation involves tensors that are part of a differentiable computational graph, especially in training-enabled paths.

## 2026-05-22 - [Optimizing Attention Stats]
**Learning:** `torch.special.entr` is significantly faster and more numerically stable than manual `- (p * p.log())` entropy calculation. `torch.amax` is faster than `torch.max(...)[0]` when indices are not needed. It can also be used for binary entropy by computing `entr(x) + entr(1-x)`.
**Action:** Prefer `torch.special` functions and `amax`/`amin` for performance-critical tensor reductions.
