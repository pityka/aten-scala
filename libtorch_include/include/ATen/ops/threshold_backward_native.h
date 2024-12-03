#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <optional>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>
#include <ATen/ops/threshold_backward_meta.h>

namespace at {
namespace native {
struct TORCH_API structured_threshold_backward_out : public at::meta::structured_threshold_backward {
void impl(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold, const at::Tensor & grad_input);
};
TORCH_API at::Tensor threshold_backwards_nested(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold);
TORCH_API at::Tensor threshold_backward_sparse(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold);
TORCH_API at::Tensor & threshold_backward_sparse_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold, at::Tensor & grad_input);
TORCH_API at::Tensor threshold_backward_sparse_compressed(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold);
TORCH_API at::Tensor & threshold_backward_sparse_compressed_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold, at::Tensor & grad_input);
TORCH_API at::Tensor mkldnn_relu_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold);
} // namespace native
} // namespace at
