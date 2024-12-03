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
#include <ATen/ops/_softmax_backward_data_meta.h>

namespace at {
namespace native {
struct TORCH_API structured_softmax_backward_cpu_out : public at::meta::structured__softmax_backward_data {
void impl(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, const at::Tensor & grad_input);
};
struct TORCH_API structured_softmax_backward_cuda_out : public at::meta::structured__softmax_backward_data {
void impl(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, const at::Tensor & grad_input);
};
TORCH_API at::Tensor nested_softmax_backward(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype);
} // namespace native
} // namespace at
