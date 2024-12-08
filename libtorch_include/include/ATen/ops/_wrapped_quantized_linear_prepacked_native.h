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


namespace at {
namespace native {
TORCH_API at::Tensor _wrapped_quantized_linear_prepacked(const at::Tensor & input, const at::Tensor & input_scale, const at::Tensor & input_zero_point, const at::Tensor & packed_weight, const at::Tensor & output_scale, const at::Tensor & output_zero_point, int64_t out_channel);
} // namespace native
} // namespace at