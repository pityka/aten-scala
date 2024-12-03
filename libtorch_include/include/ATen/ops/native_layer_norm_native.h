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
TORCH_API ::std::tuple<at::Tensor,at::Tensor,at::Tensor> math_native_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps);
TORCH_API ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_layer_norm_out_symint(const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2);
TORCH_API ::std::tuple<at::Tensor,at::Tensor,at::Tensor> layer_norm_cpu(const at::Tensor & input, at::IntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps);
TORCH_API ::std::tuple<at::Tensor,at::Tensor,at::Tensor> layer_norm_cuda(const at::Tensor & input, at::IntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps);
TORCH_API ::std::tuple<at::Tensor,at::Tensor,at::Tensor> nested_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps);
} // namespace native
} // namespace at
