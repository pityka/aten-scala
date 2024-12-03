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
TORCH_API at::Tensor _scaled_mm_cuda(const at::Tensor & self, const at::Tensor & mat2, const at::Tensor & scale_a, const at::Tensor & scale_b, const ::std::optional<at::Tensor> & bias={}, const ::std::optional<at::Tensor> & scale_result={}, ::std::optional<at::ScalarType> out_dtype=::std::nullopt, bool use_fast_accum=false);
TORCH_API at::Tensor & _scaled_mm_out_cuda(const at::Tensor & self, const at::Tensor & mat2, const at::Tensor & scale_a, const at::Tensor & scale_b, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & scale_result, ::std::optional<at::ScalarType> out_dtype, bool use_fast_accum, at::Tensor & out);
} // namespace native
} // namespace at
