#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API ::std::tuple<at::Tensor,::std::vector<at::Tensor>> histogramdd(const at::Tensor & self, at::IntArrayRef bins, c10::optional<at::ArrayRef<double>> range=c10::nullopt, const c10::optional<at::Tensor> & weight={}, bool density=false);
TORCH_API ::std::tuple<at::Tensor,::std::vector<at::Tensor>> histogramdd(const at::Tensor & self, int64_t bins, c10::optional<at::ArrayRef<double>> range=c10::nullopt, const c10::optional<at::Tensor> & weight={}, bool density=false);
TORCH_API ::std::tuple<at::Tensor,::std::vector<at::Tensor>> histogramdd(const at::Tensor & self, at::TensorList bins, c10::optional<at::ArrayRef<double>> range=c10::nullopt, const c10::optional<at::Tensor> & weight={}, bool density=false);
} // namespace native
} // namespace at
