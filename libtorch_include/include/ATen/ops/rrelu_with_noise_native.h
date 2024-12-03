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
TORCH_API at::Tensor rrelu_with_noise_cpu(const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower=0.125, const at::Scalar & upper=0.3333333333333333, bool training=false, ::std::optional<at::Generator> generator=::std::nullopt);
TORCH_API at::Tensor & rrelu_with_noise_out_cpu(const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, ::std::optional<at::Generator> generator, at::Tensor & out);
TORCH_API at::Tensor & rrelu_with_noise_cpu_(at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower=0.125, const at::Scalar & upper=0.3333333333333333, bool training=false, ::std::optional<at::Generator> generator=::std::nullopt);
TORCH_API at::Tensor rrelu_with_noise_cuda(const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower=0.125, const at::Scalar & upper=0.3333333333333333, bool training=false, ::std::optional<at::Generator> generator=::std::nullopt);
TORCH_API at::Tensor & rrelu_with_noise_out_cuda(const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, ::std::optional<at::Generator> generator, at::Tensor & out);
TORCH_API at::Tensor & rrelu_with_noise_cuda_(at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower=0.125, const at::Scalar & upper=0.3333333333333333, bool training=false, ::std::optional<at::Generator> generator=::std::nullopt);
} // namespace native
} // namespace at
