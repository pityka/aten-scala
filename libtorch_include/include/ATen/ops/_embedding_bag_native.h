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

TORCH_API ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag_cpu(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq=false, int64_t mode=0, bool sparse=false, const c10::optional<at::Tensor> & per_sample_weights={}, bool include_last_offset=false, int64_t padding_idx=-1);
TORCH_API ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag_cuda(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq=false, int64_t mode=0, bool sparse=false, const c10::optional<at::Tensor> & per_sample_weights={}, bool include_last_offset=false, int64_t padding_idx=-1);

} // namespace native
} // namespace at
