#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/_new_zeros_with_same_feature_meta_ops.h>

namespace at {


// aten::_new_zeros_with_same_feature_meta(Tensor self, Tensor other, *, int self_num_batch_dims=0) -> Tensor
TORCH_API inline at::Tensor _new_zeros_with_same_feature_meta(const at::Tensor & self, const at::Tensor & other, int64_t self_num_batch_dims=0) {
    return at::_ops::_new_zeros_with_same_feature_meta::call(self, other, self_num_batch_dims);
}

}
