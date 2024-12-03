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
#include <optional>



#include <ATen/ops/_embedding_bag_per_sample_weights_backward_ops.h>

namespace at {


// aten::_embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode, int padding_idx=-1) -> Tensor
inline at::Tensor _embedding_bag_per_sample_weights_backward(const at::Tensor & grad, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, int64_t mode, int64_t padding_idx=-1) {
    return at::_ops::_embedding_bag_per_sample_weights_backward::call(grad, weight, indices, offsets, offset2bag, mode, padding_idx);
}

// aten::_embedding_bag_per_sample_weights_backward.out(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode, int padding_idx=-1, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _embedding_bag_per_sample_weights_backward_out(at::Tensor & out, const at::Tensor & grad, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, int64_t mode, int64_t padding_idx=-1) {
    return at::_ops::_embedding_bag_per_sample_weights_backward_out::call(grad, weight, indices, offsets, offset2bag, mode, padding_idx, out);
}
// aten::_embedding_bag_per_sample_weights_backward.out(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode, int padding_idx=-1, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _embedding_bag_per_sample_weights_backward_outf(const at::Tensor & grad, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, int64_t mode, int64_t padding_idx, at::Tensor & out) {
    return at::_ops::_embedding_bag_per_sample_weights_backward_out::call(grad, weight, indices, offsets, offset2bag, mode, padding_idx, out);
}

}
