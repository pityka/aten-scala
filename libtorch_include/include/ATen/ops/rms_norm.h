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



#include <ATen/ops/rms_norm_ops.h>

namespace at {


// aten::rms_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor
inline at::Tensor rms_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight={}, ::std::optional<double> eps=::std::nullopt) {
    return at::_ops::rms_norm::call(input, normalized_shape, weight, eps);
}

}