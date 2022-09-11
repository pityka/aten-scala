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



#include <ATen/ops/greater_equal_ops.h>

namespace at {


// aten::greater_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & greater_equal_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other) {
    return at::_ops::greater_equal_Scalar_out::call(self, other, out);
}

// aten::greater_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & greater_equal_outf(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    return at::_ops::greater_equal_Scalar_out::call(self, other, out);
}

// aten::greater_equal.Scalar(Tensor self, Scalar other) -> Tensor
TORCH_API inline at::Tensor greater_equal(const at::Tensor & self, const at::Scalar & other) {
    return at::_ops::greater_equal_Scalar::call(self, other);
}

// aten::greater_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & greater_equal_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
    return at::_ops::greater_equal_Tensor_out::call(self, other, out);
}

// aten::greater_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & greater_equal_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    return at::_ops::greater_equal_Tensor_out::call(self, other, out);
}

// aten::greater_equal.Tensor(Tensor self, Tensor other) -> Tensor
TORCH_API inline at::Tensor greater_equal(const at::Tensor & self, const at::Tensor & other) {
    return at::_ops::greater_equal_Tensor::call(self, other);
}

}
