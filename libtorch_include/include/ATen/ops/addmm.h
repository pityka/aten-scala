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



#include <ATen/ops/addmm_ops.h>

namespace at {


// aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & addmm_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta=1, const at::Scalar & alpha=1) {
    return at::_ops::addmm_out::call(self, mat1, mat2, beta, alpha, out);
}

// aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & addmm_outf(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    return at::_ops::addmm_out::call(self, mat1, mat2, beta, alpha, out);
}

// aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
TORCH_API inline at::Tensor addmm(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta=1, const at::Scalar & alpha=1) {
    return at::_ops::addmm::call(self, mat1, mat2, beta, alpha);
}

}
