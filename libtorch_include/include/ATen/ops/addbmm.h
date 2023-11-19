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



#include <ATen/ops/addbmm_ops.h>

namespace at {


// aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & addbmm_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta=1, const at::Scalar & alpha=1) {
    return at::_ops::addbmm_out::call(self, batch1, batch2, beta, alpha, out);
}
// aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & addbmm_outf(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    return at::_ops::addbmm_out::call(self, batch1, batch2, beta, alpha, out);
}

// aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
inline at::Tensor addbmm(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta=1, const at::Scalar & alpha=1) {
    return at::_ops::addbmm::call(self, batch1, batch2, beta, alpha);
}

}