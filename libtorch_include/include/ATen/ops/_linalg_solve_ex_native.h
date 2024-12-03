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
#include <ATen/ops/_linalg_solve_ex_meta.h>

namespace at {
namespace native {
struct TORCH_API structured__linalg_solve_ex_out : public at::meta::structured__linalg_solve_ex {
void impl(const at::Tensor & A, const at::Tensor & B, bool left, bool check_errors, const at::Tensor & result, const at::Tensor & LU, const at::Tensor & pivots, const at::Tensor & info);
};
} // namespace native
} // namespace at
